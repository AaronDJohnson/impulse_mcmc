"""On-disk chain encodings: binary and text must be interchangeable.

The binary encoding is the default, so the properties that matter are (a) it
stores exactly what text stores, and (b) every operation the resume path needs
-- append, count, truncate, read -- behaves identically under both.
"""

import os

import numpy as np
import pytest

from impulse import PTSampler, chain_io

NCOLS = 6


def _rows(n, seed=0):
    return np.random.default_rng(seed).normal(size=(n, NCOLS))


class TestRoundTrip:
    @pytest.mark.parametrize("fmt", ["binary", "text"])
    def test_append_then_read_is_exact(self, tmp_path, fmt):
        """Both encodings are lossless for float64: %.18e round-trips exactly."""
        path = str(tmp_path / ("c" + chain_io.chain_suffix(fmt)))
        rows = _rows(50)
        assert chain_io.append_rows(path, rows, fmt) == 50
        back = chain_io.read_rows(path, NCOLS, fmt)
        assert back.shape == rows.shape
        np.testing.assert_array_equal(back, rows)

    @pytest.mark.parametrize("fmt", ["binary", "text"])
    def test_appends_accumulate(self, tmp_path, fmt):
        path = str(tmp_path / ("c" + chain_io.chain_suffix(fmt)))
        a, b = _rows(10, 1), _rows(7, 2)
        chain_io.append_rows(path, a, fmt)
        chain_io.append_rows(path, b, fmt)
        assert chain_io.count_rows(path, NCOLS, fmt) == 17
        np.testing.assert_array_equal(chain_io.read_rows(path, NCOLS, fmt), np.vstack([a, b]))

    @pytest.mark.parametrize("fmt", ["binary", "text"])
    def test_empty_append_is_a_noop(self, tmp_path, fmt):
        path = str(tmp_path / ("c" + chain_io.chain_suffix(fmt)))
        assert chain_io.append_rows(path, np.empty((0, NCOLS)), fmt) == 0
        assert chain_io.count_rows(path, NCOLS, fmt) == 0

    @pytest.mark.parametrize("fmt", ["binary", "text"])
    def test_count_rows_on_missing_file_is_zero(self, tmp_path, fmt):
        path = str(tmp_path / ("nope" + chain_io.chain_suffix(fmt)))
        assert chain_io.count_rows(path, NCOLS, fmt) == 0

    def test_binary_stores_exactly_the_expected_bytes(self, tmp_path):
        """No header, no padding: the file IS nrows*ncols*8 bytes."""
        path = str(tmp_path / "c.bin")
        chain_io.append_rows(path, _rows(13), "binary")
        assert os.path.getsize(path) == 13 * NCOLS * 8

    def test_binary_and_text_agree_value_for_value(self, tmp_path):
        rows = _rows(40, 3)
        pb, pt = str(tmp_path / "c.bin"), str(tmp_path / "c.txt")
        chain_io.append_rows(pb, rows, "binary")
        chain_io.append_rows(pt, rows, "text")
        np.testing.assert_array_equal(
            chain_io.read_rows(pb, NCOLS, "binary"),
            chain_io.read_rows(pt, NCOLS, "text"),
        )


class TestTruncate:
    @pytest.mark.parametrize("fmt", ["binary", "text"])
    def test_truncate_keeps_the_prefix(self, tmp_path, fmt):
        """This is what makes an interrupted resume bit-exact."""
        path = str(tmp_path / ("c" + chain_io.chain_suffix(fmt)))
        rows = _rows(30, 4)
        chain_io.append_rows(path, rows, fmt)
        chain_io.truncate_rows(path, 12, NCOLS, fmt)
        assert chain_io.count_rows(path, NCOLS, fmt) == 12
        np.testing.assert_array_equal(chain_io.read_rows(path, NCOLS, fmt), rows[:12])

    @pytest.mark.parametrize("fmt", ["binary", "text"])
    def test_truncate_beyond_length_is_a_noop(self, tmp_path, fmt):
        path = str(tmp_path / ("c" + chain_io.chain_suffix(fmt)))
        chain_io.append_rows(path, _rows(5), fmt)
        chain_io.truncate_rows(path, 99, NCOLS, fmt)
        assert chain_io.count_rows(path, NCOLS, fmt) == 5

    @pytest.mark.parametrize("fmt", ["binary", "text"])
    def test_truncate_missing_file_is_a_noop(self, tmp_path, fmt):
        chain_io.truncate_rows(str(tmp_path / ("gone" + chain_io.chain_suffix(fmt))), 3, NCOLS, fmt)


class TestTornWrites:
    def test_partial_binary_row_is_dropped_not_misread(self, tmp_path):
        """A crash mid-write leaves a partial row; it must not shift the data."""
        path = str(tmp_path / "c.bin")
        rows = _rows(10, 5)
        chain_io.append_rows(path, rows, "binary")
        with open(path, "ab") as fp:  # half a row
            fp.write(np.zeros(NCOLS // 2).tobytes())
        assert chain_io.count_rows(path, NCOLS, "binary") == 10
        np.testing.assert_array_equal(chain_io.read_rows(path, NCOLS, "binary"), rows)


class TestFormatSelection:
    def test_detect_format_finds_either(self, tmp_path):
        base = str(tmp_path / "chain_0")
        assert chain_io.detect_format(base) is None
        chain_io.append_rows(base + ".txt", _rows(2), "text")
        assert chain_io.detect_format(base) == "text"
        chain_io.append_rows(base + ".bin", _rows(2), "binary")
        assert chain_io.detect_format(base) == "binary"

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="chain_format must be one of"):
            chain_io.validate_format("hdf5")

    def test_sampler_rejects_unknown_format(self):
        with pytest.raises(ValueError, match="chain_format must be one of"):
            PTSampler(
                ndim=2,
                lnlike=lambda x: 0.0,
                lnprior=lambda x: 0.0,
                ntemps=2,
                chain_format="parquet",
            )


class TestSamplerIntegration:
    @staticmethod
    def _ll(x):
        return -0.5 * float(np.sum(np.asarray(x) ** 2))

    @staticmethod
    def _lp(x):
        return 0.0 if np.all(np.abs(np.asarray(x)) < 10) else -np.inf

    def _run(self, outdir, fmt, iters=400, resume=False):
        s = PTSampler(
            ndim=3,
            lnlike=self._ll,
            lnprior=self._lp,
            ntemps=4,
            seed=7,
            outdir=outdir,
            save_freq=100,
            verbose=False,
            chain_format=fmt,
            resume=resume,
        )
        s.sample(np.zeros(3), num_iterations=iters)
        return s

    def test_encoding_does_not_change_the_chain(self, tmp_path):
        """The two encodings must be numerically indistinguishable."""
        a = self._run(str(tmp_path / "b"), "binary").load_chain()
        b = self._run(str(tmp_path / "t"), "text").load_chain()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    @pytest.mark.parametrize("fmt", ["binary", "text"])
    def test_load_chain_reads_what_was_written(self, tmp_path, fmt):
        outdir = str(tmp_path / fmt)
        s = self._run(outdir, fmt)
        suffix = chain_io.chain_suffix(fmt)
        assert os.path.exists(os.path.join(outdir, "chain_0" + suffix))
        assert s.load_chain()["samples"].shape == (4, 400, 3)

    @pytest.mark.parametrize("fmt", ["binary", "text"])
    def test_resume_is_bit_exact_in_both_encodings(self, tmp_path, fmt):
        full = str(tmp_path / f"full_{fmt}")
        split = str(tmp_path / f"split_{fmt}")
        self._run(full, fmt, iters=400)
        self._run(split, fmt, iters=200)
        self._run(split, fmt, iters=400, resume=True)
        a = self._run(full, fmt, iters=400, resume=True).load_chain()
        b = PTSampler(
            ndim=3,
            lnlike=self._ll,
            lnprior=self._lp,
            ntemps=4,
            seed=7,
            outdir=split,
            save_freq=100,
            verbose=False,
            chain_format=fmt,
        ).load_chain()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    def test_load_chain_reads_a_text_run_from_a_default_sampler(self, tmp_path):
        """Encoding is discovered from disk, not assumed from the constructor."""
        outdir = str(tmp_path / "legacy")
        self._run(outdir, "text")
        default = PTSampler(
            ndim=3,
            lnlike=self._ll,
            lnprior=self._lp,
            ntemps=4,
            seed=7,
            outdir=outdir,
            verbose=False,  # chain_format defaults to binary
        )
        assert default.load_chain()["samples"].shape == (4, 400, 3)
