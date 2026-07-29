"""The supported core must not depend on ``impulse.experimental``.

``impulse.experimental`` is versioned separately from the rest of the package
(see ``impulse/experimental/__init__.py``). That separation is only real if the
dependency runs one way: experimental may import core, core may not import
experimental. These tests pin that, because a stray convenience import would
re-couple them silently and nothing else would fail.
"""

import ast
import pathlib
import subprocess
import sys

CORE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "impulse"


def _core_modules():
    """Every .py file in the package except the experimental subpackage."""
    return [
        p for p in CORE_ROOT.rglob("*.py") if "experimental" not in p.relative_to(CORE_ROOT).parts
    ]


class TestCoreDoesNotImportExperimental:
    def test_no_experimental_module_loaded_by_importing_impulse(self):
        """``import impulse`` must not pull in the experimental package.

        Run in a subprocess: this test module's siblings import experimental,
        so an in-process check would see their leftovers in sys.modules.
        """
        code = (
            "import sys, impulse; "
            "leaked = sorted(m for m in sys.modules if 'impulse.experimental' in m); "
            "print(','.join(leaked))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        leaked = [m for m in out.stdout.strip().split(",") if m]
        assert leaked == [], f"`import impulse` loaded experimental modules: {leaked}"

    def test_no_module_level_import_of_experimental_in_core(self):
        """Core may import experimental only lazily, inside a function body.

        The one sanctioned use is the legacy birth/death checkpoint migration
        in ``_pt_base``: it must repair a product-space ``PTSampler`` resumed
        from a pre-fix checkpoint, so the dispatch has to live in core. It is
        deliberately written as a function-local import guarded by a proposal
        name check, so importing core never touches experimental.
        """
        offenders = []
        for path in _core_modules():
            tree = ast.parse(path.read_text(), filename=str(path))

            # walk only module-level statements, plus class bodies (which also
            # execute at import time); function bodies are the lazy escape.
            def module_level(node, depth=0):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                        continue  # function bodies run lazily -- allowed
                    yield child
                    yield from module_level(child, depth + 1)

            for node in module_level(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("impulse.experimental"):
                        offenders.append(f"{path.name}:{node.lineno} from {node.module}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("impulse.experimental"):
                            offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
        assert offenders == [], (
            "core imported impulse.experimental at module level: "
            + "; ".join(offenders)
            + ". Move it inside the function that needs it."
        )

    def test_moved_names_are_not_re_exported_from_the_top_level(self):
        """The 2.0 move was a hard break: no top-level compatibility aliases."""
        import impulse

        for name in ("HybridPTSampler", "BirthDeathProductSpace", "load_hybrid_checkpoint"):
            assert not hasattr(impulse, name), (
                f"impulse.{name} still exists; it moved to impulse.experimental "
                "and the move was documented as a hard break"
            )
            assert name not in impulse.__all__

    def test_moved_names_are_importable_from_experimental(self):
        from impulse.experimental import (  # noqa: F401
            BirthDeathProductSpace,
            HybridPTSampler,
            load_hybrid_checkpoint,
            make_product_space_sampler,
        )

    def test_pt_sampler_has_no_product_space_constructor(self):
        """``from_product_space`` moved off the supported sampler."""
        from impulse import PTSampler

        assert not hasattr(PTSampler, "from_product_space")
