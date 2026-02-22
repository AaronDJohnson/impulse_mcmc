import pytest
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path

from impulse.utils import prepare_files, shift_array


class TestPrepareFiles:
    """Test suite for prepare_files function"""

    def test_prepare_files_new_files(self, temp_dir):
        """Test creating new files when they don't exist"""
        filepaths = [
            os.path.join(temp_dir, "test1.txt"),
            os.path.join(temp_dir, "subdir", "test2.txt")
        ]
        
        prepare_files(filepaths, resume=False)
        
        # Check that all files exist
        for filepath in filepaths:
            assert os.path.exists(filepath)
            assert os.path.isfile(filepath)
            # Check files are empty
            assert os.path.getsize(filepath) == 0

    def test_prepare_files_overwrite_existing(self, temp_dir):
        """Test overwriting existing files when resume=False"""
        filepath = os.path.join(temp_dir, "existing.txt")
        
        # Create file with content
        with open(filepath, 'w') as f:
            f.write("existing content")
        
        original_size = os.path.getsize(filepath)
        assert original_size > 0
        
        prepare_files([filepath], resume=False)
        
        # File should exist but be empty
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) == 0

    def test_prepare_files_resume_keeps_existing(self, temp_dir):
        """Test keeping existing files when resume=True"""
        filepath = os.path.join(temp_dir, "existing.txt")
        content = "existing content"
        
        # Create file with content
        with open(filepath, 'w') as f:
            f.write(content)
        
        prepare_files([filepath], resume=True)
        
        # File should exist with original content
        assert os.path.exists(filepath)
        with open(filepath, 'r') as f:
            assert f.read() == content

    def test_prepare_files_creates_directories(self, temp_dir):
        """Test that parent directories are created"""
        nested_path = os.path.join(temp_dir, "deep", "nested", "path", "file.txt")
        
        prepare_files([nested_path], resume=False)
        
        assert os.path.exists(nested_path)
        assert os.path.exists(os.path.dirname(nested_path))

    def test_prepare_files_empty_list(self, temp_dir):
        """Test with empty filepath list"""
        prepare_files([], resume=False)
        # Should not raise any errors

    def test_prepare_files_mixed_existing_new(self, temp_dir):
        """Test with mix of existing and new files"""
        existing_file = os.path.join(temp_dir, "existing.txt")
        new_file = os.path.join(temp_dir, "new.txt")
        
        # Create one existing file
        with open(existing_file, 'w') as f:
            f.write("content")
        
        prepare_files([existing_file, new_file], resume=False)
        
        # Both should exist and be empty
        assert os.path.exists(existing_file)
        assert os.path.exists(new_file)
        assert os.path.getsize(existing_file) == 0
        assert os.path.getsize(new_file) == 0


class TestShiftArray:
    """Test suite for shift_array function"""

    def test_shift_right_positive(self):
        """Test shifting array to the right (positive num)"""
        arr = np.array([1, 2, 3, 4, 5])
        result = shift_array(arr, 2)
        expected = np.array([0, 0, 1, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_shift_left_negative(self):
        """Test shifting array to the left (negative num)"""
        arr = np.array([1, 2, 3, 4, 5])
        result = shift_array(arr, -2)
        expected = np.array([3, 4, 5, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_shift_zero(self):
        """Test no shift (num=0)"""
        arr = np.array([1, 2, 3, 4, 5])
        result = shift_array(arr, 0)
        np.testing.assert_array_equal(result, arr)

    def test_shift_with_custom_fill_value(self):
        """Test shifting with custom fill value"""
        arr = np.array([1, 2, 3, 4, 5])
        result = shift_array(arr, 2, fill_value=-1)
        expected = np.array([-1, -1, 1, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_shift_2d_array(self):
        """Test shifting 2D array"""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        result = shift_array(arr, 1)
        expected = np.array([[0, 0], [1, 2], [3, 4]])
        np.testing.assert_array_equal(result, expected)

    def test_shift_float_array(self):
        """Test shifting with float array"""
        arr = np.array([1.5, 2.7, 3.9])
        result = shift_array(arr, 1, fill_value=np.nan)
        expected = np.array([np.nan, 1.5, 2.7])
        # Use allclose for NaN comparison
        assert np.isnan(result[0]) and np.isnan(expected[0])
        np.testing.assert_array_equal(result[1:], expected[1:])

    def test_shift_full_length_right(self):
        """Test shifting by full array length to the right"""
        arr = np.array([1, 2, 3])
        result = shift_array(arr, 3)
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_shift_full_length_left(self):
        """Test shifting by full array length to the left"""
        arr = np.array([1, 2, 3])
        result = shift_array(arr, -3)
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_shift_beyond_length(self):
        """Test shifting by more than array length"""
        arr = np.array([1, 2, 3])
        result = shift_array(arr, 5)
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_shift_preserves_dtype(self):
        """Test that shifting preserves array dtype"""
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = shift_array(arr, 1)
        assert result.dtype == arr.dtype

    def test_shift_empty_array(self):
        """Test shifting empty array"""
        arr = np.array([])
        result = shift_array(arr, 1)
        np.testing.assert_array_equal(result, arr)

    def test_shift_single_element(self):
        """Test shifting single element array"""
        arr = np.array([42])
        result_right = shift_array(arr, 1)
        result_left = shift_array(arr, -1)
        result_zero = shift_array(arr, 0)
        
        np.testing.assert_array_equal(result_right, np.array([0]))
        np.testing.assert_array_equal(result_left, np.array([0]))
        np.testing.assert_array_equal(result_zero, np.array([42]))