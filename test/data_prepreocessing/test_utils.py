import datetime as dt
import pytest
from src.data_preprocessing.utils import (
    is_path_excluded,
    is_path_included,
    extract_turbine_name,
    extract_timestamp
)

class TestIsPathExcluded:
    def test_filter_files_with_excluded_words(self):
        # Test cases where excluded words are in the path
        assert is_path_excluded("/path/to/Trash/file.txt") is True
        assert is_path_excluded("/Trash/file.txt") is True
        assert is_path_excluded("/path/TrashFolder/file.txt") is True
        assert is_path_excluded("/path/to/TrashBin/file.txt", exclude_words=["Trash", "Bin"]) is True

    def test_filter_files_without_excluded_words(self):
        # Test cases where excluded words are not in the path
        assert is_path_excluded("/path/to/file.txt") is False
        assert is_path_excluded("/path/to/trash/file.txt", exclude_words=["TrashBin"]) is False


class TestIsPathIncluded:
    def test_include_files_with_included_words(self):
        # Test cases where included words are in the path
        assert is_path_included("/path/to/Important/file.txt", include_words=["Important"]) is True
        assert is_path_included("/path/to/Critical/file.txt", include_words=["Critical", "Important"]) is True

    def test_include_files_without_included_words(self):
        # Test cases where included words are not in the path
        assert is_path_included("/path/to/file.txt", include_words=["Important", "Critical"]) is False
        assert is_path_included("/path/to/file.txt") is True  # Default behavior includes all files


class TestExtractTurbineName:
    def test_extract_valid_turbine_names(self):
        # Test cases where turbine name should be extracted
        assert extract_turbine_name("/NRT_TurbineA/data/") == "NRT_TurbineA"
        assert extract_turbine_name("/path/to/NRT_TurbineB/info/") == "NRT_TurbineB"
        assert extract_turbine_name("/path/NRT123/") == "NRT123"
        assert extract_turbine_name("/multiple/paths/NRT_TurbineC/info/") == "NRT_TurbineC"
        assert extract_turbine_name("/path/to/nrt_TurbineX/data/") == "nrt_TurbineX"  # Case-insensitive

    def test_extract_invalid_turbine_names(self):
        # Test cases where turbine name should not be extracted
        assert extract_turbine_name("/path/to/Turbine_NRT123/data/") is None
        assert extract_turbine_name("/path/to/Other_Turbine/data/") is None
        assert extract_turbine_name("/NRT/") is None  # Only 'NRT' without additional characters

    def test_extract_with_custom_pattern(self):
        # Test with different patterns
        pattern = r'/NRT\d+/'
        assert extract_turbine_name("/path/to/NRT123/", pattern=pattern) == "NRT123"
        assert extract_turbine_name("/path/to/NRTabc/", pattern=pattern) is None


class TestExtractTimestamp:
    def test_extract_valid_timestamps(self):
        # Test cases where timestamps should be extracted
        input_path = "/media/owilab/7759343A6EC07E1C/data_primary_norther/NRTA01/TDD/TDD_IOT/2023/01/05/df-20230105_210000.parquet.gzip"
        assert extract_timestamp(input_path) == dt.datetime(2023, 1, 5, 21, 0, 0)

        input_path = "/path/to/some/data/df-20231231_235959.parquet"
        assert extract_timestamp(input_path) == dt.datetime(2023, 12, 31, 23, 59, 59)

    def test_extract_invalid_timestamps(self):
        # Test cases where timestamps cannot be extracted
        input_path = "/path/to/some/data/no_timestamp_here.parquet"
        assert extract_timestamp(input_path) is None # Expect failure due to missing timestamp
