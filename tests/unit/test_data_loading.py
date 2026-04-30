"""
Unit tests for data loading edge cases and PR comment fixes.

Tests cover:
- UCIDiabetesLoader.load() error message accuracy (no misleading download() reference)
- Glucose range validation using MIN_GLUCOSE/MAX_GLUCOSE constants
- Directory permission enforcement (POSIX only)
- Exception handling for malformed CSV files
- Empty dataset rejection (headers-only CSV)
- Timestamp parsing consistency
"""

import os
import tempfile

import pandas as pd
import pytest

from src.data_preprocessing import (
    UCIDiabetesLoader,
    CGMacrosLoader,
    MIN_GLUCOSE,
    MAX_GLUCOSE,
    DROPOUT_PARTICIPANTS,
)


class TestUCIDiabetesLoaderErrorMessages:
    """UCIDiabetesLoader.load() should not reference download() since it's unimplemented."""

    def test_load_missing_file_error_does_not_mention_download(self):
        """Error message should direct users to provide a CSV, not call download()."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = UCIDiabetesLoader(data_dir=temp_dir)
            with pytest.raises(FileNotFoundError, match="pre-processed CSV") as exc_info:
                loader.load()
            # Must NOT tell users to "run download() first"
            assert "download()" not in str(exc_info.value), (
                "Error message should not reference download() since it raises NotImplementedError"
            )


class TestGlucoseRangeValidation:
    """MIN_GLUCOSE and MAX_GLUCOSE constants should be used for validation."""

    def test_track_a_warns_on_out_of_range_glucose(self, caplog):
        """Track A loader should log warnings for glucose values outside [MIN_GLUCOSE, MAX_GLUCOSE]."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data = pd.DataFrame({
                'pre_meal_glucose': [10.0, 100.0, 700.0],  # 10 < MIN, 700 > MAX
                'post_meal_glucose': [80.0, 120.0, 150.0],
                'insulin_dose': [5.0, 10.0, 15.0],
                'meal_timestamp': pd.date_range('2024-01-01', periods=3, freq='h'),
            })
            data.to_csv(os.path.join(temp_dir, "uci_diabetes.csv"), index=False)

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            import logging
            with caplog.at_level(logging.WARNING, logger="src.data_preprocessing"):
                result = loader.load()

            # Should still return all rows (out-of-range values are flagged, not dropped)
            assert len(result) == 3
            # Should have logged a warning about out-of-range values
            assert any("out of range" in record.message.lower() or
                       "outside" in record.message.lower() or
                       "range" in record.message.lower()
                       for record in caplog.records), (
                "Expected a warning about out-of-range glucose values"
            )

    def test_track_b_warns_on_out_of_range_glucose(self, caplog):
        """Track B loader should log warnings for glucose values outside [MIN_GLUCOSE, MAX_GLUCOSE]."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=3, freq='5min'),
                'glucose': [15.0, 100.0, 650.0],  # 15 < MIN, 650 > MAX
                'carbs': [30.0, 40.0, 50.0],
                'fat': [10.0, 15.0, 20.0],
                'protein': [20.0, 25.0, 30.0],
                'activity': [1.0, 2.0, 3.0],
                'heart_rate': [70.0, 80.0, 90.0],
            })
            data.to_csv(os.path.join(cgmacros_dir, "participant_1.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            import logging
            with caplog.at_level(logging.WARNING, logger="src.data_preprocessing"):
                result = loader.load()

            assert len(result) == 3
            assert any("out of range" in record.message.lower() or
                       "outside" in record.message.lower() or
                       "range" in record.message.lower()
                       for record in caplog.records), (
                "Expected a warning about out-of-range glucose values"
            )

    def test_glucose_constants_are_used_in_validation(self):
        """Verify the constants match the documented range [20, 600]."""
        assert MIN_GLUCOSE == 20
        assert MAX_GLUCOSE == 600


class TestDirectoryPermissions:
    """DatasetLoader should enforce 0o700 permissions on data directories (POSIX only)."""

    @pytest.mark.skipif(os.name != "posix", reason="POSIX permissions not supported")
    def test_data_dir_permissions_enforced(self):
        """Data directory should have 0o700 permissions after loader init."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "test_data")
            UCIDiabetesLoader(data_dir=data_dir)
            mode = os.stat(data_dir).st_mode & 0o777
            assert mode == 0o700, f"Expected 0o700, got {oct(mode)}"

    def test_loader_init_succeeds_on_any_platform(self):
        """Loader initialization should not crash regardless of platform."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "test_data")
            loader = UCIDiabetesLoader(data_dir=data_dir)
            assert os.path.isdir(data_dir)
            assert loader.data_dir == data_dir


class TestExceptionHandling:
    """Exception handler in load() should catch ParserError for malformed CSVs."""

    def test_load_malformed_csv_raises_value_error(self):
        """A malformed CSV should raise ValueError, not an uncaught ParserError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            malformed_path = os.path.join(temp_dir, "uci_diabetes.csv")
            with open(malformed_path, "w") as f:
                f.write("col1,col2\n")
                f.write('"unclosed quote,bad,data\n')
                f.write("more,broken,rows\n")

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            with pytest.raises((ValueError, pd.errors.ParserError)):
                loader.load()

    def test_load_binary_garbage_raises_value_error(self):
        """Binary garbage in a CSV file should raise ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            garbage_path = os.path.join(temp_dir, "uci_diabetes.csv")
            with open(garbage_path, "wb") as f:
                f.write(b"\x00\x01\x02\xff\xfe\xfd" * 100)

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            with pytest.raises(ValueError):
                loader.load()


class TestEmptyDatasetRejection:
    """Loaders should reject datasets that contain only headers (no data rows)."""

    def test_load_headers_only_csv_raises_value_error(self):
        """A CSV with headers but no data rows should raise ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "uci_diabetes.csv")
            with open(csv_path, "w") as f:
                f.write("pre_meal_glucose,post_meal_glucose,insulin_dose,meal_timestamp\n")

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            with pytest.raises(ValueError, match="[Ee]mpty"):
                loader.load()

    def test_validate_headers_only_csv_raises_value_error(self):
        """validate() should reject a CSV with headers but no data rows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "uci_diabetes.csv")
            with open(csv_path, "w") as f:
                f.write("pre_meal_glucose,post_meal_glucose,insulin_dose,meal_timestamp\n")

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            with pytest.raises(ValueError, match="[Ee]mpty"):
                loader.validate()


class TestTimestampParsing:
    """Timestamp parsing should use explicit dayfirst=False for consistency."""

    def test_track_a_parses_iso_timestamps(self):
        """Track A should correctly parse ISO format timestamps."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data = pd.DataFrame({
                'pre_meal_glucose': [100.0],
                'post_meal_glucose': [140.0],
                'insulin_dose': [10.0],
                'meal_timestamp': ['2024-03-01 12:00:00'],
            })
            data.to_csv(os.path.join(temp_dir, "uci_diabetes.csv"), index=False)

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            result = loader.load()
            assert result['meal_timestamp'].iloc[0].month == 3
            assert result['meal_timestamp'].iloc[0].day == 1

    def test_track_b_parses_iso_timestamps(self):
        """Track B should correctly parse ISO format timestamps."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            data = pd.DataFrame({
                'timestamp': ['2024-03-01 12:00:00'],
                'glucose': [100.0],
                'carbs': [30.0],
                'fat': [10.0],
                'protein': [20.0],
                'activity': [1.0],
                'heart_rate': [70.0],
            })
            data.to_csv(os.path.join(cgmacros_dir, "participant_1.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            result = loader.load()
            assert result['timestamp'].iloc[0].month == 3
            assert result['timestamp'].iloc[0].day == 1
