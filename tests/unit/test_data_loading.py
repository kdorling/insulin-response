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
            with pytest.raises(ValueError):
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

    def test_validate_binary_garbage_raises_value_error_with_corruption_message(self):
        """validate() should catch UnicodeDecodeError and re-raise as ValueError
        with a clear corruption message, not leak the raw decode error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            garbage_path = os.path.join(temp_dir, "uci_diabetes.csv")
            # Use bytes that trigger UnicodeDecodeError specifically (not ParserError)
            with open(garbage_path, "wb") as f:
                f.write(b"\x80\x81\x82\x83\x84\x85" * 100)

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            with pytest.raises(ValueError, match="corrupted") as exc_info:
                loader.validate()
            # Should NOT be a raw UnicodeDecodeError — should be wrapped
            assert type(exc_info.value) is ValueError, (
                f"Expected plain ValueError with corruption message, "
                f"got {type(exc_info.value).__name__}"
            )


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


class TestUCIValidateChecksColumns:
    """UCIDiabetesLoader.validate() should verify required columns are present."""

    def test_validate_rejects_csv_with_wrong_columns(self):
        """validate() should raise ValueError when CSV has wrong schema."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "uci_diabetes.csv")
            data = pd.DataFrame({
                'wrong_col_a': [1.0],
                'wrong_col_b': [2.0],
            })
            data.to_csv(csv_path, index=False)

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            with pytest.raises(ValueError, match="[Mm]issing required columns"):
                loader.validate()

    def test_validate_passes_with_correct_columns(self):
        """validate() should pass when CSV has all required columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "uci_diabetes.csv")
            data = pd.DataFrame({
                'pre_meal_glucose': [100.0],
                'post_meal_glucose': [140.0],
                'insulin_dose': [10.0],
                'meal_timestamp': ['2024-01-01 12:00:00'],
            })
            data.to_csv(csv_path, index=False)

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            assert loader.validate() is True


class TestCGMacrosValidateChecksParseability:
    """CGMacrosLoader.validate() should verify at least one file is parseable with required columns."""

    def test_validate_rejects_unparseable_csv(self):
        """validate() should raise ValueError when CSV files can't be parsed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            # Write a CSV with wrong columns
            data = pd.DataFrame({'wrong_col': [1.0]})
            data.to_csv(os.path.join(cgmacros_dir, "participant_1.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            with pytest.raises(ValueError, match="[Nn]o .* valid|[Mm]issing|parseable|required columns"):
                loader.validate()

    def test_validate_passes_with_valid_participant_file(self):
        """validate() should pass when at least one participant file has required columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            data = pd.DataFrame({
                'timestamp': ['2024-01-01 12:00:00'],
                'glucose': [100.0],
                'carbs': [30.0],
                'fat': [10.0],
                'protein': [20.0],
                'activity': [1.0],
                'heart_rate': [70.0],
            })
            data.to_csv(os.path.join(cgmacros_dir, "participant_1.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            assert loader.validate() is True


class TestCGMacrosErrorTypeConsistency:
    """CGMacrosLoader should use consistent error types for 'no participant files' condition."""

    def test_validate_raises_valueerror_for_no_participant_files(self):
        """validate() should raise ValueError when directory has CSVs but no participant_*.csv files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            # Write a non-participant CSV so the directory isn't empty
            data = pd.DataFrame({'col': [1]})
            data.to_csv(os.path.join(cgmacros_dir, "random.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            # validate() should reject datasets that contain no participant_*.csv files.
            with pytest.raises(ValueError):
                loader.validate()

    def test_load_raises_valueerror_for_no_participant_files(self):
        """load() should raise ValueError (not FileNotFoundError) when no participant files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            # Write a non-participant CSV
            data = pd.DataFrame({'col': [1]})
            data.to_csv(os.path.join(cgmacros_dir, "random.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            with pytest.raises(ValueError):
                loader.load()


class TestCGMacrosValidateLoadConsistency:
    """validate() and load() should use the same file discovery logic."""

    def test_validate_rejects_non_numeric_participant_files(self):
        """validate() should reject directories with only non-numeric participant files
        like participant_backup.csv (matching startswith/endswith but not regex)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            # This file matches startswith("participant_") and endswith(".csv")
            # but does NOT match the regex ^participant_(\d+)\.csv$
            data = pd.DataFrame({
                'timestamp': ['2024-01-01 12:00:00'],
                'glucose': [100.0],
                'carbs': [30.0],
                'fat': [10.0],
                'protein': [20.0],
                'activity': [1.0],
                'heart_rate': [70.0],
            })
            data.to_csv(os.path.join(cgmacros_dir, "participant_backup.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            with pytest.raises(ValueError):
                loader.validate()

    def test_validate_rejects_dropout_only_participants(self):
        """validate() should reject directories containing only dropout participant files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            data = pd.DataFrame({
                'timestamp': ['2024-01-01 12:00:00'],
                'glucose': [100.0],
                'carbs': [30.0],
                'fat': [10.0],
                'protein': [20.0],
                'activity': [1.0],
                'heart_rate': [70.0],
            })
            # All dropout participants
            for pid in DROPOUT_PARTICIPANTS:
                data.to_csv(os.path.join(cgmacros_dir, f"participant_{pid}.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            with pytest.raises(ValueError):
                loader.validate()

    def test_validate_accepts_valid_non_dropout_participant(self):
        """validate() should pass when at least one non-dropout numeric participant exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            data = pd.DataFrame({
                'timestamp': ['2024-01-01 12:00:00'],
                'glucose': [100.0],
                'carbs': [30.0],
                'fat': [10.0],
                'protein': [20.0],
                'activity': [1.0],
                'heart_rate': [70.0],
            })
            data.to_csv(os.path.join(cgmacros_dir, "participant_1.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            assert loader.validate() is True


class TestCGMacrosLoadOSError:
    """load() should raise ValueError (not FileNotFoundError) for permission errors on listdir."""

    def test_load_listdir_oserror_raises_valueerror(self, monkeypatch):
        """When os.listdir fails on an existing directory, raise ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            loader = CGMacrosLoader(data_dir=temp_dir)

            original_listdir = os.listdir

            def mocked_listdir(path):
                if path == cgmacros_dir:
                    raise PermissionError("Permission denied")
                return original_listdir(path)

            monkeypatch.setattr(os, "listdir", mocked_listdir)

            with pytest.raises(ValueError, match="Cannot access"):
                loader.load()


class TestTimestampParsing:
    """Timestamp parsing should use explicit ISO8601 parsing for consistency."""

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


class TestGlucoseValidationInBaseClass:
    """PR Comment 1: Glucose range validation should be a shared helper in DatasetLoader."""

    def test_base_class_has_validate_glucose_range_method(self):
        """DatasetLoader should expose a _validate_glucose_range helper method."""
        from src.data_preprocessing import DatasetLoader
        assert hasattr(DatasetLoader, '_validate_glucose_range'), (
            "DatasetLoader base class should have a _validate_glucose_range method"
        )

    def test_track_a_uses_shared_glucose_validation(self, caplog):
        """Track A loader should use the base class glucose validation helper."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data = pd.DataFrame({
                'pre_meal_glucose': [10.0, 100.0],
                'post_meal_glucose': [80.0, 700.0],
                'insulin_dose': [5.0, 10.0],
                'meal_timestamp': pd.date_range('2024-01-01', periods=2, freq='h'),
            })
            data.to_csv(os.path.join(temp_dir, "uci_diabetes.csv"), index=False)

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            import logging
            with caplog.at_level(logging.WARNING, logger="src.data_preprocessing"):
                result = loader.load()

            assert len(result) == 2
            warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
            assert any("out of range" in msg.lower() for msg in warning_messages)

    def test_track_b_uses_shared_glucose_validation(self, caplog):
        """Track B loader should use the base class glucose validation helper."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=2, freq='5min'),
                'glucose': [5.0, 700.0],
                'carbs': [30.0, 40.0],
                'fat': [10.0, 15.0],
                'protein': [20.0, 25.0],
                'activity': [1.0, 2.0],
                'heart_rate': [70.0, 80.0],
            })
            data.to_csv(os.path.join(cgmacros_dir, "participant_1.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            import logging
            with caplog.at_level(logging.WARNING, logger="src.data_preprocessing"):
                result = loader.load()

            assert len(result) == 2
            warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
            assert any("out of range" in msg.lower() for msg in warning_messages)


class TestTimestampFormatISO8601:
    """PR Comment 2: Timestamp parsing should use format='ISO8601' for performance."""

    def test_track_a_handles_iso8601_timestamps(self):
        """Track A should parse ISO8601 timestamps correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data = pd.DataFrame({
                'pre_meal_glucose': [100.0],
                'post_meal_glucose': [140.0],
                'insulin_dose': [10.0],
                'meal_timestamp': ['2024-06-15T14:30:00'],
            })
            data.to_csv(os.path.join(temp_dir, "uci_diabetes.csv"), index=False)

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            result = loader.load()
            ts = result['meal_timestamp'].iloc[0]
            assert ts.year == 2024
            assert ts.month == 6
            assert ts.day == 15
            assert ts.hour == 14
            assert ts.minute == 30

    def test_track_b_handles_iso8601_timestamps(self):
        """Track B should parse ISO8601 timestamps correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            data = pd.DataFrame({
                'timestamp': ['2024-06-15T14:30:00'],
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
            ts = result['timestamp'].iloc[0]
            assert ts.year == 2024
            assert ts.month == 6
            assert ts.day == 15
            assert ts.hour == 14
            assert ts.minute == 30


class TestParticipantIdParsing:
    """PR Comment: Participant ID extraction should use regex for robustness."""

    def test_standard_participant_filename_parsed(self):
        """Standard participant_N.csv filenames should be parsed correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=2, freq='5min'),
                'glucose': [100.0, 110.0],
                'carbs': [30.0, 40.0],
                'fat': [10.0, 15.0],
                'protein': [20.0, 25.0],
                'activity': [1.0, 2.0],
                'heart_rate': [70.0, 80.0],
            })
            data.to_csv(os.path.join(cgmacros_dir, "participant_5.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            result = loader.load()
            assert all(result['participant_id'] == 5)

    def test_non_participant_files_ignored(self):
        """Files not matching participant_N.csv pattern should be ignored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            valid_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1, freq='5min'),
                'glucose': [100.0],
                'carbs': [30.0],
                'fat': [10.0],
                'protein': [20.0],
                'activity': [1.0],
                'heart_rate': [70.0],
            })
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_1.csv"), index=False)

            # Non-matching files should be ignored
            pd.DataFrame({'col': [1]}).to_csv(
                os.path.join(cgmacros_dir, "metadata.csv"), index=False
            )
            pd.DataFrame({'col': [1]}).to_csv(
                os.path.join(cgmacros_dir, "participant_notes.csv"), index=False
            )

            loader = CGMacrosLoader(data_dir=temp_dir)
            result = loader.load()
            # Should only have data from participant_1.csv
            assert len(result) == 1
            assert all(result['participant_id'] == 1)

    def test_filename_with_non_numeric_suffix_ignored(self):
        """Files like participant_backup.csv (non-numeric ID) should be ignored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            valid_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1, freq='5min'),
                'glucose': [100.0],
                'carbs': [30.0],
                'fat': [10.0],
                'protein': [20.0],
                'activity': [1.0],
                'heart_rate': [70.0],
            })
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_1.csv"), index=False)

            # This file has "participant_" prefix and ".csv" suffix but non-numeric ID
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_backup.csv"), index=False)
            # This has extra text after the number
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_1_old.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            result = loader.load()
            # Should only load participant_1.csv, ignoring the non-numeric ones
            assert len(result) == 1
            assert all(result['participant_id'] == 1)

    def test_only_exact_numeric_participant_filenames_are_loaded(self):
        """Only files matching participant_<number>.csv should contribute rows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            valid_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1, freq='5min'),
                'glucose': [100.0],
                'carbs': [30.0],
                'fat': [10.0],
                'protein': [20.0],
                'activity': [1.0],
                'heart_rate': [70.0],
            })
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_2.csv"), index=False)

            # Similar-looking but invalid filenames should be ignored.
            pd.DataFrame({'col': [1]}).to_csv(
                os.path.join(cgmacros_dir, "participant_02_backup.csv"), index=False
            )
            pd.DataFrame({'col': [1]}).to_csv(
                os.path.join(cgmacros_dir, "participant_.csv"), index=False
            )
            pd.DataFrame({'col': [1]}).to_csv(
                os.path.join(cgmacros_dir, "participant_abc.csv"), index=False
            )
            pd.DataFrame({'col': [1]}).to_csv(
                os.path.join(cgmacros_dir, "xparticipant_3.csv"), index=False
            )

            loader = CGMacrosLoader(data_dir=temp_dir)
            result = loader.load()

            assert len(result) == 1
            assert all(result['participant_id'] == 2)


class TestFutureAnnotationsCompatibility:
    """PR Comment: Type annotations should be compatible with Python 3.8+."""

    def test_module_uses_future_annotations(self):
        """data_preprocessing module should use 'from __future__ import annotations'
        so that list[str] annotations work on Python 3.8."""
        import importlib
        source = importlib.util.find_spec('src.data_preprocessing')
        assert source is not None
        with open(source.origin, 'r') as f:
            content = f.read()
        assert 'from __future__ import annotations' in content, (
            "Module should use 'from __future__ import annotations' for Python 3.8+ compatibility"
        )


class TestCGMacrosValidateHandlesOSError:
    """PR Comment (Copilot): validate() should handle OSError/PermissionError on os.listdir."""

    def test_validate_unreadable_directory_raises_valueerror(self, monkeypatch):
        """validate() should raise ValueError (not OSError) when directory exists but is unreadable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            loader = CGMacrosLoader(data_dir=temp_dir)

            original_listdir = os.listdir

            def mocked_listdir(path):
                if path == cgmacros_dir:
                    raise PermissionError("Permission denied")
                return original_listdir(path)

            monkeypatch.setattr(os, "listdir", mocked_listdir)

            with pytest.raises(ValueError, match="Cannot access dataset directory"):
                loader.validate()


class TestCGMacrosParseParticipantBroadExceptions:
    """_parse_participant() should catch UnicodeDecodeError and OSError (but not TypeError)."""

    def test_parse_participant_unicode_error_returns_none(self):
        """A participant file with invalid encoding should return None, not crash load()."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            # Write a valid participant file
            valid_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1, freq='5min'),
                'glucose': [100.0],
                'carbs': [30.0],
                'fat': [10.0],
                'protein': [20.0],
                'activity': [1.0],
                'heart_rate': [70.0],
            })
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_1.csv"), index=False)

            # Write a binary garbage file that triggers UnicodeDecodeError
            with open(os.path.join(cgmacros_dir, "participant_2.csv"), "wb") as f:
                f.write(b"\x80\x81\x82\x83\x84\x85" * 100)

            loader = CGMacrosLoader(data_dir=temp_dir)
            # load() should succeed by skipping the bad file, not crash
            result = loader.load()
            assert len(result) == 1
            assert all(result['participant_id'] == 1)

    def test_parse_participant_catches_oserror(self, monkeypatch):
        """_parse_participant should catch OSError for unreadable files, not crash load()."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            # Write a valid participant file
            valid_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1, freq='5min'),
                'glucose': [100.0],
                'carbs': [30.0],
                'fat': [10.0],
                'protein': [20.0],
                'activity': [1.0],
                'heart_rate': [70.0],
            })
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_1.csv"), index=False)
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_3.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)

            original_read_csv = pd.read_csv
            call_count = {"n": 0}

            def mock_read_csv(filepath, *args, **kwargs):
                if isinstance(filepath, str) and "participant_3.csv" in filepath:
                    raise OSError("Permission denied")
                return original_read_csv(filepath, *args, **kwargs)

            monkeypatch.setattr(pd, "read_csv", mock_read_csv)

            result = loader.load()
            # Should skip the unreadable file and load only participant_1
            assert len(result) == 1
            assert all(result['participant_id'] == 1)


class TestParticipantFileRegexConstant:
    """PR Comment (Gemini): Regex pattern should be a module-level constant."""

    def test_participant_file_regex_constant_exists(self):
        """Module should define PARTICIPANT_FILE_REGEX as a constant."""
        from src import data_preprocessing
        assert hasattr(data_preprocessing, 'PARTICIPANT_FILE_REGEX'), (
            "Module should define PARTICIPANT_FILE_REGEX constant"
        )

    def test_participant_file_regex_matches_valid_filenames(self):
        """The regex constant should match valid participant filenames."""
        import re
        from src.data_preprocessing import PARTICIPANT_FILE_REGEX
        assert re.fullmatch(PARTICIPANT_FILE_REGEX, "participant_1.csv")
        assert re.fullmatch(PARTICIPANT_FILE_REGEX, "participant_42.csv")
        assert not re.fullmatch(PARTICIPANT_FILE_REGEX, "participant_backup.csv")
        assert not re.fullmatch(PARTICIPANT_FILE_REGEX, "metadata.csv")


class TestUCILoadUnicodeDecodeError:
    """PR Comment (Copilot): UCIDiabetesLoader.load() should catch UnicodeDecodeError
    and wrap it as ValueError with the dataset path in the message."""

    def test_load_unicode_error_raises_valueerror(self):
        """A file with invalid encoding should raise ValueError, not UnicodeDecodeError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "uci_diabetes.csv")
            # Write bytes that are valid enough to not trigger ParserError first,
            # but will cause UnicodeDecodeError during read_csv
            with open(csv_path, "wb") as f:
                f.write(b"pre_meal_glucose,post_meal_glucose,insulin_dose,meal_timestamp\n")
                f.write(b"100.0,140.0,10.0,2024-01-01\xff\xfe\n")

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            with pytest.raises(ValueError) as exc_info:
                loader.load()
            # Error message should include the dataset path for debugging
            assert temp_dir in str(exc_info.value) or "uci_diabetes" in str(exc_info.value), (
                "Error message should include the dataset path for easier debugging"
            )

    def test_load_binary_file_raises_valueerror_with_path(self):
        """Binary garbage in load() should produce ValueError mentioning the dataset path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "uci_diabetes.csv")
            with open(csv_path, "wb") as f:
                f.write(b"\x80\x81\x82\x83\x84\x85" * 100)

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            with pytest.raises(ValueError) as exc_info:
                loader.load()
            assert "uci_diabetes" in str(exc_info.value) or temp_dir in str(exc_info.value)


class TestUCILoadNoTypeError:
    """PR Comment (Gemini): UCIDiabetesLoader.load() should NOT catch TypeError —
    it's overly broad and can mask programming errors."""

    def test_load_propagates_typeerror(self, monkeypatch):
        """Verify that TypeError raised during CSV loading is not caught by load()."""
        def mock_read_csv(*args, **kwargs):
            raise TypeError("programming error")

        monkeypatch.setattr(pd, "read_csv", mock_read_csv)

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "uci_diabetes.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(
                    "pre_meal_glucose,post_meal_glucose,insulin_dose,meal_timestamp\n"
                    "100.0,140.0,10.0,2024-01-01\n"
                )

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            with pytest.raises(TypeError, match="programming error"):
                loader.load()


class TestCGMacrosValidateUnicodeDecodeError:
    """PR Comment (Copilot): CGMacrosLoader.validate() should catch UnicodeDecodeError
    when parsing participant files, so a bad-encoding file is skipped rather than crashing."""

    def test_validate_skips_unicode_error_participant_file(self):
        """validate() should skip participant files with invalid encoding, not crash."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            # Write a valid participant file
            valid_data = pd.DataFrame({
                'timestamp': ['2024-01-01 12:00:00'],
                'glucose': [100.0],
                'carbs': [30.0],
                'fat': [10.0],
                'protein': [20.0],
                'activity': [1.0],
                'heart_rate': [70.0],
            })
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_1.csv"), index=False)

            # Write a binary garbage file that triggers UnicodeDecodeError
            with open(os.path.join(cgmacros_dir, "participant_2.csv"), "wb") as f:
                f.write(b"\x80\x81\x82\x83\x84\x85" * 100)

            loader = CGMacrosLoader(data_dir=temp_dir)
            # validate() should succeed — the bad file is skipped, the valid one counts
            assert loader.validate() is True


class TestCGMacrosParseParticipantNoTypeError:
    """PR Comment (Gemini): _parse_participant() should NOT catch TypeError."""

    def test_parse_participant_typeerror_is_not_swallowed(self, monkeypatch):
        """TypeError raised by _parse_participant() should propagate out of load()."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            valid_data = pd.DataFrame({
                'timestamp': ['2024-01-01 12:00:00'],
                'glucose': [100.0],
                'carbs': [30.0],
                'fat': [10.0],
                'protein': [20.0],
                'activity': [1.0],
                'heart_rate': [70.0],
            })
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_1.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)

            original_parse = CGMacrosLoader._parse_participant

            def raise_typeerror(self_inner, participant_id):
                raise TypeError("programming error should not be swallowed")

            monkeypatch.setattr(CGMacrosLoader, "_parse_participant", raise_typeerror)

            with pytest.raises(TypeError, match="should not be swallowed"):
                loader.load()


class TestPyprojectDependencies:
    """PR Comment (Gemini): pyproject.toml should declare pandas>=2.0.0 as a dependency."""

    def test_pyproject_has_pandas_dependency(self):
        """pyproject.toml should list pandas>=2.0.0 in dependencies."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        pyproject_path = os.path.join(project_root, "pyproject.toml")
        with open(pyproject_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert (
            '"pandas>=2.0.0"' in content
            or "'pandas>=2.0.0'" in content
            or '"pandas"' in content
            or "'pandas'" in content
        ), "pyproject.toml should declare pandas as a project dependency"


class TestSrcInitExists:
    """PR Comment (Copilot): src/ needs __init__.py for setuptools package discovery."""

    def test_src_init_py_exists(self):
        """src/__init__.py should exist for proper package discovery."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        init_path = os.path.join(project_root, "src", "__init__.py")
        assert os.path.exists(init_path), (
            "src/__init__.py must exist for setuptools to discover the package"
        )


class TestConftestNoSysPathManipulation:
    """PR Comment (Gemini): conftest.py should not manipulate sys.path."""

    def test_conftest_does_not_manipulate_sys_path(self):
        """conftest.py should not contain sys.path.insert or sys.path manipulation."""
        # Walk up from this test file to find the project root conftest.py
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        conftest_path = os.path.join(project_root, "conftest.py")
        if not os.path.exists(conftest_path):
            # conftest.py was removed entirely, which also satisfies the requirement
            return
        with open(conftest_path, 'r') as f:
            content = f.read()
        assert 'sys.path' not in content, (
            "conftest.py should not manipulate sys.path; use pip install -e . instead"
        )


class TestParseParticipantEmptyDataFrame:
    """PR Comment (Copilot): _parse_participant() should treat header-only files as invalid."""

    def test_parse_participant_headers_only_returns_none(self):
        """A participant CSV with headers but no data rows should be treated as invalid."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            # Write a header-only participant file (no data rows)
            with open(os.path.join(cgmacros_dir, "participant_1.csv"), "w") as f:
                f.write("timestamp,glucose,carbs,fat,protein,activity,heart_rate\n")

            loader = CGMacrosLoader(data_dir=temp_dir)
            result = loader._parse_participant(1)
            assert result is None, (
                "_parse_participant should return None for header-only CSV files"
            )

    def test_load_skips_header_only_participant_files(self):
        """load() should skip header-only participant files and still load valid ones."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            # Write a header-only participant file
            with open(os.path.join(cgmacros_dir, "participant_1.csv"), "w") as f:
                f.write("timestamp,glucose,carbs,fat,protein,activity,heart_rate\n")

            # Write a valid participant file
            valid_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=2, freq='5min'),
                'glucose': [100.0, 110.0],
                'carbs': [30.0, 40.0],
                'fat': [10.0, 15.0],
                'protein': [20.0, 25.0],
                'activity': [1.0, 2.0],
                'heart_rate': [70.0, 80.0],
            })
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_2.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            result = loader.load()
            assert len(result) == 2
            assert all(result['participant_id'] == 2)

    def test_load_raises_when_all_participants_are_header_only(self):
        """load() should raise ValueError when all participant files are header-only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            # Write only header-only participant files
            for pid in [1, 2, 3]:
                with open(os.path.join(cgmacros_dir, f"participant_{pid}.csv"), "w") as f:
                    f.write("timestamp,glucose,carbs,fat,protein,activity,heart_rate\n")

            loader = CGMacrosLoader(data_dir=temp_dir)
            with pytest.raises(ValueError, match="No valid participant data"):
                loader.load()


class TestCGMacrosLoadEmptyCombinedDataFrame:
    """PR Comment (Copilot): load() should guard against empty concatenated result."""

    def test_load_raises_when_combined_df_is_empty(self):
        """load() should raise ValueError if concatenated result has zero rows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            # Write a header-only file — _parse_participant returns None for this
            # so all_data will be empty and we get "No valid participant data"
            with open(os.path.join(cgmacros_dir, "participant_1.csv"), "w") as f:
                f.write("timestamp,glucose,carbs,fat,protein,activity,heart_rate\n")

            loader = CGMacrosLoader(data_dir=temp_dir)
            with pytest.raises(ValueError):
                loader.load()


class TestCGMacrosDiscoverParticipantIds:
    """PR Comment (Gemini): Participant ID discovery should be a shared helper method."""

    def test_discover_participant_ids_returns_sorted_unique_ids(self):
        """_discover_participant_ids() should return sorted unique participant IDs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            for pid in [5, 1, 3, 1]:  # duplicates and unsorted
                pd.DataFrame({
                    'timestamp': ['2024-01-01 12:00:00'],
                    'glucose': [100.0],
                    'carbs': [30.0],
                    'fat': [10.0],
                    'protein': [20.0],
                    'activity': [1.0],
                    'heart_rate': [70.0],
                }).to_csv(os.path.join(cgmacros_dir, f"participant_{pid}.csv"), index=False)

            # Also write a non-matching file
            pd.DataFrame({'col': [1]}).to_csv(
                os.path.join(cgmacros_dir, "metadata.csv"), index=False
            )

            loader = CGMacrosLoader(data_dir=temp_dir)
            ids = loader._discover_participant_ids()
            assert ids == [1, 3, 5], f"Expected [1, 3, 5], got {ids}"

    def test_discover_participant_ids_raises_on_unreadable_dir(self, monkeypatch):
        """_discover_participant_ids() should raise ValueError for unreadable directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            loader = CGMacrosLoader(data_dir=temp_dir)

            original_listdir = os.listdir

            def mocked_listdir(path):
                if path == cgmacros_dir:
                    raise PermissionError("Permission denied")
                return original_listdir(path)

            monkeypatch.setattr(os, "listdir", mocked_listdir)

            with pytest.raises(ValueError, match="Cannot access"):
                loader._discover_participant_ids()

    def test_discover_participant_ids_empty_dir_returns_empty(self):
        """_discover_participant_ids() should return empty list for dir with no participant files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            # Only non-matching files
            pd.DataFrame({'col': [1]}).to_csv(
                os.path.join(cgmacros_dir, "metadata.csv"), index=False
            )

            loader = CGMacrosLoader(data_dir=temp_dir)
            ids = loader._discover_participant_ids()
            assert ids == []


class TestValidationLogConsistency:
    """PR Comment (Copilot): Validation summary log should report counts consistently."""

    def test_validate_log_reports_non_dropout_counts(self, caplog):
        """validate() log message should report valid count out of non-dropout total,
        not out of all participant files (which includes dropouts)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            valid_data = pd.DataFrame({
                'timestamp': ['2024-01-01 12:00:00'],
                'glucose': [100.0],
                'carbs': [30.0],
                'fat': [10.0],
                'protein': [20.0],
                'activity': [1.0],
                'heart_rate': [70.0],
            })
            # One valid non-dropout participant
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_1.csv"), index=False)
            # One dropout participant
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_24.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            import logging
            with caplog.at_level(logging.INFO, logger="src.data_preprocessing"):
                loader.validate()

            # The log should say "1 valid non-dropout ... out of 1 total"
            # (not "out of 2 total" which would include the dropout file)
            info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
            validation_msgs = [m for m in info_messages if "validation passed" in m.lower()]
            assert len(validation_msgs) == 1
            msg = validation_msgs[0]
            assert "non-dropout" in msg, (
                f"Log message should mention 'non-dropout' to clarify what's being counted: {msg}"
            )
            assert "1 valid" in msg and "out of 1" in msg, (
                f"Log should report 1 valid out of 1 non-dropout (not 2 total): {msg}"
            )
