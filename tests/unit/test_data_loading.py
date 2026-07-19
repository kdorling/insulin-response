"""
Unit tests for data loading edge cases and defensive coding practices.

Tests cover:
- UCIDiabetesLoader.load() error message accuracy (no misleading download() reference)
- Glucose range validation using MIN_GLUCOSE/MAX_GLUCOSE constants
- Directory permission enforcement (POSIX only)
- Exception handling for malformed CSV files
- Empty dataset rejection (headers-only CSV)
- Timestamp parsing consistency
"""

import logging
import os
import re
import sys
import tempfile

import pandas as pd
import pytest

from insulin_response.data_preprocessing import (
    DatasetLoader,
    UCIDiabetesLoader,
    CGMacrosLoader,
    MIN_GLUCOSE,
    MAX_GLUCOSE,
    DROPOUT_PARTICIPANTS,
    glucose_flag_column,
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
            with caplog.at_level(logging.WARNING, logger="insulin_response.data_preprocessing"):
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
            with caplog.at_level(logging.WARNING, logger="insulin_response.data_preprocessing"):
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

    @pytest.mark.skipif(os.name != "posix", reason="POSIX permissions not supported")
    def test_loader_init_succeeds_when_permissions_enforceable(self):
        """On POSIX, loader init succeeds when 0o700 can be enforced on the data dir."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "test_data")
            loader = UCIDiabetesLoader(data_dir=data_dir)
            assert os.path.isdir(data_dir)
            assert loader.data_dir == data_dir

    @pytest.mark.skipif(os.name != "posix", reason="POSIX permissions not supported")
    def test_init_fails_closed_when_chmod_fails(self, monkeypatch):
        """If 0o700 cannot be enforced, init must fail closed rather than log-and-continue.

        Sensitive health data must never be written to a directory whose access we
        could not lock down (see defensive-coding 'File System Security').
        """
        def raise_permission_error(*args, **kwargs):
            raise PermissionError("cannot enforce permissions")

        monkeypatch.setattr(os, "chmod", raise_permission_error)
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "test_data")
            with pytest.raises(RuntimeError, match="could not enforce 0o700"):
                UCIDiabetesLoader(data_dir=data_dir)


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
            with pytest.raises(ValueError, match=r"[Ee]mpty"):
                loader.load()

    def test_validate_headers_only_csv_raises_value_error(self):
        """validate() should reject a CSV with headers but no data rows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "uci_diabetes.csv")
            with open(csv_path, "w") as f:
                f.write("pre_meal_glucose,post_meal_glucose,insulin_dose,meal_timestamp\n")

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            with pytest.raises(ValueError, match=r"[Ee]mpty"):
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
            with pytest.raises(ValueError, match=r"[Mm]issing required columns"):
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
            with pytest.raises(ValueError, match=r"[Nn]o .* valid|[Mm]issing|parseable|required columns"):
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

    @staticmethod
    def _participant_frame(timestamps):
        """Build a required-column participant frame with the given timestamps."""
        n = len(timestamps)
        return pd.DataFrame({
            'timestamp': timestamps,
            'glucose': [100.0] * n,
            'carbs': [30.0] * n,
            'fat': [10.0] * n,
            'protein': [20.0] * n,
            'activity': [1.0] * n,
            'heart_rate': [70.0] * n,
        })

    def test_validate_rejects_files_with_invalid_timestamps(self):
        """A required-column CSV with unparseable timestamps is rejected by
        _parse_participant(), so validate() must not report success on it."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            self._participant_frame(['not-a-timestamp', 'also-bad']).to_csv(
                os.path.join(cgmacros_dir, "participant_1.csv"), index=False
            )

            loader = CGMacrosLoader(data_dir=temp_dir)
            with pytest.raises(ValueError):
                loader.validate()

    def test_load_rejects_files_with_invalid_timestamps(self):
        """Paired with validate(): load() must also reject a dataset whose only
        file has unparseable timestamps (no valid participant data)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            self._participant_frame(['not-a-timestamp', 'also-bad']).to_csv(
                os.path.join(cgmacros_dir, "participant_1.csv"), index=False
            )

            loader = CGMacrosLoader(data_dir=temp_dir)
            with pytest.raises(ValueError):
                loader.load()

    def test_validate_and_load_agree_on_mixed_timestamp_validity(self):
        """With one valid file and one invalid-timestamp file, validate() passes
        and load() returns only the valid participant, keeping the two in sync."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            self._participant_frame(
                ['2024-01-01 12:00:00', '2024-01-01 12:05:00']
            ).to_csv(os.path.join(cgmacros_dir, "participant_1.csv"), index=False)
            self._participant_frame(['bad', 'worse']).to_csv(
                os.path.join(cgmacros_dir, "participant_2.csv"), index=False
            )

            loader = CGMacrosLoader(data_dir=temp_dir)
            assert loader.validate() is True

            result = loader.load()
            assert set(result['participant_id'].unique()) == {1}


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
        """When os.scandir fails on an existing directory, raise ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            loader = CGMacrosLoader(data_dir=temp_dir)

            original_scandir = os.scandir

            def mocked_scandir(path):
                if path == cgmacros_dir:
                    raise PermissionError("Permission denied")
                return original_scandir(path)

            monkeypatch.setattr(os, "scandir", mocked_scandir)

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
    """Glucose validation and cleaning should be a shared helper in DatasetLoader."""

    def test_base_class_has_validate_and_clean_glucose_method(self):
        """DatasetLoader should expose a _validate_and_clean_glucose helper method."""
        from insulin_response.data_preprocessing import DatasetLoader
        assert hasattr(DatasetLoader, '_validate_and_clean_glucose'), (
            "DatasetLoader base class should have a _validate_and_clean_glucose method"
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
            with caplog.at_level(logging.WARNING, logger="insulin_response.data_preprocessing"):
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
            with caplog.at_level(logging.WARNING, logger="insulin_response.data_preprocessing"):
                result = loader.load()

            assert len(result) == 2
            warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
            assert any("out of range" in msg.lower() for msg in warning_messages)


class TestTimestampFormatISO8601:
    """Timestamp parsing should use format='ISO8601' for performance."""

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
    """Participant ID extraction should use regex for robustness."""

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
    """Type annotations should use modern syntax compatible with Python 3.9+."""

    def test_annotations_are_deferred(self):
        """Annotations should be evaluated lazily, so builtin generics like list[str]
        work on Python 3.9. Asserted behaviourally rather than by grepping the source,
        so the test survives refactors, vendoring, and packaging changes."""
        annotations = DatasetLoader._validate_and_clean_glucose.__annotations__
        assert annotations['glucose_columns'] == 'list[str]', (
            "Annotations should be stored as strings (deferred evaluation); "
            f"got {annotations['glucose_columns']!r}"
        )


class TestCGMacrosValidateHandlesOSError:
    """validate() should handle OSError/PermissionError on os.scandir."""

    def test_validate_unreadable_directory_raises_valueerror(self, monkeypatch):
        """validate() should raise ValueError (not OSError) when directory exists but is unreadable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            loader = CGMacrosLoader(data_dir=temp_dir)

            original_scandir = os.scandir

            def mocked_scandir(path):
                if path == cgmacros_dir:
                    raise PermissionError("Permission denied")
                return original_scandir(path)

            monkeypatch.setattr(os, "scandir", mocked_scandir)

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
                    call_count["n"] += 1
                    raise OSError("Permission denied")
                return original_read_csv(filepath, *args, **kwargs)

            monkeypatch.setattr(pd, "read_csv", mock_read_csv)

            result = loader.load()
            assert call_count["n"] == 1
            # Should skip the unreadable file and load only participant_1
            assert len(result) == 1
            assert all(result['participant_id'] == 1)


class TestParticipantFileRegexConstant:
    """Participant filename validation should use a module-level regex constant."""

    def test_participant_file_regex_constant_exists(self):
        """Module should define PARTICIPANT_FILE_REGEX as a constant."""
        from insulin_response import data_preprocessing
        assert hasattr(data_preprocessing, 'PARTICIPANT_FILE_REGEX'), (
            "Module should define PARTICIPANT_FILE_REGEX constant"
        )

    def test_participant_file_regex_matches_valid_filenames(self):
        """The regex constant should match valid participant filenames."""
        import re
        from insulin_response.data_preprocessing import PARTICIPANT_FILE_REGEX
        assert re.fullmatch(PARTICIPANT_FILE_REGEX, "participant_1.csv")
        assert re.fullmatch(PARTICIPANT_FILE_REGEX, "participant_42.csv")
        assert not re.fullmatch(PARTICIPANT_FILE_REGEX, "participant_backup.csv")
        assert not re.fullmatch(PARTICIPANT_FILE_REGEX, "metadata.csv")


class TestUCILoadUnicodeDecodeError:
    """UCIDiabetesLoader.load() should catch UnicodeDecodeError
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
    """UCIDiabetesLoader.load() should NOT catch TypeError — it's overly broad and can mask programming errors."""

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
    """CGMacrosLoader.validate() should catch UnicodeDecodeError when parsing participant files,
    so a bad-encoding file is skipped rather than crashing."""

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
    """_parse_participant() should NOT catch TypeError."""

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

            def raise_typeerror(self_inner, participant_id, filename=None):
                raise TypeError("programming error should not be swallowed")

            monkeypatch.setattr(CGMacrosLoader, "_parse_participant", raise_typeerror)

            with pytest.raises(TypeError, match="should not be swallowed"):
                loader.load()


class TestPyprojectDependencies:
    """pyproject.toml should declare pandas>=2.0.0 as a dependency."""

    def test_pyproject_has_pandas_dependency(self):
        """pyproject.toml should declare pandas>=2.0.0 in [project.dependencies]."""
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        pyproject_path = os.path.join(project_root, "pyproject.toml")
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        dependencies = data.get("project", {}).get("dependencies", [])
        assert "pandas>=2.0.0" in dependencies, (
            "pyproject.toml [project.dependencies] should pin pandas>=2.0.0, "
            f"got: {dependencies}"
        )


class TestSrcLayoutPackaging:
    """The project uses a src-layout: the package lives in src/insulin_response/,
    and src/ itself must NOT be a package (no src/__init__.py)."""

    def test_package_init_py_exists(self):
        """src/insulin_response/__init__.py should exist for package discovery."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        init_path = os.path.join(project_root, "src", "insulin_response", "__init__.py")
        assert os.path.exists(init_path), (
            "src/insulin_response/__init__.py must exist for setuptools to discover the package"
        )

    def test_src_is_not_a_package(self):
        """src/ is a container directory in a src-layout, not a package itself."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        src_init = os.path.join(project_root, "src", "__init__.py")
        assert not os.path.exists(src_init), (
            "src/__init__.py must NOT exist in a src-layout; src/ is a container, "
            "not an importable package"
        )


class TestConftestNoSysPathManipulation:
    """conftest.py should not manipulate sys.path."""

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
    """_parse_participant() should treat header-only files as invalid."""

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
            with pytest.raises(ValueError, match="No parseable participant data"):
                loader.load()


class TestCGMacrosLoadEmptyCombinedDataFrame:
    """load() should guard against empty concatenated result."""

    def test_load_raises_when_combined_df_is_empty(self):
        """load() should raise ValueError if concatenated result has zero rows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            # Write a header-only file — _parse_participant returns None for this
            # so all_data will be empty and we get "No parseable participant data"
            with open(os.path.join(cgmacros_dir, "participant_1.csv"), "w") as f:
                f.write("timestamp,glucose,carbs,fat,protein,activity,heart_rate\n")

            loader = CGMacrosLoader(data_dir=temp_dir)
            with pytest.raises(ValueError):
                loader.load()


class TestGlucoseValidationNonNumericData:
    """_validate_and_clean_glucose should handle non-numeric glucose columns
    using pd.to_numeric with errors='coerce' to avoid TypeError on corrupted data."""

    def test_track_a_non_numeric_glucose_does_not_raise(self, caplog):
        """load() should not crash when glucose columns contain non-numeric strings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "uci_diabetes.csv")
            # Write a CSV where glucose columns have non-numeric values
            with open(csv_path, "w") as f:
                f.write("pre_meal_glucose,post_meal_glucose,insulin_dose,meal_timestamp\n")
                f.write("not_a_number,also_bad,10.0,2024-01-01T12:00:00\n")
                f.write("100.0,140.0,10.0,2024-01-01T13:00:00\n")

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            # Should not raise TypeError from the glucose range comparison
            result = loader.load()
            assert len(result) == 2

    def test_validate_and_clean_glucose_coerces_non_numeric(self, caplog):
        """_validate_and_clean_glucose should coerce non-numeric values to NaN
        rather than raising TypeError."""
        import logging

        with tempfile.TemporaryDirectory() as temp_dir:
            loader = UCIDiabetesLoader(data_dir=temp_dir)
            df = pd.DataFrame({
                'glucose': ['bad', '100.0', 'NaN', '700.0'],
            })
            # Should not raise — non-numeric values are coerced to NaN
            with caplog.at_level(logging.WARNING, logger="insulin_response.data_preprocessing"):
                loader._validate_and_clean_glucose(df, ['glucose'])

            # Should still detect the out-of-range value (700.0 > MAX_GLUCOSE)
            warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
            assert any("out of range" in msg.lower() for msg in warning_messages), (
                "Should warn about 700.0 being out of range even when other values are non-numeric"
            )


class TestUCILoadInvalidTimestampWrapsValueError:
    """UCIDiabetesLoader.load() should catch ValueError from
    pd.to_datetime and wrap it with the dataset path for consistent error reporting."""

    def test_load_invalid_timestamp_raises_valueerror_with_path(self):
        """Invalid timestamps should produce ValueError mentioning the dataset path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "uci_diabetes.csv")
            data = pd.DataFrame({
                'pre_meal_glucose': [100.0],
                'post_meal_glucose': [140.0],
                'insulin_dose': [10.0],
                'meal_timestamp': ['not-a-valid-timestamp'],
            })
            data.to_csv(csv_path, index=False)

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            with pytest.raises(ValueError) as exc_info:
                loader.load()
            # The error message should include the dataset path for debugging
            assert "uci_diabetes" in str(exc_info.value) or temp_dir in str(exc_info.value), (
                "ValueError from invalid timestamps should include the dataset path"
            )

    def test_load_mixed_valid_invalid_timestamps_raises_valueerror_with_path(self):
        """A mix of valid and invalid timestamps should still produce a path-aware ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "uci_diabetes.csv")
            data = pd.DataFrame({
                'pre_meal_glucose': [100.0, 110.0],
                'post_meal_glucose': [140.0, 150.0],
                'insulin_dose': [10.0, 12.0],
                'meal_timestamp': ['2024-01-01T12:00:00', 'garbage-timestamp'],
            })
            data.to_csv(csv_path, index=False)

            loader = UCIDiabetesLoader(data_dir=temp_dir)
            with pytest.raises(ValueError) as exc_info:
                loader.load()
            assert "uci_diabetes" in str(exc_info.value) or temp_dir in str(exc_info.value), (
                "ValueError from invalid timestamps should include the dataset path"
            )


class TestCGMacrosDiscoverParticipantIds:
    """Participant ID discovery should be a shared helper method."""

    def test_discover_participant_ids_returns_sorted_unique_ids(self):
        """_discover_participant_ids() should return sorted unique participant ID→filename mapping."""
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
            assert list(ids.keys()) == [1, 3, 5], f"Expected keys [1, 3, 5], got {list(ids.keys())}"
            assert ids[1] == "participant_1.csv"
            assert ids[3] == "participant_3.csv"
            assert ids[5] == "participant_5.csv"

    def test_discover_participant_ids_raises_on_unreadable_dir(self, monkeypatch):
        """_discover_participant_ids() should raise ValueError for unreadable directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            loader = CGMacrosLoader(data_dir=temp_dir)

            original_scandir = os.scandir

            def mocked_scandir(path):
                if path == cgmacros_dir:
                    raise PermissionError("Permission denied")
                return original_scandir(path)

            monkeypatch.setattr(os, "scandir", mocked_scandir)

            with pytest.raises(ValueError, match="Cannot access"):
                loader._discover_participant_ids()

    def test_discover_participant_ids_empty_dir_returns_empty(self):
        """_discover_participant_ids() should return empty dict for dir with no participant files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            # Only non-matching files
            pd.DataFrame({'col': [1]}).to_csv(
                os.path.join(cgmacros_dir, "metadata.csv"), index=False
            )

            loader = CGMacrosLoader(data_dir=temp_dir)
            ids = loader._discover_participant_ids()
            assert ids == {}


class TestValidationLogConsistency:
    """Validation summary log should report counts consistently."""

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
            with caplog.at_level(logging.INFO, logger="insulin_response.data_preprocessing"):
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


class TestPyprojectPythonVersionFloor:
    """requires-python should declare the project's supported Python floor (>=3.9)."""

    def test_requires_python_is_at_least_3_9(self):
        """requires-python should be >=3.9, the project's declared minimum interpreter."""
        try:
            import tomllib
        except ModuleNotFoundError:  # Python < 3.11
            import tomli as tomllib

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        pyproject_path = os.path.join(project_root, "pyproject.toml")
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)

        # Inspect the project.requires-python field directly rather than searching
        # the raw file text, so unrelated version strings (dependency pins, tool
        # config) elsewhere in pyproject.toml can't satisfy or break the assertion.
        requires_python = pyproject["project"]["requires-python"]
        assert ">=3.9" in requires_python, (
            f"requires-python should declare a >=3.9 floor (found: {requires_python!r})"
        )
        assert ">=3.8" not in requires_python, (
            "requires-python should not claim Python 3.8 support; the project's "
            f"minimum supported interpreter is 3.9 (found: {requires_python!r})"
        )


class TestLeadingZeroParticipantFilenames:
    """_discover_participant_ids() should handle filenames with
    leading zeros like participant_02.csv correctly."""

    def test_leading_zero_participant_file_is_loaded(self):
        """participant_02.csv should be loaded correctly (as participant ID 2)."""
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
            # Use a leading-zero filename
            data.to_csv(os.path.join(cgmacros_dir, "participant_02.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            result = loader.load()
            assert len(result) == 2
            assert all(result['participant_id'] == 2)

    def test_leading_zero_and_canonical_both_loaded(self):
        """Both participant_2.csv and participant_02.csv should be discoverable.
        If both exist, the canonical form (participant_2.csv) data should be loaded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            data_canonical = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1, freq='5min'),
                'glucose': [100.0],
                'carbs': [30.0],
                'fat': [10.0],
                'protein': [20.0],
                'activity': [1.0],
                'heart_rate': [70.0],
            })
            data_leading_zero = pd.DataFrame({
                'timestamp': pd.date_range('2024-02-01', periods=1, freq='5min'),
                'glucose': [200.0],
                'carbs': [50.0],
                'fat': [20.0],
                'protein': [30.0],
                'activity': [2.0],
                'heart_rate': [80.0],
            })
            data_canonical.to_csv(os.path.join(cgmacros_dir, "participant_2.csv"), index=False)
            data_leading_zero.to_csv(os.path.join(cgmacros_dir, "participant_02.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            # Both map to pid=2, but _discover_participant_ids deduplicates and
            # deterministically prefers the canonical participant_2.csv filename
            # regardless of directory iteration order.
            ids = loader._discover_participant_ids()
            assert 2 in ids
            assert ids[2] == "participant_2.csv"

            result = loader.load()
            assert all(result['participant_id'] == 2)
            # The canonical file's data (glucose 100.0) must be loaded, and the
            # leading-zero file's distinct data (glucose 200.0) must be excluded.
            assert len(result) == 1
            assert result['glucose'].iloc[0] == 100.0
            assert 200.0 not in result['glucose'].values

    def test_ambiguous_non_canonical_duplicates_raise(self):
        """Two distinct non-canonical filenames mapping to the same ID should be
        rejected rather than silently resolved by directory iteration order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)

            data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1, freq='5min'),
                'glucose': [100.0],
                'carbs': [30.0],
                'fat': [10.0],
                'protein': [20.0],
                'activity': [1.0],
                'heart_rate': [70.0],
            })
            # Both participant_02.csv and participant_002.csv map to pid=2, and
            # neither is the canonical participant_2.csv form -> ambiguous.
            data.to_csv(os.path.join(cgmacros_dir, "participant_02.csv"), index=False)
            data.to_csv(os.path.join(cgmacros_dir, "participant_002.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            with pytest.raises(ValueError, match=r"[Aa]mbiguous participant files"):
                loader._discover_participant_ids()

    def test_canonical_resolution_independent_of_scan_order(self, monkeypatch):
        """When the canonical participant_2.csv coexists with participant_02.csv
        and participant_002.csv aliases, discovery must resolve to the canonical
        file regardless of the order os.scandir() yields the entries."""

        class _FakeEntry:
            def __init__(self, name):
                self.name = name

            def is_file(self):
                return True

        class _FakeScandir:
            def __init__(self, names):
                self._names = names

            def __enter__(self):
                return iter(_FakeEntry(n) for n in self._names)

            def __exit__(self, *exc):
                return False

        scan_orders = [
            ["participant_2.csv", "participant_02.csv", "participant_002.csv"],
            ["participant_002.csv", "participant_02.csv", "participant_2.csv"],
        ]
        for order in scan_orders:
            with tempfile.TemporaryDirectory() as temp_dir:
                loader = CGMacrosLoader(data_dir=temp_dir)
                monkeypatch.setattr(
                    os, "scandir", lambda path, names=order: _FakeScandir(names)
                )
                ids = loader._discover_participant_ids()
                assert ids == {2: "participant_2.csv"}, (
                    f"Scan order {order} should resolve to the canonical filename, "
                    f"got {ids}"
                )
                monkeypatch.undo()


class TestGlucoseValidationNonNumericWarning:
    """_validate_and_clean_glucose should log a warning when
    non-numeric values are encountered in glucose columns."""

    def test_non_numeric_glucose_values_trigger_warning(self, caplog):
        """Non-numeric values in glucose columns should produce a data quality warning."""
        import logging

        with tempfile.TemporaryDirectory() as temp_dir:
            loader = UCIDiabetesLoader(data_dir=temp_dir)
            df = pd.DataFrame({
                'glucose': ['bad_value', 'also_bad', '100.0'],
            })
            with caplog.at_level(logging.WARNING, logger="insulin_response.data_preprocessing"):
                loader._validate_and_clean_glucose(df, ['glucose'])

            warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
            assert any("non-numeric" in msg.lower() for msg in warning_messages), (
                "Should warn about non-numeric values in glucose columns as a data quality issue"
            )

    def test_all_numeric_glucose_no_non_numeric_warning(self, caplog):
        """All-numeric glucose columns should not trigger a non-numeric warning."""
        import logging

        with tempfile.TemporaryDirectory() as temp_dir:
            loader = UCIDiabetesLoader(data_dir=temp_dir)
            df = pd.DataFrame({
                'glucose': [100.0, 200.0, 300.0],
            })
            with caplog.at_level(logging.WARNING, logger="insulin_response.data_preprocessing"):
                loader._validate_and_clean_glucose(df, ['glucose'])

            warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
            assert not any("non-numeric" in msg.lower() for msg in warning_messages), (
                "Should not warn about non-numeric values when all values are numeric"
            )


class TestGlucoseValidationUpdatesDataFrame:
    """_validate_and_clean_glucose should update the DataFrame columns to numeric types
    so downstream operations don't encounter string data."""

    def test_validate_and_clean_glucose_updates_df_columns_to_numeric(self):
        """After validation, the DataFrame column should contain numeric values (not strings)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = UCIDiabetesLoader(data_dir=temp_dir)
            df = pd.DataFrame({
                'glucose': ['100.0', '200.0', 'bad', '300.0'],
            })
            loader._validate_and_clean_glucose(df, ['glucose'])
            # After validation, column should be numeric, not object/string
            assert pd.api.types.is_numeric_dtype(df['glucose']), (
                "_validate_and_clean_glucose should update the DataFrame column to numeric dtype"
            )
            # 'bad' should have been coerced to NaN
            assert df['glucose'].isna().sum() == 1
            # Valid values should be preserved
            assert df['glucose'].iloc[0] == 100.0
            assert df['glucose'].iloc[1] == 200.0
            assert df['glucose'].iloc[3] == 300.0

    def test_validate_and_clean_glucose_preserves_already_numeric(self):
        """If column is already numeric, validation should not corrupt values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = UCIDiabetesLoader(data_dir=temp_dir)
            df = pd.DataFrame({
                'glucose': [100.0, 200.0, 300.0],
            })
            loader._validate_and_clean_glucose(df, ['glucose'])
            assert pd.api.types.is_numeric_dtype(df['glucose'])
            assert df['glucose'].tolist() == [100.0, 200.0, 300.0]


class TestNullHandlerDeduplication:
    """Logger NullHandler should not accumulate duplicates on module reload."""

    def test_logger_has_at_most_one_null_handler(self):
        """The module logger should have at most one NullHandler."""
        import importlib
        import insulin_response.data_preprocessing as mod
        # Reload the module to simulate repeated import
        importlib.reload(mod)
        importlib.reload(mod)
        logger = logging.getLogger('insulin_response.data_preprocessing')
        null_handlers = [h for h in logger.handlers if isinstance(h, logging.NullHandler)]
        assert len(null_handlers) <= 1, (
            f"Logger should have at most 1 NullHandler, but has {len(null_handlers)}. "
            "Guard addHandler with 'if not logger.handlers'."
        )


class TestParticipantIdRangeFiltering:
    """_discover_participant_ids() should filter out IDs outside the declared range [1, MAX_PARTICIPANT_ID]."""

    def test_discover_excludes_ids_above_max(self):
        """Participant IDs > MAX_PARTICIPANT_ID should be excluded from discovery."""
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
            # Valid participant within range
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_1.csv"), index=False)
            # Participant ID above MAX_PARTICIPANT_ID
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_999.csv"), index=False)

            from insulin_response.data_preprocessing import MAX_PARTICIPANT_ID
            loader = CGMacrosLoader(data_dir=temp_dir)
            ids = loader._discover_participant_ids()
            assert 999 not in ids, (
                f"Participant ID 999 > MAX_PARTICIPANT_ID ({MAX_PARTICIPANT_ID}) "
                "should be excluded from discovery"
            )
            assert 1 in ids

    def test_discover_excludes_ids_below_one(self):
        """Participant IDs < 1 should be excluded from discovery."""
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
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_0.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            ids = loader._discover_participant_ids()
            assert 0 not in ids, "Participant ID 0 should be excluded from discovery"
            assert 1 in ids

    def test_discover_logs_warning_for_out_of_range_ids(self, caplog):
        """Out-of-range participant IDs should produce a log warning."""
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
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_100.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            with caplog.at_level(logging.WARNING, logger="insulin_response.data_preprocessing"):
                loader._discover_participant_ids()

            warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
            assert any("100" in msg and "range" in msg.lower() for msg in warning_messages), (
                "Should log a warning about out-of-range participant ID 100"
            )

    def test_load_excludes_out_of_range_participant_ids(self):
        """load() should not include data from participants outside [1, MAX_PARTICIPANT_ID]."""
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
            valid_data.to_csv(os.path.join(cgmacros_dir, "participant_999.csv"), index=False)

            loader = CGMacrosLoader(data_dir=temp_dir)
            result = loader.load()
            assert all(result['participant_id'] == 1), (
                "Only participant 1 should be loaded; participant 999 is out of range"
            )


class TestReadmePythonVersion:
    """README.md should state the correct minimum Python version."""

    @staticmethod
    def _python_prereq_line():
        """Return the README line declaring the Python version prerequisite.

        Targets the specific declaration (e.g. ``- Python 3.9 or higher``)
        rather than scanning the whole file, so unrelated version strings
        (dependency pins, changelog entries) cannot satisfy or break the
        assertions.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        readme_path = os.path.join(project_root, "README.md")
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        for line in content.splitlines():
            if re.search(r"\bPython\s+3\.\d+", line):
                return line
        return None

    def test_readme_does_not_claim_python_38(self):
        """The Python prerequisite declaration should not claim 3.8 support."""
        prereq_line = self._python_prereq_line()
        assert prereq_line is not None, "README should declare a Python version prerequisite"
        assert "3.8" not in prereq_line, (
            "README's Python prerequisite should not reference Python 3.8; "
            f"pyproject.toml requires-python is >=3.9 (found: {prereq_line!r})"
        )

    def test_readme_states_python_39_or_higher(self):
        """The Python prerequisite declaration should state Python 3.9."""
        prereq_line = self._python_prereq_line()
        assert prereq_line is not None, "README should declare a Python version prerequisite"
        assert "3.9" in prereq_line, (
            "README's Python prerequisite should state Python 3.9 as the minimum "
            f"version (found: {prereq_line!r})"
        )


class TestValidateAndCleanGlucoseMethodName:
    """The glucose validation/cleaning method should be named _validate_and_clean_glucose
    to accurately reflect its side effects (modifying the DataFrame in-place)."""

    def test_method_is_named_validate_and_clean_glucose(self):
        """DatasetLoader should have _validate_and_clean_glucose, not _validate_glucose_range."""
        from insulin_response.data_preprocessing import DatasetLoader
        assert hasattr(DatasetLoader, '_validate_and_clean_glucose'), (
            "Method should be renamed from _validate_glucose_range to _validate_and_clean_glucose"
        )

    def test_validate_and_clean_glucose_coerces_and_warns(self, caplog):
        """_validate_and_clean_glucose should coerce non-numeric values and warn about out-of-range."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = UCIDiabetesLoader(data_dir=temp_dir)
            df = pd.DataFrame({
                'glucose': ['bad', '100.0', '700.0'],
            })
            with caplog.at_level(logging.WARNING, logger="insulin_response.data_preprocessing"):
                loader._validate_and_clean_glucose(df, ['glucose'])

            assert pd.api.types.is_numeric_dtype(df['glucose'])
            warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
            assert any("non-numeric" in msg.lower() for msg in warning_messages)
            assert any("out of range" in msg.lower() for msg in warning_messages)


def _write_track_a_csv(path, timestamps=None, rows=2):
    """Write a minimal valid Track A CSV, optionally overriding meal_timestamp values."""
    if timestamps is None:
        timestamps = pd.date_range('2024-01-01', periods=rows, freq='h').astype(str)
    df = pd.DataFrame({
        'pre_meal_glucose': [100.0] * len(timestamps),
        'post_meal_glucose': [140.0] * len(timestamps),
        'insulin_dose': [5.0] * len(timestamps),
        'meal_timestamp': list(timestamps),
    })
    df.to_csv(path, index=False)
    return df


def _write_participant_csv(path, timestamps, glucose=None):
    """Write a minimal valid CGMacros participant CSV."""
    n = len(timestamps)
    pd.DataFrame({
        'timestamp': list(timestamps),
        'glucose': glucose if glucose is not None else [100.0] * n,
        'carbs': [30.0] * n,
        'fat': [10.0] * n,
        'protein': [20.0] * n,
        'activity': [1.0] * n,
        'heart_rate': [70.0] * n,
    }).to_csv(path, index=False)


class TestDiscoverParticipantIdsPreservesFileNotFoundError:
    """_discover_participant_ids() should not translate a missing directory into ValueError."""

    def test_missing_directory_raises_filenotfounderror(self):
        """FileNotFoundError must survive the generic OSError branch (Error Type Consistency)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = CGMacrosLoader(data_dir=temp_dir)
            # Directory is absent, so os.scandir raises FileNotFoundError.
            assert not os.path.exists(loader.dataset_dir)

            with pytest.raises(FileNotFoundError):
                loader._discover_participant_ids()

    def test_unreadable_directory_still_raises_valueerror(self, monkeypatch):
        """Other OSError cases (e.g. PermissionError) are still mapped to ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)
            loader = CGMacrosLoader(data_dir=temp_dir)

            original_scandir = os.scandir

            def deny(path):
                # Only deny the dataset directory; other paths (e.g. tempfile
                # cleanup) must keep working.
                if path == cgmacros_dir:
                    raise PermissionError("Permission denied")
                return original_scandir(path)

            monkeypatch.setattr(os, 'scandir', deny)
            with pytest.raises(ValueError, match="Cannot access dataset directory"):
                loader._discover_participant_ids()


class TestTrackAValidateChecksTimestamps:
    """validate() should reject timestamps that load() would fail on."""

    def test_validate_rejects_unparseable_timestamp(self):
        """A CSV with a non-date meal_timestamp should fail preflight, not just load()."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = UCIDiabetesLoader(data_dir=temp_dir)
            os.makedirs(os.path.dirname(loader.dataset_path), exist_ok=True)
            _write_track_a_csv(loader.dataset_path, timestamps=['not-a-date', 'also-bad'])

            with pytest.raises(ValueError, match="Invalid timestamp format"):
                loader.validate()

    def test_validate_rejects_bad_timestamp_after_first_row(self):
        """A valid first row must not mask a malformed timestamp further down the file.

        validate() is a preflight check; if it only inspects row 0 it reports success
        on exactly the corrupted files it exists to catch, and load() fails later.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = UCIDiabetesLoader(data_dir=temp_dir)
            os.makedirs(os.path.dirname(loader.dataset_path), exist_ok=True)
            _write_track_a_csv(
                loader.dataset_path,
                timestamps=['2024-01-01 00:00:00', 'not-a-date'],
            )

            with pytest.raises(ValueError, match="Invalid timestamp format"):
                loader.validate()

    def test_validate_accepts_iso_timestamps(self):
        """Well-formed ISO8601 timestamps should still pass validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = UCIDiabetesLoader(data_dir=temp_dir)
            os.makedirs(os.path.dirname(loader.dataset_path), exist_ok=True)
            _write_track_a_csv(loader.dataset_path)

            assert loader.validate() is True

    def test_missing_file_raises_filenotfounderror_not_valueerror(self):
        """validate() should preserve FileNotFoundError for an absent dataset file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = UCIDiabetesLoader(data_dir=temp_dir)
            with pytest.raises(FileNotFoundError):
                loader.validate()


class TestTrackALoadNarrowsValueError:
    """load() should not swallow unrelated ValueErrors in its broad handler."""

    def test_missing_columns_error_is_not_rewrapped(self):
        """A missing-column ValueError should propagate with its own message intact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = UCIDiabetesLoader(data_dir=temp_dir)
            os.makedirs(os.path.dirname(loader.dataset_path), exist_ok=True)
            pd.DataFrame({'pre_meal_glucose': [100.0]}).to_csv(loader.dataset_path, index=False)

            with pytest.raises(ValueError, match="Missing required columns") as exc_info:
                loader.load()
            assert "Error loading dataset" not in str(exc_info.value)

    def test_bad_timestamp_reports_timestamp_error(self):
        """A timestamp parse failure should be reported as such."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = UCIDiabetesLoader(data_dir=temp_dir)
            os.makedirs(os.path.dirname(loader.dataset_path), exist_ok=True)
            _write_track_a_csv(loader.dataset_path, timestamps=['not-a-date', 'also-bad'])

            with pytest.raises(ValueError, match="Invalid timestamp format"):
                loader.load()


class TestParticipantTimestampOrdering:
    """Each participant's rows should be returned in chronological order."""

    def test_out_of_order_rows_are_sorted(self, caplog):
        """A participant file written out of order should load sorted, with a warning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)
            shuffled = ['2024-01-01 02:00:00', '2024-01-01 00:00:00', '2024-01-01 01:00:00']
            _write_participant_csv(
                os.path.join(cgmacros_dir, 'participant_1.csv'), shuffled
            )

            loader = CGMacrosLoader(data_dir=temp_dir)
            with caplog.at_level(logging.WARNING, logger="insulin_response.data_preprocessing"):
                result = loader.load()

            assert result['timestamp'].is_monotonic_increasing
            assert any("chronological order" in r.message for r in caplog.records)

    def test_already_ordered_rows_do_not_warn(self, caplog):
        """An in-order file should not emit a sorting warning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)
            ordered = pd.date_range('2024-01-01', periods=3, freq='h').astype(str)
            _write_participant_csv(os.path.join(cgmacros_dir, 'participant_1.csv'), ordered)

            loader = CGMacrosLoader(data_dir=temp_dir)
            with caplog.at_level(logging.WARNING, logger="insulin_response.data_preprocessing"):
                result = loader.load()

            assert result['timestamp'].is_monotonic_increasing
            assert not any("chronological order" in r.message for r in caplog.records)


class TestGlucoseOutOfRangeFlag:
    """Out-of-range glucose readings should be flagged per row, not just logged."""

    def test_track_b_flags_out_of_range_rows(self):
        """CGMacros load() should mark rows outside [MIN_GLUCOSE, MAX_GLUCOSE]."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
            os.makedirs(cgmacros_dir)
            timestamps = pd.date_range('2024-01-01', periods=3, freq='h').astype(str)
            _write_participant_csv(
                os.path.join(cgmacros_dir, 'participant_1.csv'),
                timestamps,
                glucose=[MIN_GLUCOSE - 1, 100.0, MAX_GLUCOSE + 1],
            )

            result = CGMacrosLoader(data_dir=temp_dir).load()

            flag = glucose_flag_column('glucose')
            assert flag in result.columns
            assert list(result[flag]) == [True, False, True]

    def test_track_a_flags_both_glucose_columns(self):
        """Track A load() should flag pre- and post-meal glucose independently."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = UCIDiabetesLoader(data_dir=temp_dir)
            os.makedirs(os.path.dirname(loader.dataset_path), exist_ok=True)
            pd.DataFrame({
                'pre_meal_glucose': [100.0, MAX_GLUCOSE + 1],
                'post_meal_glucose': [MIN_GLUCOSE - 1, 140.0],
                'insulin_dose': [5.0, 5.0],
                'meal_timestamp': list(pd.date_range('2024-01-01', periods=2, freq='h').astype(str)),
            }).to_csv(loader.dataset_path, index=False)

            result = loader.load()

            assert list(result[glucose_flag_column('pre_meal_glucose')]) == [False, True]
            assert list(result[glucose_flag_column('post_meal_glucose')]) == [True, False]

    def test_missing_values_are_not_flagged(self):
        """NaN glucose is absent, not out of range, so it should flag False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = UCIDiabetesLoader(data_dir=temp_dir)
            df = pd.DataFrame({'glucose': [float('nan'), 100.0, 700.0]})
            loader._validate_and_clean_glucose(df, ['glucose'])

            assert list(df[glucose_flag_column('glucose')]) == [False, False, True]

    def test_flag_columns_are_not_required_of_source_files(self):
        """Derived flags are outputs only; source CSVs must not need to supply them."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = CGMacrosLoader(data_dir=temp_dir)
            assert glucose_flag_column('glucose') not in loader.required_columns
            assert glucose_flag_column('glucose') not in loader._expected_file_columns
            assert glucose_flag_column('glucose') in loader.output_columns
