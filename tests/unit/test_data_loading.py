"""
Unit tests for data loading edge cases and PR comment fixes.

Tests cover:
- UCIDiabetesLoader.load() error message accuracy (no misleading download() reference)
- Glucose range validation using MIN_GLUCOSE/MAX_GLUCOSE constants
- Directory permission enforcement
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
    """DatasetLoader should enforce 0o700 permissions on data directories."""

    def test_data_dir_permissions_enforced(self):
        """Data directory should have 0o700 permissions after loader init."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "test_data")
            UCIDiabetesLoader(data_dir=data_dir)
            mode = os.stat(data_dir).st_mode & 0o777
            assert mode == 0o700, f"Expected 0o700, got {oct(mode)}"
