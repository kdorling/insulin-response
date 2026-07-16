"""
Property-based tests for data loading and processing.

Feature: insulin-response-modeling
"""

from hypothesis import given, strategies as st, settings, HealthCheck
import pandas as pd
import pytest
import shutil
import tempfile
import os

from src.data_preprocessing import UCIDiabetesLoader, CGMacrosLoader, DROPOUT_PARTICIPANTS, MAX_PARTICIPANT_ID, MIN_GLUCOSE, MAX_GLUCOSE


# Generators for test data
@st.composite
def track_a_dataframe_generator(draw):
    """Generate valid Track A dataframes for property testing."""
    n_rows = draw(st.integers(min_value=1, max_value=100))

    data = {
        'pre_meal_glucose': draw(st.lists(
            st.floats(min_value=MIN_GLUCOSE, max_value=MAX_GLUCOSE, allow_nan=False, allow_infinity=False),
            min_size=n_rows, max_size=n_rows
        )),
        'post_meal_glucose': draw(st.lists(
            st.floats(min_value=MIN_GLUCOSE, max_value=MAX_GLUCOSE, allow_nan=False, allow_infinity=False),
            min_size=n_rows, max_size=n_rows
        )),
        'insulin_dose': draw(st.lists(
            st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=n_rows, max_size=n_rows
        )),
        'meal_timestamp': pd.date_range(
            start=draw(st.datetimes(min_value=pd.Timestamp('2000-01-01'), max_value=pd.Timestamp('2025-12-31'))),
            periods=n_rows, freq='h'
        )
    }

    return pd.DataFrame(data)


# Feature: insulin-response-modeling, Property 1: Complete Field Extraction
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(track_a_data=track_a_dataframe_generator())
def test_track_a_field_extraction(track_a_data, tmp_path):
    """
    Property 1: Complete Field Extraction
    Validates: Requirements 1.2

    For any Track A dataset, when the Data Pipeline processes it,
    all required fields must be present in the output DataFrame.
    """
    track_a_data.to_csv(os.path.join(tmp_path, "uci_diabetes.csv"), index=False)

    loader = UCIDiabetesLoader(data_dir=str(tmp_path))
    result = loader.load()

    required_fields = ['pre_meal_glucose', 'post_meal_glucose',
                      'insulin_dose', 'meal_timestamp']

    assert all(field in result.columns for field in required_fields), \
        f"Missing required fields. Expected: {required_fields}, Got: {list(result.columns)}"

    assert len(result) == len(track_a_data), \
        f"Row count mismatch. Expected: {len(track_a_data)}, Got: {len(result)}"


@st.composite
def track_b_dataframe_generator(draw):
    """Generate valid Track B dataframes for property testing."""
    n_rows = draw(st.integers(min_value=1, max_value=100))

    valid_participants = [i for i in range(1, MAX_PARTICIPANT_ID + 1) if i not in DROPOUT_PARTICIPANTS]
    participant_id = draw(st.sampled_from(valid_participants))

    data = {
        'timestamp': pd.date_range(
            start=draw(st.datetimes(min_value=pd.Timestamp('2000-01-01'), max_value=pd.Timestamp('2025-12-31'))),
            periods=n_rows, freq='5min'
        ),
        'glucose': draw(st.lists(
            st.floats(min_value=MIN_GLUCOSE, max_value=MAX_GLUCOSE, allow_nan=False, allow_infinity=False),
            min_size=n_rows, max_size=n_rows
        )),
        'carbs': draw(st.lists(
            st.floats(min_value=0, max_value=200, allow_nan=False, allow_infinity=False),
            min_size=n_rows, max_size=n_rows
        )),
        'fat': draw(st.lists(
            st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=n_rows, max_size=n_rows
        )),
        'protein': draw(st.lists(
            st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=n_rows, max_size=n_rows
        )),
        'activity': draw(st.lists(
            st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False),
            min_size=n_rows, max_size=n_rows
        )),
        'heart_rate': draw(st.lists(
            st.floats(min_value=40, max_value=200, allow_nan=False, allow_infinity=False),
            min_size=n_rows, max_size=n_rows
        ))
    }

    return pd.DataFrame(data), participant_id


# Feature: insulin-response-modeling, Property 1: Complete Field Extraction
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(track_b_data_tuple=track_b_dataframe_generator())
def test_track_b_field_extraction(track_b_data_tuple, tmp_path):
    """
    Property 1: Complete Field Extraction
    Validates: Requirements 1.4

    For any Track B dataset, when the Data Pipeline processes it,
    all required fields must be present in the output DataFrame.
    """
    track_b_data, participant_id = track_b_data_tuple

    cgmacros_dir = os.path.join(tmp_path, 'cgmacros')
    # Clean up any leftover files from previous Hypothesis iterations
    # since tmp_path is reused across @given examples
    shutil.rmtree(cgmacros_dir, ignore_errors=True)
    os.makedirs(cgmacros_dir, exist_ok=True)

    participant_file = os.path.join(cgmacros_dir, f'participant_{participant_id}.csv')
    track_b_data.to_csv(participant_file, index=False)

    loader = CGMacrosLoader(data_dir=str(tmp_path))
    result = loader.load()

    required_fields = ['participant_id', 'timestamp', 'glucose', 'carbs',
                      'fat', 'protein', 'activity', 'heart_rate', 'health_group']

    assert all(field in result.columns for field in required_fields), \
        f"Missing required fields. Expected: {required_fields}, Got: {list(result.columns)}"

    assert len(result) == len(track_b_data), \
        f"Row count mismatch. Expected: {len(track_b_data)}, Got: {len(result)}"

    assert (result['participant_id'] == participant_id).all(), \
        f"Participant ID mismatch. Expected: {participant_id}, Got: {result['participant_id'].unique()}"


def test_track_a_load_raises_on_missing_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        loader = UCIDiabetesLoader(data_dir=temp_dir)
        with pytest.raises(FileNotFoundError):
            loader.load()


def test_track_a_load_raises_on_missing_columns():
    with tempfile.TemporaryDirectory() as temp_dir:
        incomplete = pd.DataFrame({'pre_meal_glucose': [100.0], 'post_meal_glucose': [120.0]})
        incomplete.to_csv(os.path.join(temp_dir, "uci_diabetes.csv"), index=False)

        loader = UCIDiabetesLoader(data_dir=temp_dir)
        with pytest.raises(ValueError, match="Missing required columns"):
            loader.load()


def test_track_a_validate_raises_on_empty_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        open(os.path.join(temp_dir, "uci_diabetes.csv"), 'w').close()

        loader = UCIDiabetesLoader(data_dir=temp_dir)
        with pytest.raises(ValueError):
            loader.validate()


def test_track_b_load_raises_on_missing_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        loader = CGMacrosLoader(data_dir=temp_dir)
        with pytest.raises(FileNotFoundError):
            loader.load()


def test_track_b_load_raises_on_no_valid_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
        os.makedirs(cgmacros_dir)
        # Write a file with missing required columns
        pd.DataFrame({'glucose': [100.0]}).to_csv(
            os.path.join(cgmacros_dir, "participant_1.csv"), index=False
        )

        loader = CGMacrosLoader(data_dir=temp_dir)
        with pytest.raises(ValueError, match="No valid participant data"):
            loader.load()
