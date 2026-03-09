"""
Property-based tests for data loading and processing.

Feature: insulin-response-modeling
"""

from hypothesis import given, strategies as st
import pandas as pd
import tempfile
import os

from src.data_preprocessing import UCIDiabetesLoader, CGMacrosLoader, DROPOUT_PARTICIPANTS


# Generators for test data
@st.composite
def track_a_dataframe_generator(draw):
    """Generate valid Track A dataframes for property testing."""
    n_rows = draw(st.integers(min_value=1, max_value=100))

    data = {
        'pre_meal_glucose': draw(st.lists(
            st.floats(min_value=20, max_value=600, allow_nan=False, allow_infinity=False),
            min_size=n_rows, max_size=n_rows
        )),
        'post_meal_glucose': draw(st.lists(
            st.floats(min_value=20, max_value=600, allow_nan=False, allow_infinity=False),
            min_size=n_rows, max_size=n_rows
        )),
        'insulin_dose': draw(st.lists(
            st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=n_rows, max_size=n_rows
        )),
        'meal_timestamp': pd.date_range(start='2024-01-01', periods=n_rows, freq='h')
    }

    return pd.DataFrame(data)


# Feature: insulin-response-modeling, Property 1: Complete Field Extraction
@given(track_a_data=track_a_dataframe_generator())
def test_track_a_field_extraction(track_a_data):
    """
    Property 1: Complete Field Extraction
    Validates: Requirements 1.2

    For any Track A dataset, when the Data Pipeline processes it,
    all required fields must be present in the output DataFrame.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
        track_a_data.to_csv(temp_path, index=False)

    try:
        temp_dir = os.path.dirname(temp_path)
        loader = UCIDiabetesLoader(data_dir=temp_dir)
        loader.dataset_path = temp_path

        result = loader.load()

        required_fields = ['pre_meal_glucose', 'post_meal_glucose',
                          'insulin_dose', 'meal_timestamp']

        assert all(field in result.columns for field in required_fields), \
            f"Missing required fields. Expected: {required_fields}, Got: {list(result.columns)}"

        assert len(result) == len(track_a_data), \
            f"Row count mismatch. Expected: {len(track_a_data)}, Got: {len(result)}"

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@st.composite
def track_b_dataframe_generator(draw):
    """Generate valid Track B dataframes for property testing."""
    n_rows = draw(st.integers(min_value=1, max_value=100))

    valid_participants = [i for i in range(1, 46) if i not in DROPOUT_PARTICIPANTS]
    participant_id = draw(st.sampled_from(valid_participants))

    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=n_rows, freq='5min'),
        'glucose': draw(st.lists(
            st.floats(min_value=20, max_value=600, allow_nan=False, allow_infinity=False),
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
@given(track_b_data_tuple=track_b_dataframe_generator())
def test_track_b_field_extraction(track_b_data_tuple):
    """
    Property 1: Complete Field Extraction
    Validates: Requirements 1.4

    For any Track B dataset, when the Data Pipeline processes it,
    all required fields must be present in the output DataFrame.
    """
    track_b_data, participant_id = track_b_data_tuple

    with tempfile.TemporaryDirectory() as temp_dir:
        cgmacros_dir = os.path.join(temp_dir, 'cgmacros')
        os.makedirs(cgmacros_dir)

        participant_file = os.path.join(cgmacros_dir, f'participant_{participant_id}.csv')
        track_b_data.to_csv(participant_file, index=False)

        loader = CGMacrosLoader(data_dir=temp_dir)
        loader.dataset_dir = cgmacros_dir

        result = loader.load()

        required_fields = ['participant_id', 'timestamp', 'glucose', 'carbs',
                          'fat', 'protein', 'activity', 'heart_rate', 'health_group']

        assert all(field in result.columns for field in required_fields), \
            f"Missing required fields. Expected: {required_fields}, Got: {list(result.columns)}"

        assert len(result) == len(track_b_data), \
            f"Row count mismatch. Expected: {len(track_b_data)}, Got: {len(result)}"

        assert all(result['participant_id'] == participant_id), \
            f"Participant ID mismatch"
