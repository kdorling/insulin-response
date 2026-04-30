"""
Data preprocessing module for insulin response modeling system.

This module provides data loading capabilities for both
Track A (UCI Diabetes dataset) and Track B (CGMacros dataset).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional
import os
import logging
import re

# Library-safe logging: let callers configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

# Constants
MIN_GLUCOSE = 20  # mg/dL
MAX_GLUCOSE = 600  # mg/dL
DROPOUT_PARTICIPANTS = [24, 25, 37, 40]
MAX_PARTICIPANT_ID = 45  # Inclusive upper bound for CGMacros participant IDs
PARTICIPANT_FILE_REGEX = r"participant_(\d+)\.csv$"


class DatasetLoader(ABC):
    """Base class for dataset loading with abstract methods."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the dataset loader.

        Args:
            data_dir: Directory where datasets will be stored
        """
        self.data_dir = data_dir
        # data_dir is always developer-supplied, not user-facing input, so path
        # traversal validation is intentionally omitted here.
        # Restrict permissions to 0o700 for health data privacy (POSIX only)
        os.makedirs(data_dir, mode=0o700, exist_ok=True)
        if os.name == "posix":
            try:
                os.chmod(data_dir, 0o700)
            except (PermissionError, OSError) as exc:
                logger.warning(
                    "Could not set permissions on data directory '%s' to 0o700: %s",
                    data_dir,
                    exc,
                )

    def _validate_glucose_range(self, df: pd.DataFrame, glucose_columns: list[str],
                               context: str = "") -> None:
        """
        Validate glucose values are within the expected range and log warnings
        for out-of-range values.

        Args:
            df: DataFrame containing glucose columns to validate
            glucose_columns: List of column names containing glucose values
            context: Optional context string for log messages (e.g., participant ID)
        """
        for col in glucose_columns:
            if col not in df.columns:
                continue
            out_of_range = (df[col] < MIN_GLUCOSE) | (df[col] > MAX_GLUCOSE)
            n_out = out_of_range.sum()
            if n_out > 0:
                ctx = f"{context}: " if context else ""
                logger.warning(
                    f"{ctx}{n_out} values in '{col}' are out of range "
                    f"[{MIN_GLUCOSE}, {MAX_GLUCOSE}] mg/dL"
                )

    @abstractmethod
    def download(self) -> bool:
        """
        Download the dataset from its source.

        Returns:
            True if download successful, False otherwise

        Raises:
            NotImplementedError: When subclass has not implemented download
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate that the dataset is complete and not corrupted.

        Returns:
            True if validation passes, False otherwise

        Raises:
            ValueError: If dataset is corrupted or incomplete
        """
        pass

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load the dataset into a pandas DataFrame.

        Returns:
            DataFrame containing the loaded dataset

        Raises:
            FileNotFoundError: If dataset file is missing
            ValueError: If required columns are missing or data is corrupted
        """
        pass


class UCIDiabetesLoader(DatasetLoader):
    """Loads UCI Diabetes dataset (Track A)."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the UCI Diabetes loader.

        Args:
            data_dir: Directory where dataset will be stored
        """
        super().__init__(data_dir)
        self.dataset_path = os.path.join(data_dir, "uci_diabetes.csv")
        self.required_columns = [
            'pre_meal_glucose',
            'post_meal_glucose',
            'insulin_dose',
            'meal_timestamp'
        ]

    def download(self) -> bool:
        """
        Download the UCI Diabetes dataset from UCI Machine Learning Repository.

        Returns:
            True if download successful

        Raises:
            NotImplementedError: UCI Diabetes download is not yet implemented
        """
        raise NotImplementedError(
            "UCI Diabetes dataset download is not yet implemented. "
            "The source format (tab-separated with coded values) requires transformation "
            "to match the expected CSV schema (pre_meal_glucose, post_meal_glucose, "
            "insulin_dose, meal_timestamp). Please provide a pre-processed CSV file "
            "that matches the required columns."
        )

    def validate(self) -> bool:
        """
        Validate that the dataset file exists, is not corrupted, and contains
        the required columns.

        Returns:
            True if validation passes

        Raises:
            FileNotFoundError: If dataset file is missing
            ValueError: If dataset is corrupted, empty, or missing required columns
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        try:
            file_size = os.path.getsize(self.dataset_path)
            if file_size == 0:
                raise ValueError(f"Dataset file is empty: {self.dataset_path}")

            sample = pd.read_csv(self.dataset_path, nrows=1)
            if sample.empty:
                raise ValueError(f"Dataset file has no data rows (empty): {self.dataset_path}")

            missing_columns = [col for col in self.required_columns if col not in sample.columns]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in {self.dataset_path}: {missing_columns}"
                )

            logger.info(f"Dataset validation passed: {self.dataset_path}")
            return True
        except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            raise ValueError(f"Dataset file is corrupted: {self.dataset_path}. Error: {e}") from e

    def load(self) -> pd.DataFrame:
        """
        Load the UCI Diabetes dataset into a pandas DataFrame.

        Returns:
            DataFrame with columns: pre_meal_glucose, post_meal_glucose,
                                   insulin_dose, meal_timestamp

        Raises:
            FileNotFoundError: If dataset file is missing
            ValueError: If required columns are missing or data is corrupted
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}. "
                                   f"Please provide a pre-processed CSV file.")

        try:
            df = pd.read_csv(self.dataset_path)

            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            df['meal_timestamp'] = pd.to_datetime(df['meal_timestamp'], format='ISO8601')

            # Validate glucose ranges and warn about out-of-range values
            self._validate_glucose_range(df, ['pre_meal_glucose', 'post_meal_glucose'])

            if df.empty:
                raise ValueError(f"Dataset file has no data rows (empty): {self.dataset_path}")

            logger.info(f"Loaded {len(df)} records from UCI Diabetes dataset")
            return df[self.required_columns]

        except pd.errors.EmptyDataError as e:
            raise ValueError(f"Dataset file is empty or corrupted: {self.dataset_path}") from e
        except ValueError:
            raise
        except (OSError, TypeError, pd.errors.ParserError) as e:
            raise ValueError(f"Error loading dataset: {e}") from e


class CGMacrosLoader(DatasetLoader):
    """Loads CGMacros dataset (Track B)."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the CGMacros loader.

        Args:
            data_dir: Directory where dataset will be stored
        """
        super().__init__(data_dir)
        self.dataset_dir = os.path.join(data_dir, "cgmacros")
        self.required_columns = [
            'participant_id',
            'timestamp',
            'glucose',
            'carbs',
            'fat',
            'protein',
            'activity',
            'heart_rate',
            'health_group'
        ]

        # Health group categorization based on CGMacros dataset
        # 15 healthy, 16 pre-diabetic, 14 Type 2 diabetic
        self.health_groups = {
            'healthy': list(range(1, 16)),  # Participants 1-15
            'pre-diabetic': list(range(16, 32)),  # Participants 16-31
            't2d': list(range(32, MAX_PARTICIPANT_ID + 1))  # Participants 32-45
        }

    @property
    def _expected_file_columns(self) -> list[str]:
        """Columns expected in raw participant CSV files (excludes derived columns)."""
        return [
            c for c in self.required_columns
            if c not in ('participant_id', 'health_group')
        ]

    def download(self) -> bool:
        """
        Download/clone the CGMacros dataset repository.

        Returns:
            True if download successful

        Raises:
            NotImplementedError: CGMacros download is not yet implemented
        """
        raise NotImplementedError(
            "CGMacros dataset download is not yet implemented. "
            "Please provide the dataset files manually or configure a real repository source. "
            "Expected structure: data/cgmacros/participant_*.csv files."
        )

    def validate(self) -> bool:
        """
        Validate that the dataset directory exists, contains parseable participant
        files, and that at least one file has the required columns.

        Returns:
            True if validation passes

        Raises:
            FileNotFoundError: If dataset directory is missing
            ValueError: If dataset is incomplete or no valid participant files found
        """
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        # Use the same regex-based discovery as load() for consistency
        try:
            dataset_entries = os.listdir(self.dataset_dir)
        except OSError as e:
            raise ValueError(
                f"Cannot read dataset directory: {self.dataset_dir}"
            ) from e

        participant_files = [
            f for f in dataset_entries
            if re.fullmatch(PARTICIPANT_FILE_REGEX, f)
        ]
        if len(participant_files) == 0:
            raise ValueError(
                f"No participant CSV files found in dataset directory: {self.dataset_dir}"
            )

        # Filter out dropout participants, consistent with load()
        non_dropout_files = []
        for filename in participant_files:
            match = re.fullmatch(PARTICIPANT_FILE_REGEX, filename)
            if match:
                pid = int(match.group(1))
                if pid not in DROPOUT_PARTICIPANTS:
                    non_dropout_files.append(filename)

        if len(non_dropout_files) == 0:
            raise ValueError(
                f"No non-dropout participant files found in: {self.dataset_dir}"
            )

        # Verify at least one file is parseable with required columns
        valid_count = 0
        for filename in non_dropout_files:
            filepath = os.path.join(self.dataset_dir, filename)
            try:
                sample = pd.read_csv(filepath, nrows=1)
                missing = [c for c in self._expected_file_columns if c not in sample.columns]
                if not missing and not sample.empty:
                    valid_count += 1
            except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError):
                continue

        if valid_count == 0:
            raise ValueError(
                f"No valid participant files with required columns found in: "
                f"{self.dataset_dir}"
            )

        logger.info(
            f"Dataset validation passed: found {valid_count} valid participant files "
            f"out of {len(participant_files)} total"
        )
        return True

    def _categorize_health_group(self, participant_id: int) -> str:
        """
        Categorize participant into health group.

        Args:
            participant_id: Participant ID

        Returns:
            Health group: 'healthy', 'pre-diabetic', or 't2d'
        """
        for group, ids in self.health_groups.items():
            if participant_id in ids:
                return group
        logger.warning(f"Unknown participant ID {participant_id}, cannot assign health group")
        return 'unknown'

    def _parse_participant(self, participant_id: int) -> Optional[pd.DataFrame]:
        """
        Parse data for a single participant.

        Args:
            participant_id: Participant ID

        Returns:
            DataFrame with participant data, or None if file not found or missing columns
        """
        participant_file = os.path.join(self.dataset_dir, f"participant_{participant_id}.csv")

        if not os.path.exists(participant_file):
            logger.debug(f"Participant file not found: {participant_file}")
            return None

        try:
            df = pd.read_csv(participant_file)

            expected_file_columns = self._expected_file_columns
            missing = [c for c in expected_file_columns if c not in df.columns]
            if missing:
                logger.error(f"Participant {participant_id} missing columns: {missing}")
                return None

            df['participant_id'] = participant_id
            df['health_group'] = self._categorize_health_group(participant_id)

            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')

            # Validate glucose range and warn about out-of-range values
            self._validate_glucose_range(
                df, ['glucose'], context=f"Participant {participant_id}"
            )

            return df
        except (
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
            ValueError,
            UnicodeDecodeError,
            OSError,
            TypeError,
        ) as e:
            logger.error(f"Error parsing participant {participant_id}: {e}")
            return None

    def load(self) -> pd.DataFrame:
        """
        Load the CGMacros dataset into a pandas DataFrame.

        Returns:
            DataFrame with columns: participant_id, timestamp, glucose, carbs,
                                   fat, protein, activity, heart_rate, health_group

        Raises:
            FileNotFoundError: If dataset directory is missing
            ValueError: If required columns are missing or no valid data found
        """
        if not os.path.exists(self.dataset_dir):
            error_msg = (f"Dataset directory not found: {self.dataset_dir}. "
                        f"Please provide the dataset files manually.")
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Discover available participant files
        try:
            discovered_ids = []
            for filename in os.listdir(self.dataset_dir):
                match = re.fullmatch(PARTICIPANT_FILE_REGEX, filename)
                if match:
                    discovered_ids.append(int(match.group(1)))
        except OSError as e:
            raise RuntimeError(
                f"Error accessing dataset directory '{self.dataset_dir}': {e}"
            ) from e

        if not discovered_ids:
            raise ValueError(
                f"No participant files found in '{self.dataset_dir}'"
            )
        participant_ids = sorted(set(discovered_ids))

        all_data = []
        for participant_id in participant_ids:
            if participant_id in DROPOUT_PARTICIPANTS:
                logger.info(f"Excluding dropout participant: {participant_id}")
                continue

            df = self._parse_participant(participant_id)
            if df is not None:
                all_data.append(df)

        if len(all_data) == 0:
            raise ValueError("No valid participant data found in dataset")

        combined_df = pd.concat(all_data, ignore_index=True)

        logger.info(f"Loaded {len(combined_df)} records from {len(all_data)} participants")
        return combined_df[self.required_columns]
