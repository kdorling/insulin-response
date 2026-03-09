"""
Data preprocessing module for insulin response modeling system.

This module provides data loading and feature engineering capabilities for both
Track A (UCI Diabetes dataset) and Track B (CGMacros dataset).
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional
import os
import logging

# Library-safe logging: let callers configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

# Constants
MIN_GLUCOSE = 20  # mg/dL
MAX_GLUCOSE = 600  # mg/dL
DROPOUT_PARTICIPANTS = [24, 25, 37, 40]


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
        os.makedirs(data_dir, exist_ok=True)

    @abstractmethod
    def download(self) -> bool:
        """
        Download the dataset from its source.

        Returns:
            True if download successful, False otherwise

        Raises:
            ConnectionError: If network errors occur after retries
            FileNotFoundError: If source URL is invalid
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
            ConnectionError: If network errors occur after retries
        """
        import urllib.request
        import urllib.error
        import time

        # UCI Diabetes dataset URL (placeholder - actual URL would be used in production)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/diabetes/diabetes.data"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading UCI Diabetes dataset (attempt {attempt + 1}/{max_retries})...")
                urllib.request.urlretrieve(url, self.dataset_path)
                logger.info(f"Dataset downloaded successfully to {self.dataset_path}")
                return True
            except urllib.error.URLError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Download failed: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Download failed after {max_retries} attempts")
                    raise ConnectionError(f"Failed to download dataset after {max_retries} attempts: {e}") from e

    def validate(self) -> bool:
        """
        Validate that the dataset file exists and is not corrupted.

        Returns:
            True if validation passes

        Raises:
            FileNotFoundError: If dataset file is missing
            ValueError: If dataset is corrupted or empty
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        try:
            file_size = os.path.getsize(self.dataset_path)
            if file_size == 0:
                raise ValueError(f"Dataset file is empty: {self.dataset_path}")

            # Try to read first few lines to check for corruption
            with open(self.dataset_path, 'r') as f:
                f.read(100)

            logger.info(f"Dataset validation passed: {self.dataset_path}")
            return True
        except ValueError:
            raise
        except OSError as e:
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
                                   f"Please run download() first.")

        try:
            df = pd.read_csv(self.dataset_path)

            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            if 'meal_timestamp' in df.columns:
                df['meal_timestamp'] = pd.to_datetime(df['meal_timestamp'])

            logger.info(f"Loaded {len(df)} records from UCI Diabetes dataset")
            return df[self.required_columns]

        except pd.errors.EmptyDataError as e:
            raise ValueError(f"Dataset file is empty or corrupted: {self.dataset_path}") from e
        except ValueError:
            raise
        except Exception as e:
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
            't2d': list(range(32, 46))  # Participants 32-45
        }

    def download(self) -> bool:
        """
        Download/clone the CGMacros dataset repository.

        Returns:
            True if download successful

        Raises:
            ConnectionError: If network errors occur after retries
        """
        import subprocess
        import time

        # CGMacros dataset repository URL (placeholder - replace with real URL when available)
        repo_url = "https://github.com/example/cgmacros.git"

        if os.path.exists(self.dataset_dir):
            logger.info(f"CGMacros dataset already exists at {self.dataset_dir}")
            return True

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Cloning CGMacros repository (attempt {attempt + 1}/{max_retries})...")
                subprocess.run(
                    ['git', 'clone', repo_url, '--', self.dataset_dir],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info(f"Repository cloned successfully to {self.dataset_dir}")
                return True
            except subprocess.CalledProcessError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Clone failed: {e}. Stderr: {e.stderr}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Clone failed after {max_retries} attempts")
                    logger.error("CGMacros dataset unavailable. Please provide synthetic data or manual download.")
                    raise ConnectionError(f"Failed to clone repository after {max_retries} attempts: {e}") from e
            except FileNotFoundError as e:
                logger.error("Git is not installed. Please install git or download the dataset manually.")
                raise ConnectionError("Git command not found. Please install git.") from e

    def validate(self) -> bool:
        """
        Validate that the dataset directory exists and contains participant files.

        Returns:
            True if validation passes

        Raises:
            FileNotFoundError: If dataset directory is missing
            ValueError: If dataset is incomplete
        """
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        csv_files = [f for f in os.listdir(self.dataset_dir) if f.endswith('.csv')]
        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in dataset directory: {self.dataset_dir}")

        logger.info(f"Dataset validation passed: found {len(csv_files)} participant files")
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
        return 'unknown'

    def _parse_participant(self, participant_id: int) -> Optional[pd.DataFrame]:
        """
        Parse data for a single participant.

        Args:
            participant_id: Participant ID

        Returns:
            DataFrame with participant data, or None if file not found
        """
        participant_file = os.path.join(self.dataset_dir, f"participant_{participant_id}.csv")

        if not os.path.exists(participant_file):
            logger.debug(f"Participant file not found: {participant_file}")
            return None

        try:
            df = pd.read_csv(participant_file)
            df['participant_id'] = participant_id
            df['health_group'] = self._categorize_health_group(participant_id)

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            return df
        except Exception as e:
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
                        f"Please run download() first or provide synthetic data.")
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Discover available participant files rather than assuming range 1..45
        try:
            discovered_ids = []
            for filename in os.listdir(self.dataset_dir):
                if not (filename.startswith("participant_") and filename.endswith(".csv")):
                    continue
                id_str = filename[len("participant_"):-len(".csv")]
                if id_str.isdigit():
                    discovered_ids.append(int(id_str))
        except OSError as e:
            logger.warning(f"Failed to list dataset directory '{self.dataset_dir}': {e}")
            discovered_ids = []

        # Fall back to full range if no files discovered
        participant_ids = sorted(set(discovered_ids)) if discovered_ids else list(range(1, 46))

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

        missing_columns = [col for col in self.required_columns if col not in combined_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        logger.info(f"Loaded {len(combined_df)} records from {len(all_data)} participants")
        return combined_df[self.required_columns]
