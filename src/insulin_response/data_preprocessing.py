"""
Data preprocessing module for insulin response modeling system.

This module provides data loading capabilities for both
Track A (UCI Diabetes dataset) and Track B (CGMacros dataset).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
import re
from typing import Optional

import pandas as pd

# Library-safe logging: let callers configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# Constants
MIN_GLUCOSE = 20  # mg/dL
MAX_GLUCOSE = 600  # mg/dL
DROPOUT_PARTICIPANTS = [24, 25, 37, 40]
MAX_PARTICIPANT_ID = 49  # Inclusive upper bound for CGMacros participant IDs (originals 1-49; 24, 25, 37, 40 are dropouts)
PARTICIPANT_FILE_REGEX = r"participant_(\d+)\.csv$"


def glucose_flag_column(column: str) -> str:
    """
    Name of the derived per-row out-of-range flag for a glucose column.

    These flags are derived outputs, not part of the input contract: loaders add
    them to the returned DataFrame but never require them in source files.
    """
    return f"{column}_out_of_range"


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
        # Restrict permissions to 0o700 for health data privacy. Fail closed: if the
        # directory cannot be locked down, refuse to use it rather than risk storing
        # sensitive health data in a location whose access we could not secure.
        os.makedirs(data_dir, mode=0o700, exist_ok=True)
        if os.name == "posix":
            try:
                os.chmod(data_dir, 0o700)
            except (PermissionError, OSError) as exc:
                raise RuntimeError(
                    f"Refusing to use '{data_dir}' for sensitive data: "
                    "could not enforce 0o700 permissions"
                ) from exc
        else:
            # os.chmod cannot set POSIX bits on non-POSIX platforms. Access must be
            # restricted via a platform-appropriate mechanism (e.g. Windows ACLs)
            # before storing sensitive data; until that is in place, fail closed.
            raise RuntimeError(
                f"Refusing to use '{data_dir}' for sensitive data: POSIX permissions "
                "cannot be enforced on this platform; configure and verify ACLs explicitly"
            )

    @property
    def output_columns(self) -> list[str]:
        """
        Columns returned by load(): the required columns plus the derived
        out-of-range flag for each glucose column. `required_columns` remains the
        input contract, so derived flags are never demanded of source files.
        """
        return self.required_columns + [
            glucose_flag_column(c) for c in self.glucose_columns
        ]

    def _validate_and_clean_glucose(self, df: pd.DataFrame, glucose_columns: list[str],
                                    context: str = "") -> None:
        """
        Validate and clean glucose columns: coerce to numeric types, log warnings
        for out-of-range values. Non-numeric values are coerced to NaN and
        logged as a data quality warning.

        For each glucose column present, adds a boolean flag column (see
        `glucose_flag_column`) marking rows outside the physiological range, so
        downstream consumers can exclude them without re-implementing the check.
        Missing values (NaN) are flagged False — absent, not out of range.

        Args:
            df: DataFrame containing glucose columns to validate
            glucose_columns: List of column names containing glucose values
            context: Optional context string for log messages (e.g., participant ID)
        """
        for col in glucose_columns:
            if col not in df.columns:
                continue

            ctx = f"{context}: " if context else ""
            if not pd.api.types.is_numeric_dtype(df[col]):
                vals = pd.to_numeric(df[col], errors='coerce')
                # Warn about non-numeric values that were coerced to NaN
                n_non_numeric = vals.isna().sum() - df[col].isna().sum()
                df[col] = vals  # Ensure numeric types for downstream tasks
                if n_non_numeric > 0:
                    logger.warning(
                        "%s%s non-numeric values in '%s' were coerced to NaN",
                        ctx,
                        n_non_numeric,
                        col,
                    )
            else:
                vals = df[col]

            # NaN compares False on both sides, so missing values are not flagged.
            out_of_range = (vals < MIN_GLUCOSE) | (vals > MAX_GLUCOSE)
            df[glucose_flag_column(col)] = out_of_range

            n_out = out_of_range.sum()
            if n_out > 0:
                logger.warning(
                    "%s%s values in '%s' are out of range [%s, %s] mg/dL",
                    ctx,
                    n_out,
                    col,
                    MIN_GLUCOSE,
                    MAX_GLUCOSE,
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
        self.glucose_columns = ['pre_meal_glucose', 'post_meal_glucose']

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

            try:
                sample = pd.read_csv(self.dataset_path, nrows=1)
            except pd.errors.EmptyDataError as e:
                raise ValueError(
                    f"Dataset file is empty (no header or data): {self.dataset_path}"
                ) from e

            if sample.empty:
                raise ValueError(f"Dataset file has no data rows (empty): {self.dataset_path}")

            missing_columns = [col for col in self.required_columns if col not in sample.columns]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in {self.dataset_path}: {missing_columns}"
                )

            # Parse the whole timestamp column with the same format load() uses, so
            # validate() cannot pass on a file whose timestamps load() would reject.
            # Sampling only the first row would report success on exactly the
            # corrupted files this preflight exists to catch; reading one column is
            # cheap next to the full load().
            # Read outside the handler below: UnicodeDecodeError subclasses
            # ValueError, so a decode failure here would otherwise be reported as a
            # timestamp-format problem instead of falling through to the corruption
            # handler that names the real cause.
            timestamps = pd.read_csv(self.dataset_path, usecols=['meal_timestamp'])
            try:
                pd.to_datetime(timestamps['meal_timestamp'], format='ISO8601')
            except ValueError as e:
                raise ValueError(
                    f"Invalid timestamp format in dataset {self.dataset_path}: {e}"
                ) from e

            logger.info("Dataset validation passed: %s", self.dataset_path)
            return True
        except FileNotFoundError:
            raise  # file does not exist — keep FileNotFoundError
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as e:
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
            # Check columns first for a clear error message, then use usecols for memory efficiency
            header = pd.read_csv(self.dataset_path, nrows=0)
            missing_columns = [col for col in self.required_columns if col not in header.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            df = pd.read_csv(self.dataset_path, usecols=self.required_columns)

            # Scope this ValueError narrowly to the timestamp parse so the broad
            # handler below cannot swallow unrelated ValueErrors.
            try:
                df['meal_timestamp'] = pd.to_datetime(df['meal_timestamp'], format='ISO8601')
            except ValueError as e:
                raise ValueError(
                    f"Invalid timestamp format in dataset {self.dataset_path}: {e}"
                ) from e

            # Validate glucose ranges and warn about out-of-range values
            self._validate_and_clean_glucose(df, self.glucose_columns)

            if df.empty:
                raise ValueError(f"Dataset file has no data rows (empty): {self.dataset_path}")

            logger.info("Loaded %d records from UCI Diabetes dataset", len(df))
            return df[self.output_columns]

        except FileNotFoundError:
            raise  # file does not exist — keep FileNotFoundError
        except pd.errors.EmptyDataError as e:
            raise ValueError(f"Dataset file is empty or corrupted: {self.dataset_path}") from e
        except UnicodeDecodeError as e:
            raise ValueError(f"Error loading dataset {self.dataset_path}: {e}") from e
        except (OSError, pd.errors.ParserError) as e:
            raise ValueError(f"Error loading dataset {self.dataset_path}: {e}") from e


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
        self.glucose_columns = ['glucose']

        # Health group categorization based on CGMacros dataset. The released
        # cohort uses original participant IDs 1-49 with 24, 25, 37, 40 as
        # dropouts; after excluding those this yields 15 healthy, 16 pre-diabetic,
        # 14 Type 2 diabetic.
        self.health_groups = {
            'healthy': list(range(1, 16)),  # IDs 1-15 -> 15 healthy
            'pre-diabetic': list(range(16, 34)),  # IDs 16-33 minus dropouts 24, 25 -> 16
            't2d': list(range(34, MAX_PARTICIPANT_ID + 1))  # IDs 34-49 minus dropouts 37, 40 -> 14
        }

    @property
    def _expected_file_columns(self) -> list[str]:
        """Columns expected in raw participant CSV files (excludes derived columns)."""
        return [
            c for c in self.required_columns
            if c not in ('participant_id', 'health_group')
        ]

    @property
    def _layout_hint(self) -> str:
        """
        Description of the on-disk layout this loader expects.

        Derived from `_expected_file_columns` so the guidance in error messages
        cannot drift from the contract actually enforced when parsing. Required by
        Requirement 1.7: an unavailable dataset must report both the expected path
        and the required layout.
        """
        return (
            f"Expected layout: one 'participant_<id>.csv' per participant "
            f"(IDs 1-{MAX_PARTICIPANT_ID}) directly inside '{self.dataset_dir}', "
            f"each containing columns: {', '.join(self._expected_file_columns)}."
        )

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
            "Please provide the dataset files manually or configure a real "
            f"repository source. {self._layout_hint}"
        )

    def _discover_participant_ids(self) -> dict[int, str]:
        """
        Discover available participant IDs and their filenames from the dataset directory.

        Scans the dataset directory for files matching the participant filename
        pattern and returns a mapping of numeric IDs to actual filenames.
        This preserves the real filename (e.g., participant_02.csv) so that
        subsequent reads use the correct path even for non-canonical names.

        Returns:
            Dict mapping participant IDs to their actual filenames, sorted by ID

        Raises:
            ValueError: If the dataset directory cannot be read
        """
        try:
            # Collect every candidate filename per participant ID first, so that
            # alias resolution does not depend on os.scandir() iteration order.
            candidates: dict[int, list[str]] = {}
            with os.scandir(self.dataset_dir) as it:
                for entry in it:
                    if entry.is_file():
                        match = re.fullmatch(PARTICIPANT_FILE_REGEX, entry.name)
                        if match:
                            pid = int(match.group(1))
                            if pid < 1 or pid > MAX_PARTICIPANT_ID:
                                logger.warning(
                                    "Skipping participant file '%s': ID %d is outside "
                                    "the valid range [1, %d]",
                                    entry.name, pid, MAX_PARTICIPANT_ID,
                                )
                                continue
                            candidates.setdefault(pid, []).append(entry.name)
        except FileNotFoundError:
            raise  # directory does not exist — keep FileNotFoundError
        except OSError as e:
            raise ValueError(
                f"Cannot access dataset directory: {self.dataset_dir}"
            ) from e

        # Resolve each ID to a single filename. When multiple files map to the
        # same ID (e.g. participant_2.csv and participant_02.csv), prefer the
        # canonical participant_{pid}.csv form. Reject genuinely ambiguous cases
        # where two or more distinct non-canonical filenames map to the same ID,
        # regardless of the order they were scanned in.
        id_to_filename: dict[int, str] = {}
        for pid, names in candidates.items():
            canonical = f"participant_{pid}.csv"
            if canonical in names:
                id_to_filename[pid] = canonical
            elif len(names) == 1:
                id_to_filename[pid] = names[0]
            else:
                raise ValueError(
                    f"Ambiguous participant files map to ID {pid}: "
                    f"{sorted(names)}. Remove or rename all but one so a single "
                    f"file maps to each participant ID."
                )
        return dict(sorted(id_to_filename.items()))

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
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dataset_dir}. {self._layout_hint}"
            )

        # Use shared discovery helper for consistency with load()
        id_to_filename = self._discover_participant_ids()

        if len(id_to_filename) == 0:
            raise ValueError(
                f"No participant CSV files found in dataset directory: {self.dataset_dir}"
            )

        # Filter out dropout participants, consistent with load()
        non_dropout = {
            pid: fname for pid, fname in id_to_filename.items()
            if pid not in DROPOUT_PARTICIPANTS
        }

        if len(non_dropout) == 0:
            raise ValueError(
                f"No non-dropout participant files found in: {self.dataset_dir}"
            )

        # Verify at least one file fully parses. Use the same _parse_participant()
        # path as load() (required-column checks, timestamp parsing, and exception
        # handling) so validate() cannot pass on files load() would reject or skip.
        valid_count = sum(
            1
            for pid, fname in non_dropout.items()
            if self._parse_participant(pid, filename=fname) is not None
        )

        if valid_count == 0:
            raise ValueError(
                f"No valid participant files with required columns found in: "
                f"{self.dataset_dir}"
            )

        logger.info(
            "Dataset validation passed: found %d valid non-dropout "
            "participant files out of %d total",
            valid_count,
            len(non_dropout),
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
        logger.warning(
            "Unknown participant ID %s, cannot assign health group", participant_id
        )
        return 'unknown'

    def _parse_participant(self, participant_id: int,
                          filename: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Parse data for a single participant.

        Args:
            participant_id: Participant ID
            filename: Actual filename to read (defaults to participant_{id}.csv)

        Returns:
            DataFrame with participant data, or None if file not found or missing columns
        """
        if filename is None:
            filename = f"participant_{participant_id}.csv"
        participant_file = os.path.join(self.dataset_dir, filename)

        if not os.path.exists(participant_file):
            logger.debug("Participant file not found: %s", participant_file)
            return None

        try:
            df = pd.read_csv(participant_file)

            if df.empty:
                logger.warning(
                    "Participant %s file has no data rows (headers only)", participant_id
                )
                return None

            expected_file_columns = self._expected_file_columns
            missing = [c for c in expected_file_columns if c not in df.columns]
            if missing:
                logger.error(
                    "Participant %s missing columns: %s", participant_id, missing
                )
                return None

            df['participant_id'] = participant_id
            df['health_group'] = self._categorize_health_group(participant_id)

            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')

            # Time-series splits downstream assume each participant's rows are in
            # chronological order; file order is not guaranteed to be. Sort here so a
            # mis-ordered file cannot leak future readings into a training split.
            if not df['timestamp'].is_monotonic_increasing:
                logger.warning(
                    "Participant %s rows were not in chronological order; sorting by timestamp",
                    participant_id,
                )
                df = df.sort_values('timestamp', ignore_index=True)

            # Validate and clean glucose values, warn about out-of-range values
            self._validate_and_clean_glucose(
                df, self.glucose_columns, context=f"Participant {participant_id}"
            )

            return df
        except (
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
            ValueError,
            UnicodeDecodeError,
            OSError,
        ) as e:
            logger.error("Error parsing participant %s: %s", participant_id, e)
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
                        f"Please provide the dataset files manually. "
                        f"{self._layout_hint}")
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Discover available participant files using shared helper
        id_to_filename = self._discover_participant_ids()

        if not id_to_filename:
            raise ValueError(
                f"No participant files found in '{self.dataset_dir}'"
            )

        all_data = []
        for participant_id, filename in id_to_filename.items():
            if participant_id in DROPOUT_PARTICIPANTS:
                logger.info("Excluding dropout participant: %s", participant_id)
                continue

            df = self._parse_participant(participant_id, filename=filename)
            if df is not None:
                all_data.append(df[self.output_columns])

        if len(all_data) == 0:
            raise ValueError(
                f"No parseable participant data found in any file under: {self.dataset_dir}"
            )

        combined_df = pd.concat(all_data, ignore_index=True)

        logger.info(
            "Loaded %d records from %d participants", len(combined_df), len(all_data)
        )
        return combined_df
