---
inclusion: auto
description: Defensive coding guidelines for error handling, validation, security, and test hygiene
---

# Defensive Coding Guidelines

## Error Messages

- Error messages must accurately reflect available recovery actions. Do not reference methods that raise `NotImplementedError` as a recovery path.
- When a method is intentionally unimplemented (e.g., `download()`), error messages in dependent methods (e.g., `load()`) should direct users to the actual workaround (e.g., "provide a pre-processed CSV file").

## Constants and Validation

- Define constants for domain-specific validation boundaries (e.g., `MIN_GLUCOSE`, `MAX_GLUCOSE`).
- Constants that are defined must be used in validation logic. Do not leave unused constants in the codebase.
- Out-of-range values should be flagged with `logger.warning()` but not dropped, per the project's data validation rules.
- When the same validation logic applies across multiple subclasses, implement it as a shared helper method in the base class to avoid duplication. Pattern:

  ```python
  # In base class
  def _validate_and_clean_glucose(self, df, glucose_columns, context=""):
      for col in glucose_columns:
          if col not in df.columns:
              continue
          vals = pd.to_numeric(df[col], errors='coerce')
          df[col] = vals  # Ensure numeric types for downstream tasks
          out_of_range = (vals < MIN_GLUCOSE) | (vals > MAX_GLUCOSE)
          n_out = out_of_range.sum()
          if n_out > 0:
              ctx = f"{context}: " if context else ""
              logger.warning("%s%s values in '%s' are out of range ...", ctx, n_out, col)
  ```

- After coercing with `pd.to_numeric(..., errors='coerce')`, write the result back to the DataFrame (`df[col] = vals`) before any further checks. This ensures the DataFrame contains numeric types for downstream mathematical operations, not the original string/object values.
- Use `pd.to_numeric(df[col], errors='coerce')` before numeric comparisons in validation helpers. This converts non-numeric values (e.g., corrupted strings) to `NaN` instead of raising `TypeError`, making the validation robust against mixed-type columns.
- After coercing with `pd.to_numeric(..., errors='coerce')`, explicitly check for and log non-numeric values as a data quality warning. Non-numeric values silently become `NaN` and would otherwise go unnoticed:

  ```python
  original_na = df[col].isna()  # capture before coercion / any reassignment of df[col]
  vals = pd.to_numeric(df[col], errors='coerce')
  n_non_numeric = (vals.isna() & ~original_na).sum()
  if n_non_numeric > 0:
      logger.warning("%s non-numeric values in '%s' were coerced to NaN", n_non_numeric, col)
  ```

## Empty Dataset Rejection

- Both `validate()` and `load()` must reject datasets that contain only headers (zero data rows). Read a sample and check `.empty` before proceeding. A truly empty (0-byte / no-header) file raises `pd.errors.EmptyDataError` from `pd.read_csv()` before `.empty` can be evaluated, so catch it explicitly and re-raise as `ValueError` with the path.
- Pattern for validate:

  ```python
  try:
      sample = pd.read_csv(path, nrows=1)
  except pd.errors.EmptyDataError as e:
      raise ValueError(f"Dataset file is empty (no header or data): {path}") from e
  if sample.empty:
      raise ValueError(f"Dataset file has no data rows (empty): {path}")
  ```

- For per-file parsers that return `None` on failure (e.g., `_parse_participant()`), treat a completely empty file the same as a header-only file: catch `pd.errors.EmptyDataError`, check `df.empty` immediately after `read_csv`, and return `None`. This prevents empty DataFrames from being included in concatenation:

  ```python
  try:
      df = pd.read_csv(filepath)
  except pd.errors.EmptyDataError:
      logger.warning("File is empty (no header or data)")
      return None
  if df.empty:
      logger.warning("File has no data rows (headers only)")
      return None
  ```

- When a multi-file `load()` aggregates per-file parsers that may each return `None`, reject the case where **every** file is skipped/unparseable: if the collected list is empty (or all results are `None`), raise `ValueError` with the dataset path rather than returning or concatenating an empty result. An all-header/all-empty dataset is an incomplete dataset, not a successful empty load:

  ```python
  frames = [df for df in (self._parse_participant(f) for f in files) if df is not None]
  if not frames:
      raise ValueError(f"No parseable data rows found in any file under: {self.dataset_dir}")
  combined = pd.concat(frames, ignore_index=True)
  ```

## Validation Completeness

- `validate()` must check everything that `load()` depends on. If `load()` requires specific columns, `validate()` must verify those columns exist. A file with rows but the wrong schema must not pass validation.
- For single-file datasets, read the header and check required columns:

  ```python
  sample = pd.read_csv(path, nrows=1)
  missing = [col for col in required_columns if col not in sample.columns]
  if missing:
      raise ValueError(f"Missing required columns in {path}: {missing}")
  ```

- For multi-file datasets (e.g., participant files), `validate()` must verify that at least one file is parseable and contains the required columns — not just that files exist.

## Error Type Consistency

- Use consistent exception types for the same logical condition across `validate()` and `load()`:
  - `FileNotFoundError` — the directory or file does not exist at all.
  - `ValueError` — the directory/file exists but the dataset is incomplete, empty, or has wrong schema.
- When a directory exists but contains no valid data files, that is a `ValueError` (incomplete dataset), not a `FileNotFoundError`.

## Datetime Parsing

- Prefer `format='ISO8601'` in `pd.to_datetime()` for ISO 8601 formatted timestamps. This is faster and more reliable than the default inference or `dayfirst=False`.
- Fall back to `dayfirst=False` only when the timestamp format is not ISO 8601.
- Remove redundant column-existence guards when the column has already been validated as present earlier in the method.

## Exception Handling

- Catch specific exception types rather than broad `Exception`. For pandas CSV operations, catch `pd.errors.ParserError` alongside `OSError`.
- **Do not catch `TypeError`** in data loading/parsing exception handlers. `TypeError` is overly broad and can mask programming errors (e.g., calling a method on `None`). If timestamp conversion or other operations raise `TypeError`, let it propagate so the bug is visible.
- **Always catch `UnicodeDecodeError`** when reading CSV files. Binary or invalidly-encoded files can raise `UnicodeDecodeError` from `pd.read_csv()`. This must be caught to maintain the method's documented exception contract.
- **Beware of `UnicodeDecodeError` and `except ValueError` ordering**: In Python, `UnicodeDecodeError` is a subclass of `ValueError`. If you have `except ValueError: raise` before `except UnicodeDecodeError`, the `ValueError` clause catches it first and re-raises the raw `UnicodeDecodeError`. Place `except UnicodeDecodeError` before `except ValueError` in the handler chain. When `ValueError` is only being re-raised (not wrapped), the `except ValueError: raise` clause is redundant and should be removed — `ValueError` will propagate naturally:

  ```python
  except FileNotFoundError:
      raise  # file does not exist — keep FileNotFoundError (it is an OSError subclass)
  except pd.errors.EmptyDataError as e:
      raise ValueError(...) from e
  except UnicodeDecodeError as e:
      raise ValueError(f"Error loading dataset {path}: {e}") from e
  except (OSError, pd.errors.ParserError, ValueError) as e:
      raise ValueError(f"Error loading dataset {path}: {e}") from e
  ```

- Prefer to scope the timestamp-parse `ValueError` **narrowly**: wrap only the `pd.to_datetime(..., format='ISO8601')` call in its own `try/except ValueError` and re-raise with the dataset path. A method-wide `except ValueError` would also catch *unrelated* `ValueError`s and misreport them as load errors, so keep it out of the broad handler:

  ```python
  try:
      df[ts_col] = pd.to_datetime(df[ts_col], format='ISO8601')
  except ValueError as e:
      raise ValueError(f"Invalid timestamp format in dataset {path}: {e}") from e
  ```

  If you instead keep `ValueError` in a method-wide catch-all, place the `UnicodeDecodeError` and `pd.errors.EmptyDataError` handlers before it (since `UnicodeDecodeError` is a `ValueError` subclass and `EmptyDataError` needs distinct messaging).
- When wrapping exceptions into `ValueError`, **include the file/dataset path** in the error message for easier debugging.
- For methods that parse individual files and return `None` on failure (graceful degradation), catch a broad set of file-level exceptions to prevent a single bad file from aborting the entire pipeline:

  ```python
  except (
      pd.errors.ParserError,
      pd.errors.EmptyDataError,
      ValueError,
      UnicodeDecodeError,
      OSError,
  ) as e:
      logger.error("Error parsing file: %s", e)
      return None
  ```

- For methods that call `os.scandir()` or `os.listdir()` on directories that may be unreadable, wrap in try/except and re-raise as a consistent error type matching the method's documented exceptions. `FileNotFoundError` is a subclass of `OSError`, so re-raise it **before** the generic `OSError` branch to preserve it for a missing directory (per Error Type Consistency); the generic branch then maps other `OSError` cases (e.g., `PermissionError`) to `ValueError`:

  ```python
  try:
      with os.scandir(directory) as it:
          for entry in it:
              ...
  except FileNotFoundError:
      raise  # directory does not exist — keep FileNotFoundError
  except OSError as e:
      raise ValueError(f"Cannot access directory: {directory}") from e
  ```

- Prefer `os.scandir()` over `os.listdir()` for directory scanning. `os.scandir()` yields `DirEntry` objects with cached file attributes, avoiding extra system calls. Use `entry.is_file()` to filter out non-files. Use `os.scandir()` as a context manager (`with os.scandir(...) as it:`) to ensure the directory handle is closed promptly, even if an exception occurs mid-iteration:

  ```python
  # Good — efficient, ensures directory handle is closed
  with os.scandir(directory) as it:
      for entry in it:
          if entry.is_file():
              process(entry.name)

  # Less efficient — requires separate os.path.isfile() calls
  for name in os.listdir(directory):
      if os.path.isfile(os.path.join(directory, name)):
          process(name)
  ```

- Do not catch `KeyError` when column presence has already been validated — it is redundant and masks real bugs.
- When **wrapping or translating** an exception into a different type, always use `raise ... from e` to preserve the original traceback. When **propagating the original exception unchanged** (e.g., re-raising `FileNotFoundError` past a broader `OSError` handler), use a bare `raise` — do not attach `from e`.

## File System Security

- When creating directories for sensitive data (health data), enforce permissions explicitly with `os.chmod()` after `os.makedirs()` since the `mode` parameter is subject to umask.
- **Fail closed.** If the required permissions cannot be enforced, do **not** log-and-continue — sensitive health data must never be written to a directory whose access we could not lock down. Abort loader initialization with a fatal error so the failure is loud and no data is ever stored in an unprotected location.
- On POSIX, guard `os.chmod()` behind a POSIX check (`os.name == "posix"`). If `os.chmod()` raises `PermissionError`/`OSError`, wrap it in a fatal error with `raise ... from exc` rather than swallowing it.
- On non-POSIX platforms (e.g., Windows), `os.chmod()` cannot set POSIX permission bits. Restrict access with a platform-appropriate mechanism (e.g., Windows ACLs via `icacls`) before storing sensitive data; if no such mechanism is applied, fail closed rather than leaving the directory world-readable.
- Pattern:

  ```python
  os.makedirs(path, mode=0o700, exist_ok=True)
  if os.name == "posix":
      try:
          os.chmod(path, 0o700)
      except (PermissionError, OSError) as exc:
          # Fail closed: refuse to use a directory we cannot secure for health data.
          raise RuntimeError(
              f"Refusing to use '{path}' for sensitive data: "
              "could not enforce 0o700 permissions"
          ) from exc
  else:
      # Non-POSIX: os.chmod cannot set POSIX bits. Enforce access via platform
      # ACLs (e.g., icacls on Windows) here, or fail closed if that is not possible.
      raise RuntimeError(
          f"Refusing to use '{path}' for sensitive data: POSIX permissions cannot "
          "be enforced on this platform; configure and verify ACLs explicitly"
      )
  ```

- Test the fail-closed path by monkeypatching `os.chmod` to raise `PermissionError` and asserting that loader initialization raises the fatal error — do not rely on real `chmod`, which is flaky when running as root.
- Tests that assert POSIX permission bits must be skipped on non-POSIX platforms with `@pytest.mark.skipif(os.name != "posix", ...)`.
- **Prefer monkeypatch over chmod for permission-error tests.** Tests that use `chmod(0o000)` to trigger `PermissionError` are flaky when running as root or in CI with elevated capabilities. Use `monkeypatch` to inject `PermissionError` deterministically:

  ```python
  # Good — deterministic, works as root
  def test_unreadable_dir_raises_valueerror(self, monkeypatch):
      original_scandir = os.scandir
      def mocked_scandir(path):
          if path == target_dir:
              raise PermissionError("Permission denied")
          return original_scandir(path)
      monkeypatch.setattr(os, "scandir", mocked_scandir)
      with pytest.raises(ValueError, match="Cannot access"):
          loader.validate()

  # Fragile — fails silently as root
  os.chmod(target_dir, 0o000)
  try:
      with pytest.raises(ValueError):
          loader.validate()
  finally:
      os.chmod(target_dir, 0o700)
  ```

## Filename Parsing

- Avoid fixed-offset string slicing (e.g., `s[len("prefix"):-len(".ext")]`) for extracting parts of filenames. It is fragile if the naming convention changes.
- Prefer `re.fullmatch()` or `re.search()` with a capturing group for robust extraction of structured components (e.g., numeric IDs) from filenames.
- Extract repeated regex patterns into module-level constants to ensure consistency across methods (e.g., `validate()` and `load()`).

  ```python
  # Good — module-level constant, used everywhere
  PARTICIPANT_FILE_REGEX = r"participant_(\d+)\.csv$"

  # Good — re.fullmatch for clarity (no need for ^ anchor)
  match = re.fullmatch(PARTICIPANT_FILE_REGEX, filename)
  if match:
      participant_id = int(match.group(1))

  # Acceptable — re.search with anchored pattern
  match = re.search(r'^participant_(\d+)\.csv$', filename)

  # Fragile — breaks if prefix/suffix lengths change
  id_str = filename[len("participant_"):-len(".csv")]
  ```

## Type Annotation Compatibility

- Use `from __future__ import annotations` at the top of modules that use modern type annotation syntax (e.g., `list[str]`, `dict[str, int]`, `X | Y`).
- This ensures deferred annotation evaluation, avoiding runtime evaluation of type expressions.
- Place the `__future__` import immediately after the module docstring, before all other imports.

  ```python
  """Module docstring."""

  from __future__ import annotations

  from abc import ABC, abstractmethod
  import os
  # ... rest of imports
  ```

## Test Hygiene

- Tests should exercise public API behavior only. Avoid mutating internal attributes (e.g., `loader.dataset_path = ...`) when the constructor already sets them correctly.
- Use the constructor's parameters to control behavior in tests.
- When testing that code wraps one exception type into another (e.g., `ParserError` → `ValueError`), assert only the expected wrapper type. Accepting both types defeats the purpose of the test.

  ```python
  # Good — enforces the wrapping behavior
  with pytest.raises(ValueError):
      loader.load()

  # Bad — passes even if wrapping is broken
  with pytest.raises((ValueError, pd.errors.ParserError)):
      loader.load()
  ```

- **Prefer behavioral tests over source inspection.** Do not use `inspect.getsource()` to assert that certain exception types are or aren't in an `except` clause. This is brittle (breaks on harmless refactors/reformatting) and doesn't test runtime behavior. Instead, use `monkeypatch` to force the exception and assert it propagates (or is caught):

  ```python
  # Good — behavioral test using monkeypatch
  def test_load_propagates_typeerror(self, monkeypatch):
      def mock_read_csv(*args, **kwargs):
          raise TypeError("programming error")
      monkeypatch.setattr(pd, "read_csv", mock_read_csv)
      with pytest.raises(TypeError, match="programming error"):
          loader.load()

  # Bad — brittle source inspection
  def test_no_typeerror_in_except(self):
      import inspect
      source = inspect.getsource(SomeClass.load)
      assert 'TypeError' not in source  # breaks on comments, refactors
  ```

## Avoiding Redundant Validation

- Do not re-validate conditions that are guaranteed by prior steps. For example, if `_parse_participant()` already validates that each DataFrame has all required columns, a post-`pd.concat` column check on the combined DataFrame is redundant.
- When the same discovery or filtering logic is used in multiple methods (e.g., `validate()` and `load()`), extract it into a private helper method to avoid duplication and ensure consistency:

  ```python
  # Good — shared helper returning id→filename mapping, single source of truth
  def _discover_participant_ids(self) -> dict[int, str]:
      try:
          id_to_filename: dict[int, str] = {}
          with os.scandir(self.dataset_dir) as it:
              for entry in it:
                  if entry.is_file():
                      match = re.fullmatch(PARTICIPANT_FILE_REGEX, entry.name)
                      if match:
                          pid = int(match.group(1))
                          if pid not in id_to_filename or entry.name == f"participant_{pid}.csv":
                              id_to_filename[pid] = entry.name
          return dict(sorted(id_to_filename.items()))
      except FileNotFoundError:
          raise  # dataset directory does not exist — keep FileNotFoundError
      except OSError as e:
          raise ValueError(f"Cannot access dataset directory: {self.dataset_dir}") from e

  # Then both validate() and load() call self._discover_participant_ids()
  ```

- Discovery helpers should filter out participant IDs outside the valid range (e.g., `1..MAX_PARTICIPANT_ID`). IDs outside this range should be logged as a warning and skipped, not silently included:

  ```python
  if pid < 1 or pid > MAX_PARTICIPANT_ID:
      logger.warning(
          "Skipping participant file '%s': ID %d is outside the valid range [1, %d]",
          entry.name, pid, MAX_PARTICIPANT_ID,
      )
      continue
  ```

- When a discovery helper converts filenames to IDs (e.g., `participant_02.csv` → pid=2), return a mapping of IDs to actual filenames rather than just IDs. This prevents path reconstruction bugs when the actual filename differs from the canonical form (e.g., leading zeros).
- When iterating over a list that was already filtered by a regex, do not re-apply the same regex. Extract captured groups during the initial filtering pass to avoid redundant matching:

  ```python
  # Good — extract IDs during the first pass, no redundant regex
  participant_files = []
  for f in entries:
      m = re.fullmatch(PARTICIPANT_FILE_REGEX, f)
      if m:
          participant_files.append((f, int(m.group(1))))

  non_dropout = [
      (fname, pid) for fname, pid in participant_files
      if pid not in DROPOUT_PARTICIPANTS
  ]

  # Redundant — re-matching a regex that's already been checked.
  # (filenames is a plain list of names, each already matched once above)
  for filename in filenames:
      pid = int(re.fullmatch(PARTICIPANT_FILE_REGEX, filename).group(1))
  ```

- Redundant checks add maintenance burden and can mislead readers into thinking the condition is actually possible.

## Memory Efficiency in DataFrame Operations

- When concatenating multiple DataFrames, filter to only the required columns before `pd.concat` rather than after. This reduces the memory footprint of the intermediate combined DataFrame:

  ```python
  # Good — filter before concat, lower memory usage
  all_data.append(df[self.required_columns])
  combined_df = pd.concat(all_data, ignore_index=True)

  # Less efficient — concat all columns, then filter
  all_data.append(df)
  combined_df = pd.concat(all_data, ignore_index=True)
  return combined_df[self.required_columns]
  ```

- When loading a single-file dataset where required columns are known, use the `usecols` parameter in `pd.read_csv` to avoid loading unnecessary columns into memory. Check the header for required columns first to provide a clear error message, then use `usecols` for the data read:

  ```python
  # Good — check columns first, then read efficiently
  header = pd.read_csv(path, nrows=0)
  missing = [col for col in required_columns if col not in header.columns]
  if missing:
      raise ValueError(f"Missing required columns: {missing}")
  df = pd.read_csv(path, usecols=required_columns)
  ```

## Import Path Configuration

- Do not manipulate `sys.path` in `conftest.py` or other configuration files. Use `pip install -e .` with a `pyproject.toml` for editable installs instead.
- This ensures consistent import behavior across environments and avoids subtle path-ordering bugs.

## Packaging and Dependencies

- Ensure each real package directory under `src/` (e.g., `src/insulin_response/__init__.py`) contains its own `__init__.py` so that `setuptools` can discover it via `find_packages`. Do **not** create `src/__init__.py` — in a src-layout, `src/` is a container directory, not a package itself, and marking it as one breaks package discovery.
- Declare runtime dependencies in `pyproject.toml` under `[project.dependencies]`, not just in `requirements.txt`. This ensures `pip install` pulls in the correct versions.
- Pin minimum versions for dependencies that provide APIs the code relies on (e.g., `pandas>=2.0.0` for `format='ISO8601'` in `pd.to_datetime()`).
- Align `requires-python` with the actual minimum Python version supported by declared dependencies. When a pinned dependency drops older interpreters (e.g., `pandas>=2.1.0` requires Python 3.9+, whereas `pandas>=2.0.0` still supports 3.8), set `requires-python` to match — do not claim `>=3.8` if a dependency needs 3.9+. This prevents install failures on unsupported Python versions.
- Ensure all documentation (README, docstrings, test comments) consistently reflects the actual `requires-python` version floor. After changing `requires-python`, audit for stale references to the old version.
- For stdlib modules introduced in newer Python versions (e.g., `tomllib` in 3.11), use a try/except compatibility import with a backport fallback, and declare the backport as a conditional dependency. Match the dependency's scope to where the import is used: if only test code imports it, keep it in the `test` extra; if application/runtime code imports it, it must be a runtime dependency instead.

  ```python
  # In test code
  try:
      import tomllib
  except ModuleNotFoundError:
      import tomli as tomllib
  ```

  ```toml
  # In pyproject.toml — test-only usage
  [project.optional-dependencies]
  test = ["pytest", "hypothesis", "tomli; python_version < '3.11'"]

  # In pyproject.toml — if imported by application/runtime code instead
  [project]
  dependencies = ["tomli; python_version < '3.11'"]
  ```

- When a test only needs to verify that a value exists in `pyproject.toml` (not parse its structure), prefer a simple string check over importing a TOML parser. This avoids adding `tomli` as a test dependency for older Python versions:

  ```python
  # Good — no TOML parser needed
  with open("pyproject.toml", "r") as f:
      content = f.read()
  assert '"pandas>=2.0.0"' in content or "'pandas>=2.0.0'" in content

  # Overkill for simple checks — requires tomli on Python <3.11
  import tomllib
  config = tomllib.load(open("pyproject.toml", "rb"))
  ```

## Property Test Hygiene

- When using Hypothesis `@given` with pytest's `tmp_path` fixture, suppress the `function_scoped_fixture` health check since `tmp_path` is not reset between generated inputs.
- For property tests that perform filesystem I/O or pandas parsing, set `deadline=None` in the `@settings` decorator to prevent flaky `DeadlineExceeded` failures on slower CI runners:

  ```python
  @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
  @given(data=my_strategy())
  def test_property(data, tmp_path):
      ...
  ```

- If the test writes files to `tmp_path`, clean up between iterations using `shutil.rmtree` followed by `os.makedirs` for robustness:

  ```python
  import shutil
  shutil.rmtree(target_dir, ignore_errors=True)
  os.makedirs(target_dir, exist_ok=True)
  ```

## Logging Consistency

- Log messages that report counts must use consistent denominators. If a count is computed over a filtered subset (e.g., non-dropout files), the "total" in the message must reflect that same subset, not the unfiltered set.

  ```python
  # Good — both numbers refer to non-dropout files
  f"found {valid_count} valid non-dropout files out of {len(non_dropout_files)} total"

  # Bad — valid_count is from non-dropout, but total includes dropouts
  f"found {valid_count} valid files out of {len(all_files)} total"
  ```

- Use lazy `%`-style formatting for `logger.warning()` and other log calls instead of f-strings. This avoids eagerly formatting the message when the log level is disabled, reducing overhead:

  ```python
  # Good — lazy formatting, no cost when WARNING is disabled
  logger.warning(
      "%s%s values in '%s' are out of range [%s, %s] mg/dL",
      ctx, n_out, col, MIN_GLUCOSE, MAX_GLUCOSE,
  )

  # Bad — f-string always formats, even when WARNING is disabled
  logger.warning(
      f"{ctx}{n_out} values in '{col}' are out of range "
      f"[{MIN_GLUCOSE}, {MAX_GLUCOSE}] mg/dL"
  )
  ```

## Library-Safe Logging

- When adding a `NullHandler` to a library logger at import time, guard the call with `if not logger.handlers` to prevent accumulating duplicate handlers on module reload (e.g., by `importlib.reload()` or certain test runners):

  ```python
  # Good — no duplicates on reload
  logger = logging.getLogger(__name__)
  if not logger.handlers:
      logger.addHandler(logging.NullHandler())

  # Bad — accumulates NullHandlers on every reload
  logger = logging.getLogger(__name__)
  logger.addHandler(logging.NullHandler())
  ```

## Task Tracking

- Task checkboxes in `tasks.md` must reflect the actual state of the code. If a method raises `NotImplementedError`, the corresponding task should note this limitation rather than being marked as fully complete without qualification.

## Test Documentation Accuracy

- Test class and method docstrings must match the actual behavior being tested. If the implementation uses `format='ISO8601'`, the test docstring should not say `dayfirst=False`.
- When the implementation changes (e.g., exception type changes from `RuntimeError` to `ValueError`), update all test docstrings that reference the old behavior.
- Remove outdated inline comments that describe behavior that has already been fixed. Stale comments mislead future maintainers into thinking the fix is still pending.
