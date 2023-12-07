from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_id: int
    destination_folder: Path
    filename: str
    miscellaneous_folder: Path

@dataclass(frozen=True)
class DataValidationTrainingConfig:
    root_dir: Path
    good_raw: Path
    bad_raw: Path
    filename_regex: str
    metadata_filename: str
    description_filename: str
    training_source_dir: Path
    number_of_columns: int

@dataclass(frozen=True)
class DataTransformationTrainingConfig:
    good_raw: Path
    bad_raw: Path
    archive_bad_raw: Path
    column_names: list

@dataclass(frozen=True)
class DatabaseOperationsCredentials:
    ASTRA_TOKEN_PATH: Path
    ASTRA_DB_SECURE_BUNDLE_PATH: Path

