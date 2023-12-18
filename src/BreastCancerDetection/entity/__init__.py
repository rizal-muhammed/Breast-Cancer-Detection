from dataclasses import dataclass
from pathlib import Path
import yaml

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
class DataBaseOperationsTrainingConfig:
    root_dir: Path
    file_name: str
    good_raw: Path
    bad_raw: Path

@dataclass(frozen=True)
class DataBaseOperationsTrainingCredentials:
    ASTRA_TOKEN_PATH: Path
    ASTRA_DB_SECURE_BUNDLE_PATH: Path

@dataclass(frozen=True)
class DataBaseOperationsTrainingParams:
    ASTRA_DB_KEYSPACE: str
    db_name: str
    table_name: str
    column_names: dict

@dataclass(frozen=True)
class DataPreProcessingTrainingConfig:
    root_dir: Path
    input_file_path: Path
    correlation_dir: Path
    preprocessed_input_data_dir: Path
    test_set_dir: Path

@dataclass(frozen=True)
class DataPreProcessingTrainingParams:
    label_column_name: str
    row_threshold: int
    test_size: float


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    models_dir: Path
    path_to_preprocessed_train_X: Path
    path_to_preprocessed_train_y: Path
    best_models_dir: Path
    final_model_dir: Path

@dataclass(frozen=True)
class ModelTrainingParams:
    cv: int
    linear_regression_params: dict
    sgd_classifier_params: dict
    random_forest_classifier_params: dict
