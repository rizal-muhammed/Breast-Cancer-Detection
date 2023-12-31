from pathlib import Path
from BreastCancerDetection.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SECRETS_FILE_PATH
from BreastCancerDetection.utils import common
from BreastCancerDetection.entity import (DataIngestionConfig, 
                                        DataValidationTrainingConfig,
                                        DataTransformationTrainingConfig,
                                        DataBaseOperationsTrainingConfig,
                                        DataBaseOperationsTrainingCredentials,
                                        DataBaseOperationsTrainingParams,
                                        DataPreProcessingTrainingConfig,
                                        DataPreProcessingTrainingParams,
                                        ModelTrainingConfig,
                                        ModelTrainingParams,
                                        ModelEvaluationConfig)


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH,
                 secrets_filepath=SECRETS_FILE_PATH) -> None:
        self.config = common.read_yaml(config_filepath)
        self.params = common.read_yaml(params_filepath)
        self.credentials = common.read_yaml(secrets_filepath)

        common.create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            data_id=config.data_id,
            destination_folder=config.destination_folder,
            filename=config.filename,
            miscellaneous_folder=config.miscellaneous_folder,
        )
    
        return data_ingestion_config
    
    def get_data_validation_training_config(self, ) -> DataValidationTrainingConfig:
        config = self.config.data_validation_training

        data_validation_training_config = DataValidationTrainingConfig(
            root_dir = config.root_dir,
            good_raw = config.good_raw,
            bad_raw = config.bad_raw,
            filename_regex = config.filename_regex,
            metadata_filename = config.metadata_filename,
            description_filename = config.description_filename,
            training_source_dir = config.training_source_dir,
            number_of_columns = config.number_of_columns,
        )

        return data_validation_training_config
    
    def get_data_transformation_training_config(self, ) -> DataTransformationTrainingConfig:
        config = self.config.data_transformation_training

        data_transformation_training_config = DataTransformationTrainingConfig(
            good_raw = config.good_raw, 
            bad_raw = config.bad_raw, 
            archive_bad_raw = config.archive_bad_raw, 
            column_names = config.column_names)
        
        return data_transformation_training_config
    
    def get_data_base_operations_trainig_config(self, ) -> DataBaseOperationsTrainingConfig:
        config = self.config.database_operations_training

        data_base_operations_training_config = DataBaseOperationsTrainingConfig(
            root_dir = config.root_dir,
            file_name = config.file_name,
            good_raw = config.good_raw,
            bad_raw = config.bad_raw,
        )

        return data_base_operations_training_config

    def get_data_base_operations_training_credentials(self, ) -> DataBaseOperationsTrainingCredentials:
        credentials = self.credentials.database_credentials

        data_base_operations_training_credentials = DataBaseOperationsTrainingCredentials(
            ASTRA_TOKEN_PATH = credentials.ASTRA_TOKEN_PATH,
            ASTRA_DB_SECURE_BUNDLE_PATH = credentials.ASTRA_DB_SECURE_BUNDLE_PATH,
        )

        return data_base_operations_training_credentials
    
    def get_data_base_operations_training_params(self, ) -> DataBaseOperationsTrainingParams:
        db_params = self.params.database_insertion_training_params

        data_base_operations_training_params = DataBaseOperationsTrainingParams(
            ASTRA_DB_KEYSPACE = db_params.ASTRA_DB_KEYSPACE,
            db_name = db_params.db_name,
            table_name = db_params.table_name,
            column_names = db_params.column_names
        )

        return data_base_operations_training_params

    def get_data_preprocessing_training_config(self) -> DataPreProcessingTrainingConfig:
        data_preprocessing_training = self.config.data_preprocessing_training

        data_preprocessing_training_config = DataPreProcessingTrainingConfig(
            root_dir=data_preprocessing_training.root_dir,
            input_file_path=Path(data_preprocessing_training.input_file_path),
            correlation_dir=data_preprocessing_training.correlation_dir,
            preprocessed_input_data_dir=data_preprocessing_training.preprocessed_input_data_dir,
            test_set_dir=Path(data_preprocessing_training.test_set_dir),
        )

        return data_preprocessing_training_config

    def get_data_preprocessing_training_params(self) -> DataPreProcessingTrainingParams:
        data_preprocessing_training = self.params.data_preprocessing_training_params

        data_preprocessing_training_params = DataPreProcessingTrainingParams(
            label_column_name=data_preprocessing_training.label_column_name,
            row_threshold=data_preprocessing_training.row_threshold,
            test_size=data_preprocessing_training.test_size,
        )

        return data_preprocessing_training_params
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        model_training = self.config.model_training

        model_training_config = ModelTrainingConfig(
            root_dir=Path(model_training.root_dir),
            models_dir=Path(model_training.models_dir),
            path_to_preprocessed_train_X=model_training.path_to_preprocessed_train_X,
            path_to_preprocessed_train_y=model_training.path_to_preprocessed_train_y,
            best_models_dir=Path(model_training.best_models_dir),
            final_model_dir=Path(model_training.final_model_dir),
        )

        return model_training_config
    
    def get_model_training_params(self) -> ModelTrainingParams:
        model_training = self.params.model_training_params

        model_training_params = ModelTrainingParams(
            cv=model_training.cv,
            linear_regression_params=dict(model_training.linear_regression_params),
            sgd_classifier_params=dict(model_training.sgd_classifier_params),
            random_forest_classifier_params=dict(model_training.random_forest_classifier_params),
        )

        return model_training_params
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        model_evaluation = self.config.model_evaluation

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(model_evaluation.root_dir),
            final_model=Path(model_evaluation.final_model),
            test_set_feautres=Path(model_evaluation.test_set_feautres),
            test_set_label=Path(model_evaluation.test_set_label),
            label_column_name=str(model_evaluation.label_column_name),
            columns_with_zero_std_dev_path=Path(model_evaluation.columns_with_zero_std_dev_path),
            knn_imputer_path=Path(model_evaluation.knn_imputer_path),
            std_scaler_path=Path(model_evaluation.std_scaler_path),
        )

        return model_evaluation_config
