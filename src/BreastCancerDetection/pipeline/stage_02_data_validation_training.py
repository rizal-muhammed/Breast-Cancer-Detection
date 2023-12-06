from BreastCancerDetection.config.configuration import ConfigurationManager
from BreastCancerDetection.components.data_validation_training import DataValidationTraining


class DataValidationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_training_config = config.get_data_validation_training_config()
        data_validation_training = DataValidationTraining(
            config=data_validation_training_config)

        data_validation_training.training_raw_file_name_validation()
        data_validation_training.training_validate_column_length()
        data_validation_training.training_validate_missing_values_in_whole_column()
