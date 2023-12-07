from BreastCancerDetection.config.configuration import ConfigurationManager
from BreastCancerDetection.components.db_ops_training import DatabaseOperations


class DatabaseOperationsTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_base_operations_training_credentials = config.get_database_operations_credentials()

        data_base_training = DatabaseOperations(credentials=data_base_operations_training_credentials)
        data_base_training.database_connection_establishment()