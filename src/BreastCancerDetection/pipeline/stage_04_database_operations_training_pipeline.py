
from BreastCancerDetection.config.configuration import ConfigurationManager
from BreastCancerDetection.components.database_operations_training import DatabaseOperations
from BreastCancerDetection.components.data_transformation_training import DataTransformationTraining


class DatabaseOperationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        database_operations_training_config = config.get_data_base_operations_trainig_config()
        database_operations_training_credentials = config.get_data_base_operations_training_credentials()
        database_operations_training_params = config.get_data_base_operations_training_params()

        db_ops = DatabaseOperations(config=database_operations_training_config,
                                    credentials=database_operations_training_credentials,
                                    params=database_operations_training_params)

        db_ops.create_table_db()
        db_ops.insert_into_table_good_data()

        data_transformation_training_config = config.get_data_transformation_training_config()
        data_transformation_training = DataTransformationTraining(config=data_transformation_training_config)
        data_transformation_training.move_bad_files_to_archive_bad()
        data_transformation_training.delete_exising_bad_data_training_folder()
        data_transformation_training.delete_existing_good_data_training_folder()

        db_ops.export_data_from_table_into_csv()
        db_ops.shutdown_driver()




