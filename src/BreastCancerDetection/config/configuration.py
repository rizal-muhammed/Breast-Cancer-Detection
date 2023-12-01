from pathlib import Path
from BreastCancerDetection.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SECRETS_FILE_PATH
from BreastCancerDetection.utils import common
from BreastCancerDetection.entity import (DataIngestionConfig)


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

        common.create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            data_id=config.data_id,
            destination_folder=config.destination_folder,
            filename=config.filename,
            miscellaneous_folder=config.miscellaneous_folder,
        )
    
        return data_ingestion_config