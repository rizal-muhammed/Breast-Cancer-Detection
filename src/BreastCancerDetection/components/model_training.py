import numpy as np
import pandas as pd


from BreastCancerDetection.logging import logger
from BreastCancerDetection.utils import common
from BreastCancerDetection.entity import ModelTrainingConfig, ModelTrainingParams 



class ModelTraining:
    def __init__(self,
                 config: ModelTrainingConfig,
                 params:ModelTrainingParams,) -> None:
        self.config = config
        self.params = params

        common.create_directories([self.config.root_dir, self.config.models_dir])
    
    def load_data(self):
        """
            This method is used to load the pre-processed data for training.

            Parameters
            ----------
            None

            Returns
            -------
            X_train : pandas DataFrame like
                Ground truth (correct) input training features
            X_val : pandas DataFrame like
                Ground truth (correct) input validation features
            y_train : pandas DataFrame like
                Ground truth (correct) training target values
            y_val : pandas DataFrame like
                Ground truth (correct) validation target values

            Raises
            ------
            Exception

            Notes
            ------
        
        """
        try:
            logger.info(f"""Loading pre-processed data for model training started...""")

            X_train = pd.read_csv(self.config.preprocessed_train_X)
            X_val = pd.read_csv(self.config.preprocessed_val_X)
            y_train = pd.read_csv(self.config.preprocessed_train_y)
            y_val = pd.read_csv(self.config.preprocessed_val_y)

            logger.info(f"""Loading pre-processed data for model training successful.""")

            return X_train, X_val, y_train, y_val

        except Exception as e:
            logger.exception(f"""Exception in data loading.
                             Exception message: {str(e)}""")
            raise e