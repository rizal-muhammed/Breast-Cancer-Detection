import pandas as pd
import os
from ucimlrepo import fetch_ucirepo
from BreastCancerDetection.utils import common

from BreastCancerDetection.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config:DataIngestionConfig) -> None:
        self.config = config

        common.create_directories([config.root_dir])

    def data_ingestion(self, ):

        # fetch dataset 
        breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

        # data (as pandas dataframes) 
        X = breast_cancer_wisconsin_diagnostic.data.features 
        y = breast_cancer_wisconsin_diagnostic.data.targets 

        breast_cancer_wisconsin_diagnostic_df = pd.concat([X, y], axis=1)

        destination_folder = self.config.destination_folder
        if not os.path.exists(destination_folder or not os.path.isdir(destination_folder)):
            common.create_directories([self.config.destination_folder])
        
        breast_cancer_wisconsin_diagnostic_df.to_csv(
            os.path.join(self.config.destination_folder, self.config.filename),
            index=False,
            header=True)

        miscellaneous_folder = self.config.miscellaneous_folder
        if not os.path.exists(miscellaneous_folder) or not os.path.isdir(miscellaneous_folder):
            common.create_directories([miscellaneous_folder])

        with open(os.path.join(miscellaneous_folder, "metadata.txt"), "w") as file:
            file.write(str(breast_cancer_wisconsin_diagnostic.metadata))
        
        with open(os.path.join(miscellaneous_folder, "variables.txt"), "w") as file:
            file.write(str(breast_cancer_wisconsin_diagnostic.variables))

