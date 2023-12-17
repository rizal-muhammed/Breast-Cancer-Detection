import os
import pandas as pd
import numpy as np
from ensure import ensure_annotations
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from BreastCancerDetection.logging import logger
from BreastCancerDetection.utils import common
from BreastCancerDetection.entity import (DataPreProcessingTrainingConfig,
                                          DataPreProcessingTrainingParams)


class DataPreProcessingTraining:
    def __init__(self, config:DataPreProcessingTrainingConfig,
                        params:DataPreProcessingTrainingParams,) -> None:
        self.config = config
        self.params = params

        common.create_directories([self.config.root_dir])
    
    def load_input_data_for_training(self):
        """
            This method loads input data for training and returns the data as pandas DataFrame type.

            Parameters
            ----------
            None

            Returns
            -------
            df: Pandas DataFrame type
                Input training data as a Pandas DataFrame.

            Raises
            ------
            Exception
        
        """
        try:
            logger.info(f"""Loading input data for training...""")

            input_filepath = self.config.input_file_path
            df = pd.read_csv(input_filepath, index_col='index_column')
            df = df.reset_index(drop=True)

            logger.info(f""" Data loaded successfully.""")

            return df
        
        except Exception as e:
            logger.exception(f""" Exception while loading data.""")
            raise e
    
    @ensure_annotations
    def remove_columns(self, df:pd.DataFrame, colums_to_remove:list):
        """
            This method removes the given list of columns from a pandas DataFrame.

            Parameters
            ----------
            df: Pandas DataFrame type
                DataFrame to remove columns
            columns_to_remove: list
                List of columns to remove

            Returns
            -------
            df: Pandas DataFrame type
                Pandas DataFrame after removing the specified list of columns

            Raises
            ------
            Exception
        
        """
        try:
            logger.info(f""" Attempting to remove specified columns from DataFrame... """)

            remaining_df = df.drop(columns=colums_to_remove, axis=1)

            logger.info(f""" Columns {str(colums_to_remove)} successfully removed from DataFrame.""")

            return remaining_df
        
        except Exception as e:
            logger.exception(f"""Exception while removing columns from DataFrame. 
                             Exception message: {str(e)}""")
            raise e
    
    @ensure_annotations
    def separate_label_feature(self, df: pd.DataFrame, ):
        """
            This method separates the features and label columns.

            Parameters
            ----------
            df : pandas dataframe type
                input data in the form of dataframe

            Returns
            -------
            X : pandas DataFrame type
                Returns a DataFrame X of features
            y : pandas DataFrame type
                Returns a DataFrame y of labels

            Raises
            ------
            Exception

            Notes
            ------
            label_column_name : str type
                label column name to separate from the input data, is retrieved as parameter
        
        """
        try:
            logger.info(f"""Attempting to separate label from features...""")

            X = df.drop(columns=[self.params.label_column_name], axis=1)
            y = df[self.params.label_column_name].to_frame(name=self.params.label_column_name)

            logger.info(f"""Label separation is successful.""")

            return X, y

        except Exception as e:
            logger.exception(f"""Exception while separating labels from features.""")
            raise e
    
    @ensure_annotations
    def separate_test_set(self, df: pd.DataFrame):
        """
            This method perform stratified shuffle split and separate test set for evaluation.

            Parameters
            ----------
            df : pandas dataframe type
                input data in the form of dataframe

            Returns
            -------
            strat_train_set : pandas DataFrame type
                Returns training set in the form of pandas DataFrame
            strat_test_set : pandas DataFrame type
                Returns test set in the form of pandas DataFrame

            Raises
            ------
            Exception

            Notes
            ------
            test_size : float
                test_size, is retrieved as parameter.
                Should be between 0.0 to 1.0 and represent the proportion of the dataset to include in the test split.
        
        """
        try:
            logger.info(f"""Performing Stratified Shuffle split to separate test set for evaluation...""")

            split = StratifiedShuffleSplit(n_splits=1, test_size=self.params.test_size, random_state=42)
            for train_index, test_index in split.split(df, df['diagnosis']):
                strat_train_set = df.loc[train_index]
                strat_test_set = df.loc[test_index]
            
            logger.info(f"""Stratified shuffle split is successful.""")

            return strat_train_set, strat_test_set
        
        except Exception as e:
            logger.exception(f"""Exception during Stratified shuffle split for separating test set. 
                             Exception message: {str(e)}""")
            raise e
    
    @ensure_annotations
    def correlations(self, df:pd.DataFrame):
        """
            This method outputs correlations matrix of features w.r.t target feature.

            Parameters
            ----------
            df : pandas dataframe type
                input data in the form of dataframe

            Returns
            -------
            None

            Raises
            ------
            Exception

            Notes
            ------
            - label_column_name is received as parameter
            - An image of correlation matrix is outputted at directory specified in configuration.
            - This method assumes that there are no categorical features in the input dataframe.
            - if the number of rows in the input DataFrame is greater than 'row_threshod'(specified in the params), then
            a warning that the 'DataFrame is too large to compute correlation matrix' is generated and correlation matrix is not calculated.
        
        """
        try:

            if df.shape[0] > self.params.row_threshold:
                logger.warning(f"""DataFrame is too large to compute correlation matrix. Skipping correlation matrix generation.""")
            else:
                logger.info(f"""Plotting correlation matrix...""")
                target_feature = self.params.label_column_name
                common.create_directories([self.config.correlation_dir])

                corr_matrix = df.corr()

                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix[[target_feature]].abs().sort_values(by=target_feature, ascending=False), annot=True, cmap='coolwarm', fmt=".2f")
                plt.title(f'Correlation Matrix with Respect to {target_feature}')
                plt.savefig(os.path.join(self.config.correlation_dir, "correlation_matrix.png"))

                logger.info("Correlation matrix saved successfully.")
        
        except Exception as e:
            logger.exception(f"""Exception while plotting correlation matrix.
                             Exception message: {str(e)}""")
            raise e
    
    @ensure_annotations
    def map_target_feature_to_binary(self, df:pd.DataFrame):
        """
            This method maps 'M' to 1 and 'B' to 0 in the specified target column of DataFrame df.

            Parameters
            ----------
            df : pandas DataFrame type
                The input DataFrame

            Returns
            -------
            modified_df: pandas DataFrame
                Returns the modified DataFrame

            Raises
            ------
            Exception

            Notes
            ------
            - label_column_name is received as parameter.
        
        """
        try:
            logger.info(f"""Mapping 'M' to 1 and 'B' to 0 in the target column...""")

            target_column = self.params.label_column_name
            modified_df = df.copy()
            modified_df[target_column] = modified_df[target_column].map({'M': 1, 'B': 0})
            
            logger.info(f"""Mapping completed.""")
            return modified_df
        except Exception as e:
            logger.exception(f"""{str(e)}""")
            raise e
    
    @ensure_annotations
    def is_null_present(self, df:pd.DataFrame):
        """
            This method checks whether there are null values present in the input dataframe.

            Parameters
            ----------
            df : pandas dataframe type
                input data in the form of dataframe

            Returns
            -------
            null_present : bool type
                True if null values are present in df, False if null values are not present in df.

            Raises
            ------
            Exception

            Notes
            ------
            Saves null count information in 'preprocessed_data/null_value_counts.csv' directory
            for further reference
        
        """
        try:
            logger.info(f"""Attempting to quantify null values in the input data for training...""")

            null_present = False
            null_counts = df.isna().sum()

            for i in null_counts:
                if i > 0:
                    null_present = True
                    break
            
            if null_present:
                df_null_counts = null_counts.to_frame(name="null_count")
                filename = "null_value_counts.csv"
                df_null_counts.to_csv(os.path.join(self.config.root_dir, filename), index=False, header=True)
                logger.info(f"""The null values in the input dataframe is quanitfied at '{str(os.path.join(self.config.root_dir, filename))}',if present.""")

            logger.info(f"""Quantifying null values in the input data for training is completed""")
            
            return null_present
        
        except Exception as e:
            logger.exception(f"""Exception while counting null values in input file.""")
            raise e
    
    @ensure_annotations
    def impute_missing_values(self, df:pd.DataFrame):
        """
            This method replaces all the missing values in the input dataframe using KNNImputer.

            Parameters
            ----------
            df : pandas dataframe type
                input data in the form of a pandas DataFrame

            Returns
            -------
            df_new : pandas DataFrame type
                Returns a dataframe with all the missing values imputed.

            Raises
            ------
            Exception

            Notes
            ------
            The KNN imputer information is stored in 'preprocessed_data/knn_imputer.pkl' directory
            for further reference during prediction.
        
        """
        try:
            logger.info(f"""Impuring missing values from input data...""")

            knn_imputer = KNNImputer(weights="uniform", missing_values=np.nan)
            df_new = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

            # save model for further reference
            with open(os.path.join(self.config.root_dir, "knn_imputer.pkl"), "wb") as file:
                pickle.dump(knn_imputer, file)

            logger.info(f"""Impute missing values is successful.""")

            return df_new
        
        except Exception as e:
            logger.exception(f"""Exception while imputing missing values in input file.""")
            raise e
    
    @ensure_annotations
    def get_columns_with_zero_std_deviation(self, df:pd.DataFrame):
        """
            This method retuns a list of columns which have zero standard deviation.


            If the standard deviation is zero, then the column is populated by one value. 
            So if your goal is to prepare the data for regression or classfication, 
            you can throw the column out, since it will contribute nothing to the regression 
            or classification.

            Parameters
            ----------
            df : pandas DataFrame type
                input data in the form of a pandas DataFrame

            Returns
            -------
            columns_lst_with_zero_std_dev : list
                Returns a list of column names for which standard deviation is zero.

            Raises
            ------
            Exception

            Notes
            ------
            The list of columns with zero standard deviation is stored in 
            'preprocessed_data/columns_lst_with_zero_std_dev.pkl' directory for further 
            reference during prediction.
        
        """
        try:
            logger.info(f"""Listing columns with zero standard deviation in input file started...""")

            columns = df.columns
            df_description = df.describe()

            columns_with_zero_std_dev = []

            for col in columns:
                if df_description[col]['std'] == 0:
                    columns_with_zero_std_dev.append(col)
            
            if len(columns_with_zero_std_dev) > 0:
                logger.info(f"""Columns with Zero standard deviation are {str(columns_with_zero_std_dev)}""")
            else:
                logger.info(f""" There are no columns with Zero standard deviation.""")
            
            # saving the list for further reference
            with open(os.path.join(self.config.root_dir, "columns_with_zero_std_dev.pkl"), "wb") as file:
                pickle.dump(columns_with_zero_std_dev, file)
            
            logger.info(f"""Listing columns with zero standard deviation in input file is successful.""")

            return columns_with_zero_std_dev
        
        except Exception as e:
            logger.exception(f"""Exception while listing columns with Zero std deviation in input file.""")
            raise e
    
    @ensure_annotations
    def training_std_scaling(self, X:pd.DataFrame):
        """
            This method shall be used for performing standard scaling on training data.

            Parameters
            ----------
            X : pandas DataFrame type
                input data in the form of a pandas DataFrame

            Returns
            -------
            X_scaled : pandas DataFrame type
                Returns a X_scaled which is scaled input.

            Raises
            ------
            Exception

            Notes
            ------
            The std scaler is stored in 'preprocessed_data/std_scaler.pkl' directory 
            for further reference during prediction.
        
        """
        try:
            logger.info(f"""Scaling of input data started...""")

            std_scaler = StandardScaler()
            X_scaled = pd.DataFrame(std_scaler.fit_transform(X), columns=X.columns)

            # save model for further reference
            with open(os.path.join(self.config.root_dir, "std_scaler.pkl"), "wb") as file:
                pickle.dump(std_scaler, file)
            
            logger.info(f""" Min-Max scaling of input training data is successful. """)

            return X_scaled

        except Exception as e:
            logger.exception(f"""Exception while scaling of input data.""")
            raise e
    
    @ensure_annotations
    def save_data(self, X:pd.DataFrame, y:pd.DataFrame):
        """
            This method saves the data at the end of pre-processing step at specified directory,
            so that we can retrieve them for model training.

            Parameters
            ----------
            X : pandas DataFrame type
                input features in the form of pandas DataFrame
            y : pandas DataFrame
                Ground truth for training in the form of pandas DataFrame

            Returns
            -------
            
            None

            Raises
            ------
            Exception

            Notes
            ------
            The preprocessed input data is stored at 'artifacts/preprocessed_data/preprocessed_input.csv'
            directory for further reference during prediction.
        
        """
        try:
            logger.info(f"Attempting saving pre-processed data into 'artifacts/preprocessed_data' directory...")

            common.create_directories([self.config.preprocessed_input_data_dir])
            X.to_csv(os.path.join(self.config.preprocessed_input_data_dir, "preprocessed_train_X.csv"),
                      index=False,
                      header=True)
            # X_val.to_csv(os.path.join(self.config.preprocessed_input_data_dir, "preprocessed_val_X.csv"),
            #           index=False,
            #           header=True)
            y.to_csv(os.path.join(self.config.preprocessed_input_data_dir, "preprocessed_train_y.csv"),
                      index=False,
                      header=True)
            # y_val.to_csv(os.path.join(self.config.preprocessed_input_data_dir, "preprocessed_val_y.csv"),
            #           index=False,
            #           header=True)
            
            logger.info(f"""Saving pre-processed data is successful.""")

        except Exception as e:
            logger.exception(f"""Exception while saving pre-processed input data.""")
            raise e
    
    @ensure_annotations
    def save_test_set(self, test_set:pd.DataFrame):
        """
            This method saves the test set at directory specified in config.

            Parameters
            ----------
            test_set : pandas DataFrame type
                input features in the form of a pandas DataFrame

            Returns
            -------
            None

            Raises
            ------
            Exception

            Notes
            ------
            The test set is stored as a csv file at directory specified by 'test_set_dir' in config.
        
        """
        try:
            logger.info(f"""Saving test set for future evaluation...""")

            common.create_directories([self.config.test_set_dir])

            test_set_features = test_set.drop(self.params.label_column_name, axis=1)
            test_set_label = test_set[self.params.label_column_name].to_frame(name=self.params.label_column_name)
            
            test_set_features.to_csv(os.path.join(self.config.test_set_dir, "test_set_features.csv"), index=False, header=True)
            test_set_label.to_csv(os.path.join(self.config.test_set_dir, "test_set_label.csv"), index=False, header=True)

            logger.info(f"""Test set saved successfully.""")
        
        except Exception as e:
            logger.exception(f"""Exception while saving test data.""")
            raise e
        
    def train_val_split(self, X, y):
        """
            This method splits the data into train set and validation set.

            Parameters
            ----------
            X : pandas DataFrame type
                input features in the form of a pandas DataFrame
            y : lists, numpy arrays, scipy-sparse matrices or pandas dataframe.

            Returns
            -------
            X_train: 
            X_val: 
            y_train: 
            y_val: 

            Raises
            ------
            Exception

            Notes
            ------
            None
        
        """
        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=False, random_state=42, test_size=self.params.test_size)

            # X_train = pd.DataFrame(X_train, columns=X_train.columns)
            # X_val = pd.DataFrame(X_val, columns=X_val.columns)
            # y_train = pd.DataFrame(y_train, columns=y_train.columns)
            # y_val = pd.DataFrame(y_val, columns=y_val.columns)
            return X_train, X_val, y_train, y_val
        
        except Exception as e:
            raise e
    
    @ensure_annotations
    def impute_missing_values_validation_set(self, X_val:pd.DataFrame):
        """
            This method replaces all the missing values in the validation set.

            Parameters
            ----------
            X_val : pandas dataframe type
                Valdation set in the form of a pandas DataFrame

            Returns
            -------
            X_val_new : pandas DataFrame type
                Returns a dataframe with all the missing values imputed.

            Raises
            ------
            Exception

            Notes
            ------
            The KNN imputer information is stored in 'preprocessed_data/knn_imputer.pkl' directory is refered for imputing missing values.
        
        """
        try:
            file_path = os.path.join(self.config.root_dir, "knn_imputer.pkl")

            if os.path.exists(file_path):
                knn_imputer = pd.read_pickle(file_path)
                X_val_new = pd.DataFrame(knn_imputer.transform(X_val), columns=X_val.columns)

                return X_val_new
            else:
                return X_val

        except Exception as e:
            raise e
    
    @ensure_annotations
    def validation_std_scaling(self, X:pd.DataFrame):
        """
            This method shall be used for performing standard scaling on validation data.

            Parameters
            ----------
            X : pandas DataFrame type
                input validation data in the form of a pandas DataFrame

            Returns
            -------
            X_scaled : pandas DataFrame type
                Returns a X_scaled which is scaled input.

            Raises
            ------
            Exception

            Notes
            ------
            The std scaler stored in 'preprocessed_data/std_scaler.pkl' directory is refered for scaling.
        
        """
        try:
            logger.info(f"""Scaling of input validation data started...""")

            # load model 
            with open(os.path.join(self.config.root_dir, "std_scaler.pkl"), "rb") as file:
                std_scaler = pickle.load(file)

            X_scaled = pd.DataFrame(std_scaler.transform(X), columns=X.columns)
            
            logger.info(f"""Scaling of input validation data is successful. """)

            return X_scaled

        except Exception as e:
            logger.exception(f"""Exception while scaling of input data.""")
            raise e


