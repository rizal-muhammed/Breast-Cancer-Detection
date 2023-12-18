import joblib
import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from ensure import ensure_annotations
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

from BreastCancerDetection.utils import common
from BreastCancerDetection.logging import logger
from BreastCancerDetection.entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config:ModelEvaluationConfig) -> None:
        self.config = config

        common.create_directories([self.config.root_dir])
    
    def load_final_model(self):
        """
            This method loads the final model for evaluation.

            Parameters
            ----------
            None

            Returns
            -------
            final_model : 
                Final model

            Raises
            ------
            Exception

            Notes
            ------
        
        """
        try:
            logger.info(f"""Loading final model...""")

            final_model_path = self.config.final_model
            final_model = joblib.load(final_model_path)

            final_model_dir = os.path.join(self.config.root_dir, "final_model")
            common.create_directories([final_model_dir])
            shutil.copy(final_model_path, final_model_dir)

            logger.info(f"""Final model loaded successfully.""")

            return final_model

        except Exception as e:
            logger.exception(f"""Exception while loading the final model.
                             Exception message: {str(e)}""")
            raise e
    
    def load_test_data(self):
        """
            This method loads the test data and perform pre-processing.

            Parameters
            ----------
            None

            Returns
            -------
            X_test : 
                Ground truth (correct) input test features after pre-processing
            y_test : 
                Ground truth (correct) test target values after pre-processing

            Raises
            ------
            Exception

            Notes
            ------
            This method loads the test data perform pre-processing and returns. Therefore, the data can be used 
            directly for model evaluation. further pre-processing is not required.
        
        """
        try:
            logger.info(f"""Loading test data...""")

            if os.path.exists(self.config.test_set_feautres) and os.path.exists(self.config.test_set_label):
                X_test = pd.read_csv(self.config.test_set_feautres)
                y_test = pd.read_csv(self.config.test_set_label)

                logger.info(f"""Test data loaded successfully.""")
            else:
                logger.info(f"""File doesnt exists.""")
                return
            
            X_test_preprocessed, y_test_preprocessed = self.preprocess_test_data(X_test, y_test)

            logger.info(f"""Test data loaded successfully.""")

            return X_test_preprocessed, y_test_preprocessed
        
        except Exception as e:
            logger.exception(f"""Exception while loading test data.
                             Exception message : {str(e)}""")
            raise e
    
    def preprocess_test_data(self, X_test, y_test):
        """
            This method pre-process the test data and returns the pre-processed data.

            Parameters
            ----------
            X_test : pandas DataFrame like
                Ground truth (correct) input test features
            y_test : pandas DataFrame like
                Ground truth (correct) test target values

            Returns
            -------
            X_test : 
                Ground truth (correct) input test features after pre-processing
            y_test : 
                Ground truth (correct) test target values after pre-processing

            Raises
            ------
            Exception

            Notes
            ------
            pre-processing steps are 
            - Map target feature to binary
            - Dropping columns with zero standard deviation
            - Imputing missing values
            - Scaling the data
        
        """
        try:
            logger.info(f"""Pre-processing test data...""")

            y_test = self.map_target_feature_to_binary(y_test)

            columns_lst_with_zero_std_dev = self.get_columns_lst_with_zero_std_dev()
            if len(columns_lst_with_zero_std_dev) > 0:
                X_test = self.remove_columns(X_test, columns_lst_with_zero_std_dev)
            
            is_null_present = self.is_null_present(X_test)
            if is_null_present:
                X_test = self.impute_missing_values(X_test)
            
            X_test_scaled = self.test_set_std_scaling(X_test)

            logger.info(f"""Pre-processing of test data is completed.""")

            return X_test_scaled, y_test
        
        except Exception as e:
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

            target_column = self.config.label_column_name
            modified_df = df.copy()
            modified_df[target_column] = modified_df[target_column].map({'M': 1, 'B': 0})
            
            logger.info(f"""Mapping completed.""")
            return modified_df
        except Exception as e:
            logger.exception(f"""{str(e)}""")
            raise e
    
    def get_columns_lst_with_zero_std_dev(self):
        """
            This method loads the list of column names with zero standard deviation.
            Theses columns need to be dropped since we have dropped them in training set.

            Parameters
            ----------
            None

            Returns
            -------
            columns_lst_with_zero_std_dev : list
                Returns the list of column names that have zero standard deviation.

            Raises
            ------
            Exception

            Notes
            ------
            
        
        """
        try:
            logger.info(f"""Loading list of column names with zero std dev...""")
            
            columns_lst_with_zero_std_dev = []
            if os.path.exists(self.config.columns_with_zero_std_dev_path):
                with open(self.config.columns_with_zero_std_dev_path, "rb") as pkl_file:
                    columns_lst_with_zero_std_dev = pickle.load(pkl_file)
                    return columns_lst_with_zero_std_dev
            else:
                logger.info(f"""File not exists. Returning an empty list.""")
                return columns_lst_with_zero_std_dev
        
        except Exception as e:
            logger.exception(f"""Exception while loading list of column names with zero standard deviation.""")
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
    def impute_missing_values(self, X_test:pd.DataFrame):
        """
            This method replaces all the missing values in the test set.

            Parameters
            ----------
            X_test : pandas dataframe type
                test set features in the form of a pandas DataFrame

            Returns
            -------
            X_test_new : pandas DataFrame type
                Returns a dataframe with all the missing values imputed.

            Raises
            ------
            Exception

            Notes
            ------
            The KNN imputer information is stored in 'preprocessed_data/knn_imputer.pkl' directory is refered for imputing missing values.
        
        """
        try:
            logger.info(f"""Imputing missing values from test set...""")

            file_path = self.config.knn_imputer_path
            if os.path.exists(file_path):  # train set contained missing values
                knn_imputer = pd.read_pickle(file_path)
                X_test_new = pd.DataFrame(knn_imputer.transform(X_test), columns=X_test.columns)

            else:  # train set doesn't contain any missing values, but test contains missing values
                logger.info(f"""Train set doesn't contain any missing values, but test set contains some missing values.
                            Therefore imputing missing values in the test set with new KNNImputer...""")
                knn_imputer = KNNImputer(weights="uniform", missing_values=np.nan)
                X_test_new = pd.DataFrame(knn_imputer.fit_transform(X_test), columns=X_test.columns)
            
            return X_test_new

        except Exception as e:
            raise e
    
    @ensure_annotations
    def test_set_std_scaling(self, X:pd.DataFrame):
        """
            This method shall be used for performing standard scaling on test data.

            Parameters
            ----------
            X : pandas DataFrame type
                input test data in the form of a pandas DataFrame

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

            try:
                # load model 
                with open(self.config.std_scaler_path, "rb") as file:
                    std_scaler = pickle.load(file)

                X_scaled = pd.DataFrame(std_scaler.transform(X), columns=X.columns)
                
                logger.info(f"""Scaling of input validation data is successful. """)

                return X_scaled
            
            except FileNotFoundError:
                print(f"The file {self.config.std_scaler_path} does not exist.")

        except Exception as e:
            logger.exception(f"""Exception while scaling of input data.""")
            raise e
    
    def evaluate(self, final_model, X_test, y_test):
        """
            This method shall be used for performing standard scaling on test data.

            Parameters
            ----------
            final_model :
                Final model for evaluation
            X_test : 
                Ground truth (correct) input test features after pre-processing
            y_test : 
                Ground truth (correct) input test targets after pre-processing

            Returns
            -------
            None

            Raises
            ------
            Exception

            Notes
            ------
        
        """
        try:
            logger.info(f"""Evaluating final model...""")

            final_predictions = final_model.predict(X_test)

            final_model_f1_score = f1_score(y_test, final_predictions)
            final_model_roc_auc_score = roc_auc_score(y_test, final_predictions)
            logger.info(f"""Final Model
                        Test f1 score = {str(final_model_f1_score)}.
                        Test ROC AUC Score = {str(final_model_roc_auc_score)}.""")

            dir_path = os.path.join(self.config.root_dir, "final_model")
            with open(os.path.join(dir_path, "final_model_scores.txt"), "w+") as file:
                file.write(f"""Test F1 Score : {str(final_model_f1_score)}\n""")
                file.write(f"""Test ROC AUC Score : {str(final_model_roc_auc_score)}""")
            
            cm = confusion_matrix(y_test, final_predictions)
            self.plot_confusion_matrix(cm, dir_path)

            logger.info(f"""Final model evaluation is completed.""")

        except Exception as e:
            logger.exception(f"""Exception while evaluation of final model.
                             Exception message : {str(e)}""")
            raise e
    
    def plot_confusion_matrix(self, cm, path_to_save):
        """
            This method plots confusion matrix and saves as an image at `path_to_save`

            Parameters
            ----------
            cm : numpy array type
                Confusion matrix to plot
            path_to_save : Path like
                Path to save the confusion matrix as an image for further reference.

            Returns
            -------
            None

            Raises
            ------
            Exception

            Notes
            ------
            Saves the Confusion Matrix at `path_to_save` for further reference.
        
        """
        try:
            logger.info(f"""Plotting confusion matrix...""")

            # Plot the confusion matrix using seaborn heatmap and save for further reference
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Predicted 0', 'Predicted 1'],
                        yticklabels=['Actual 0', 'Actual 1'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(path_to_save, "confusion_matrix.png"))

            logger.info(f"""Plotting confusion matrix is successful and is saved at {str(path_to_save)} directory.""")
        
        except Exception as e:
            raise e

