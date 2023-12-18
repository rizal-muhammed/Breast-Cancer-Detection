import os
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

from BreastCancerDetection.utils import common
from BreastCancerDetection.logging import logger
from BreastCancerDetection.entity import ModelTrainingConfig, ModelTrainingParams

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_recall_curve, auc



class ModelTraining:
    def __init__(self, config:ModelTrainingConfig,
                        params:ModelTrainingParams) -> None:
        self.config = config
        self.params = params

        common.create_directories([self.config.root_dir, self.config.models_dir, self.config.best_models_dir])
    
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
            y_train : pandas DataFrame like
                Ground truth (correct) training target values

            Raises
            ------
            Exception

            Notes
            ------
        
        """
        try:
            logger.info(f"""Loading pre-processed data for model training started...""")

            X_train = pd.read_csv(self.config.path_to_preprocessed_train_X)
            y_train = pd.read_csv(self.config.path_to_preprocessed_train_y)


            logger.info(f"""Loading pre-processed data for model training successful.""")

            return X_train, y_train

        except Exception as e:
            logger.exception(f"""Exception while loading preprocessed data.
                             Exception message: {str(e)}""")
            raise e
    
    def train_sgd_classifier(self, X_train, y_train):
        """
            This method trains an SGDClassifier with default parameters.

            Also performs cross validation with `sklearn.model_selection.cross_val_predict` and computes corresponding 
            `f1_score` and `roc_auc_score` and saves them for further reference.
            Also saves the model.
            Also plot and saves confusion matrix for further reference.
            Also plots precision_recall Vs decision threshold curve and precision Vs recall curve and saves them for further reference.
            These curves can be referenced for deciding optimal decision threshold for the problem statement and deciding precision that we
            should aim at.


            Parameters
            ----------
            X_train : pandas DataFrame like
                Ground truth (correct) input training features
            y_train : pandas DataFrame like
                Ground truth (correct) training target values

            Returns
            -------
            None

            Raises
            ------
            Exception

            Notes
            ------
            Saves the model, f1_score, roc_auc_score, confusion matrix in `models_dir` specified in config
            Also saves precision_recall Vs decision threshold curve as well as precision Vs recall curve in the same directory.
        
        """
        try:    
            logger.info(f"""Training SGDClassifier Model...""")

            sgd_clf = SGDClassifier(random_state=42, verbose=3)
            sgd_clf.fit(X_train, y_train.values.ravel())

            y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=self.params.cv)

            sgd_clf_f1_score = f1_score(y_train, y_train_pred)
            sgd_clf_roc_auc_score = roc_auc_score(y_train, y_train_pred)

            logger.info(f"""SGDClassifier Model
                        Training f1 score = {str(sgd_clf_f1_score)}.
                        Training ROC AUC Score = {str(sgd_clf_roc_auc_score)}.""")
            
            now = datetime.now()
            date = now.date()
            time = now.strftime("%H:%M:%S")

            dir_path = os.path.join(str(self.config.models_dir), "SGDClassifier_" + str(date) + "_" + str(time))
            common.create_directories([dir_path])

            joblib.dump(sgd_clf, os.path.join(dir_path, "sgd_classifier.joblib"))

            with open(os.path.join(dir_path, "scores.txt"), "w+") as file:
                file.write(f"""F1 Score : {str(sgd_clf_f1_score)}\n""")
                file.write(f"""ROC AUC Score : {str(sgd_clf_roc_auc_score)}.""")

            cm = confusion_matrix(y_train, y_train_pred)
            self.plot_confusion_matrix(cm, dir_path)

            stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            y_scores = cross_val_predict(sgd_clf, X_train, y_train.values.ravel(), cv=stratified_kfold, method="decision_function")
            precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
            self.plot_precision_recall_vs_threshold(precisions, recalls, thresholds, dir_path)
            self.plot_precision_vs_recall(precisions, recalls, dir_path)

        except Exception as e:
            logger.exception(f"""Error in 'train_sgd_classifier' method of 'ModelTraining' class.
                             Exception message: {str(e)}""")
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
    
    def plot_precision_recall_vs_threshold(self, precisions, recalls, thresholds, path_to_save):
        """
            This method plots the graph between Threshold Vs Precisions, Recalls, and saves as an image at `path_to_save`.

            This method plot precision and recall as functions of the threshold vale. Therefore, this graph can be referenced to 
            decide the right precision for the problem statement in hand.

            Parameters
            ----------
            precisions: numpy array type
                Precisions
            recalls : numpy array type
                Recalls
            threhoslds : numpy array type
                Thresholds
            path_to_save : Path like
                 Path to save the precision_recall Vs Threshold graph.

            Returns
            -------
            None

            Raises
            ------
            Exception

            Notes
            ------
            Saves the Precion-Recall Vs Thresholds graph as an image at `path_to_save` for further reference.
        
        """
        try:
            logger.info(f"""Plotting Precision-Recall Vs Thresholds...""")

            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
            plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
            plt.xlabel("Threshold")
            plt.grid()
            plt.legend(loc='best')
            plt.title("Precision and Recall Vs Decision Threshold")
            plt.savefig(os.path.join(path_to_save, "precision_recall_vs_threshold.png"))

            logger.info(f"""Plotting precision_recall Vs threshold is successful and is saved at {str(path_to_save)} directory.""")
        
        except Exception as e:
            raise e
    
    def plot_precision_vs_recall(self, precisions, recalls, path_to_save):
        """
            This method plots Precision Vs Recall curve and saves as an image at `path_to_save` for further reference.

            The Precision Vs Recall curve can be refered to find out what precision we should aim at, for the problem statement in hand.

            Parameters
            ----------
            precisions: numpy array type
                Precisions
            recalls : numpy array type
                Recalls
            path_to_save : Path like
                 Path to save the precision_recall Vs Threshold graph.

            Returns
            -------
            None

            Raises
            ------
            Exception

            Notes
            ------
            Saves the Precion Vs Recall graph as an image at `path_to_save` for further reference.
        
        """
        try:
            logger.info(f"""Plotting Precision Vs Recall...""")

            # Calculate the area under the curve (AUC)
            auc_score = auc(recalls, precisions)

            # Plot the precision-recall curve
            plt.figure(figsize=(8, 6))
            plt.plot(recalls, precisions, label=f'AUC = {auc_score:.2f}', color='b')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='best')
            plt.title('Precision Vs Recall')
            plt.savefig(os.path.join(path_to_save, "precision_vs_recall.png"))

            logger.info(f"""Plotting Precision Vs Recall is successful and is saved at {str(path_to_save)} directory.""")
        
        except Exception as e:
            raise e
        
    def train_random_forest_classifier(self, X_train, y_train):
        """
            This method trains a RandomForestClassifier with default parameters.

            Also performs cross validation with `sklearn.model_selection.cross_val_predict` and computes corresponding 
            `f1_score` and `roc_auc_score` and saves them for further reference.
            Also saves the model.
            Also plot and saves confusion matrix for further reference.
            Also plots precision_recall Vs decision threshold curve and precision Vs recall curve and saves them for further reference.
            These curves can be referenced for deciding optimal decision threshold for the problem statement and deciding precision that we
            should aim at.


            Parameters
            ----------
            X_train : pandas DataFrame like
                Ground truth (correct) input training features
            y_train : pandas DataFrame like
                Ground truth (correct) training target values

            Returns
            -------
            None

            Raises
            ------
            Exception

            Notes
            ------
            Saves the model, f1_score, roc_auc_score, confusion matrix in `models_dir` specified in config
            Also saves precision_recall Vs decision threshold curve as well as precision Vs recall curve in the same directory.
        
        """
        try:    
            logger.info(f"""Training RandomForestClassifier Model...""")

            rf_clf = RandomForestClassifier(random_state=42, verbose=3)
            rf_clf.fit(X_train, y_train.values.ravel())

            y_train_pred = cross_val_predict(rf_clf, X_train, y_train.values.ravel(), cv=self.params.cv)

            rf_clf_f1_score = f1_score(y_train, y_train_pred)
            rf_clf_roc_auc_score = roc_auc_score(y_train, y_train_pred)

            logger.info(f"""RandomForestClassifier Model
                        Training f1 score = {str(rf_clf_f1_score)}.
                        Training ROC AUC Score = {str(rf_clf_roc_auc_score)}.""")
            
            now = datetime.now()
            date = now.date()
            time = now.strftime("%H:%M:%S")

            dir_path = os.path.join(str(self.config.models_dir), "RandomForestClassifier_" + str(date) + "_" + str(time))
            common.create_directories([dir_path])

            joblib.dump(rf_clf, os.path.join(dir_path, "rf_classifier.joblib"))

            with open(os.path.join(dir_path, "scores.txt"), "w+") as file:
                file.write(f"""F1 Score : {str(rf_clf_f1_score)}\n""")
                file.write(f"""ROC AUC Score : {str(rf_clf_roc_auc_score)}.""")

            cm = confusion_matrix(y_train, y_train_pred)
            self.plot_confusion_matrix(cm, dir_path)

            stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            probability_scores = cross_val_predict(rf_clf, X_train, y_train.values.ravel(), cv=stratified_kfold, method="predict_proba")[:, 1]
            precisions, recalls, thresholds = precision_recall_curve(y_train, probability_scores)
            self.plot_precision_recall_vs_threshold(precisions, recalls, thresholds, dir_path)
            self.plot_precision_vs_recall(precisions, recalls, dir_path)

        except Exception as e:
            logger.exception(f"""Error in 'train_random_forest_classifier' method of 'ModelTraining' class.
                             Exception message: {str(e)}""")
            raise e

    def get_best_params_for_sgdclassifier(self, X_train, y_train):
        """
            This method is used to perform hyper parameter tuning for SGDClassifier model to obtain the best parameters.

            Parameters
            ----------
            None

            Returns
            -------
            X_train : pandas DataFrame like
                Ground truth (correct) input training features
            y_train : pandas DataFrame like
                Ground truth (correct) training target values

            Raises
            ------
            Exception

            Notes
            ------
        
        """
        try:
            logger.info(f"""Hyper parameter tuning for SGDClassifier...""")

            params_grids = self.params.sgd_classifier_params
  
            random_search = RandomizedSearchCV(estimator=SGDClassifier(),
                                      param_distributions=params_grids,
                                      cv=StratifiedKFold(),
                                      scoring="f1",
                                      verbose=2,
                                      random_state=42)
            random_search.fit(X_train, y_train.values.ravel())

            best_params = random_search.best_params_
            best_model = random_search.best_estimator_
            best_score = random_search.best_score_
            cvres = random_search.cv_results_

            now = datetime.now()
            date = now.date()
            time = now.strftime("%H:%M:%S")

            dir_path = os.path.join(str(self.config.best_models_dir), "SGDClassifier_best-model_" + str(date) + "_" + str(time))
            common.create_directories([dir_path])

            joblib.dump(best_model, os.path.join(dir_path, "sgd-classifier_best-model.joblib"))

            with open(os.path.join(dir_path, "best_params.txt"), "w+") as file:
                file.write(f"""best_params : {str(best_params)}\n""")
            
            with open(os.path.join(dir_path, "best_score.txt"), "w+") as best_score_file:
                best_score_file.write(f"""Best Score : {str(best_score)}\n""")
            
            with open(os.path.join(dir_path, "evaluation_scores.txt"), "w") as eval_file:
                for mean_f1_score, params in zip(cvres['mean_test_score'], cvres['params']):
                    eval_file.write(f"""{str(mean_f1_score)}   {str(params)}\n""")

            
        except Exception as e:
            logger.exception(f"""Error while hyper parameter tuning for SGDClassifier. 
                             Error message: {str(e)}""")
            raise e
    
    def get_best_params_for_random_forest_classifier(self, X_train, y_train):
        """
            This method is used to perform hyper parameter tuning for RandomForestClassifier model to obtain the best parameters.

            Parameters
            ----------
            None

            Returns
            -------
            X_train : pandas DataFrame like
                Ground truth (correct) input training features
            y_train : pandas DataFrame like
                Ground truth (correct) training target values

            Raises
            ------
            Exception

            Notes
            ------
        
        """
        try:
            logger.info(f"""Hyper parameter tuning for RandomForestClassifier...""")

            params_grids = self.params.random_forest_classifier_params
  
            random_search = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                      param_distributions=params_grids,
                                      cv=StratifiedKFold(),
                                      scoring="f1",
                                      verbose=2,
                                      random_state=42)
            random_search.fit(X_train, y_train.values.ravel())

            best_params = random_search.best_params_
            best_model = random_search.best_estimator_
            best_score = random_search.best_score_
            cvres = random_search.cv_results_

            now = datetime.now()
            date = now.date()
            time = now.strftime("%H:%M:%S")

            dir_path = os.path.join(str(self.config.best_models_dir), "RandomForestClassifier_best-model_" + str(date) + "_" + str(time))
            common.create_directories([dir_path])

            joblib.dump(best_model, os.path.join(dir_path, "random-forest-classifier_best-model.joblib"))

            with open(os.path.join(dir_path, "best_params.txt"), "w+") as file:
                file.write(f"""best_params : {str(best_params)}\n""")

            with open(os.path.join(dir_path, "best_score.txt"), "w+") as best_score_file:
                best_score_file.write(f"""Best Score : {str(best_score)}\n""")
            
            with open(os.path.join(dir_path, "evaluation_scores.txt"), "w") as eval_file:
                for mean_f1_score, params in zip(cvres['mean_test_score'], cvres['params']):
                    eval_file.write(f"""{str(mean_f1_score)}   {str(params)}\n""")

            
        except Exception as e:
            logger.exception(f"""Error while hyper parameter tuning for RandomForestClassifier. 
                             Error message: {str(e)}""")
            raise e
    
    def get_final_model(self):
        """
            This method returns the best model with highest f1 score among the models that we've trained so far.

            Parameters
            ----------
            None

            Returns
            -------
            final_model : 
                Returns the best model with highest f1 score.

            Raises
            ------
            Exception

            Notes
            ------
        
        """
        max_score = float('-inf')
        final_model = ""

        for root, dirs, files in os.walk(self.config.best_models_dir):
            if "best_score.txt" in files:
                score_file_path = os.path.join(root, "best_score.txt")

                with open(score_file_path, "r") as score_file:
                    score_line = score_file.readline()
                    try:
                        score = float(score_line.split(':')[-1].strip())
                    except ValueError:
                        continue

                    if score > max_score:
                        max_score = score
                        joblib_files = [f for f in os.listdir(root) if f.endswith('.joblib')]
                    
                        if joblib_files:
                            final_model = os.path.join(root, joblib_files[0])
        
        return final_model
    
    def save_final_model(self):
        """
            This method saves the final model in the directory specified at 'final_model_dir' in config.

            Parameters
            ----------
            None

            Returns
            -------
            None

            Raises
            ------
            Exception

            Notes
            ------
            final model is saved at `final_model_dir`(specified in config.)
        
        """

        common.create_directories([self.config.final_model_dir])
        logger.info(f"""Saving final model...""")

        final_model = self.get_final_model()
        if final_model:
            shutil.copy(final_model, os.path.join(self.config.final_model_dir, "final_model.joblib"))
        
        logger.info(f"""Final model is successfully saved at {str(self.config.final_model_dir)}""")




