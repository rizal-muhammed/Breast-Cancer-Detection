from BreastCancerDetection.config.configuration import ConfigurationManager
from BreastCancerDetection.components.data_preprocessing_training import DataPreProcessingTraining


class DataPreProcessingTrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_preprocessing_training_config = config.get_data_preprocessing_training_config()
        data_preprocessing_training_params = config.get_data_preprocessing_training_params()

        data_preprocessing_training = DataPreProcessingTraining(
            config=data_preprocessing_training_config,
            params=data_preprocessing_training_params,
        )

        # loading data
        df = data_preprocessing_training.load_input_data_for_training()

        # separate test set
        train_set, test_set = data_preprocessing_training.separate_test_set(df)

        # save test set
        data_preprocessing_training.save_test_set(test_set)

        # map target features to binary. 'M' to 1 and 'B' to 0.
        train_set_modified = data_preprocessing_training.map_target_feature_to_binary(train_set)

        # documenting correlations w.r.t target feature
        data_preprocessing_training.correlations(train_set_modified)

        # get columns with zero standard deviation
        columns_lst_with_zero_std_dev = data_preprocessing_training.get_columns_with_zero_std_deviation(train_set_modified)
        if len(columns_lst_with_zero_std_dev) > 0:
            train_set_modified = data_preprocessing_training.remove_columns(train_set_modified, columns_lst_with_zero_std_dev)

        # separate features and label
        X,y = data_preprocessing_training.separate_label_feature(train_set_modified)

        X_train, X_val, y_train, y_val = data_preprocessing_training.train_val_split(X, y)

        # dealing with missing values
        is_null_present = data_preprocessing_training.is_null_present(X_train)
        if is_null_present:
            X_train = data_preprocessing_training.impute_missing_values(X_train)
            X_val = data_preprocessing_training.impute_missing_values_validation_set(X_val)

        # scaling
        X_train_scaled = data_preprocessing_training.training_std_scaling(X_train)
        X_val_scaled = data_preprocessing_training.validation_std_scaling(X_val)

        # saving pre processed data for training
        data_preprocessing_training.save_data(X_train_scaled, X_val_scaled, y_train, y_val)

        
