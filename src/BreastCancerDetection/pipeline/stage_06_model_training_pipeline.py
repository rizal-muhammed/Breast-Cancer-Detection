from BreastCancerDetection.config.configuration import ConfigurationManager
from BreastCancerDetection.components.model_training import ModelTraining


class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training_params = config.get_model_training_params()

        model_training = ModelTraining(config=model_training_config,
                                        params=model_training_params)
        
        X_train, y_train = model_training.load_data()
        model_training.train_sgd_classifier(X_train, y_train)
        model_training.train_random_forest_classifier(X_train, y_train)
        model_training.get_best_params_for_sgdclassifier(X_train, y_train)
        model_training.get_best_params_for_random_forest_classifier(X_train, y_train)
        model_training.save_final_model()