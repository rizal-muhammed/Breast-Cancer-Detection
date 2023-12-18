from BreastCancerDetection.config.configuration import ConfigurationManager
from BreastCancerDetection.components.model_evaluation import ModelEvaluation


class ModelEvaluationPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()

        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        final_model = model_evaluation.load_final_model()
        if model_evaluation.load_test_data() is not None:
            X_test, y_test = model_evaluation.load_test_data()
            
            model_evaluation.evaluate(final_model, X_test, y_test)
