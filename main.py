from BreastCancerDetection.logging import logger
from BreastCancerDetection.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from BreastCancerDetection.pipeline.stage_02_data_validation_training import DataValidationTrainingPipeline


STAGE_NAME = f"""Data Ingestion"""
try:
    logger.info(f""">>>>>>> Stage {STAGE_NAME} started... <<<<<<<""")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f""">>>>>>> Stage {STAGE_NAME} Completed... <<<<<<<\n\n""")
except Exception as e:
    logger.exception(e)
    raise

STAGE_NAME = f"""Data Validation Training"""
try:
    logger.info(f""">>>>>>> Stage {STAGE_NAME} started... <<<<<<<""")
    data_validation_training = DataValidationTrainingPipeline()
    data_validation_training.main()
    logger.info(f""">>>>>>> Stage {STAGE_NAME} Completed... <<<<<<<\n\n""")
except Exception as e:
    logger.exception(e)
    raise e
