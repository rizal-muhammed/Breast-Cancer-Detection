from BreastCancerDetection.logging import logger
from BreastCancerDetection.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from BreastCancerDetection.pipeline.stage_02_data_validation_training import DataValidationTrainingPipeline
from BreastCancerDetection.pipeline.stage_03_data_transformation_training import DataTransformationTrainingPipeline
from BreastCancerDetection.pipeline.stage_04_db_ops_training_pipeline import DatabaseOperationsTrainingPipeline


# STAGE_NAME = f"""Data Ingestion"""
# try:
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} started... <<<<<<<""")
#     data_ingestion = DataIngestionPipeline()
#     data_ingestion.main()
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} Completed... <<<<<<<\n\n""")
# except Exception as e:
#     logger.exception(e)
#     raise

# STAGE_NAME = f"""Data Validation Training"""
# try:
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} started... <<<<<<<""")
#     data_validation_training = DataValidationTrainingPipeline()
#     data_validation_training.main()
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} Completed... <<<<<<<\n\n""")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = f"""Data Transformation Training"""
# try:
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} started... <<<<<<<""")
#     data_transformation_training = DataTransformationTrainingPipeline()
#     data_transformation_training.main()
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} Completed... <<<<<<<\n\n""")
# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = f"""Database Operations Training"""
try:
    logger.info(f""">>>>>>> Stage {STAGE_NAME} started... <<<<<<<""")
    data_base_operations = DatabaseOperationsTrainingPipeline()
    data_base_operations.main()
    logger.info(f""">>>>>>> Stage {STAGE_NAME} Completed... <<<<<<<\n\n""")
except Exception as e:
    logger.exception(e)
    raise e