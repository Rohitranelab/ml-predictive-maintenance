import os
from datetime import date

# For MongoDB Connection
DATABASE_NAME = "predictive_maintenance_ml"
COLLECTION_NAME = "Predictive-Data"
MONGODB_URL_KEY = "MONGODB_URL"

PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME = "predictive_maintenance_model.pkl"

TARGET_COLUMN = "Machine failure"
CURRENT_YEAR = date.today().year
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILE_NAME: str = "ai4i2020.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

'''
Data Ingestion related constant start with DATA_INJESTION VAR NAME
'''
DATA_INJESTION_COLLECTION_NAME: str = "Predictive-Data"
DATA_INJESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIT: str = "ingested"
DATA_INGESTION_TRAIN_AND_TEST_SPLIT_RATIO: float = 0.2

'''
DATA VALIDATION related constant start with DATA_VALIDATION VAR NAME
'''
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALDATION_REPORT_FILE_NAME: str = "report.yaml"

# '''
# Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
# '''
# DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
# DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
# DATA_TRANSFORMED_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
# 
# '''
# Model Trainer related constant start with MODEL_TRAINER VAR NAME
# '''
# MODEL_TRAINER_DIR_NAME: str = "model_trainer"
# MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
# MODEL_TRAINER_TRAINED_MODEL_NAME: str = "predictive_maintenance_model.pkl"
# MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
# MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
# MODELTRAINER_N_ESTIMATORS = 100
# MODEL_TRAINER_MIN_SAMPLES_SPLIT: int = 2
# MOEDL_TRAINER_MIN_SAMPLES_LEAF: int = 3
# MIN_SAMPLES_SPLIT_MAX_DEPTH: int = None
# MIN_SAMPLES_SPLIT_CRITERATION: str = 'entropy'
# MIN_SAMPLES_SPLIT_RANDOM_STATE: int = 42
# 
# '''
# Model Evaluation related constant
# '''
# MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
# MODEL_BUCKET_NAME = "predictive-model"
# MODEL_PUSHER_S3_KEY = "model-registry"
# 
# 
# APP_HOST = "0.0.0.0"
# APP_PORT = 5000