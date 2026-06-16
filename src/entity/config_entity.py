import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INJESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIT, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIT, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_AND_TEST_SPLIT_RATIO
    collection_name: str = DATA_INJESTION_COLLECTION_NAME

@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    validation_report_file_path: str = os.path.join(data_validation_dir, DATA_VALDATION_REPORT_FILE_NAME)

@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformation_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                       TRAIN_FILE_NAME.replace('csv', 'npy'))
    transformation_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                      TEST_FILE_NAME.replace('csv', 'npy'))
    transformed_object_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMED_TRANSFORMED_OBJECT_DIR,
                                                     PREPROCESSING_OBJECT_FILE_NAME)
    
# @dataclass
# class ModelTrainerConfig:
#     model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
#     trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)
#     excepted_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
#     model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
#     _n_estimators = MODELTRAINER_N_ESTIMATORS
#     _min_samples_split = MODEL_TRAINER_MIN_SAMPLES_SPLIT
#     _min_samples_leaf = MOEDL_TRAINER_MIN_SAMPLES_LEAF
#     _max_depth  = MIN_SAMPLES_SPLIT_MAX_DEPTH
#     _criteration = MIN_SAMPLES_SPLIT_CRITERATION
#     _random_state = MIN_SAMPLES_SPLIT_RANDOM_STATE
# 
# @dataclass
# class ModelEvaluationConfig:
#     changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
#     bucket_name: str = MODEL_BUCKET_NAME
#     s3_model_key_path: str = MODEL_FILE_NAME
# 
# @dataclass
# class ModelPusherConfig:
#     bucket_name: str = MODEL_BUCKET_NAME
#     s3_model_key_path: str = MODEL_FILE_NAME
# 
# @dataclass
# class PredictionMaintenanceConfig:
#     model_file_path: str = MODEL_FILE_NAME
#     model_bucket_name: str = MODEL_BUCKET_NAME