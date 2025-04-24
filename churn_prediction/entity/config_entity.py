import os
from churn_prediction.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m-%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR)
    timestamp: str = TIMESTAMP

#variable: object = class_name()
training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME

@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, 
                                                DATA_TRANSFORMATION_DIR_NAME)
    transformed_trained_file_path: str = os.path.join(data_transformation_dir,
                                                   DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                   TRAIN_FILE_NAME.replace("csv","npy"))
    transformed_test_file_path: str = os.path.join(data_transformation_dir,
                                                   DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                   TEST_FILE_NAME.replace("csv","npy"))
    transformed_object_file_path: str = os.path.join(data_transformation_dir,
                                                     DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                     PREPROCESSING_OBJECT_FILE_PATH)
    
@dataclass
class ModelTrainerConfig:
    model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)
    expected_recall: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH

@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME

@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME

@dataclass
class ModelPredictConfig:
    model_file_path: str = MODEL_FILE_NAME # because we are using s3 model so this model is used
    model_bucket_name: str = MODEL_BUCKET_NAME