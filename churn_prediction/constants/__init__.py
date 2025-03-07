import os
from datetime import date

DATABASE_NAME = "Churn_Prediction"
COLLECTION_NAME = "customerdata"
MONGODB_URL_KEY = "MONGO_DB_CHURN"

PIPELINE_NAME: str = 'customerchurn'
ARTIFACT_DIR:str = "artifact"

FILE_NAME: str = "customer_churn_data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

MODEL_FILE_NAME = "model.pkl"

"""
Data Ingestion related constants
"""
DATA_INGESTION_COLLECTION_NAME: str = "customerdata"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2