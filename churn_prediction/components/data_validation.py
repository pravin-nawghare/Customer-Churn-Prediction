import json
import sys

import pandas as pd
from pandas import DataFrame

from churn_prediction.exception import CustomerChurnException
from churn_prediction.logger import logging
from churn_prediction.utils.main_utils import read_yaml_file
from churn_prediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from churn_prediction.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact):
        """
        :param data_ingestion_artifact: output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise CustomerChurnException(e,sys)
    
    def validate_number_of_columns(self, dataframe:pd.DataFrame)-> bool:
        """
        Method Name: validate_number_of_columns
        Description: This method validates the number of columns

        Output: Returns bool value based on validation results
        On Failure: Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config['columns'])
            logging.info(f"Is required columns present: [{status}]")
            return status
        except Exception as e:
            raise CustomerChurnException(e,sys)
        
    def does_all_columns_exist(self, df:pd.DataFrame)-> bool:
        """
        Method Name: does_all_columns_exist
        Description: This method checks does all numerical and categorical columns exists

        Output: Returns bool value based on validation results
        On Failure: Write an exception log and then raise an exception
        """        
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in self._schema_config['numerical_columns']:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)
                
            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")
            
            for column in self._schema_config['categorical_columns']:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)
                
            if len(missing_categorical_columns)>0:
                logging.info(f"Missing numerical column: {missing_categorical_columns}")
            
            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        
        except Exception as e:
            raise CustomerChurnException(e,sys) from e
        
    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomerChurnException(e,sys) from e
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name: initiate_data_validation
        Description: This method initiates the data validation component for the pipeline

        Output: Returns bool value based on validation results
        On Failure: Write an exception log and then raise an exception
        """
        try:
            validation_error_message = ""
            logging.info("Starting data validation")
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                                DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))
            
            # Check for train file
            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                validation_error_message += f"Columns are missing in training file"
            
            # Check test file
            status = self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_message += f"Columns are missing in test file"

            status = self.does_all_columns_exist(df=train_df)
            if not status:
                validation_error_message += f"Columns are missing in training dataframe"

            status = self.does_all_columns_exist(df=test_df)
            if not status:
                validation_error_message += f"Columns are missing in test dataframe"
            
            validation_status = len(validation_error_message) == 0

            
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_message
            )
            logging.info(f"Data Validation Artifact: {data_validation_artifact}")
            return data_validation_artifact
        
        except Exception as e:
            raise CustomerChurnException(e,sys)
        
