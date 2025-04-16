import sys
from churn_prediction.exception import CustomerChurnException
from churn_prediction.logger import logging
from churn_prediction.components.data_ingestion import DataIngestion
from churn_prediction.components.data_validation import DataValidation
from churn_prediction.components.data_transformation import DataTransformation

from churn_prediction.entity.config_entity import (DataIngestionConfig,
                                                   DataTransformationConfig)
from churn_prediction.entity.artifact_entity import (DataIngestionArtifact, 
                                                     DataValidationArtifact,
                                                     DataTransformationArtifact)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()

    def start_data_ingestion(self)-> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data fron mongodb")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact= data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys) from e
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact)-> DataValidationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data validation component
        """
        try:
            logging.info("Entered the start_data_validation method of TrainPipeline class")
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data validation completed")
            logging.info("Exited the start_data_validation method of TrainPipeline class")
            return data_validation_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys) from e
        
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact:DataValidationArtifact)-> DataTransformationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data transformation component
        """
        try:
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                    data_transformation_config=self.data_transformation_config,
                                                    data_validation_artifact=data_validation_artifact)
            
            data_transformation_artifact = data_transformation.initialize_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys)
    def run_pipeline(self,)-> None:
        """
        This method of TrainPipeline class is responsible for runing complete pipeline
        """
        try:
            data_ingestion_artifact= self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
                                                                           data_validation_artifact=data_validation_artifact)
            # model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

        except Exception as e:
            raise CustomerChurnException(e,sys)