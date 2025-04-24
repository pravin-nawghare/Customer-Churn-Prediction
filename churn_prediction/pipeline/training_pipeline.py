import sys
from churn_prediction.exception import CustomerChurnException
from churn_prediction.logger import logging
from churn_prediction.components.data_ingestion import DataIngestion
from churn_prediction.components.data_validation import DataValidation
from churn_prediction.components.data_transformation import DataTransformation
from churn_prediction.components.model_trainer import ModelTrainer
from churn_prediction.components.model_evaluation import ModelEvaluation
from churn_prediction.components.model_pusher import ModelPusher
from churn_prediction.entity.config_entity import (DataIngestionConfig,
                                                   DataTransformationConfig,
                                                   ModelTrainerConfig,
                                                   ModelEvaluationConfig,
                                                   ModelPusherConfig)
from churn_prediction.entity.artifact_entity import (DataIngestionArtifact, 
                                                     DataValidationArtifact,
                                                     DataTransformationArtifact,
                                                     ModelTrainerArtifact,
                                                     ModelEvaluationArtifact,
                                                     ModelPusherArtifact)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

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
            logging.info("Entered the start_data_transformation method of TrainPipeline class")
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                    data_transformation_config=self.data_transformation_config,
                                                    data_validation_artifact=data_validation_artifact)
            
            data_transformation_artifact = data_transformation.initialize_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys)
        
    def start_model_trainer(self,data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        This method of TrainPipeline class is responsible for starting model training component
        """
        try:
            logging.info("Entered the start_model_trainer method of TrainPipeline class")
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                               model_trainer_config=self.model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys)
    
    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact, 
                               model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        """
        This method of TrainPipeline class is responsible for starting model evaluation component
        """
        try:
            logging.info("Entered the start_model_evaluation method of TrainePipeline class")
            model_evaluation = ModelEvaluation(data_ingestion_artifact=data_ingestion_artifact,
                                               model_eval_config=self.model_evaluation_config,
                                               model_trainer_artifact=model_trainer_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys)
        
    def start_model_to_push_in_cloud_environment(self,model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        This method of TrainPipeline class is responsible to push the best trained model in cloud environment
        """
        try:
            logging.info('Entered the start_model_to_push_in_cloud_environment method of TrainPipeline class')
            model_pusher = ModelPusher(model_evaluation_artifact=model_evaluation_artifact,
                                       model_pusher_config=self.model_pusher_config)
            start_model_pushing = model_pusher.initiate_model_pusher()
            return start_model_pushing
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
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                    model_trainer_artifact=model_trainer_artifact)
            
            if not model_evaluation_artifact.ismodelaccepted:
                logging.info(f"Model not accepted")
                return None 
            model_pusher_artifact = self.start_model_to_push_in_cloud_environment(model_evaluation_artifact=model_evaluation_artifact)
        except Exception as e:
            raise CustomerChurnException(e,sys)