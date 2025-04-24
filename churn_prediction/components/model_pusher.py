import sys

from churn_prediction.exception import CustomerChurnException
from churn_prediction.logger import logging
from churn_prediction.entity.config_entity import ModelPusherConfig
from churn_prediction.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from churn_prediction.entity.s3_estimator import CustomerChurnEstimator
from churn_prediction.cloud_storage.aws_storage import StorageService

class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact, 
                 model_pusher_config: ModelPusherConfig):
        self.s3 = StorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.mode_pusher_config = model_pusher_config
        self.churn_estimator = CustomerChurnEstimator(bucket_name=model_pusher_config.bucket_name,
                                                      model_path= model_pusher_config.s3_model_key_path)
        
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name: initiate_model_pusher
        Description: This function is used to initite model to push in cloud storage

        Output: Returns model pusher artifact
        On Failure: Wrties an exeception log and raise a custom exception
        """
        logging.info("Entered the intitate_model_pusher method of ModelPusher class")
        try:
            logging.info("Uploading artifacts folder to s3 bucket")
            self.churn_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path,)

            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.mode_pusher_config.bucket_name,
                                                        s3_model_path=self.mode_pusher_config.s3_model_key_path)

            logging.info("Uploading artifacts folder to s3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of class ModelPusher")

            return model_pusher_artifact    
        except Exception as e:
            raise CustomerChurnException(e,sys)