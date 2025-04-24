import sys
import pandas as pd

from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import recall_score

from churn_prediction.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from churn_prediction.logger import logging
from churn_prediction.exception import CustomerChurnException
from churn_prediction.entity.config_entity import ModelEvaluationConfig
from churn_prediction.entity.artifact_entity import ModelTrainerArtifact,DataIngestionArtifact,ModelEvaluationArtifact
from churn_prediction.entity.estimator import TargetValueMapping, ChurnModel
from churn_prediction.entity.s3_estimator import CustomerChurnEstimator
from churn_prediction.utils.main_utils import drop_columns, read_yaml_file

@dataclass
class EvaluateModelResponse:
    trained_model_recall_score: float
    best_model_recall_score: float
    is_model_accepted: bool
    difference: float

class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomerChurnException(e, sys) from e

    def get_best_model(self) -> Optional[CustomerChurnEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model in production
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            customer_churn_estimator = CustomerChurnEstimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if customer_churn_estimator.is_model_present_in_bucket(model_path=model_path):
                return customer_churn_estimator
            return None
        except Exception as e:
            raise  CustomerChurnException(e,sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            bins = [0, 10, 20, 35, 50]
            labels = ['No Refunds', 'Low Refunds', 'Medium Refunds', 'High Refunds']
            test_df['refund_category'] = pd.cut(test_df['total_refunds'], bins=bins, labels=labels,include_lowest=True)

            drop_cols = self._schema_config['drop_columns']
            test_df = drop_columns(df=test_df,cols=drop_cols)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            y = y.replace(
                TargetValueMapping()._asdict()
            )
            # trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            trained_model_recall_score = self.model_trainer_artifact.metric_artifact.recall_score

            best_model_recall_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                y_hat_best_model = best_model.give_prediction(x)
                best_model_recall_score = recall_score(y, y_hat_best_model)
            
            tmp_best_model_score = 0 if best_model_recall_score is None else best_model_recall_score
            result = EvaluateModelResponse(trained_model_recall_score=trained_model_recall_score,
                                           best_model_recall_score=best_model_recall_score,
                                           is_model_accepted=trained_model_recall_score > tmp_best_model_score,
                                           difference=trained_model_recall_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                ismodelaccepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_recall_score=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise CustomerChurnException(e, sys) from e

