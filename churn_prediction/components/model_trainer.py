import sys
import os

import numpy as np
import pandas as pd

from typing import Tuple
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from neuro_mf import ModelFactory

from churn_prediction.logger import logging
from churn_prediction.exception import CustomerChurnException
from churn_prediction.entity.config_entity import ModelTrainerConfig
from churn_prediction.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact, ClassificationMetricArtifact 
from churn_prediction.entity.estimator import ChurnModel
from churn_prediction.utils.main_utils import load_numpy_array_data, read_yaml_file, load_object, save_object

class ModelTrainer:
    def __init__(self, data_transformaion_artifact: DataTransformationArtifact, 
                 model_trainer_config: ModelTrainerConfig):
        """
        :params data_transformation_artifact: Output reference of data_transformation_artifact stage
        :params model_trainer_config: Configuration for trained model
        """
        self.data_transformation_artifact = data_transformaion_artifact
        self.model_trainer_config = model_trainer_config
    
    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method name: get_model_object_and_report
        Description: This function uses neuro_mf to get the best model object and report of the best model

        Output: Returns metric artifact object and best model
        On Failure: Write an exception log and then raise an exception
        """
        try:
            logging.info("Using neuro_mf to get best model")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            
            x_train, x_test, y_train, y_test = train[:,:-1], train[:,-1], test[:,:-1], test[:,-1]

            best_model_detail = model_factory.get_best_model(
                X=x_train, y=y_train, base_accuracy=self.model_trainer_config.expected_accuracy
            )
            model_obj = best_model_detail.best_model

            y_pred = model_obj.predict(x_test)

            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)

            metric_artifact =  ClassificationMetricArtifact(f1_score=f1, precision_score=precision,recall_score=recall,accuracy_score=accuracy)

            return best_model_detail,metric_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys) from e
        
    def initiate_model_trainer(self, ) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer methd of ModelTrainer class")
        """
        Method name: initiate_model_trainer
        Description: This function intiates model trainer steps

        Output: Returns model trainer artifact
        On Failure: Write an exception log and then raise an exception
        """
        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            best_model_detail, metric_artifact = self.get_model_object_and_report(train=train_arr,test=test_arr)

            preprocessing_obj = load_object(file_path= self.data_transformation_artifact.transformed_object_file_path)

            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")
            
            churnmodel = ChurnModel(preprocessing_object=preprocessing_obj,
                                    trained_model_object=best_model_detail.best_model)
            logging.info("Created churn model object and preprocessor and model")
            logging.info("Created best model file path")
            save_object(self.model_trainer_config.trained_model_file_path, churnmodel)

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                          metric_artifact=metric_artifact)
            
            logging.info(f"Model Trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys) from e