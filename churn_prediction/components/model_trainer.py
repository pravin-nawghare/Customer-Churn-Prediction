import sys
import os

import numpy as np
import pandas as pd

from typing import Tuple
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from churn_prediction.logger import logging
from churn_prediction.exception import CustomerChurnException
from churn_prediction.entity.config_entity import ModelTrainerConfig
from churn_prediction.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact, ClassificationMetricArtifact 
from churn_prediction.entity.estimator import ChurnModel
from churn_prediction.utils.main_utils import load_numpy_array_data, load_object, save_object

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, 
                 model_trainer_config: ModelTrainerConfig):
        """
        :params data_transformation_artifact: Output reference of data_transformation_artifact stage
        :params model_trainer_config: Configuration for trained model
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
    
    def divide_data_into_train_and_test_set(self, train: np.array, test: np.array): #-> Tuple[object, object]:
        """
        Method name: get_model_object_and_report
        Description: This function uses neuro_mf to get the best model object and report of the best model

        Output: Returns metric artifact object and best model
        On Failure: Write an exception log and then raise an exception
        """
        try:
            logging.info("Dividing data into x_train, x_test, y_train, y_test")
            x_train, x_test, y_train, y_test = train[:,:-1], test[:,:-1], train[:,-1],  test[:,-1]
            return x_train, x_test, y_train, y_test
        except Exception as e:
            raise CustomerChurnException(e,sys)
            
        
    def get_best_model(self,model_class, x_train, x_test, y_train, y_test, param_grid):
        try:
            scorer = make_scorer(recall_score, average='binary', zero_division=0)

            # create a pipeline
            #logging.info("Creating pipeline for model training")
            pipeline = Pipeline([('model',model_class())])

            #Perform grid search
            #logging.info("Performing grid search")
            gs = GridSearchCV(pipeline,param_grid, cv=5, scoring=scorer, n_jobs=-1)
            gs.fit(x_train, y_train)

            # Get the best model
            
            best_model_pipeline = gs.best_estimator_
            best_model = best_model_pipeline.named_steps['model']
            best_model_pipeline.set_params(**gs.best_params_)
            #best_model.fit(x_train, y_train)
            y_pred = best_model_pipeline.predict(x_test)

            # Evaluate the best model
            recall = round(recall_score(y_test, y_pred),4)
            # print('Recall',recall)
            # print('best model', best_model)
            return best_model, recall
        except Exception as e:
            raise CustomerChurnException(e,sys)
            
    def start_hyperparameter_tuning(self,x_train, x_test, y_train, y_test):
        try:
            logging.info("Creating model dictionary for hyperparameter tuning")
            model_dict = {
                        'RandomForest': (
                                        RandomForestClassifier,
                                                {
                                                    'model__n_estimators': [100,200,300,400,500],
                                                    'model__max_depth':[7,8,9,10,12,14],
                                                    'model__min_samples_split':[3,4,5,6]    
                                        }
                         ),
                        'LogisticRegression': (
                                        LogisticRegression,
                                                {
                                                    'model__C':[0.001,0.01,0.1,1,10,100,1000],
                                                    'model__penalty':['l1','l2'],
                                                    'model__solver':['saga','liblinear']
                                        }
                        ),
                        'StochasticGradientDescent': (
                                        SGDClassifier,
                                                {
                                                    'model__loss':['log_loss','modified_huber'],
                                                    'model__penalty':['l2','elasticnet'],
                                                    'model__n_iter_no_change':[5,7,9,11],
                                                    'model__warm_start':[True,False],
                                                    'model__shuffle':[True,False]
                                        }
                        ),
                        'DecisionTree': (
                                        DecisionTreeClassifier,
                                                {
                                            'model__criterion':['gini', 'entropy', 'log_loss'],
                                            'model__splitter':['best', 'random'],
                                            'model__max_depth':[10,12,14,16],
                                            'model__min_samples_split':[3,4,5,6],
                                            'model__max_features':['sqrt','log2',None]
                                        }
                        )
            }

            best_overall_model = None
            best_recall_score = 0.0
            best_model_name = ""
            logging.info("Starting hyperparameter tuning")

            for name, (model_class, param_grid) in model_dict.items():
                model,recall = self.get_best_model(model_class, x_train, x_test, y_train, y_test, param_grid)
                if recall > best_recall_score:
                    best_overall_model = model 
                    best_recall_score = recall
                    best_model_name = name

            logging.info(f'Best Model is {best_model_name}')
            logging.info(f'Best recall score of model is {best_recall_score}')

            y_pred = best_overall_model.predict(x_test)
            logging.info(f"Best model: {best_overall_model}")

            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)

            metric_artifact =  ClassificationMetricArtifact(f1_score=f1, precision_score=precision,recall_score=recall,accuracy_score=accuracy)

            return best_overall_model,metric_artifact,recall
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

            x_train, x_test, y_train, y_test = self.divide_data_into_train_and_test_set(train=train_arr,test=test_arr)
            best_model, metric_artifact, best_recall = self.start_hyperparameter_tuning(x_train, x_test, y_train, y_test)

            preprocessing_obj = load_object(file_path= self.data_transformation_artifact.transformed_object_file_path)

            if best_recall < self.model_trainer_config.expected_recall:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")
            
            churnmodel = ChurnModel(preprocessing_object=preprocessing_obj,
                                    trained_model_object=best_model)
            logging.info("Created churn model object and preprocessor and model")
            logging.info("Created best model file path")
            save_object(self.model_trainer_config.trained_model_file_path, churnmodel)

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                          metric_artifact=metric_artifact)
            
            logging.info(f"Model Trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys) from e