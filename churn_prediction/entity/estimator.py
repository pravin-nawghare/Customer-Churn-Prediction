import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from churn_prediction.exception import CustomerChurnException
from churn_prediction.logger import logging

class TargetValueMapping:
    def __init__(self):
        self.Stayed:int = 0
        self.Churned:int = 1
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))

class ChurnModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocessor
        :param trained_model_object: Input Object of trained model
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
    
    def predict(self, dataframe: DataFrame) -> DataFrame:
        """
        Function accepts raw input and then transformed raw input using preprocessing_object which guarantees that the
        inputs are in the same format as the training data.
        At last it performs predictions on transformed features
        """
        logging.info("Entered predict method of ChurnModel class")
        try:
            logging.info("Using the trained model to get predictions")

            transformed_feature = self.preprocessing_object.transform(dataframe)
            logging.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise CustomerChurnException(e,sys) from e
        
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"