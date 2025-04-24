import sys
from pandas import DataFrame

from churn_prediction.cloud_storage.aws_storage import StorageService
from churn_prediction.logger import logging
from churn_prediction.exception import CustomerChurnException
from churn_prediction.entity.estimator import ChurnModel

class CustomerChurnEstimator:
    """
    This class is used to save and retrieve customer_churn_model in s3 bucket and do the prediction
    """

    def __init__(self,bucket_name,model_path,):
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """
        self.bucket_name = bucket_name
        self.s3 = StorageService()
        self.model_path = model_path
        self.loaded_model:ChurnModel=None

    def is_model_present_in_bucket(self,model_path):
        try:
            return self.s3.s3_key_path(bucket_name=self.bucket_name, s3_key=model_path)
        except CustomerChurnException as e:
            print(e)
            return False

    def load_model(self,)->ChurnModel:
        """
        Load the model from the model_path
        :return:
        """
        return self.s3.load_model(self.model_path,bucket_name=self.bucket_name)

    def save_model(self,from_file,remove:bool=False)->None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        :return:
        """
        try:
            self.s3.upload_file(from_file,
                                to_filename=self.model_path,
                                bucket_name=self.bucket_name,
                                remove=remove
                                )
        except Exception as e:
            raise CustomerChurnException(e, sys)
    
    def give_prediction(self,dataframe:DataFrame):
        """
        :param dataframe:
        :return:
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise CustomerChurnException(e, sys)
