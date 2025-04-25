import os
import sys

import numpy as np
import pandas as pd
from churn_prediction.entity.config_entity import ModelPredictConfig
from churn_prediction.entity.s3_estimator import  CustomerChurnEstimator
from churn_prediction.exception import CustomerChurnException
from churn_prediction.logger import logging
from churn_prediction.utils.main_utils import read_yaml_file
from pandas import DataFrame


class CustomerChurnData:
    def __init__(self,
                married,
                city,
                multiple_lines,
                number_of_dependents,
                number_of_referrals,
                device_protection_plan,
                total_revenue ,
                online_security,
                total_extra_data_charges,
                payment_method,
                paperless_billing,
                internet_service ,
                internet_type,
                online_backup,
                premium_tech_support,
                contract,
                refund_category,
                streaming_tv,
                streaming_movies,
                streaming_music,
            ):
        """
        Customer Churn Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.married = married
            self.city = city
            self.multiple_lines = multiple_lines
            self.number_of_dependents = number_of_dependents
            self.number_of_referrals = number_of_referrals
            self.device_protection_plan = device_protection_plan
            self.total_revenue = total_revenue 
            self.online_security = online_security
            self.total_extra_data_charges = total_extra_data_charges
            self.payment_method = payment_method
            self.paperless_billing = paperless_billing
            self.internet_service = internet_service 
            self.internet_type = internet_type
            self.online_backup = online_backup
            self.premium_tech_support = premium_tech_support
            self.contract = contract
            self.refund_category = refund_category
            self.streaming_tv = streaming_tv
            self.streaming_movies = streaming_movies
            self.streaming_music = streaming_music

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


    def get_customer_data_as_dict(self): # data in the form of dict is required as it easy to convert into a
                                       # dataframe from dict
        """                            
        This function returns a dictionary from USvisaData class input 
        """
        logging.info("Entered get_customer_data_as_dict method as CustomerChurnData class")

        try:
            input_data = {
               "married": [self.married],
               "city" : [self.city], 
               "multiple_lines" : [self.multiple_lines],
               "number_of_dependents" : [self.number_of_dependents],
               "number_of_referrals" : [self.number_of_referrals], 
               "device_protection_plan" : [self.device_protection_plan], 
               "total_revenue" : [self.total_revenue],   
               "online_security" : [self.online_security],  
               "total_extra_data_charges" : [self.total_extra_data_charges],  
               "payment_method" : [self.payment_method],  
               "paperless_billing" : [self.paperless_billing], 
               "internet_service" : [self.internet_service],   
               "internet_type" : [self.internet_type], 
               "online_backup" : [self.online_backup],  
               "premium_tech_support": [self.premium_tech_support],  
               "contract" : [self.contract], 
               "refund_category" : [self.refund_category],
               "streaming_tv" : [self.streaming_tv], 
               "streaming_movies" : [self.streaming_movies],  
               "streaming_music" : [self.streaming_music], 
            }

            logging.info("Created customer churn data as dict")

            logging.info("Exited get_customer_data_as_dict method as CustomerChurnData class")

            return input_data

        except Exception as e:
            raise CustomerChurnException(e, sys) from e
        
    def get_customer_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from CustomerChurnData class input
        """
        try:
            
            customer_data_input_dict = self.get_customer_data_as_dict()
            return DataFrame(customer_data_input_dict)
        
        except Exception as e:
            raise CustomerChurnException(e, sys) from e

class CustomerChurnClassifier:
    def __init__(self,prediction_pipeline_config: ModelPredictConfig = ModelPredictConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise CustomerChurnException(e, sys)


    def predict(self, dataframe) -> str:
        """
        This is the method of CustomerChurnClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of CustomerChurnClassifier class")
            model = CustomerChurnEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise CustomerChurnException(e, sys)