import sys
import os
import pymongo
# import certifi -> use if server timeout issuse occurs

from churn_prediction.exception import CustomerChurnException
from churn_prediction.logger import logging
from churn_prediction.constants import DATABASE_NAME, MONGODB_URL_KEY

# ca = certifi.where() 

class MongoDBClient():
    """
    Class Name: export_data_into_feature_store
    Description: This method exports the dataframe from feature store as dataframe

    Output: connection to mongodb database
    On Failure: raises an exception
    """

    client = None  

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment key: {MONGODB_URL_KEY} is not set")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url)#, tlsCAFile=ca) -> add it if server timeout issuse comes
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection successfull")
        except Exception as e:
            raise CustomerChurnException(e, sys)  