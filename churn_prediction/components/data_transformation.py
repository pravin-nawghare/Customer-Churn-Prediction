import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from churn_prediction.entity.artifact_entity import (DataIngestionArtifact, 
                                                     DataTransformationArtifact, 
                                                     DataValidationArtifact)
from churn_prediction.entity.config_entity import DataTransformationConfig
from churn_prediction.logger import logging
from churn_prediction.exception import CustomerChurnException
from churn_prediction.constants import SCHEMA_FILE_PATH, TARGET_COLUMN, SAMPLING_RATIO
from churn_prediction.utils.main_utils import save_object, save_numpy_array_data,read_yaml_file,drop_columns
from churn_prediction.entity.estimator import TargetValueMapping

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomerChurnException(e,sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomerChurnException(e,sys)
        
    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name: get_data_transformer_object
        Description: This method creates and returns a data transformer object for the data

        Output: data transformer object is created and returned
        On Failure: Write an exception log and then raise an exception
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            logging.info("Got numerical columns from schema config")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(sparse_output=False, handle_unknown='error')
            or_transformer = OrdinalEncoder()

            logging.info("Initialized standardscaler, ohehotencoder, ordinalecoder objects")

            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']

            logging.info("Initialize Power Transformer")

            transformer_pipeline = Pipeline(
                steps=[
                    ('transformer',PowerTransformer(method='yeo-johnson'))
                ]
            )
            
            preprocessor = ColumnTransformer(
                [
                    ('OneHotEncoder',oh_transformer, oh_columns),
                    ('OrdinalEncoder',or_transformer, or_columns),
                    ('transformer', transformer_pipeline, transform_columns),
                    ('StandardScaler',numeric_transformer, num_features)
                ], remainder='passthrough'
            )

            logging.info("Created preprocessor object from Column Transformer")

            logging.info("Exited get_data_transformer_object method of DataTransformation class")

            return preprocessor
        except Exception as e:
            raise CustomerChurnException(e,sys)

    def initialize_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name: initialize_data_transformation
        Description: This method initiates the data transformation component for the pipeline

        Output: data transformer steps are preformed and preprocessor object is created
        On Failure: Write an exception log and then raise an exception
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                logging.info(f"Train df loaded from data_transformation.read_data shape: {train_df.shape}")
                logging.info(f"Test df loaded from data_transformation.read_data shape: {test_df.shape}")

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]
                logging.info("Got train features and test features of Training dataset")

                logging.info(f"Input feature train set shape after dropping target feature: {input_feature_train_df.shape}")
                logging.info(f"Target feature train set shape after dropping target feature: {target_feature_train_df.shape}")
                
                bins = [0, 10, 20, 35, 50]
                labels = ['No Refunds', 'Low Refunds', 'Medium Refunds', 'High Refunds']
                input_feature_train_df['refund_category'] = pd.cut(input_feature_train_df['total_refunds'], bins=bins, labels=labels,include_lowest=True)
                logging.info("Created refund_category column in train features")
                logging.info(f"Input feature train df shape after creating refund_category: {input_feature_train_df.shape}")

                logging.info(f"Refund category from input feature train df value counts:{input_feature_train_df['refund_category'].value_counts()}")

                drop_cols = self._schema_config['drop_columns']

                logging.info("Drop the columns in drop_cols of Training dataset")

                input_feature_train_df = drop_columns(df=input_feature_train_df,cols=drop_cols)
                logging.info(f"Input feature train df shape after dropping irrevelant features: {input_feature_train_df.shape}")
                
                target_feature_train_df = target_feature_train_df.replace(
                    TargetValueMapping()._asdict()
                )

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                logging.info(f"Input feature test df shape after dropping target feature: {input_feature_train_df.shape}")
                logging.info(f"Target feature test df shape after dropping target feature: {target_feature_train_df.shape}")

                input_feature_test_df['refund_category'] = pd.cut(input_feature_test_df['total_refunds'], bins=bins, labels=labels,include_lowest=True)
                logging.info("Created refund_category column in test features")
                logging.info(f"Input feature test df shape after creating refund_category: {input_feature_test_df.shape}")

                logging.info(f"Refund category from input feature test df value counts:{input_feature_test_df['refund_category'].value_counts()}")
                
                input_feature_test_df = drop_columns(df=input_feature_test_df,cols=drop_cols)

                logging.info("Drop the columns in drop_cols of Test dataset")
                
                target_feature_test_df = target_feature_test_df.replace(
                    TargetValueMapping()._asdict()
                )
                logging.info("Got train features and test features of test dataset")
                logging.info("Applying preprocessing object on training dataset and testing dataset")

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                logging.info("Used the preprocessor object to fit_transform the train features")

                input_feature_test_arr = preprocessor.fit_transform(input_feature_test_df)
                logging.info("Used the preprocessor object to fit_transform the test features")

                logging.info("Applying ADASYN on training dataset")
                ada = ADASYN(sampling_strategy=SAMPLING_RATIO, random_state=0)
                input_feature_train_final, target_feature_train_final = ada.fit_resample(input_feature_train_arr, target_feature_train_df)
                logging.info("Used the ADASYN to fit_resample the train features")

                logging.info("Applying ADASYN on test dataset")
                input_feature_test_final, target_feature_test_final = ada.fit_resample(input_feature_test_arr, target_feature_test_df)
                logging.info("Used the ADASYN to fit_resample the test features")

                logging.info("Created train array and test array")

                train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
                test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_trained_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
                logging.info("Saved the preprocessor object")

                logging.info("Exited initiate_data_transformation method of Data_Transformation class")

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path= self.data_transformation_config.transformed_object_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_trained_file_path
                )
                return data_transformation_artifact
            
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise CustomerChurnException(e,sys)

