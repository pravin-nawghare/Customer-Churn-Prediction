import os
import sys
import pandas as pd
import numpy as np

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

from churn_prediction.entity.config_entity import DataIngestionConfig
from churn_prediction.entity.artifact_entity import DataIngestionArtifact
from churn_prediction.logger import logging
from churn_prediction.exception import CustomerChurnException
from churn_prediction.data_access.customer_churn_data import CustomerChurnData

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomerChurnException(e,sys)

    def export_data_into_feature_store(self)->DataFrame:
        """
        Method name: export_data_into_feature_store
        Description: This method exports data from mongodb to csv file

        Output: data is returned as artifact of data ingestion components
        On Failure: Write an exception log and then raise an exception 
        """
        try:
            logging.info(f"Exporting data from mongodb")
            customer_data = CustomerChurnData()
            dataframe = customer_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path= os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")

            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        
        except Exception as e:
            raise CustomerChurnException(e, sys)

    def clean_target_variable(self,dataframe: DataFrame) -> pd.DataFrame:
        """
        Method name: clean_target_variable
        Description: This method cleans the target variable in the dataframe

        Output: data is returned as datframe 
        On Failure: Write an exception log and then raise an exception 
        """
        try:
            logging.info("Cleaning target variable started")
            col_to_drop_before_missing_data = ['latitude','longitude']
            logging.info(f'Dataframe shape before dropping columns:{dataframe.shape}')
            dataframe.drop(columns=col_to_drop_before_missing_data,axis=1,inplace=True)
            logging.info("Latitude and Longitude columns dropped")
            logging.info(f'Dataframe shape after dropping columns:{dataframe.shape}')

            dataframe['customer_status'] = dataframe['customer_status'].replace({"Joined":'Stayed'})

            return dataframe

        except KeyError as e:
            print(f"Error: Missing  columns {e} in the dataframe")

        except Exception as e:
            raise CustomerChurnException(e,sys)
        
    def preprocess_data_for_missing_value_imputation(self,dataframe: DataFrame) -> pd.DataFrame:
        """
        Method name: preprocess_data_for_missing_value_imputation
        Description: This method converts the categorical data into numerical values for missing value imputation

        Output: data is returned as datframe 
        On Failure: Write an exception log and then raise an exception 
        """
        try:

            logging.info("Data conversion for missing value imputation started")

            city_counts = dataframe['city'].value_counts()
            cities_with_count_50_or_more = city_counts[city_counts >= 50].index.tolist()
            dataframe['city'] = dataframe['city'].apply(lambda x: x if x in cities_with_count_50_or_more else 'Other')

            categorical_features = [col for col in dataframe.columns if dataframe[col].dtype == 'O']
            numerical_features = [col for col in dataframe.columns if dataframe[col].dtype != 'O']
            logging.info("Categorical and Numerical features seperated")
            logging.info(f"length of features: {len(categorical_features+numerical_features)}")

            categories = [
                ['Male', 'Female'],
                ['No','Yes'],
                ['Escondido','Oakland','Long Beach','Fresno','San Francisco','Sacramento','San Jose','San Diego','Los Angeles','Other'],
                ['No','Yes'],
                ['No','Yes'],
                ['No', 'Yes'],
                ['Cable','DSL','Fiber Optic'],
                ['Yes','No'],
                ['Yes','No'],
                ['Yes','No'],
                ['Yes','No'],
                ['No','Yes'],
                ['No','Yes'],
                ['Yes','No'],
                ['No','Yes'],
                ['Month-to-Month','One Year','Two Year'],
                ['No','Yes'],
                ['Mailed Check','Credit Card','Bank Withdrawal'],
                ['Stayed','Churned']
                ]
            logging.info(f"Missing value counts: {dataframe.isna().sum().sum()}")
            logging.info("Ordinal Encoding started")

            categorical_transformer = Pipeline(steps=[
                ('ordinal_encoder',OrdinalEncoder(categories=categories,
                                                  handle_unknown='use_encoded_value',
                                                  unknown_value=np.nan))
                ]
            )
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
                ]
            )
            transformer = ColumnTransformer([
                ('numeric',numeric_transformer, numerical_features),
                ('category',categorical_transformer,categorical_features)
                ], remainder='passthrough'
            )
            
            preprocess_data = transformer.fit_transform(dataframe)
            logging.info("Data successfully encoded for missing value imputation")
            logging.info(f"Preprocess data shape: {preprocess_data.shape}")

            # Extract feature names 
            feature_names = numerical_features + categorical_features
            # Create a DataFrame from preprocess_data with the correct column names
            processed_data = pd.DataFrame(preprocess_data, columns=feature_names, index=dataframe.index)
            logging.info("Transformed data converted to dataframe")
            logging.info(f"Processed data shape: {processed_data.shape}")

            if processed_data.shape[1] != len(feature_names):
                feature_names = feature_names[:processed_data.shape[1]]
                raise ValueError(f"Mismatch in number of columns: {processed_data.shape[1]} vs {len(feature_names)}")
                
            logging.info("Missing data imputation starts")
            mice_imputer = IterativeImputer(estimator=RandomForestRegressor(
                                                n_estimators=120,
                                                max_depth=8,
                                                warm_start=True,
                                                n_jobs=-1,
                                                max_features='sqrt',
                                                min_samples_split=5,
                                                min_samples_leaf=3
                                                ), 
                                            max_iter=20, random_state=0)
            imputed_data = mice_imputer.fit_transform(processed_data)
            logging.info("Missing value imputation completed")
            logging.info(f"Imputed data shape: {imputed_data.shape}")

            if imputed_data.shape[1] != len(feature_names):
                raise ValueError(f"Mismatch in number of columns: {imputed_data.shape[1]} vs {len(feature_names)}")
            
            imputed = pd.DataFrame(imputed_data, columns=feature_names,index=dataframe.index)
            logging.info(f"Missing value counts after imputation: {imputed.isna().sum().sum()}")

            logging.info("Inverse transform for scaled numeric values start")
            scaler_values = transformer.named_transformers_['numeric']['scaler']
            inverse_scaled_values = scaler_values.inverse_transform(imputed_data[:,:len(numerical_features)])
            logging.info("Inverse transform done")

            inversed_transformed_df = pd.DataFrame(inverse_scaled_values,columns=numerical_features,index=dataframe.index)
            inversed_transformed_df = pd.concat([inversed_transformed_df,imputed[categorical_features]],axis=1)
            logging.info(f"Inversed transformed dataframe shape: {inversed_transformed_df.shape}")
            logging.info(f"Inversed transformed df missing value count: {inversed_transformed_df.isna().sum().sum()}")
            
            return inversed_transformed_df

        except Exception as e:
            raise CustomerChurnException(e,sys)
    
    def convert_back_to_original_dataframe(self, dataframe:pd.DataFrame)-> pd.DataFrame:
        """
        Method name: convert_back_to_original_dataframe
        Description: This method retrives dataframe with categorical values like in the original dataframe

        Output: Dataframe with categorical values is returned
        On Failure: Write an exception in log and then raise an exception 
        """
        try:
            mapping = {
                'gender': {0:'Male', 1:"Female"},
                'married' : {0:'No',1:'Yes'},
                'city' : {0:'Escondido',1:'Oakland',2:'Long Beach',3:'Fresno',4:'San Francisco',5:'Sacramento',6:'San Jose',
                          7:'San Diego',8:'Los Angeles',9:'Other'},
                'phone_service' : {0:'No',1:'Yes'},
                'multiple_lines' : {0:'No',1:'Yes'},
                'internet_service' : { 0:'No', 1:'Yes'},
                'internet_type' : {0:'Cable',1:'DSL',2:'Fiber Optic'},
                'online_security' : {0:'Yes',1:'No'},
                'online_backup' : {0:'Yes',1:'No'},
                'device_protection_plan' : {0:'Yes',1:'No'},
                'premium_tech_support' : {0:'Yes',1:'No'},
                'streaming_tv' : {0:'No',1:'Yes'},
                'streaming_movies' : {0:'No',1:'Yes'},
                'streaming_music' : {0:'Yes',1:'No'},
                'unlimited_data' : {0:'No',1:'Yes'},
                'contract' : {0:'Month-to-Month',1:'One Year',2:'Two Year'},
                'paperless_billing': {0:'No',1:'Yes'},
                'payment_method' : {0:'Mailed Check',1:'Credit Card',2:'Bank Withdrawal'},
                'customer_status' : {0:'Stayed',1:'Churned'}
            }
            categorical_features = ['gender','married', 'city', 'phone_service', 'multiple_lines', 'internet_service', 'internet_type', 
                                    'online_security', 'online_backup', 'device_protection_plan', 'premium_tech_support','streaming_tv', 
                                    'streaming_movies', 'streaming_music', 'unlimited_data', 'contract', 'paperless_billing', 
                                    'payment_method', 'customer_status']
            for column in categorical_features:
                dataframe[column] = dataframe[column].astype(int).map(mapping[column])
            logging.info("Column mapping completed")
            logging.info(f"Missing value counts: {dataframe.isna().sum().sum()}")

            return dataframe
        except Exception as e:
            raise CustomerChurnException(e,sys)

    def split_data_as_train_test(self, dataframe: DataFrame)-> None:
        """
        Method name: split_data_as_train_test
        Description: This method splits the dataframe into train set and test set

        Output: folder is created in s3 bucket
        On Failure: Write an exception in log and then raise an exception 
        """
        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio,random_state=0)
            logging.info(f"Train set missing value count: {train_set.isna().sum().sum()}")
            logging.info(f"Test set missing value count: {test_set.isna().sum().sum()}")
            logging.info("Performed train test split on the dataframe")
            logging.info("Exited split_data_as_train_test method of Data_Ingestion class")
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)#
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)#

            logging.info(f"Exported train and test file path")
        except Exception as e:
            raise CustomerChurnException(e,sys) from e
        
    def initiate_data_ingestion(self)->DataIngestionArtifact: # -> return type of function
            """
            Method name: initiate_data_ingestion
            Description: This method initiates the data ingestion components of training pipeline

            Output: train set and test set are returned as the artifact of data ingestion components
            On Failure: Write an exception in log and then raise an exception 
            """
            logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")
            try:
                dataframe = self.export_data_into_feature_store()
                logging.info("Got the data from mongodb")

                clean_target_data = self.clean_target_variable(dataframe)
                logging.info("Target variable cleaned")

                handle_missing_data = self.preprocess_data_for_missing_value_imputation(clean_target_data)
                logging.info("Converted data for missing value imputation")

                original_dataframe = self.convert_back_to_original_dataframe(handle_missing_data)
                logging.info("Original dataframe is retrived after missing value imputation")

                self.split_data_as_train_test(original_dataframe)
                logging.info("Performed train test split on the dataset")

                logging.info("Exited initiate_data_ingestion method of Data_Ingestion class")

                data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                          test_file_path = self.data_ingestion_config.testing_file_path)

                logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
                return data_ingestion_artifact
            except Exception as e:
                raise CustomerChurnException(e,sys) from e