import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass
from pathlib import Path


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.preprocessing import LabelEncoder

from src.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
        self.city_encoder=None
        self.locality_means=None
        
    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated')
            
            categorical_cols=['seller_type', 'layout_type', 'property_type', 'furnish_type']
            numerical_cols=['bedroom', 'area', 'bathroom']
            
            #Define categories
            sell_cat=['OWNER', 'AGENT', 'BUILDER']
            layout_cat=['BHK', 'RK']
            property_cat=['Apartment', 'Studio Apartment', 'Independent House', 'Villa','Independent Floor', 'Penthouse']
            furnished_cat=['Furnished', 'Semi-Furnished', 'Unfurnished']
            
            logging.info('Pipeline Initiated')
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )
            
            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("ordinalencoder",OrdinalEncoder(categories=[sell_cat,layout_cat,property_cat,furnished_cat]))

                ]
            )
            
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])            
            
            return preprocessor
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)
        
    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read train and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            preprocessing_obj = self.get_data_transformation()
            target_column_name = 'price'            
            drop_columns = [target_column_name,'locality','city']
            
            # For city encoding_train_data
            city_encoder = LabelEncoder()
            train_df['city_encoded'] = city_encoder.fit_transform(train_df['city'])
            self.city_encoder=city_encoder
            
            # For city encoding_test_data
            city_encoder = LabelEncoder()
            test_df['city_encoded'] = city_encoder.fit_transform(test_df['city'])
            
            
            # For locality encoding_train_data
            locality_means = train_df.groupby('locality')['price'].mean()
            train_df['locality_encoded'] = train_df['locality'].map(locality_means)
            self.locality_means=locality_means
            
            # For locality encoding_test_data
            locality_means = test_df.groupby('locality')['price'].mean()
            test_df['locality_encoded'] = test_df['locality'].map(locality_means)  
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name] 
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            return (
                train_arr,
                test_arr
            )                                                                     
            
      
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)            