import sys
import os

from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
#from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from imblearn.over_sampling import RandomOverSampler

from dataclasses import dataclass

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_columns=['satisfaction_level', 'last_evaluation', 'number_project','average_montly_hours', 
                               'time_spend_company', 'Work_accident', 'left','promotion_last_5years']
            categorical_columns=['Departments ', 'salary']

            logging.info(f'numerical columns:{numerical_columns}')
            logging.info(f'categorical columns:{categorical_columns}')

            preprocessor=ColumnTransformer(transformers=
                                           [('ohe_encoder',OneHotEncoder(sparse_output=False,drop='first'),['Departments ']),
                                            ('ord_encoder',OrdinalEncoder(categories=[['low','medium','high']]),['salary']),
                                            ('std_scaler',StandardScaler(),['average_montly_hours'])
                                            ],remainder='passthrough'
                                            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train data and test data completed')

            #drop the duplicates 
            train_df.drop_duplicates(keep='first',inplace=True,ignore_index=True)
            test_df.drop_duplicates(keep='first',inplace=True,ignore_index=True)

            logging.info('Removed the duplicates')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformation_object()

            target_column_name='left'
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info('Applying preprocessing object on training dataframe and testing dataframe')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info('Oversampling started')

            os=RandomOverSampler(random_state=42)
            input_feature_train_arr,target_feature_train_df=os.fit_resample(input_feature_train_arr,target_feature_train_df)
            #input_feature_test_arr,target_feature_test_df=os.fit_resample(input_feature_test_arr,target_feature_test_df)

            logging.info('Forming train array and test array')
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('Saved preprocessing object')

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj)
            return(train_arr,test_arr)
            

        except Exception as e:
            raise CustomException(e,sys)