"""
In this file we are going to load the data and other ml pipeline techniques which are needed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import sys
import os
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('main')
from sklearn.model_selection import train_test_split
from var_out import variable_transformation_outliers
#from feature_selection import complete_feature_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from balanced_data import scale_data
from all_models import linear_regression


class CALORIES_PREDICTION:
    def __init__(self,path1,path2):
        try:
            self.path1 = path1
            self.path2 = path2
            self.df1 = pd.read_csv(self.path1)
            self.df2 = pd.read_csv(self.path2)
            self.df = pd.merge(self.df1,self.df2,on = 'User_ID',how = 'inner')
            logger.info(f'Data Loaded Successfully')
            logger.info(f'Total number of rows and columns : {self.df.shape}')
            logger.info(f'sample data : {self.df.head(5)}')
            logger.info(f'Total number of rows : {self.df.shape[0]}')
            logger.info(f'Total number of columns : {self.df.shape[1]}')

            logger.info(f'Checking null values : {self.df.isnull().sum()}')
            logger.info(f'{self.df.info()}')

            self.df = self.df.drop(['User_ID'],axis = 1)
            self.df.reset_index(drop=True, inplace=True)

            #splitting X and y
            self.X = self.df.iloc[: ,:-1]
            self.y = self.df.iloc[:,-1]

            logger.info(f'X shape : {self.X.shape}')
            logger.info(f'y shape : {self.y.shape}')

            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)

            logger.info(f'total training data : {self.X_train.shape}')
            logger.info(f'{self.X_train.columns}')
            logger.info(f'Total testing data : {self.X_test.shape}')
            logger.info(f'{self.y_test.shape}')
            logger.info(f'{self.y_train.shape}')
            logger.info(f'{self.y_train.sample(5)}')
            logger.info(f'{self.y_test.sample(5)}')


        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
    def var_outli(self):
        try:
            logger.info(f'{self.X_train.columns}')
            logger.info(f'{self.X_test.columns}')
            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_train_cat = self.X_train.select_dtypes(include='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')

            logger.info(f'{self.X_train_num.columns}')
            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_num.columns}')
            logger.info(f'{self.X_test_cat.columns}')

            logger.info(f'{self.X_train_num.shape}')
            logger.info(f'{self.X_train_cat.shape}')
            logger.info(f'{self.X_test_num.shape}')
            logger.info(f'{self.X_test_cat.shape}')
            self.X_train_num,self.X_test_num = variable_transformation_outliers(self.X_train_num,self.X_test_num)
            logger.info(f"{self.X_train_num.columns} -> {self.X_train_num.shape}")
            logger.info(f"{self.X_test_num.columns} -> {self.X_test_num.shape}")
            logger.info(f'sample data :{self.X_train_num.sample(10)}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
    #
    #def fs(self):
      #  try:
      #      logger.info(f" Before : {self.X_train_num.columns} -> {self.X_train_num.shape}")
      #      logger.info(f"Before : {self.X_test_num.columns} -> {self.X_test_num.shape}")
     #       self.X_train_num,self.X_test_num = complete_feature_selection(self.X_train_num,self.X_test_num,self.y_train)
     #       logger.info(f" After : {self.X_train_num.columns} -> {self.X_train_num.shape}")
    #       logger.info(f" After : {self.X_test_num.columns} -> {self.X_test_num.shape}")
      #  except Exception as e:
    #       error_type, error_msg, error_line = sys.exc_info()
    #       logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def cat_to_num(self):

        try:
            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_cat.columns}')

            for i in self.X_train_cat.columns:
                    logger.info(f'{i} --> {self.X_train_cat[i].unique()}')

            logger.info(f'Before Converting : {self.X_train_cat}')
            logger.info(f'Before Converting : {self.X_test_cat}')

            # One-Hot Encoding
            one_hot = OneHotEncoder(drop='first')
            one_hot.fit(self.X_train_cat[['Gender']])
            res = one_hot.transform(self.X_train_cat[['Gender']]).toarray()

            f = pd.DataFrame(data=res, columns=one_hot.get_feature_names_out())
            self.X_train_cat.reset_index(drop=True, inplace=True)
            f.reset_index(drop=True, inplace=True)

            self.X_train_cat = pd.concat([self.X_train_cat, f], axis=1)
            self.X_train_cat = self.X_train_cat.drop(['Gender'], axis=1)

            res1 = one_hot.transform(self.X_test_cat[['Gender']]).toarray()
            f1 = pd.DataFrame(data=res1, columns=one_hot.get_feature_names_out())

            self.X_test_cat.reset_index(drop=True, inplace=True)
            f1.reset_index(drop=True, inplace=True)

            self.X_test_cat = pd.concat([self.X_test_cat, f1], axis=1)
            self.X_test_cat = self.X_test_cat.drop(['Gender'], axis=1)

            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_cat.columns}')

            logger.info(f"After Converting : {self.X_train_cat}")
            logger.info(f"After Converting : {self.X_test_cat}")

            logger.info(f"{self.X_train_cat.shape}")
            logger.info(f"{self.X_test_cat.shape}")

            logger.info(f"{self.X_train_cat.isnull().sum()}")
            logger.info(f"{self.X_test_cat.isnull().sum()}")

            self.X_train_num.reset_index(drop=True, inplace=True)
            self.X_train_cat.reset_index(drop=True, inplace=True)

            self.X_test_num.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)

            self.training_data = pd.concat([self.X_train_num, self.X_train_cat], axis=1)
            self.testing_data = pd.concat([self.X_test_num, self.X_test_cat], axis=1)


            logger.info(f"{self.training_data.shape}")
            logger.info(f"{self.testing_data.shape}")

            logger.info(f"{self.training_data.isnull().sum()}")
            logger.info(f"{self.testing_data.isnull().sum()}")

            logger.info(f"=======================================================")

            logger.info(f"Training Data : {self.training_data.sample(10)}")
            logger.info(f"Testing Data : {self.testing_data.sample(10)}")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def bal_data(self):

        try:
            self.training_data, self.testing_data = scale_data(
            self.training_data,
            self.testing_data
                )

            logger.info(f'Final Train Shape: {self.training_data.shape}')
            logger.info(f'Final Test Shape: {self.testing_data.shape}')

            linear_regression(
                    self.training_data,
                    self.y_train,
                    self.testing_data,
                    self.y_test
                )
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')


if __name__ == '__main__':
    try:
        obj = CALORIES_PREDICTION(f'C:\\Users\\hp\\Downloads\\Colorie_Brunt_projjct\\exercise.csv',
                               f'C:\\Users\\hp\\Downloads\\Colorie_Brunt_projjct\\calories.csv')
        obj.var_outli()
        #obj.fs()
        obj.cat_to_num()
        obj.bal_data()
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')