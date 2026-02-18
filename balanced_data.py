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
logger = setup_logging('balanced_data')
from sklearn.preprocessing import StandardScaler
import pickle



def scale_data(X_train, X_test):
    try:

      sc = StandardScaler()


      X_train_scaled = pd.DataFrame(sc.fit_transform(X_train),columns=X_train.columns,index  = X_train.index )

      X_test_scaled = pd.DataFrame(sc.transform(X_test),columns=X_test.columns,index = X_test.index )

      logger.info(f'After Scaling : {X_train_scaled}')
      logger.info(f'After Scaling : {X_test_scaled}')
      logger.info(f'{X_train_scaled.shape} -->  {X_train_scaled.columns}')
      logger.info(f'{X_test_scaled.shape} -->  {X_test_scaled.columns}')

      with open('scaler.pkl', 'wb') as f:
          pickle.dump(sc, f)

      return X_train_scaled,X_test_scaled



    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')