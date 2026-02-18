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
logger = setup_logging('var_out')
from scipy import stats


def variable_transformation_outliers(X_train_num,X_test_num):

        try:
            logger.info(f'X_train columns : {X_train_num.columns} --> {X_train_num.shape}')
            logger.info(f'X_test columns  : {X_test_num.columns} --> {X_test_num.shape}')

            PLOT_PATH = "plots_path"
            os.makedirs(PLOT_PATH, exist_ok=True)

            for i in X_train_num.columns:
                plt.figure()
                X_train_num[i].plot(kind='kde', color='r')
                plt.title(f'KDE-{i}')
                plt.savefig(f'{PLOT_PATH}/kde_{i}.png')
                plt.close()

            for i in X_train_num.columns:
                plt.figure()
                sns.boxplot(x=X_train_num[i])
                plt.title(f'Boxplot-{i}')
                plt.savefig(f'{PLOT_PATH}/boxplot_{i}.png')
                plt.close()


            for k in X_train_num.columns:
                # Log transform
                X_train_num[k] = np.log1p(X_train_num[k])
                X_test_num[k] = np.log1p(X_test_num[k])

                #  Quantile capping (from train only)
                lower = X_train_num[k].quantile(0.01)
                upper = X_train_num[k].quantile(0.99)

                X_train_num[k] = X_train_num[k].clip(lower,upper)
                X_test_num[k] = X_test_num[k].clip(lower,upper)

            logger.info(f'After transform {X_train_num.shape}')
            logger.info(f'After transform {X_test_num.shape}')



            for i in X_train_num.columns:
                plt.figure()
                X_train_num[i].plot(kind='kde', color='r')
                plt.title(f'KDE-{i}')
                plt.savefig(f'{PLOT_PATH}/kde_{i}.png')
                plt.close()
            for i in X_train_num.columns:
                plt.figure()
                sns.boxplot(x=X_train_num[i])
                plt.title(f'Boxplot-{i}')
                plt.savefig(f'{PLOT_PATH}/boxplot_{i}.png')
                plt.close()

            logger.info(f"{X_train_num.columns} -> {X_train_num.shape}")
            logger.info(f"{X_test_num.columns} -> {X_test_num.shape}")

            return X_train_num,X_test_num


        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error in Line {error_line.tb_lineno}: {error_msg}')