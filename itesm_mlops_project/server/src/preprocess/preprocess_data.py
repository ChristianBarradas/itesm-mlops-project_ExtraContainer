import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array
from custom_loggin import CustomLogger

FILE_LOG_DIR = 'preprocess/preprocess.log'
logger = CustomLogger(__name__, FILE_LOG_DIR).logger    


COLUMNS_TO_DROP = ['id', 'zipcode', 'date']


class CustomTransformer(BaseEstimator, TransformerMixin):
    #the constructor
    logger.info("add columns")
    '''setting the add_bedrooms_per_room to True helps us check if the hyperparameter is useful'''
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    #estimator method
    def fit(self, X, y = None):
        return self
    #transfprmation
    def transform(self, X, y = None):
        #agregar 2 columnas
        X_copy = X.copy()
        X_copy['date'] = pd.to_datetime(X_copy['date'])
        X_copy['month'] = X_copy['date'].apply(lambda date: date.month)
        X_copy['year'] = X_copy['date'].apply(lambda date: date.year)
        #X_copy = X_copy.drop('date', axis=1)
        return X_copy
    
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    logger.info("drop columns")
    def __init__(self):
        self.COLUMNS_TO_DROP = COLUMNS_TO_DROP
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.drop(self.COLUMNS_TO_DROP, axis=1)
        return 
    
class CustomMinMaxScaler(BaseEstimator, TransformerMixin):
    logger.info("CustomMinMaxScaler")
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def fit(self, X, y=None):
        # Ajusta el escalador en los datos de entrenamiento
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        # Transforma los datos usando el escalador ajustado
        X_scaled = self.scaler.transform(X)
        return X_scaled