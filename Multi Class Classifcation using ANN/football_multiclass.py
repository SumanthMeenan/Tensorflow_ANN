import pandas as pd 
from cv2 import normalize
import pandas as pd
import numpy as np 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit, cross_val_score

from keras.models import Sequential
from keras.layers import Dense 
from keras.wrappers.scikit_learn import KerasClassifier 

df = pd.read_csv('data.csv') 
print(df.Position.value_counts())

print(df.isnull().sum().sort_values(ascending=False)/len(df) *100)

print(len(df[pd.notnull(df['Position'])]))