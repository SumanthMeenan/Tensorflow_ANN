import pandas as pd
import numpy as np 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit, cross_val_score

from keras.models import Sequential
from keras.layers import Dense 
from keras.wrappers.scikit_learn import KerasClassifier 

iris_data = pd.read_csv("iris.csv", header=None)

print(iris_data[4].value_counts(normalize=True))

print(iris_data.isnull().sum())
print(iris_data.info())

col_tranformer = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[4])], remainder = "passthrough")
transformed_data  = np.array(col_tranformer.fit_transform(iris_data))

ip_features = transformed_data[:,3:]
output = transformed_data[:, :3]

def multiclass_model():
    model = Sequential()
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation="softmax")) 
    model.compile(loss = "categorical_crossentropy", optimizer='rmsprop', metrics = 'accuracy')
    return model 

estimator = KerasClassifier(build_fn = multiclass_model, batch_size = 5, epochs = 50)

kfold_cv = KFold(n_splits = 10, shuffle = True)
skf = StratifiedKFold(n_splits=10, shuffle=True)
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=2) 

kfold_predictions = cross_val_score(estimator, ip_features, output, cv = kfold_cv)
# skf_predictions = cross_val_score(estimator, ip_features, output, cv = skf)
# rskf_predictions = cross_val_score(estimator, ip_features, output, cv = rskf)

print("kfold_predictions" , kfold_predictions)

# datasize - 150 rows, k = 10 (10 folds), cut data into 10 chunks,
# training - 9chunks , eval - 1 chunk, every chunk has 15 rows
# for training -> 150 - 15 = 135 rows, 135/(batchsize = 5) = 27 steps for every epoch

# kf1 kf2 kf3 kf4 kf5 kf6 kf7 kf8 kf9 kf10 

# model1 -> train: kf2 kf3 kf4 kf5 kf6 kf7 kf8 kf9 kf10  test: kf1 
# model2 -> train: kf2 kf3 kf4 kf5 kf6 kf7 kf8 kf9 kf1 test : kf10 

# kfold: divides dataset into k folds 
# stratified kfold: ensures each fold of dataset has same proportion of observations with a given label
