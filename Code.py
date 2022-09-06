* linguist-vendored
*.python linguist-vendored=false

# import necessary libraries 
import os
import random
import tqdm
import numpy as np
import pandas as pd
 
 
import tensorflow as tf
from tensorflow import keras
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras import backend as K
 
from keras.layers import Dense, Embedding, SimpleRNN
from keras.models import Sequential
from scipy import stats
 
from pprint import pprint
from pylab import plt, mpl
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
pd.set_option('precision', 4)
np.set_printoptions(suppress=True, precision=4)
os.environ['PYTHONHASHSEED'] = '0'
 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import *
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.metrics import plot_confusion_matrix
 
%matplotlib inline
 
# These options determine the way floating point numbers are displayed.
np.set_printoptions(precision=4, suppress=True)
 
tf.random.set_seed(100)
 
from google.colab import drive
 
drive.mount('/drive')
 
df = pd.read_csv('/drive/My Drive/Colab Notebooks/Data/equity_2000_2020top5.csv', parse_dates=['DATE'])
 
df.sort_values('DATE',inplace=True)
df.set_index('DATE', inplace=True)
 
# remove an useless column in raw dataset
df.drop(columns=['Unnamed: 0'], inplace=True)
df.dropna(inplace=True)
 
df.head(-10)
# construct a new column with tag of train or test
 
 
# total days in dataset
Time_diff = df.index[-1] - df.index[0]
print(Time_diff.days, 'days in the dataset')
print(np.count_nonzero(df['permno'].unique()), 'stocks in total in the datasets') 
 
# ratio of data for test #training data 80%
test_ratio = 0.2
 
# index splitting train and test data
test_index = df.index[0] + (Time_diff*(1-test_ratio))
 
# create a new column of test flag
df['test_flag'] = False                                  #all false for training
df.loc[ test_index : df.index[-1], 'test_flag' ] = True  #some data for testing based on test_ratio
def plot_stock(stockID):
    
    plt.figure(figsize=(9,6))
    
    df[(df['permno']==stockID)&(df['test_flag']==False)]['RET'].plot()
    df[(df['permno']==stockID)&(df['test_flag']==True)]['RET'].plot()
    
    plt.title('Stock Return'+' '+str(stockID))
    plt.legend(['Training','Testing'])
    plt.show()
    
### CREATE GENERATOR FOR LSTM WINDOWS AND LABELS ###
 
# the length of each training sample (how many time steps in each sample)
# we read features in previous 4 time steps to forecast return in next time step
sequence_length = 4
 
def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
 
# zip takes an iterable (list, tuple, set, or dictionary), generates a list of tuples that contain elements from each iterable
# construct LSTM input features for LSTM
def gen_sequence(id_df, seq_length, seq_cols):
 
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
 
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :] #include all columns (features)
 
 
# construct LSTM output
def gen_labels(id_df, seq_length, label):
 
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
 
    return data_matrix[seq_length:num_elements, :]
### CREATE TRAIN/TEST PRICE DATA ### 
 
col = df.columns
 
X_train, X_test = [], []
y_train, y_test = [], []
 
for (stock,is_test), _df in df.groupby(['permno', 'test_flag']):     
    for seq in gen_sequence(_df, sequence_length, col):        
        if is_test:
            X_test.append(seq)
        else:
            X_train.append(seq)
                
    for seq in gen_labels(_df, sequence_length, ['RET']):        
        if is_test:
            y_test.append(seq)
        else:
            y_train.append(seq)
 
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
 
print('shape of the intput X and output y\ninput training data dimension: ')
print(X_train.shape)
print('\noutput training data dimension: ')
print(y_train.shape)
 
print('\ninput testing data dimension: ')
print(X_test.shape)
print('\noutput testing data dimension: ')
print(y_test.shape)
 
# print several examples, only print 3 out of 101 features
for i in range(5):
    print('input ', i, ': \n',X_train[i,:,0:5]) 
    print('==> output ', i, ': \n', y_train[i], '\n')
 
    #training input(191220samples,4time steps,101features)
    #training output(191220samples,1time step)
scaler = StandardScaler() #do the standardization 
X_train = scaler.fit_transform(X_train.reshape(-1,X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1,X_test.shape[-1])).reshape(X_test.shape)
 
print(X_train.shape)
 
# shape of the training data is (191220, 4, 101), it means:
# 191220 rows
# 4 rows --> 1 output (LSTM many to many application)
# 101 features in each row
# construct model
 
set_seed(33)
 
### DEFINE LSTM AUTOENCODER ###
 
# input shape is (4, 101), using 4 rows to forecast one output, each row has 101 features
inputs_ae = Input(shape=(X_train.shape[1:]))
 
# encoding LSTM layer, receive inputs_ae from input layer
encoded_ae_2 = LSTM( int(X_train.shape[2]*1.2) , return_sequences=True, dropout=0.5)(inputs_ae, training=True)
 
# decoding LSTM layer, recive encoded_ae from encoding LSTM layer
decoded_ae = LSTM( int(X_train.shape[2]/2) , return_sequences=True, dropout=0.5)(encoded_ae_2, training=True)
 
out_ae = TimeDistributed(Dense(1))(decoded_ae)
 
sequence_autoencoder = Model(inputs_ae, out_ae) 
sequence_autoencoder.compile(optimizer='adam', loss='mse') #optimization
 
### TRAIN AUTOENCODER ###
 
# early stop
es = EarlyStopping(patience=6, verbose=2, min_delta=0.001, 
                   monitor='val_loss', mode='auto', restore_best_weights=True)
 
# train autoencoder with output reconstructing the input
sequence_autoencoder.fit(X_train, X_train, validation_data=(X_train, X_train),
                         batch_size=128, epochs=100, verbose=1, callbacks=[es])
### ENCODE PRICE AND CONCATENATE REGRESSORS ###
 
# to obtain the encoded information and concatenating it with the 
# original features, along with the last imension: 
# 101 original features + 121 encoded features = 222
encoder = Model(inputs_ae, encoded_ae_2) 
encoded_feature_train = encoder.predict(X_train) 
encoded_feature_test = encoder.predict(X_test)
 
# dimension of encoded features
print('encoded features: ', encoded_feature_train.shape) #encoded features
 
X_train_ = np.concatenate([X_train, encoder.predict(X_train)], axis=-1) 
X_test_ = np.concatenate([X_test, encoder.predict(X_test)], axis=-1)
 
# each row of X_train_ and X_test_ has 222 features
X_train_.shape, X_test_.shape
%%time
 
set_seed(33)
 
### DEFINE STANDARD LSTM FORECASTER ###
 
inputs = Input(shape=(X_train_.shape[1:]))
lstm = LSTM(128, return_sequences=True, dropout=0.5)(inputs, training=True)
lstm = LSTM(32, return_sequences=False, dropout=0.5)(lstm, training=True)
dense = Dense(50)(lstm)
out = Dense(1)(dense)
 
model = Model(inputs, out)
model.compile(loss='mse', optimizer='adam')
 
### FIT FORECASTER ###
es = EarlyStopping(patience=6, verbose=2, min_delta=0.001, 
                   monitor='val_loss', mode='auto', restore_best_weights=True)
model.fit(X_train_, y_train, validation_data=(X_train_, y_train), 
          epochs=100, batch_size=128, verbose=1, callbacks=[es])
### COMPUTE STOCHASTIC DROPOUT ###
%%time
scores = []
for i in tqdm.tqdm(range(0,100)):
    scores.append(mean_absolute_error(y_test, model.predict(X_test_).ravel()))
 
print(np.mean(scores), np.std(scores))
results = {'LSTM':None, 'Autoencoder+LSTM':None}
results['Autoencoder+LSTM'] = {'mean':np.mean(scores), 'std':np.std(scores)}
print(results)
 
set_seed(33)
 
### DEFINE FORECASTER ###
 
inputs = Input(shape=(X_train.shape[1:]))
lstm = LSTM(128, return_sequences=True, dropout=0.5)(inputs, training=True)
lstm = LSTM(32, return_sequences=False, dropout=0.5)(lstm, training=True)
dense = Dense(50)(lstm)
out = Dense(1)(dense)
 
model = Model(inputs, out)
model.compile(loss='mse', optimizer='adam')
 
### FIT FORECASTER ###
es = EarlyStopping(patience=6, verbose=2, min_delta=0.001, monitor='val_loss', mode='auto', restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_train, y_train), 
          epochs=100, batch_size=128, verbose=1, callbacks=[es])
### COMPUTE STOCHASTIC DROPOUT ###
 
scores = []
for i in tqdm.tqdm(range(0,100)):
    scores.append(mean_absolute_error(y_test, model.predict(X_test).ravel()))
 
print(np.mean(scores), np.std(scores))
results['LSTM'] = {'mean':np.mean(scores), 'std':np.std(scores)}
 
for key, value in results.items():
    print(key, ': ', value, '\n')
#Feature Importance
# split features and target
X = df.drop(['RET'], axis=1)
y = df['RET']
 
# Spilt into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
 
# make the y as the dataframe with a column name
y_train.name = 'Default'
y_train = y_train.to_frame()
y_test.name = 'Default'
y_test = y_test.to_frame()
 
 
# sklearn.neural_network is not supported by SHAP 
 
# construct a NN classification model by TensorFlow
 
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
model.fit(X_train, y_train, epochs=100, verbose=False)
 
#### calculate the feature importance by SHAP ####
 
# install SHAP 
!pip install shap
 
# import shap package
import shap
 
# define the explainer
# shap has many explainers, Kernelexplainer supports tensorflow NN model
explainer = shap.KernelExplainer(model,X_train.iloc[:50,:])
 
# calculate the SHAP values for each feature using 300 rows, 
# you may use less for shorter calculation time
# this takes ~3 minutes
shap_values = explainer.shap_values(X_train.iloc[:300,:])
#### show some results ####
 
expected_value = explainer.expected_value
 
# visulize the feature importance
shap.summary_plot(shap_values, X_train, plot_type="bar")
 
# Linear Tree
 
!pip install -U linear-tree
 
from lineartree import LinearTreeRegressor
#def plot_stock(stockID):
    
 
    #df[(df['permno']==stockID)&(df['test_flag']==False)]['RET'].plot()
    #df[(df['permno']==stockID)&(df['test_flag']==True)]['RET'].plot()
df[(df['permno']==df['permno'])&(df['test_flag']==False)]['RET'].plot()
df[(df['permno']==df['permno'])&(df['test_flag']==True)]['RET'].plot()
 
#plt.title('Stock Return'+' '+str(stockID))
plt.legend(['Training','Testing'])
plt.show()
#X_train.shape, X_test.shape
X_train, X_test, y_train, y_test = train_test_split(
  df[df['permno']==df['permno']].drop(['RET'], axis=1), #
  df['RET'], test_size=0.2, shuffle=False)
X_train.shape, X_test.shape
 
y_train.plot(label='train', figsize=(16,6))
y_test.plot(label='test')
#plt.title("store: {}".format(id_shop)); plt.legend()
### TUNING LINEAR TREE FOR SINGLE STORE ###
 
model = GridSearchCV(estimator=LinearTreeRegressor(Ridge(), criterion='rmse'),
                     param_grid={'max_depth': [1, 2, 3, 4, 5], 'min_samples_split': [0.5, 0.4, 0.3, 0.2]}, 
                     n_jobs=-1, cv=2, scoring='neg_mean_absolute_error', refit=True)
%time model.fit(X_train, y_train)
### PLOT MODEL DECISION PATH ###
model.best_estimator_.plot_model(feature_names=X_train.columns)
 
### COMPUTE TEST ERROR FOR LINEAR TREE ###
 
pred_lt = pd.Series(model.predict(X_test), index = y_test.index)
#mean_squared_error(y_test, pred_lt, squared=False)
linear_tree_mae=mean_absolute_error(y_test, pred_lt)
linear_tree_mae
