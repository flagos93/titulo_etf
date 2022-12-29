import pandas as pd
import os
import seaborn as sn
import plotly.graph_objects as go
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from matplotlib import pyplot as plt
import numpy as np


import warnings
warnings.filterwarnings('ignore')

df_list=[]
for i in os.listdir(r'./etf/Instruments'):
    for j in os.listdir(r'./etf/Instruments/'+i):
        size = len(j)
        name = j[:size - 4]
        globals()[name] = pd.read_csv(r'./etf/Instruments/'+i+'/'+j)
        globals()[name].dfname = name
        df_list.append(globals()[name])

for i in df_list:
    i['Datetime'] = i['Date'] + ' ' + i['Time']
    i.drop('Date', inplace = True, axis = 1)
    i.drop('Time', inplace = True, axis = 1)
    i.set_index('Datetime', inplace = True)
    print(i.dfname)
    print(i.describe())
    print('\n')

# Correlation Matrix formation
c=0
for i in df_list:
    corr_matrix = i.corr()
    print(i.dfname)
    #Using heatmap to visualize the correlation matrix
    sn.heatmap(corr_matrix, annot=True, fmt=".3f")
    print('\n')
    sn.pairplot(i)

for i in df_list:
    if(i.dfname.endswith('daily')):
        fig = go.Figure(data=[go.Candlestick(x=i.index,
                        open=i.Open,
                        high=i.High ,
                        low=i.Low,
                        close=i.Close)])
        fig.show()

train_X_list=[]
test_X_list=[]
train_y_list=[]
test_y_list=[]
for i in df_list:
    for j in range(4):
        '''
        print(i.drop(i.columns[j+2], axis=1))
        print(i.columns[j+2])
        '''
        
        X = i.drop(i.columns[j], axis=1).values
        y = i.iloc[:,j].values

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 100)
        train_X_list.append(X_train)
        test_X_list.append(X_test)
        train_y_list.append(y_train)
        test_y_list.append(y_test)

scaled_train_X=[]
for i in train_X_list:
    sc = MinMaxScaler(feature_range = (0, 1))
    training_scaled = sc.fit_transform(i)
    scaled_train_X.append(training_scaled)

figure, axis = plt.subplots(8, 4)

for a in range(8):
    for b in range(4):
        for i in range(len(scaled_train_X)):
            predicted_price = 'predicted_price_'+str(i)
            model_name = 'model_'+str(df_list[i].dfname)+'_'+str(b)
            print(model_name)
            globals()[model_name] = Sequential()
            globals()[model_name].add(LSTM(units = 50, return_sequences = True, input_shape = (scaled_train_X[i].shape[1], 1)))
            globals()[model_name].add(Dropout(0.2))
            globals()[model_name].add(LSTM(units = 50, return_sequences = True))
            globals()[model_name].add(Dropout(0.2))
            globals()[model_name].add(LSTM(units = 50, return_sequences = True))
            globals()[model_name].add(Dropout(0.2))
            globals()[model_name].add(LSTM(units = 50))
            globals()[model_name].add(Dropout(0.2))
            globals()[model_name].add(Dense(units = 1))
            globals()[model_name].compile(optimizer = 'adam', loss = 'mean_squared_error')
            globals()[model_name].fit(scaled_train_X[i], train_y_list[i], epochs = 100, batch_size = 32)
            globals()[model_name].save('etf_model', overwrite=False)

            globals()[predicted_price] = globals()[model_name].predict(test_X_list[i])

            axis[a,b].plot(df_list[i].index, X, color = 'red', label = 'Real' + df_list[i].dfname + 'Stock Price')
            axis[a,b].plot(df_list[i].index,globals()[predicted_price], color = 'blue', label = 'Predicted ' + df_list[i].dfname + ' Stock Price')
            axis[a,b].xticks(np.arange(0,459,50))
            axis[a,b].title(df_list[i].dfname +'Stock Price Prediction')
            axis[a,b].xlabel('Time')
            axis[a,b].ylabel(df_list[i].dfname +' Stock Price')
            axis[a,b].legend()

plt.show()