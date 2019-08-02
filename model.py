#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import keras
from keras.layers import LSTM, Input, Dense, TimeDistributed, Concatenate, GRU
import keras.backend as K
from keras import regularizers, Model
import tensorflow as tf
import scipy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import csv
import pickle
import datetime
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

data_1992 = pd.read_excel(r'./data/1992.xlsx', sheet_name='Export_Output1992')
data_1992 = data_1992.drop('POINTID', axis=1)
data_1992 = data_1992[data_1992['DEM']>=0]
data_1992 = data_1992[data_1992['NDVI']>=0]
data_1992new = pd.read_excel(r'./data/1992_new.xlsx', sheet_name='Export_Output1992')
data_1992new = data_1992new.drop('POINTID', axis=1)
data_1992new = data_1992new[data_1992new['DEM']>=0]
data_1992new = data_1992new[data_1992new['NDVI']>=0]

data_1998 = pd.read_excel(r'./data/1998.xlsx', sheet_name='Export_Output1998')
data_1998 = data_1998.drop('POINTID', axis=1)
data_1998 = data_1998[data_1998['DEM']>=0]
data_1998 = data_1998[data_1998['NDVI']>=0]
data_1998new = pd.read_excel(r'./data/1998_new.xlsx', sheet_name='Export_Output1998')
data_1998new = data_1998new.drop('POINTID', axis=1)
data_1998new = data_1998new[data_1998new['DEM']>=0]
data_1998new = data_1998new[data_1998new['NDVI']>=0]

data_2007 = pd.read_excel(r'./data/2007.xlsx', sheet_name='Export_Output2007')
data_2007 = data_2007.drop('POINTID', axis=1)
data_2007 = data_2007[data_2007['NDVI']>=0]
data_2007 = data_2007[data_2007['DEM']>=0]
data_2007new = pd.read_excel(r'./data/2007_new.xlsx', sheet_name='Export_Output2007')
data_2007new = data_2007new.drop('POINTID', axis=1)
data_2007new = data_2007new[data_2007new['NDVI']>=0]
data_2007new = data_2007new[data_2007new['DEM']>=0]

data_2018 = pd.read_excel(r'./data/2018.xlsx', sheet_name='Export_Output_2018')
data_2018 = data_2018.drop('POINTID', axis=1)
data_2018 = data_2018[data_2018['DEM']>=0]
data_2018 = data_2018[data_2018['NDVI']>=0]
data_2018new = pd.read_excel(r'./data/2018_new.xlsx', sheet_name='Export_Output2018')
data_2018new = data_2018new.drop('POINTID', axis=1)
data_2018new = data_2018new[data_2018new['DEM']>=0]
data_2018new = data_2018new[data_2018new['NDVI']>=0]

#Original data point
data_dict1992 = dict()
data_dict1998 = dict()
data_dict2007 = dict()
data_dict2018 = dict()

for index,row in data_1992.iterrows():
    cur_id = (int(row['X']),int(row['Y']))
    data_dict1992[cur_id] = {}
    data_dict1992[cur_id]['year'] = 1992
    data_dict1992[cur_id]['x'] = cur_id[0]
    data_dict1992[cur_id]['y'] = cur_id[1]
    data_dict1992[cur_id]['height'] = row['DEM']
    data_dict1992[cur_id]['dis'] = row['MEAN_NEAR_D']
    data_dict1992[cur_id]['ndvi'] = row['NDVI']
for index,row in data_1992new.iterrows():
    cur_id = (int(row['X']),int(row['Y']))
    if cur_id in data_dict1992:
        continue
    data_dict1992[cur_id] = {}
    data_dict1992[cur_id]['year'] = 1992
    data_dict1992[cur_id]['x'] = cur_id[0]
    data_dict1992[cur_id]['y'] = cur_id[1]
    data_dict1992[cur_id]['height'] = row['DEM']
    data_dict1992[cur_id]['dis'] = 0
    data_dict1992[cur_id]['ndvi'] = row['NDVI']

for index,row in data_1998.iterrows():
    cur_id = (int(row['X']),int(row['Y']))
    data_dict1998[cur_id] = {}
    data_dict1998[cur_id]['year'] = 1998
    data_dict1998[cur_id]['x'] = cur_id[0]
    data_dict1998[cur_id]['y'] = cur_id[1]
    data_dict1998[cur_id]['height'] = row['DEM']
    data_dict1998[cur_id]['dis'] = row['MIN_NEAR_D']
    data_dict1998[cur_id]['ndvi'] = row['NDVI']
for index,row in data_1998new.iterrows():
    cur_id = (int(row['X']),int(row['Y']))
    if cur_id in data_dict1998:
        continue
    data_dict1998[cur_id] = {}
    data_dict1998[cur_id]['year'] = 1998
    data_dict1998[cur_id]['x'] = cur_id[0]
    data_dict1998[cur_id]['y'] = cur_id[1]
    data_dict1998[cur_id]['height'] = row['DEM']
    data_dict1998[cur_id]['dis'] = 0
    data_dict1998[cur_id]['ndvi'] = row['NDVI']

for index,row in data_2007.iterrows():
    cur_id = (int(row['X']),int(row['Y']))
    data_dict2007[cur_id] = {}
    data_dict2007[cur_id]['year'] = 2007
    data_dict2007[cur_id]['x'] = cur_id[0]
    data_dict2007[cur_id]['y'] = cur_id[1]
    data_dict2007[cur_id]['height'] = row['DEM']
    data_dict2007[cur_id]['dis'] = row['MIN_NEAR_D']
    data_dict2007[cur_id]['ndvi'] = row['NDVI']
for index,row in data_2007new.iterrows():
    cur_id = (int(row['X']),int(row['Y']))
    if cur_id in data_dict2007:
        continue
    data_dict2007[cur_id] = {}
    data_dict2007[cur_id]['year'] = 2007
    data_dict2007[cur_id]['x'] = cur_id[0]
    data_dict2007[cur_id]['y'] = cur_id[1]
    data_dict2007[cur_id]['height'] = row['DEM']
    data_dict2007[cur_id]['dis'] = 0
    data_dict2007[cur_id]['ndvi'] = row['NDVI']

for index,row in data_2018.iterrows():
    cur_id = (int(row['X']),int(row['Y']))
    data_dict2018[cur_id] = {}
    data_dict2018[cur_id]['year'] = 2018
    data_dict2018[cur_id]['x'] = cur_id[0]
    data_dict2018[cur_id]['y'] = cur_id[1]
    data_dict2018[cur_id]['height'] = row['DEM']
    data_dict2018[cur_id]['dis'] = row['MIN_NEAR_D']
    data_dict2018[cur_id]['ndvi'] = row['NDVI']
for index,row in data_2018new.iterrows():
    cur_id = (int(row['X']),int(row['Y']))
    if cur_id in data_dict2018:
        continue
    data_dict2018[cur_id] = {}
    data_dict2018[cur_id]['year'] = 2018
    data_dict2018[cur_id]['x'] = cur_id[0]
    data_dict2018[cur_id]['y'] = cur_id[1]
    data_dict2018[cur_id]['height'] = row['DEM']
    data_dict2018[cur_id]['dis'] = 0
    data_dict2018[cur_id]['ndvi'] = row['NDVI']

#Merge data
data_dict = dict()
for each_id in data_dict1992:
    data_dict[each_id] = {}
    data_dict[each_id] = {}
    data_dict[each_id]['year'] = [1992]
    data_dict[each_id]['x'] = each_id[0]
    data_dict[each_id]['y'] = each_id[1]
    data_dict[each_id]['height'] = data_dict1992[each_id]['height']
    data_dict[each_id]['dis'] = [data_dict1992[each_id]['dis']]
    data_dict[each_id]['ndvi'] = [data_dict1992[each_id]['ndvi']]
    
for each_id in data_dict1998:
    if each_id in data_dict:
        data_dict[each_id]['year'].append(1998)
        data_dict[each_id]['dis'] .append(data_dict1998[each_id]['dis'])
        data_dict[each_id]['ndvi'].append(data_dict1998[each_id]['ndvi'])
    else:
        data_dict[each_id] = {}
        data_dict[each_id] = {}
        data_dict[each_id]['year'] = [1998]
        data_dict[each_id]['x'] = each_id[0]
        data_dict[each_id]['y'] = each_id[1]
        data_dict[each_id]['height'] = data_dict1998[each_id]['height']
        data_dict[each_id]['dis'] = [data_dict1998[each_id]['dis']]
        data_dict[each_id]['ndvi'] = [data_dict1998[each_id]['ndvi']]
    
for each_id in data_dict2007:
    if each_id in data_dict:
        data_dict[each_id]['year'].append(2007)
        data_dict[each_id]['dis'] .append(data_dict2007[each_id]['dis'])
        data_dict[each_id]['ndvi'].append(data_dict2007[each_id]['ndvi'])
    else:
        data_dict[each_id] = {}
        data_dict[each_id] = {}
        data_dict[each_id]['year'] = [2007]
        data_dict[each_id]['x'] = each_id[0]
        data_dict[each_id]['y'] = each_id[1]
        data_dict[each_id]['height'] = data_dict2007[each_id]['height']
        data_dict[each_id]['dis'] = [data_dict2007[each_id]['dis']]
        data_dict[each_id]['ndvi'] = [data_dict2007[each_id]['ndvi']]
        
for each_id in data_dict2018:
    if each_id in data_dict:
        data_dict[each_id]['year'].append(2018)
        data_dict[each_id]['dis'] .append(data_dict2018[each_id]['dis'])
        data_dict[each_id]['ndvi'].append(data_dict2018[each_id]['ndvi'])
    else:
        data_dict[each_id] = {}
        data_dict[each_id] = {}
        data_dict[each_id]['year'] = [2018]
        data_dict[each_id]['x'] = each_id[0]
        data_dict[each_id]['y'] = each_id[1]
        data_dict[each_id]['height'] = data_dict2018[each_id]['height']
        data_dict[each_id]['dis'] = [data_dict2018[each_id]['dis']]
        data_dict[each_id]['ndvi'] = [data_dict2018[each_id]['ndvi']]

counter_1,counter_2,counter_3,counter_4 = 0,0,0,0
err_list = []
for each_id in data_dict:
    if 0 in data_dict[each_id]['dis']:
        err_list.append(each_id)
        continue
    if len(data_dict[each_id]['year']) == 1:
        counter_1+=1
        err_list.append(each_id)
    elif len(data_dict[each_id]['year']) == 2:
        counter_2+=1
        err_list.append(each_id)
    elif len(data_dict[each_id]['year']) == 3:
        counter_3+=1
        err_list.append(each_id)
    elif len(data_dict[each_id]['year']) == 4:
        counter_4+=1
    else:
        raise(ValueError)
print(counter_1,counter_2,counter_3,counter_4)
for each_id in err_list:
    del data_dict[each_id]


# ## Import soil data

soil_data = pd.read_excel(r'./data/soil.xlsx', sheet_name='soil')
soil_data = soil_data.to_dict(orient='record')

#one-hot label
label_list = []
for each_id in soil_data:
    if each_id['soil_1'] in label_list:
        continue
    else:
        label_list.append(each_id['soil_1'])

for each_id in soil_data:
    cur_ind = (each_id['X'],each_id['Y'])
    if cur_ind not in data_dict:
        continue
    cur_soil = label_list.index(each_id['soil_1'])
    cur_onehot = np.zeros(len(label_list))
    cur_onehot[cur_soil] = 1
    data_dict[cur_ind]['soil'] = cur_onehot


# ## Import social data

additional_data = pd.read_excel(r'./data/total.xlsx', sheet_name='Sheet1')
additional_data = additional_data.to_dict(orient='record')
additional_feature = ['area_orchard', 'area_farm', 'area_water', 'area_town', 'area_unuse', 'spring_temp', 'spring_hum', 'spring_pre', 'spring_eva', 'spring_speed', 'spring_sun', 'spring_surface', 'summer_temp', 'summer_hum', 'summer_pre', 'summer_eva', 'summer_speed', 'summer_sun', 'summer_surface', 'autumn_temp', 'autumn_hum', 'autumn_pre', 'autumn_eva', 'autumn_speed', 'autumn_sun', 'autumn_surface', 'winter_temp', 'winter_hum', 'winter_pre', 'winter_eva', 'winter_speed', 'winter_sun', 'winter_surface', 'high', 'low', 'snow', 'population', 'gdp', 'oil']
for each_id in data_dict:
    for each_feature in additional_feature:
        data_dict[each_id][each_feature] = []
    for each_feature in additional_feature:
        for i in range(4):
            data_dict[each_id][each_feature].append(additional_data[i][each_feature])


# ## Dataset construction

dynamic = []
static = []
space = []
y = []
loc_id = []
year = []

dynamic_feature = ['ndvi','dis']
static_feature = additional_feature
space_feature = ['x','y','height','soil']
for each_id in data_dict:
    cur_static = []
    cur_space = []
    cur_dynamic = []
    
    for i in range(4):
        tmp = []
        for each_feature in dynamic_feature:
            tmp.append(data_dict[each_id][each_feature][i])
        cur_dynamic.append(tmp)
    cur_dynamic = np.array(cur_dynamic)
    
    for i in range(4):
        tmp = []
        for each_feature in static_feature:
            tmp.append(data_dict[each_id][each_feature][i])
        cur_static.append(tmp)
    cur_static = np.array(cur_static)
    
    for i in range(4):
        tmp = []
        for each_feature in space_feature:
            if each_feature == 'soil':
                tmp+=list(data_dict[each_id][each_feature])
            else:
                tmp.append(data_dict[each_id][each_feature])
        cur_space.append(tmp)
    cur_space = np.array(cur_space)
    
    dynamic.append(cur_dynamic)
    static.append(cur_static)
    space.append(cur_space)
    y.append(np.expand_dims(np.array(data_dict[each_id]['ndvi'])[1:], axis=-1))
    loc_id.append(each_id)
    year.append(data_dict[each_id]['year'])


# ## Scaling

dynamic_transform = []
space_transform = []
static_transform = []
y_transform = []

for each_id in range(len(static)):
    for each_year in range(4):
        dynamic_transform.append(dynamic[each_id][each_year])
        static_transform.append(static[each_id][each_year])
        space_transform.append(space[each_id][each_year])

dynamic_transform = np.array(dynamic_transform)
static_transform = np.array(static_transform)
space_transform = np.array(space_transform)

dynamic_scaler = sklearn.preprocessing.StandardScaler()
static_scaler = sklearn.preprocessing.StandardScaler()
space_scaler = sklearn.preprocessing.StandardScaler()
dynamic_scaler = dynamic_scaler.fit(dynamic_transform)
static_scaler = static_scaler.fit(static_transform)
space_scaler = space_scaler.fit(space_transform)

for each_id in range(len(static)):
    dynamic[each_id] = dynamic_scaler.transform(dynamic[each_id])
    static[each_id] = static_scaler.transform(static[each_id])
    space[each_id] = space_scaler.transform(space[each_id])


dynamic = np.array(dynamic)
static = np.array(static)
space = np.array(space)
y = np.array(y)


# ## Split dataset

all_ind = list(range(len(static)))
train_ind, test_ind = train_test_split(all_ind, test_size=0.1, random_state=12345)
train_ind, valid_ind = train_test_split(train_ind, test_size=0.1, random_state=12345)

dynamic_train = []
dynamic_valid = []
dynamic_test = []
static_train = []
static_valid = []
static_test = []
space_train = []
space_valid = []
space_test = []
label_train = []
label_valid = []
label_test = []

for ind in train_ind:
    dynamic_train.append(dynamic[ind][:-1])
    static_train.append(static[ind][:-1])
    label_train.append(y[ind])
    space_train.append(space[ind][:-1])

for ind in valid_ind:
    dynamic_valid.append(dynamic[ind][:-1])
    static_valid.append(static[ind][:-1])
    label_valid.append(y[ind])
    space_valid.append(space[ind][:-1])

for ind in test_ind:
    dynamic_test.append(dynamic[ind])
    static_test.append(static[ind])
    label_test.append(y[ind])
    space_test.append(space[ind])
    
dynamic_train = np.array(dynamic_train)
dynamic_valid = np.array(dynamic_valid)
dynamic_test = np.array(dynamic_test)
static_train = np.array(static_train)
static_valid = np.array(static_valid)
static_test = np.array(static_test)
label_train = np.array(label_train)
label_valid = np.array(label_valid)
label_test = np.array(label_test)
space_train = np.array(space_train)
space_valid = np.array(space_valid)
space_test = np.array(space_test)


# # Model structure

def det_coeff(y_true, y_pred):
    u = K.sum(K.square(y_true - y_pred))
    v = K.sum(K.square(y_true - K.mean(y_true)))
    return K.ones_like(v) - (u / v)


print(space_train.shape)
print(label_train.shape)
print(static_train.shape)
print(dynamic_train.shape)


# ## Model define

def create_model():
    feature_size = 2
    space_size = 16
    static_size = 39
    rnn_size = 16
    l2_regular = regularizers.l2(1e-3)
    
    input_states = Input(shape=(None, feature_size), name='input_states')
    input_static = Input(shape=(None, static_size), name='input_statics')
    input_space = Input(shape=(None, space_size), name='input_space')
    
    concat_1 = Concatenate(name='concat1')([input_states, input_static])
    gru = GRU(rnn_size, activation='tanh', name='gru', kernel_regularizer=l2_regular, return_sequences=True)(input_states) 
    concat_2 = Concatenate(name='concat2')([gru, input_space])
    dense = TimeDistributed(Dense(8, activation='relu', kernel_regularizer=l2_regular), name='dense')(concat_2)
    output = TimeDistributed(Dense(1, kernel_regularizer=l2_regular, activation='relu'), name='output')(dense)
    model = Model([input_states, input_static, input_space], output)
    opt = keras.optimizers.Adam(lr=1e-3)
    model.compile(loss='mean_squared_error',optimizer=opt,  metrics=['mse',det_coeff])
    return model
print(create_model().summary())
model_test = create_model()


# # Train

epoch = 300
batch_size = 1000
save_dir = './trained_weights/gru'

valid_loss = []
train_loss = []

model = create_model()
save_callback = keras.callbacks.ModelCheckpoint(save_dir, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
history = model.fit(x=[dynamic_train, static_train, space_train], y=label_train, batch_size=batch_size, epochs=epoch, callbacks=[save_callback], verbose=1, validation_data=([dynamic_valid, static_valid, space_valid],label_valid), shuffle=True)
pickle.dump(history,open(r'./trained_weights/m_history','wb'))


# # Model evaluation


save_dir = './trained_weights/'
model_test.load_weights(save_dir+'gru')


y_pred = []
y_true = []

y_pred = model_test.predict([dynamic_test[:, :-1, :], static_test[:, :-1, :], space_test[:, :-1, :]])
y_true = label_test

y_pred = y_pred.flatten()
y_true = y_true.flatten()


print('Mean Squared Error: %.6f'%sklearn.metrics.mean_squared_error(y_pred=y_pred, y_true=y_true))
print('Mean Absolute Error: %.6f'%sklearn.metrics.mean_absolute_error(y_pred=y_pred, y_true=y_true))
print('R2 Score: %.6f'%sklearn.metrics.r2_score(y_pred=y_pred, y_true=y_true))
adj_r2 = 1-(1-sklearn.metrics.r2_score(y_pred=y_pred, y_true=y_true))*(len(y_pred)-1)/(len(y_pred)-5-1)
print('Adjusted R2 Score: %.6f'%adj_r2)