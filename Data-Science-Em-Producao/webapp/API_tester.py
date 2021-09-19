import pandas as pd
import math
import numpy as np
import datetime
import xgboost as xgb
import json
import requests

# loading test dataset
df10 = pd.read_csv('/Users/Igor/repos/Data-Science-Em-Producao/data/test.csv')
df_store_raw = pd.read_csv('/Users/Igor/repos/Data-Science-Em-Producao/data/store.csv', low_memory=False)

#merge test dataset + store
df_test = pd.merge(df10, df_store_raw, how = 'left', on = 'Store')

# Choose store for prediction
df_test = df_test[df_test['Store'] == 22]
# df_test = df_test[df_test['Store'].isin([24, 12, 22])]

# remove closed days
df_test = df_test[df_test['Open'] != 0]
df_test = df_test[~df_test['Store'].isnull()]
df_test = df_test.drop('Id', axis = 1)

df_test.head()

# convert Dataframe to json
data = json.dumps(df_test.to_dict(orient = 'records'))

# API Call
# url = 'http://127.0.0.1:5000/rossmann/predict'
url = 'https://rosmann-prediction-model.herokuapp.com/rossmann/predict'
header = {'Content-type': 'application/json'}
data = data

r = requests.post(url, data = data, headers = header)
print('Status Code {}'.format(r.status_code))

d1 = pd.DataFrame(r.json(), columns = r.json()[0].keys())

d1.head()

# aki se faz o modelo, trabalha nas transformacoes do dataset, faz o eda, e tudo mais, depois coloca o modelo salvo em producao,
# e por fim se faz uma requisicao de fora via api, o modelo recebe essa requisicao, roda, fornece a predicao e devolve via api.

d2 = d1[['store', 'prediction']].groupby('store').sum().reset_index()

d2

for i in range(len(d2)):
    print('Store Number {} will sell ${:,.2f} in the next 6 weeks'.format(
            d2.loc[i,'store'],
            d2.loc[i, 'prediction']))

print(d1.head())