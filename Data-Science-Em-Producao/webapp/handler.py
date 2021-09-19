import pickle
import pandas as pd
from flask             import Flask, request, Response
from rossmann.Rossmann import Rossmann
import numpy as np

# loading model
model = pickle.load(open('/Users/Igor/repos/Data-Science-Em-Producao/model/model_rossmann.pkl', 'rb'))

# initialize API
app = Flask(__name__)

@app.route('/rossmann/predict', methods = ['POST']) # endpoint, onde os dados oringianis com a previsao serao enviados
def rossmann_predict():
    test_json = request.get_json() # aqui puxa or aquivos originais de csv tanto o train quanto o store
  
    if test_json: # there is data
          # conversao do json em dataframe
        if isinstance(test_json, dict): # Unique example
            test_raw = pd.DataFrame(test_json, index =[0])
            
        else: # Multiple examples
            test_raw = pd.DataFrame(test_json, columns = test_json[0].keys())
            
        # instantiate Rossmann class # pega as informacoes la do rossmann class
        pipeline = Rossmann()
        
        # data cleaning # preparacao 1 do modelo 
        df1 = pipeline.data_cleaning(test_raw)
        
        # feature engineering # preparacao 2 do modelo 
        df2 = pipeline.feature_engineering(df1)
        
        # data preparation # preparacao 3 do modelo 
        df3 = pipeline.data_preparation(df2)
        
        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)# test raw sao os dados originais e p df3 sao os dados preparados

       
        return df_response
    
    else:
        return Response('{}', status = 200, mimetype = 'application/json')
    
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host = '0.0.0.0', port = port)