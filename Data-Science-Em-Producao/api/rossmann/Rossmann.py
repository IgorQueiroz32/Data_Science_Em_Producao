import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann(object):
    
    def __init__(self):
        self.competition_distance_scaler   = pickle.load(open('/Users/Igor/repos/Data-Science-Em-Producao/parameter/competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(open('/Users/Igor/repos/Data-Science-Em-Producao/parameter/competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_week_scaler        = pickle.load(open('/Users/Igor/repos/Data-Science-Em-Producao/parameter/promo_time_week_scaler.pkl', 'rb'))
        self.year_scaler                   = pickle.load(open('/Users/Igor/repos/Data-Science-Em-Producao/parameter/year_scaler.pkl', 'rb'))
        self.store_type_scaler             = pickle.load(open('/Users/Igor/repos/Data-Science-Em-Producao/parameter/store_type_scaler.pkl', 'rb'))
        
    def data_cleaning(self, df1):

        ## 1.1. Rename Columns

        # alterando os nomes das colunas
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
               'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
               'CompetitionDistance', 'CompetitionOpenSinceMonth',
               'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
               'Promo2SinceYear', 'PromoInterval']

        # funcao que altera o nome das colunas para o tipo snakecase
        snakecase = lambda x: inflection.underscore(x)

        # map faz o mapeamento da funcao snakecase em todas as palavras da variavel cols_old
        cols_new = list(map(snakecase, cols_old))

        #rename
        df1.columns = cols_new
        
        ## 1.3. Data Types
        df1['date'] = pd.to_datetime(df1['date']) # alterando o tipo da coluna date para date time

        ## 1.5. Fillout NA using business logic

        # competition_distance 
        # Nesta coluna, NA significa q a loja nao tem competicao proxima ou entao q a loja competidora esta muito longe. 
        #Com isso, para remover os NAs desta coluna, podemos colocar uma distancia maior do que a distancia maxima entre as lojas. 

        #funcao q substitui todos os NAs das colunas por uma distancia maior do que a distancia maxima entre as lojas, neste caso, 
        #200000
        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: 200000.0 if math.isnan(x) else x)
        #obs a funcao math.isnan so funcionanna funcao lambda mais apply, sozinho a funcao math.isnan nao funciona

        # competition_open_since_month
        # Nesta coluna, se for NA, copia-se o mes da data de venda daquela linha, da coluna date, para a coluna competition_open_since_month.
        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else
                                                                  x['competition_open_since_month'], axis =1)
        #obs: tem o nome das colunas do lado do x nesse pois usa-se mais de uma coluna 

        #competition_open_since_year 
        # Nesta coluna, se for NA, copia-se o ano da data de venda daquela linha, da coluna date, para a coluna competition_open_since_year.
        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else
                                                                  x['competition_open_since_year'], axis =1)

        #promo2_since_week     
        # Nesta coluna, se for NA, copia-se a semana da data de venda daquela linha, da coluna date, para a coluna promo2_since_week .
        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else
                                                                  x['promo2_since_week'], axis =1)
        #promo2_since_year  
        # Nesta coluna, se for NA, copia-se a ano da data de venda daquela linha, da coluna date, para a coluna promo2_since_week .
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else
                                                                  x['promo2_since_year'], axis =1)

        #promo_interval   
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

        df1['promo_interval'].fillna(0, inplace = True) # substitui os NAs da coluna por 0, e inplace = True serve deixar essa substituicao como definitivo

        df1['month_map'] = df1['date'].dt.month.map(month_map) # cria-se uma nova coluna, onde se copia os meses da coluna date, 
        #de cada linha, para esta nova coluna, e transforma estes meses em string  com o auxulio do dicionario criado acima e da funcao .map.

        df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 
                                                                               1 if x['month_map'] in x['promo_interval'].split(',') else 
                                                                               0,axis = 1)
        # split quebra o array separado por virgula e transforma em lsita
        #cria-se uma coluna nova chamada is_promo, onde e criada baseada em duas condicoes:

        #primeira:
        # Se o mes da coluna 'month_map estiver incluso na coluna promo_interval, o valor da coluna is_promo e 1, 
        # Se o mes da coluna 'month_map nao estiver incluso na coluna promo_interval, o valor da coluna is_promo e 0.

        #segunda
        # Se o valor da linha da coluna 'promo_interval for 0, o valor da coluna is_promo e 0,
        # Se o valor da linha da coluna 'promo_interval NAO for 0, o valor da coluna is_promo e 1.

        ## 1.6. Change Types

        # Depois de criar ou aletrar colunas e bom checar os tipos das colunas novamente

        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(np.int64)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(np.int64)
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(np.int64)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(np.int64)
        
        return df1

    def feature_engineering(sel, df2):
        
        ## 2.4. Feature Engineering

        # Variaveis a serem derivadas da variavel original date

        # year
        df2['year'] = df2['date'].dt.year

        # month
        df2['month'] = df2['date'].dt.month

        # day
        df2['day'] = df2['date'].dt.day

        # week of year
        df2['week_of_year'] = df2['date'].dt.isocalendar().week.astype(np.int64)

        # year week
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

        # competition since (calcula,em meses, desde quando a competicao exsite ate a da data de compra)
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year = x['competition_open_since_year'], 
                                                                         month = x['competition_open_since_month'], 
                                                                         day = 1), axis=1) # junta 3 colunas separadas (ano, mes e dia) 
                                                                                           # em uma coluna so com as 3 informacoes

        df2['competition_time_month'] = ((df2['date'] - df2['competition_since'])/30).apply(lambda x: x.days).astype(np.int64)
        # diminui data de compra pela data de competicao, e divide por 30, para manter a granularidade de mes

        # promo since ( desde quando tem a promocao ativa)
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str) # juntando as colunas 
                                                                                                            # em forma de string

        df2['promo_since'] = df2['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))
        # transformando a coluna do tipo str para o tipo data.

        df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).apply(lambda x: x.days).astype(np.int64)
        # diminui data de compra pela data de promocao, e divide por 7, para manter a granularidade de semanas


        # assortment (trocar as letras por nome por extenso)
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic'    if x == 'a' else
                                                              'extra'    if x == 'b' else
                                                              'extended'                  )

        # state holiday (trocar as letras por nome por extenso)
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public holiday' if x == 'a' else
                                                                    'Easter holiday' if x == 'b' else
                                                                    'Christmas'      if x == 'c' else
                                                                    'regular_day'                     )

        # 3.0. STEP 03 - VARIABLES FILTER

        ## 3.1. Rows Filter

        ### Restricao de Negocio:

        # Open: PQ nao se usa as linhas onde as lojas estao fechadas, pois nao tem venda nessas linhas, logo joga-se fora todas as 
        # linhas que tem como valor 0 fora.
        #'open' != 0#

        # Sales: Excluir as linhas q tem valor 0, pois essas nao sao uteis para predicao.
        #'sales' > 0#

        df2 = df2[df2['open'] !=0]

        ## 3.2. Columns Selection

        ### Restricao de Negocio:

        #Customers: PQ como se precisa de uma predicao somente para daki a 6 semanas, nao teremos a quantidade de clientes por dia, 
        # a partir de hj ate daki a 6 semanas, so temos de hj para traz. Por isso joga-se essa coluna fora.

        cols_drop = ['open', 'promo_interval', 'month_map']
        df2 = df2.drop(cols_drop, axis=1)

        # open deletada pq so tem 1 entao nao e relevante, e 'promo_interval', 'month_map' deletadas pq sao colunas auxiliares para
        # outras colunas.
        
        return df2
    
    def data_preparation(self, df5):
        
        # 5.0. STEP 05 - DATA PREPARATION

        ## 5.1. Normalization

        # como nao tem variaveis numericas com distribuicao normal, nao se usa a normalization entao.

        ## 5.2. Rescaling

        # todas as variaveis numericas com natureza nao ciclica

        # competition_distance uses Robust Scaler
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform(df5[['competition_distance']].values)
        
        # competition_time_month uses Robust Scaler
        df5['competition_time_month'] = self.competition_time_month_scaler.fit_transform(df5[['competition_time_month']].values)
       
        # promo_time_week uses Min-Max Scaler
        df5['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df5[['promo_time_week']].values)
        
        # year uses Min-Max Scaler
        df5['year'] = self.year_scaler.fit_transform(df5[['year']].values)

        ## 5.3. TRansformation

        ### 5.3.1. Encoding

        # variaveis categoricas

        # state_holiday
        # como e uma variavel que apresenta ideia de estado, usa-se o one hot encoding
        df5 = pd.get_dummies(df5,prefix=['state_holiday'], columns=['state_holiday'])

        # store_type
        # como e uma variavel que nao apresenta ordem ou escala, cada valor e independente, usa-se o one label encoding
        df5['store_type'] = self.store_type_scaler.fit_transform(df5['store_type'])

        # assortment
        # como e uma variavel que apresenta ordem ou escala, usa-se o ordinal encoding
        assortment_dict = {'basic' : 1, 'extra' : 2, 'extended' : 3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)

        ### 5.3.2 Nature Rtansformation

        # inclui todas as variaveis numericas com natureza ciclica

        # day
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x * (2. * np.pi/30))) # 30 pois e a quantidade de dias do mes
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * (2. * np.pi/30)))

        # day_of_week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x * (2. * np.pi/7))) # 7 pois e a quantidade de dias da semana
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * (2. * np.pi/7)))

        # month
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x * (2. * np.pi/12))) # 12 pois e a quantidade de meses existentes
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * (2. * np.pi/12)))

        # week_of_year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x * (2. * np.pi/52))) # 52 pois e a quantidade de semanas do ano
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * (2. * np.pi/52)))
        
        cols_selected = ['store','promo', 'store_type', 'assortment', 'competition_distance',
             'competition_open_since_month', 'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year',
             'competition_time_month', 'promo_time_week', 'day_sin', 'day_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin',
             'month_cos', 'week_of_year_sin', 'week_of_year_cos']
        
        return df5[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict(test_data) # test_data sao os dados preparados
        
        # join pred into the original data
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient = 'records', date_format = 'iso')  