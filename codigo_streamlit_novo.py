import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Título do Aplicativo
st.set_page_config(page_title='Catapimbas: Modelo Preditivo de Vendas', page_icon='🚗')
st.title('🚗 Catapimbas: Modelo Preditivo de Vendas')

with st.expander('Sobre este Aplicativo'):
  st.markdown('**O aplicativo propõe integrar todos os dados de entrada no modelo preditivo que vai entregar, de forma visual e inteligente, diversos insight para tomada estretégica de decisão.**')
  st.info('Este aplicativo auxilia o usuário a construir um modelo preditivo utilizando o conceito de regressão. Basta adicionar o seu arquivo base e ver a magia acontecer diante dos seus olhos!')

  st.markdown('**Como usar este aplicativo?**')
  st.warning('É muito simples. A sua esquerda, no tópico 01, você irá adicionar a sua base de estudo. No tópico 02, é onde serão ajustado os parâmetros do modelo. Como resultado, o modelo será iniciados, apresentando seus resultados e permitindo que você faça os downloads dos modelos gerados.')

  st.markdown('**Vem ver por baixo dos panos:**')
  st.markdown('Bibliotecas utilizadas:')
  st.code('''- Pandas para fazer a análise exploratória;
- Scikit-learn para construir o modelo de machine learning;
- Altair para criação visual das apresentações;
- Streamlit para criação da interface final do usuário.
  ''', language='markdown')


# Criação da barra lateral para colocar os dados de entrada
with st.sidebar:
    # Database
    st.header('1. Database de Entrada')

    st.markdown('**Use sua base de dados**')
    arquivo_upload = st.file_uploader("Faça o upload do seu csv aqui:")
    if arquivo_upload is not None:
        df = pd.read_csv(arquivo_upload, index_col=False)

f = '%Y-%m-%d %H:%M:%S UTC'

df_datetime = df_inicial.loc[df_inicial['tipo'] == 'Superesportivo'].reset_index(drop=True)

df_datetime['data_venda'] = pd.to_datetime(df_datetime['data_venda'], format=f)

df_datetime.columns

df2 = df_datetime[['data_venda', 'tipo', 'valor_venda']]

df2 = df2.assign(only_day = df2['data_venda'].dt.date)

df2.sort_values(by='only_day')

df_agg = df2.groupby('only_day').sum('valor_venda').reset_index()

plt.plot(df_agg['only_day'], df_agg['valor_venda'])

df_sales = df_agg.copy()

df_sales['sales_diff'] = df_agg['valor_venda'].diff()

df_sales = df_sales.dropna()

supervised_data = df_sales.drop(['only_day', 'valor_venda'], axis=1)

for i in range(1, 13):
  col_name = 'month_' + str(i)
  supervised_data[col_name] = supervised_data['sales_diff'].shift(i)
supervised_data = supervised_data.dropna().reset_index(drop=True)

train_data = supervised_data[:-12]
test_data = supervised_data[-12:]

scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)

test_data = scaler.transform(test_data.to_numpy())

X_train, y_train = train_data[:,1:], train_data[:,0:1]

X_test, y_test   =  test_data[:,1:],   test_data[:,0:1]

y_train = y_train.ravel()
y_test = y_test.ravel()

print("X-train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("x_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

sales_dates = df_agg['only_day'][-12:].reset_index(drop=True)

predict_df = pd.DataFrame(sales_dates)

act_sales = df_agg['valor_venda'][-12:].to_list()
print(act_sales)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pre = lr_model.predict(X_test)

lr_pre = lr_pre.reshape(-1, 1)
lr_pre_test_set = np.concatenate([lr_pre, X_test], axis=1)
lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)

len(X_test)

result_list = []
for index in range(0, len(lr_pre_test_set)):
  result_list.append(lr_pre_test_set[index][0] + act_sales[index])
lr_pre_series = pd.Series(result_list, name='Linear Prediction')
predict_df = predict_df.merge(lr_pre_series, left_index=True, right_index=True)

lr_mse = np.sqrt(mean_squared_error(predict_df['Linear Prediction'], df_agg['valor_venda'][-12:]))
lr_mae = mean_absolute_error(predict_df['Linear Prediction'], df_agg['valor_venda'][-12:])
lr_r2 = r2_score(predict_df['Linear Prediction'], df_agg['valor_venda'][-12:])
print("Linear Regression MSE", lr_mse)
print("Linear Regression MSE", lr_mae)
print("Linear Regression MSE", lr_r2)

df_agg['only_day']

plt.figure(figsize=(15, 5))
# Actual Sales

startdate = pd.to_datetime("2024-05-25").date()

plt.plot(df_agg['only_day'].loc[df_agg['only_day'] <= startdate], df_agg['valor_venda'].loc[df_agg['only_day'] <= startdate])

# Predicted Sales
plt.plot(predict_df['only_day'], predict_df['Linear Prediction'])


plt.title('Actual Sales vs Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend(['Actual Sales', 'Predicted Sales'])
plt.legend()
plt.show()
