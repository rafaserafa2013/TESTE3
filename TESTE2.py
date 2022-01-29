import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt
import investpy as inv
import streamlit as st
import yfinance as yf
from datetime import date
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objects as go

data = date.today().strftime('%d/%m/%Y')

ticker = "CMCS34"
inicio = "01/01/2020"
fim = data

st.title('ANÁLISE DE AÇÕES BOVESPA')

# criando a sidebar
st.sidebar.header('ESCOLHA A AÇÃO:')

n_dias = st.slider('QUANTIDADE DE DIAS PARA ESTIMATIVA DE PREVISÃO:', 30, 365)

def pegar_dados_acoes():
    path = 'C:/Users/Sofia/Downloads/acoes.csv'
    return pd.read_csv(path, delimiter=';')

df = pegar_dados_acoes()

acao = df['snome']

nome_acao_escolhida = st.sidebar.selectbox('ESCOLHA UMA AÇÃO:', acao)

df_acao = df[df['snome'] == nome_acao_escolhida]

acao_escolhida = df_acao.iloc[0]['sigla_acao']

def pegar_valores_online(sigla_acao):
    dados = inv.get_stock_historical_data(stock = acao_escolhida
                                      , country = "Brazil"
                                      , from_date = inicio
                                      , to_date = fim)
    dados.reset_index(inplace=True)
    return dados

df_valores=pegar_valores_online(acao_escolhida)

st.subheader('TABELA DE VALORES - ' + nome_acao_escolhida)

st.write(df_valores.tail(10))

#criar grafico

st.subheader('GRAFICO DE PREÇOS')
fig=go.Figure()
fig.add_trace(go.Scatter(x=df_valores['Date'],
                         y=df_valores['Close'],
                         name='Preco Fechamento',
                         line_color='yellow'))

fig.add_trace(go.Scatter(x=df_valores['Date'],
                         y=df_valores['Open'],
                         name='Preco Abertura',
                         line_color='blue'))

st.plotly_chart(fig)

#PREVISAO DE PREÇOS

df_treino=df_valores[['Date','Close']]

#renomear coluna

df_treino=df_treino.rename(columns={"Date":'ds',"Close":'y'})

#criar modelo

modelo=Prophet()
modelo.fit(df_treino)

futuro=modelo.make_future_dataframe(periods=n_dias,freq='B')

previsao=modelo.predict(futuro)

st.subheader('PREVISAO')

st.write(previsao[['ds','yhat','yhat_lower','yhat_upper']].tail(n_dias))

#grafico1

grafico1=plot_plotly(modelo,previsao)
st.plotly_chart(grafico1)

#grafico2
grafico2=plot_components_plotly(modelo,previsao)
st.plotly_chart(grafico2)
