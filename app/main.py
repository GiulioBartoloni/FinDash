import streamlit as st

import pandas as pd
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

from fpdf import FPDF

from PIL import Image
import io
import base64
from datetime import datetime, date, timedelta
import os
import time
import json
import configparser
import shutil
import hashlib

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import scipy

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

symbol_to_company = {}

# Leggo le informazioni necessarie dal file di configurazione per avere la data di aggiornamento del dataset
config = configparser.ConfigParser()
config.read('config.ini')
latest_update = datetime.strptime(config.get('DATASET','latest_update'), '%Y-%m-%d').date()
today = datetime.now().date()

# Carico il logo dell'applicazione
img = Image.open('images/FinDash_logo.png')
buffered = io.BytesIO()
img.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Imposto la modalità 'wide' per la fase di login
if 'wide_mode' not in st.session_state:
    st.session_state.wide_mode = "centered"
st.set_page_config(layout=st.session_state.wide_mode)

# La funzione controlla la validità del login
def check_login(username, password):
    hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
    for user in users:
        if user["username"] == username and user["password"] == hashed_password:
            return user
    return None

# La funzione crea un utente e salva il json nel file
def create_user(users,first_name,last_name,username,password,sex):
    hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
    new_user={'first_name':first_name, 'last_name':last_name, 'username':username,'password':hashed_password,'sex':sex}
    users.append(new_user)
    with open('login_data.json', 'w') as file:
        json.dump(users, file, indent=4)

# Leggo il file e salvo in memoria gli utenti
users = []
if not os.stat('login_data.json').st_size==0:
    with open('login_data.json', 'r') as file:
        users = json.load(file)

# Variabili di stato per realizzare la fase di login/register
if 'update_dataset_flag' not in st.session_state:
    st.session_state.update_dataset_flag = False
if 'updating' not in st.session_state:
    st.session_state.updating = False
if 'ignoring' not in st.session_state:
    st.session_state.ignoring = False

if 'user' not in st.session_state:
    st.session_state.user = None
if 'register_flag' not in st.session_state:
    st.session_state.register_flag = False

# Questa parte del codice gestisce la logica del login/register e aggiornamento dei dati
if not st.session_state.ignoring and( not os.path.exists('datasets/stock_data') or today>latest_update):
    st.session_state.update_dataset_flag = True

# Se devo fare l'aggiornamento, eseguo gli script di update e aggiorno le flag
if(st.session_state.updating):
    os.system('python3 build_dataset.py')
    os.system('python3 compute_financial_indicators.py')
    st.session_state.update_dataset_flag = False
    st.session_state.updating = False

# Se il dataset non è aggiornato, chiedo se l'utente vuole farlo
if st.session_state.update_dataset_flag:
    st.error(f"The dataset is not up to date, with the latest update being on {latest_update}, would you like to update?\nWARNING: the operation may take a while!")
    col1,_,_,_,_,_,_,_,_,_,col2 = st.columns(11)
    with col1:
        if st.button("Yes"):
            st.session_state.updating=True
            st.rerun()
            
    with col2:
        if st.button("No"):
            st.session_state.update_dataset_flag = False
            st.session_state.ignoring = True
            st.rerun()
elif st.session_state.user is None:
    # Schermata di login
    if(st.session_state.register_flag==False):
        st.title("Login")
        username = st.text_input("Username", key="login_Username")
        password = st.text_input("Password", type="password", key="login_Password")

        if st.button("Login"):
            user = check_login(username, password)
            if user:
                st.session_state.user = user  
                st.session_state.wide_mode="wide"
                st.success("Login was successful!")
                time.sleep(2)
                st.rerun()
            else:
                st.error("Username or password is incorrect. Try again.")
    
        if st.button("Need to register?"):
            st.session_state.register_flag = True
            st.rerun()
            
            
    else:
        # Schermata di registrazione
        st.title("Register")
        first_name = st.text_input("First Name", key="register_first_name")
        last_name = st.text_input("First Name", key="register_last_name")
        username = st.text_input("Username", key="register_Username")
        password = st.text_input("Password", type="password", key="register_Password")
        sex = st.radio("Sex",["M","F"])
        if st.button("Register"):
            if username in [u["username"] for u in users]:
                st.error("Username is already taken!")
            elif len(username)==0:
                st.error("Username must be longer than 0 characters!")
            elif len(password)<8:
                st.error("Password must be longer than 7 characters!")
            else:
                user = create_user(users,first_name, last_name, username,password,sex)
                
                st.session_state.user = user 
                st.success("Account registered succesfully! You can now login.")
                st.session_state.register_flag = False
                time.sleep(2)
                st.rerun()
                
        if st.button("Go back"):
            st.session_state.register_flag = False
            st.rerun()    
else:
    
    # Crea il dizionario {Symbol: Company Name}
    folder_path = 'datasets/stock_data'
    files = os.listdir(folder_path)
    stock_names = [file.replace('.csv', '') for file in files if file.endswith('.csv')]
    us_conversion_table = pd.read_csv("datasets/us_conversion_table.csv", usecols=['Symbol', 'Company Name'])
    filtered_conversion_table = us_conversion_table[us_conversion_table['Symbol'].isin(stock_names)]
    symbol_to_company = dict(zip(filtered_conversion_table['Symbol'], filtered_conversion_table['Company Name']))
    
    # Creazione della sidebar con logo, messaggio di welcome, selezione delle sezioni del sito e bottone di logout
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{img_str}" style="width:50%; max-width:100%; margin-top: -50px;margin-bottom: 100px">',
        unsafe_allow_html=True
    )     
    
    st.sidebar.subheader(f"Welcome to FinDash, {st.session_state.user["first_name"]}!")
    st.sidebar.markdown("##")
    panel_selection = st.sidebar.selectbox(
        "Choose an option",
        ["Guide","Explore stocks","Study correlations","Research clusters", "Analyze metrics"]
    )
    
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.sidebar.success("Logout was successful!")
        time.sleep(2)
        st.session_state.wide_mode="centered"
        st.rerun()
        
    st.sidebar.markdown(
        '*<div style="margin-top: 400px; text-align: left; font-size: 16px;">~ Swim with the sharks ~</div>*',
        unsafe_allow_html=True
    ) 
    
    
    
    ############################################################################################################################################
    #                   SEZIONE PRINCIPALE DEL CODICE, CONTIENE LE 5 SEZIONI                                                                   #
    ############################################################################################################################################
    
    if panel_selection=="Guide":     
        st.markdown("# **FinDash**, the FINancial DASHboard that has anything you'll need.")
        st.markdown("#### Access historical data, discover asset relationships and identify investment opportunities with advanced yet easy-to-use tools.")
        
        
        st.divider()
        
        st.subheader("What you can do with FinDash:")
        st.markdown(r"$\textsf{\tiny (Infographics generated with sample data)}$")
    
        st.markdown("#")
        st.info("**Explore Stocks**: View historical trends and fundamental data of any title.")
        st.text("Choose a title, period of interest and analyse all relevant indicators:")
        
        col1,col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.text("Graph n.1")
                st.image("images/explore_1.png")
                st.markdown("""
                            Candlestick chart with:
                            - **Price**: stock price
                            - **WMA 10**: Weighted Moving Average on a 10 day period
                            - **TWAP 10**: Time Weighted Moving Average on a 10 day period
                            - **SVMA 20**: Simple Volume Moving Average on a 20 day period
                            """)
        with col2:
            with st.container(border=True):
                st.text("Graph n.2")
                st.image("images/explore_2.png")
                st.markdown("###")
                st.markdown("#####")
                st.markdown("""
                            RSI chart with:
                            - **RSI**: Relative Strength Index
                            - **Red limit line**: Overbought line
                            - **Blue limit line**: Oversold line
                            ######
                            """)
        col3,col4 = st.columns(2)
        with col3:
            with st.container(border=True):
                st.text("Graph n.3")
                st.image("images/explore_3.png")
                st.markdown("""
                            MACD chart with:
                            - **MACD Histogram**: Green bars for positive momentum, red bars for negative momentum
                            - **MACD Line (blue)**: Moving Average Convergence Divergence line 
                            - **Signal Line (orange)**: Signal line  for crossover signals
                            """)
        with col4:
            with st.container(border=True):
                st.text("Graph n.4")
                st.image("images/explore_4.png")
                st.markdown("""
                            ATR chart with:  
                            - **ATR Line**: Average True Range (volatility measure) 
                            ###
                            """)
        st.divider()
        st.success("**Study correlations**: Analyze relationships between different assets with heatmaps and graphs.")
        col5,col6 = st.columns(2)
        with col5:
            with st.container(border=True):
                st.text("Graph n.1")
                st.image("images/heatmap.png")
                st.markdown("""
                            Heatmap with correlation coefficients
                            """)
        with col6:
            with st.container(border=True):
                st.text("Graph n.2")
                st.image("images/graph.png")
                st.markdown("##")
                st.markdown("""
                            Graph with edges between nodes with a significant correlation coefficient
                            """)
        with st.container(border=True):
                st.text("Graph n.3")
                st.image("images/centrality_metrics.png")
                st.markdown("""
                            Barchart with centrality metrics
                            """)
    
        st.divider()
        st.warning("**Research clusters**: Identify groups of stocks with similar behaviours.")
        
        col5,col6 = st.columns(2)
        with col5:
            with st.container(border=True):
                st.text("Graph n.1")
                st.image("images/epsilon.png")
                st.markdown("""
                            ######
                            Choose the optimal epsilon in the elbow of the curve
                            """)
        with col6:
            with st.container(border=True):
                st.text("Graph n.2")
                st.image("images/dbscan.png")
                st.markdown("""
                            ###
                            ###
                            ###
                            Explore the results
                            """)
        
        st.divider()
        st.error("**Analyze metrics**: Evaluate key metrics and relevant news.")
        
        col5,col6 = st.columns(2)
        with col5:
            with st.container(border=True):
                st.text("Graph n.1")
                st.image("images/gauges.png")
                st.markdown("""
                            ######
                            Gauges with sentiment and community metrics
                            """)
        with col6:
            with st.container(border=True):
                st.text("Graph n.2")
                st.image("images/news.png")
                st.markdown("""
                            Latest news about chosen asset
                            """)
        
    ###############################################################################################
    #                   SEZIONE PER ESPLORARE TUTTI GLI INDICATORI FINANZIARI                     #
    ###############################################################################################  
    elif panel_selection == "Explore stocks":    
        
        st.title("Stock analytics")
        # Selezione della stock di interessse
        selected_stock = st.selectbox(
            "Choose a stock to analyze:", 
            options= symbol_to_company.values(),
            index=0
        )
        symbol = list(symbol_to_company.keys())[list(symbol_to_company.values()).index(selected_stock)]

        # Il dataset viene caricato
        filename = f"datasets/stock_data/{symbol}.csv"
        data = pd.read_csv(filename, usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'WMA', 'MACD_LINE', 'SIGNAL_LINE', 'MACD_HISTOGRAMS' , "RSI", 'TWAP_10', 'ATR', 'SVMA_20'])
        data['Date'] = pd.to_datetime(data['Date'])

        min_date = data['Date'].min().date()
        max_date = data['Date'].max().date()
        data['Date'] = pd.to_datetime(data['Date']).dt.date
        
        # Faccio scegliere l'tervallo di tempo e filtro il dataset in quello specificato
        selected_range = st.radio(
            "Interval:",
            options=["3Y", "1Y", "6M", "1M", "1S"],
            horizontal=True,
            label_visibility="collapsed",
            index=2
        )

        if selected_range == "1S":
            start_date = max_date - timedelta(weeks=1)
        elif selected_range == "1M":
            start_date = max_date - timedelta(days=30)
        elif selected_range == "6M":
            start_date = max_date - timedelta(days=6*30)
        elif selected_range == "1Y":
            start_date = max_date - timedelta(days=365)
        else:
            start_date = max_date - timedelta(days=3*365)

        filtered_data = data[data['Date'] >= start_date]


        # Candlestick chart
        chart1 = go.Figure()
        
        chart1.add_trace(go.Candlestick(
            x=filtered_data['Date'],
            open=filtered_data['Open'],
            high=filtered_data['High'],
            low=filtered_data['Low'],
            close=filtered_data['Close'],
            name='Price',
            increasing_line_color='#2CA453',
            decreasing_line_color='#F04730',
            opacity=0.8
        ))

        chart1.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['WMA'],
            name='WMA 10',
            line=dict(color='#8A2BE2', width=2),
            yaxis='y1'
        ))

        chart1.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['TWAP_10'],
            name='TWAP 10',
            line=dict(color='#eed202', width=2),
            yaxis='y1'
        ))


        chart1.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['SVMA_20'],
            name='SVMA 20',
            line=dict(color='#00ff00', width=2),
            yaxis='y2'
        ))

        chart1.update_layout(
            title=f'Candlestick chart: {selected_stock} ({symbol})',
            xaxis=dict(
                rangeslider=dict(visible=False),
                type='date'
            ),
            yaxis=dict(
                title='Price',
                side='left',
                showgrid=True
            ),
            yaxis2=dict(
                title='SVMA 20',
                overlaying='y',
                side='right',  
                showgrid=False
            ),
            hovermode='x unified',
            height=700,
            template='plotly_white',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1,
                xanchor='right',
                x=1
            )
        )

        chart1.update_xaxes(
            type = 'category',
            rangebreaks=[
                dict(bounds=["sat", "mon"])
            ],
            tickangle=-45
        )
        
        st.plotly_chart(chart1, use_container_width=True, config={
            'modeBarButtonsToAdd': [
                'zoom2d',
                'pan2d',
                'select2d',
                'lasso2d',
                'zoomIn2d',
                'zoomOut2d',
                'autoScale2d',
                'resetScale2d'
            ],
            'scrollZoom': True
        })
            
        ######################################################################
        # RSI index chart
        chart2 = go.Figure()

        chart2.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='#0000ff')
        ))

        chart2.add_shape(
            type='line',
            x0=filtered_data['Date'].min(),
            y0=70,
            x1=filtered_data['Date'].max(),
            y1=70,
            line=dict(color='red', width=2),
        )

        chart2.add_shape(
            type='line',
            x0=filtered_data['Date'].min(),
            y0=30,
            x1=filtered_data['Date'].max(),
            y1=30,
            line=dict(color='green', width=2),
        )

        chart2.add_annotation(
            x=filtered_data['Date'].min(),
            y=70,
            xref="x",
            yref="y",
            text="Overbought",
            showarrow=False,
            font=dict(color="red", size=12),
            xanchor="left",
            yanchor="bottom"
        )

        chart2.add_annotation(
            x=filtered_data['Date'].min(),
            y=30,
            xref="x",
            yref="y",
            text="Oversold",
            showarrow=False,
            font=dict(color="green", size=12),
            xanchor="left",
            yanchor="top"
        )

        chart2.update_layout(
            title='Relative Strenght Index with overbought/oversold lines',
            xaxis_title='Date',
            yaxis_title='RSI',
            yaxis=dict(range=[0, 100]),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1,
                xanchor='right',
                x=1
            )
        )

        chart2.update_xaxes(
            type = 'category',
            rangebreaks=[
                dict(bounds=["sat", "mon"])
            ],
            tickangle=-45
        )
        
        st.plotly_chart(chart2, use_container_width=True, config={
            'modeBarButtonsToAdd': [
                'zoom2d',
                'pan2d',
                'select2d',
                'lasso2d',
                'zoomIn2d',
                'zoomOut2d',
                'autoScale2d',
                'resetScale2d'
            ],
            'scrollZoom': True
        })

        ######################################################################
        # MACD chart
        chart3 = go.Figure()
        
        chart3.add_bar(
            x=filtered_data['Date'],
            y=filtered_data['MACD_HISTOGRAMS'],
            name='MACD Histograms',
            marker=dict(
                color=np.where(filtered_data['MACD_HISTOGRAMS'] >= 0, 'green', 'red'),
                opacity=0.5
            )
        )

        chart3.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['MACD_LINE'],
            name='MACD Line',
            line=dict(color='blue', width=2)
        ))
        chart3.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['SIGNAL_LINE'],
            name='Signal Line',
            line=dict(color='orange', width=2)
        ))
        

        chart3.update_yaxes(
            range=[min(min(data['SIGNAL_LINE'].min(),data['MACD_LINE'].min(),data['MACD_HISTOGRAMS'].min()),-max(data['SIGNAL_LINE'].max(),data['MACD_LINE'].max(),data['MACD_HISTOGRAMS'].max()))
                ,-min(min(data['SIGNAL_LINE'].min(),data['MACD_LINE'].min(),data['MACD_HISTOGRAMS'].min()),-max(data['SIGNAL_LINE'].max(),data['MACD_LINE'].max(),data['MACD_HISTOGRAMS'].max()))]
            )
        
        chart3.update_layout(
            title='MACD index',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1,
                xanchor='right',
                x=1
            ),
            barmode='overlay'
        )
        
        chart3.update_xaxes(
            type = 'category',
            rangebreaks=[
                dict(bounds=["sat", "mon"])
            ],
            tickangle=-45
        )
        
        st.plotly_chart(chart3, use_container_width=True, config={
            'modeBarButtonsToAdd': [
                'zoom2d',
                'pan2d',
                'select2d',
                'lasso2d',
                'zoomIn2d',
                'zoomOut2d',
                'autoScale2d',
                'resetScale2d'
            ],
            'scrollZoom': True
        })

        ######################################################################
        # ATR chart

        chart4 = go.Figure()

        chart4.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['ATR'],
            mode='lines',
            name='RSI',
            line=dict(color='purple')
        ))

        chart4.update_xaxes(
            type = 'category',
            rangebreaks=[
                dict(bounds=["sat", "mon"])
            ],
            tickangle=-45
        )
        
        chart4.update_layout(
            title='Average True Range indicator',
            xaxis_title='Date',
            yaxis_title='ATR',
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1,
                xanchor='right',
                x=1
            )
        )

        st.plotly_chart(chart4, use_container_width=True)

        # Bottone per generare il report della stock selezionata
        if st.button("Generate report"):
            with st.spinner('Generating report...'):
                # Creo la cartella temporanea
                os.makedirs("temp_images", exist_ok=True)
                
                try:
                    # Estraggo le immagini dei chart 
                    chart1.write_image("temp_images/chart1.png", width=1100, height=800)
                    chart2.write_image("temp_images/chart2.png", width=1100, height=800)
                    chart3.write_image("temp_images/chart3.png", width=1100, height=800)
                    chart4.write_image("temp_images/chart4.png", width=1100, height=800)
                    
                    # Genero il PDF e aggiungo una pagina per ciascun grafico
                    pdf = FPDF()
                    for i in range(1, 5):
                        img_path = f"temp_images/chart{i}.png"
                        with Image.open(img_path) as img:
                            width, height = img.size
                            width_mm = width * 0.264583
                            height_mm = height * 0.264583
                            
                            pdf.add_page(orientation ="L")
                            pdf.image(img_path, 0, 0, width_mm, height_mm)
                    pdf_bytes = pdf.output(dest='S').encode('latin1')
                    
                    # Mostro il bottone per scaricare il pdf generato
                    st.download_button(
                        label="Download Report PDF",
                        data=pdf_bytes,
                        file_name="report.pdf",
                        mime="application/pdf",
                        key='pdf_download'
                    )
                    
                finally:
                    # Elimino la cartella temporanea 
                    shutil.rmtree("temp_images", ignore_errors=True)
            
    ###############################################################################################
    #                   SEZIONE CON LO STUDIO DELLE CORRELAZIONI                                  #
    ###############################################################################################          
    elif panel_selection == "Study correlations":
        st.title("Bivariate analysis")

        # Multiselect per scegliere le stock da analizzare
        selected = st.multiselect(
            "Choose up to 25 stocks to analyze:",
            symbol_to_company.values(),
            default=None,
            placeholder="Search or select...",
            max_selections=25
        )
        # Scelta dei coefficenti di correlazione da utilizzare
        selected_correlation = st.selectbox(
            "Choose type of correlation coefficients to calculate:", 
            options= ["Pearson", "Kendall", "Spearman"],
            index=0
        )

        if st.button("Analyze", type="primary"):
            if len(selected)<2:
                st.error("Choose at least 2 stocks")
            else:
                # Costruisco il df per il calcolo delle correlazioni sui ritorni, creando un df con colonna Date e una per ciascuna stock selezionata
                correlation_df = None
                for el in selected:
                    symbol = list(symbol_to_company.keys())[list(symbol_to_company.values()).index(el)]
                    filepath = f"datasets/stock_data/{symbol}.csv"      
                    df = pd.read_csv(filepath)
                    
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df[['Date', 'returns']].rename(columns={'returns': symbol})
                    df = df.iloc[1:]
                    
                    if correlation_df is None:
                        correlation_df = df
                    else:
                        correlation_df = pd.merge(correlation_df, df, on='Date', how='inner')
                # Mantengo solo le colonne dei ritorni e calcolo i coefficienti di correlazione
                correlation_df = correlation_df.drop('Date', axis=1)
                correlation_matrix = correlation_df.corr(method=selected_correlation.lower())
                
                
                # Mostro la heatmap dei coefficienti calcolati
                with st.container(border=True):  
                    st.subheader(f"Correlation heatmap with {selected_correlation} coefficients")
                    font_size = max(6,12-0.2*len(selected))        
                    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                    heatmap, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(correlation_matrix, 
                                annot=True, fmt=".2f", center=0, cmap='coolwarm',
                                square=True, linewidths=0.5, cbar_kws={"shrink": .75},  annot_kws={"size": font_size}, ax=ax, mask=mask)
                    st.pyplot(heatmap)
                
                # Mostro il grafico con la forza delle relazioni
                with st.container(border=True):  
                    st.subheader(f"Correlation graph")
                    G = nx.Graph()
                    # Aggiungo tutti i nodi
                    for col in correlation_matrix.columns:
                        G.add_node(col)
                    # Calcolo il 40esimo percentile (arbitrario) 
                    all_correlations = correlation_matrix.abs().values.flatten()
                    threshold = np.percentile(all_correlations, 40)
                    # Aggiungo gli archi tra i nodi i cui coefficienti superano il 40esimi percentile
                    for i in range(len(correlation_matrix)):
                        for j in range(i+1, len(correlation_matrix)):
                            corr_value = correlation_matrix.iloc[i, j]
                            if abs(corr_value) > threshold:
                                G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j], 
                                        weight=corr_value)
                    # Definisco la posizione e plotto il grafico
                    pos = nx.spring_layout(G, k=1.5, iterations=100, weight='weight', seed=42)
                    
                    graph, ax = plt.subplots(figsize=(8, 6))
                    edges = G.edges(data=True)
                    weights = [abs(data['weight']) for _, _, data in edges]
                    
                    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=400)
                    nx.draw_networkx_labels(G, pos, font_size=5)
                    
                    for i, (u, v, d) in enumerate(edges):
                        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                            alpha=weights[i], 
                                            width=1, 
                                            edge_color=plt.cm.Blues(weights[i]))
                    plt.box(False)
                    st.pyplot(graph)
                    
                    # Calcolo alcune metriche semplici e le mostro
                    density_column, average_degree_column, clustering_coefficient_column = st.columns(3)
                    with density_column:
                        st.metric(label="**Graph density**", value=round(nx.density(G),2), border=True)
                    with average_degree_column:
                        st.metric(label="**Average degree**", value=round(sum(dict(G.degree()).values()) / len(G),2), border=True)
                    with clustering_coefficient_column:
                        st.metric(label="**Clustering coefficient**", value=round(nx.average_clustering(G),2), border=True)
                    
                    # Calcolo metriche più avanzate e le plotto con un barchart per metterle tutte a confronto
                    nodes = [list(symbol_to_company.keys())[list(symbol_to_company.values()).index(el)] for el in selected]
                    betweenness=list(nx.betweenness_centrality(G).values())
                    closeness=list(nx.closeness_centrality(G).values())
                    degree=list(nx.degree_centrality(G).values())
                    
                    graph_metrics = go.Figure()
                    graph_metrics.add_trace(go.Bar(
                        x=nodes,
                        y=betweenness,
                        name='Betweenness',
                        marker_color='#1f77b4',
                        hovertemplate="<b>%{x}</b><br>Betweenness: %{y:.3f}<extra></extra>"
                    ))

                    graph_metrics.add_trace(go.Bar(
                        x=nodes,
                        y=closeness,
                        name='Closeness',
                        marker_color='#ff7f0e',
                        hovertemplate="<b>%{x}</b><br>Closeness: %{y:.3f}<extra></extra>"
                    ))

                    graph_metrics.add_trace(go.Bar(
                        x=nodes,
                        y=degree,
                        name='Degree',
                        marker_color='#2ca02c',
                        hovertemplate="<b>%{x}</b><br>Degree: %{y:.3f}<extra></extra>"
                    ))

                    graph_metrics.update_layout(
                        title='Centrality metrics',
                        xaxis_title='Nodes',
                        yaxis_title='Values',
                        barmode='group',
                        hovermode='x unified',
                        template='plotly_white',
                        height=600,
                        width=1000,
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1,
                            xanchor='right',
                            x=1
                        )
                    )

                    st.plotly_chart(graph_metrics)
        
    ###############################################################################################
    #                   SEZIONE CON IL CLUSTERING DBSCAN                                          #
    ###############################################################################################
    elif panel_selection == "Research clusters":
        st.title("DBSCAN clustering")
        
        # Selezione dei parametri per l'algoritmo DBSCAN, richiedo almeno due parametri scelti
        st.divider()
        st.subheader("Choose parameters:")
        selected_features_dict = {}
        selected_features = []
        col = st.columns(7)
        with col[0]:
            if st.checkbox("Mean Return", value = True):
                selected_features_dict["Mean Return"]='mean_return'
        with col[1]:
            if st.checkbox("Volatility", value = True):
                selected_features_dict["Volatility"]='volatility'
        with col[2]:
            if st.checkbox("Max drawdown", value = True):
                selected_features_dict["Max drawdown"]='max_drawdown'
        with col[3]:
            if st.checkbox("Average RSI", value = True):
                selected_features_dict["Average RSI"]='avg_rsi'
        with col[4]:
            if st.checkbox("MACD histograms mean", value = True):
                selected_features_dict["MACD histograms mean"]='macd_hist_mean'
        with col[5]:
            if st.checkbox("Mean volume", value = True):
                selected_features_dict["Mean volume"]='volume_mean'
        with col[6]:
            if st.checkbox("Average ATR", value = True):
                selected_features_dict["Average ATR"]='atr_mean'
        
        if(len(selected_features_dict)>1):
            # Costruisco un dataframe che servirà per il DBSCAN, richiede una colonna per il nome della stock e una per ciascun parametro
            features_df = pd.DataFrame(index=stock_names, columns=selected_features_dict.values())
            
            for ticker in stock_names:
                filepath = f"datasets/stock_data/{ticker}.csv"      
            
                df = pd.read_csv(filepath)
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Calcola e assegna solo le metriche selezionate
                if 'mean_return' in selected_features_dict.values():
                    features_df.loc[ticker, 'mean_return'] = df['returns'].mean()
                if 'volatility' in selected_features_dict.values():
                    features_df.loc[ticker, 'volatility'] = df['returns'].std()
                if 'max_drawdown' in selected_features_dict.values():
                    features_df.loc[ticker, 'max_drawdown'] = (df['Close'] / df['Close'].cummax() - 1).min()
                if 'avg_rsi' in selected_features_dict.values():
                    features_df.loc[ticker, 'avg_rsi'] = df['RSI'].mean()
                if 'macd_hist_mean' in selected_features_dict.values():
                    features_df.loc[ticker, 'macd_hist_mean'] = df['MACD_HISTOGRAMS'].mean()
                if 'volume_mean' in selected_features_dict.values():
                    features_df.loc[ticker, 'volume_mean'] = df['Volume'].mean()
                if 'atr_mean' in selected_features_dict.values():
                    features_df.loc[ticker, 'atr_mean'] = df['ATR'].mean()
            
            # Faccio una normalizzazione dei valori e utilizzo il NearestNeighbors per il calcolo dell'iperparametro epsilon di DBSCAN
            scaler = StandardScaler()
            X = scaler.fit_transform(features_df)
            nn = NearestNeighbors(n_neighbors=len(selected_features_dict.values())+1).fit(X)
            distances, _ = nn.kneighbors(X)
            distances = np.sort(distances[:, -1])
            # Plotto il grafico per far scegliere dinamicamente il valore ottimale dell'iperparametro 
            elbow_df = pd.DataFrame({
                "Index": np.arange(len(distances)),
                "Distance": distances
            })
            fig_elbow = px.line(
                elbow_df,
                x="Index",
                y="Distance",
                title="Choose eps corresponding to the elbow point",
                labels={"Distance": f'{len(selected_features_dict.values())+1}-Distance', "Index": "Distance sorted points"}
            )
            
            
            elbow_column, dbscancolumn = st.columns(2)
            with elbow_column:
                st.plotly_chart(fig_elbow, use_container_width=True)
            with dbscancolumn:
                st.markdown("")
                eps = st.slider('Eps (max distance)', 
                            min_value=min(distances), 
                            max_value=max(distances),
                            value=min(distances), 
                            step=0.05)
                
                # Applico il DBSCAN, con iperparametro selezionato e dimensione minima del cluster pari a n.features+1
                dbscan = DBSCAN(eps=eps, min_samples=len(selected_features_dict.values())+1)
                features_df['Cluster'] = dbscan.fit_predict(X)
                # Quelli con valore di cluster '-1' sono considerabili rumore, quindi li etichetto
                features_df['Cluster'] = features_df['Cluster'].replace(-1, 'Noise').astype(str)
                # Mostro un riassunto dei risultati all'utente
                cluster_counts = features_df['Cluster'].value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Elements found']
                cluster_counts = cluster_counts.sort_values('Cluster')
                st.write("### Results")
                st.table(cluster_counts)
            
            # Plotto i risultati per vedere graficamente la classificazione in cluster
            st.title("Plotted results")
            st.subheader("Choose axis:")
            x_axis_column, y_axis_column = st.columns(2)
            with x_axis_column:
                x_axis = st.selectbox(
                    "Choose metric on X axis:",
                    options=selected_features_dict.keys(),
                    index=0
                )
            with y_axis_column:
                y_axis = st.selectbox(
                    "Choose metric on Y axis:",
                    options=selected_features_dict.keys(),
                    index=1
                )
            
            clustering_results_graph = px.scatter(
                features_df,
                x=selected_features_dict[x_axis],
                y=selected_features_dict[y_axis],
                hover_name=features_df.index,
                color='Cluster',
                height=600,
                labels={
                    selected_features_dict[x_axis]: x_axis,
                    selected_features_dict[y_axis]: y_axis
                }
            )

            st.plotly_chart(clustering_results_graph, use_container_width=True)
        else:
            st.warning("Choose at least two options!")

    ###############################################################################################
    #                   SEZIONE DELLE METRICHE AGGIUNTIVE (SCRAPING)                              #
    ###############################################################################################
    elif panel_selection == "Analyze metrics":
        st.title("Analyze metrics")
        # Selezione della stock per cui fare lo scraping
        selected_stock = st.selectbox(
            "Choose a stock to analyze:", 
            options= symbol_to_company.values(),
            index=0
        )
        symbol = list(symbol_to_company.keys())[list(symbol_to_company.values()).index(selected_stock)]
        
        # Preparo il chromedriver per lo scraping
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=chrome_options)
        # Ottengo la descrizione della stock e la mostro
        try:
            driver.get(f'https://stocktwits.com/symbol/{symbol}/about')
            element = driver.find_element(By.CSS_SELECTOR, '[data-cy="symbol-about-description"]')
            st.subheader(f"About {selected_stock}")
            st.text(element.text)
        except:
            st.error("Data not found!")
        ############################################################
        # GAUGE CHARTS
        ############################################################
        # Ottengo ora i dati dalla sezione sentiment e li mostro con dei gauge chart
        try:
            driver.get(f'https://stocktwits.com/symbol/{symbol}/sentiment')
            gauge_values = driver.find_elements(By.CSS_SELECTOR, "div.gauge_gagueNumber__Dr41m.absolute")
            
            sentiment_col, volume_col = st.columns(2)
            with sentiment_col:
                # Utilizzo il valore ottenuto dello user sentiment per plottare il gauge chart        
                categories = {
                    "Extremely Bearish": (0, 25),
                    "Bearish": (25, 45),
                    "Neutral": (45, 55),
                    "Bullish": (55, 75),
                    "Extremely Bullish": (75, 100)
                }

                if(gauge_values[0].text=="N/A"):
                    value=0
                else:
                    value = int(gauge_values[0].text) 

                current_category = next(
                    (cat for cat, (min_val, max_val) in categories.items() if min_val <= value < max_val)
                )

                tick_positions = [(min_val + max_val)/2 for min_val, max_val in categories.values()]

                sentiment_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Current user sentiment: {current_category}"},
                    gauge={
                        'axis': {
                            'range': [0, 100],
                            'tickvals': tick_positions,  
                            'ticktext': list(categories.keys()),
                            'tickmode': 'array'
                        },
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "#540302"},
                            {'range': [25, 45], 'color': "#ab3130"},
                            {'range': [45, 55], 'color': "#ece0a6"},
                            {'range': [55, 75], 'color': "#7a9163"},
                            {'range': [75, 100], 'color': "#4f7b58"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': value
                        }
                    }
                ))
                st.plotly_chart(sentiment_gauge)
                
            with volume_col:
                # Utilizzo il valore ottenuto del message volume per plottare il gauge chart                
                categories = {
                    "Extremely Low": (0, 25),
                    "Low": (25, 45),
                    "Normal": (45, 55),
                    "High": (55, 75),
                    "Extremely High": (75, 100)
                }

                if(gauge_values[1].text=="N/A"):
                    value=0
                else:
                    value = int(gauge_values[1].text) 

                current_category = next(
                    (cat for cat, (min_val, max_val) in categories.items() if min_val <= value < max_val)
                )

                tick_positions = [(min_val + max_val)/2 for min_val, max_val in categories.values()]

                message_volume_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Current message volume: {current_category}"},
                    gauge={
                        'axis': {
                            'range': [0, 100],
                            'tickvals': tick_positions,
                            'ticktext': list(categories.keys()),
                            'tickmode': 'array'
                        },
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "#540302"},
                            {'range': [25, 45], 'color': "#ab3130"},
                            {'range': [45, 55], 'color': "#ece0a6"},
                            {'range': [55, 75], 'color': "#7a9163"},
                            {'range': [75, 100], 'color': "#4f7b58"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': value
                        }
                    }
                ))

                st.plotly_chart(message_volume_gauge)
        except:
            st.error("Data not found!")
        # Utilizzo il valore ottenuto del participation ratio per plottare il gauge chart 
        categories = {
            "Extremely Low": (0, 25),
            "Low": (25, 45),
            "Normal": (45, 55),
            "High": (55, 75),
            "Extremely High": (75, 100)
        }

        if(gauge_values[2].text=="N/A"):
            value=0
        else:
            value = int(gauge_values[2].text) 

        current_category = next(
            (cat for cat, (min_val, max_val) in categories.items() if min_val <= value < max_val)
        )

        tick_positions = [(min_val + max_val)/2 for min_val, max_val in categories.values()]

        participation_ratio_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Current participation ratio: {current_category}"},
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickvals': tick_positions,
                    'ticktext': list(categories.keys()),
                    'tickmode': 'array'
                },
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "#540302"},
                    {'range': [25, 45], 'color': "#ab3130"},
                    {'range': [45, 55], 'color': "#ece0a6"},
                    {'range': [55, 75], 'color': "#7a9163"},
                    {'range': [75, 100], 'color': "#4f7b58"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))

        st.plotly_chart(participation_ratio_gauge)  
        

        ############################################################
        # OTHER METRICS
        ############################################################
        # Ottengo tutte le altre metriche dal sito presenti nelle due tabelle
        try:
            driver.get(f'https://stocktwits.com/symbol/{symbol}/fundamentals')
            table_items = driver.find_elements(
                By.CSS_SELECTOR, 
                'tr[class^="SymbolFundamentalsTable_item"]'
            )
            keys=[]
            values=[]
            for row in table_items:
                key = row.find_element(By.TAG_NAME,"td")
                keys.append(key.text)
                values.append(key.find_element(By.XPATH, "following-sibling::*[1]").text)
            fundamentals = dict(zip(keys,values))
            
            # Mostro le metriche ottenute
            divident_yield_column, beta_column, pe_ratio_column, market_capitalization_column = st.columns(4)
            with divident_yield_column:
                st.metric(label="**Dividend Yield**", value=fundamentals['Dividend Yield'], border=True)
            with beta_column:
                st.metric(label="**Beta**", value=fundamentals['Beta'], border=True)
            with pe_ratio_column:
                st.metric(label="**PE Ratio**", value=fundamentals['PE Ratio'], border=True)
            with market_capitalization_column:
                st.metric(label="**Market Capitalization**", value=fundamentals['Market Capitalization'], border=True)
            price_to_book_column, revenue_per_employee_column, ebitda_column = st.columns(3)
            with price_to_book_column:
                st.metric(label="**Price to Book**", value=fundamentals['Price to Book'], border=True)
            with revenue_per_employee_column:
                st.metric(label="**Revenue Per Employee**", value=fundamentals['Revenue Per Employee'], border=True)
            with ebitda_column:
                st.metric(label="**Enterprise Value/EBITDA**", value=fundamentals['Enterprise Value/EBITDA'], border=True)
            
            
            st.title("Relevant news")
            # Ottengo le notizie dal sito
            driver.get(f'https://stocktwits.com/symbol/{symbol}/news')
            # Prendo il titolo della notizia principale (ha tag diversi)
            main_news_title = driver.find_element(
                By.CSS_SELECTOR, 
                'div[class^="NewsItem_standaloneTitle"] span'
            )
            # Ottengo tutti gli altri titoli ed aggiungo quello principale
            news_titles = driver.find_elements(
                By.CSS_SELECTOR,
                'div[class^="NewsItem_title"] span'
            )
            news_titles.insert(0,main_news_title)

            # Prendo tutti i testi delle notizie
            news_descriptions = driver.find_elements(
                By.CSS_SELECTOR,
                'div[class^="NewsItem_description"]'
            )

            # Prendo i link, se disponibili, delle notizie
            link_divs = driver.find_elements(
                By.CSS_SELECTOR, 
                "div[class^='NewsItem_description'] + div"  
            )
            links = []
            for div in link_divs:
                try:
                    a_tag = div.find_element(By.TAG_NAME, "a")
                    href = a_tag.get_attribute("href")
                    links.append(href)
                except:
                    links.append("")  
            
            # Mostro tutte le notizie all'utente 
            for i in range(0,len(links)):
                with st.container(border=True):
                    st.subheader(news_titles[i].text)
                    st.text(news_descriptions[i].text)
                    if(links[i]!=""):
                        st.link_button("View full article",links[i])
            # Chiudo il driver
            driver.quit()
        except:
            st.error("Data not found!")
