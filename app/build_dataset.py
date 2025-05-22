
    #####################################################################################################
    #   Questo script recupera i dati storici delle stock nella lista presente nel file di config       # 
    #   utilizzando l'API pubblica di Yahoo Finance tramite la libreria yfinance.                       #
    #                                                                                                   #
    #   I dati sono poi salvati nella cartella che FinDash utilizzerà.                                  #
    #####################################################################################################

# Python modules
import yfinance as yf
import pandas as pd
from datetime import datetime
import os 
import configparser
import time
from multiprocessing import Process, Value, Lock
import numpy as np
# My libs
from progress_bar import printProgressBar

    #############################################################################################################

# Faccio il parse del file di configurazione per estrarre i dati rilevanti
config = configparser.ConfigParser()
config.read('config.ini')
STOCKS_TO_ANALYZE = tickers_list = [ticker.strip() for ticker in config.get('DATASET', 'tickers').split(',')]

    #############################################################################################################

# Funzione principale per il recupero dei dati
# Dato il ticker di un'azione, salverà il corrispettivo file CSV
def retrieve_stock_history(ticker):
    
    #Inizializzo le date come massima e minima, così da assicurarmi di prelevare tutti i dati
    start_date = '1900-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Tento di recuperare la stock tramite API
    try:
        history = None
        stock = yf.Ticker(ticker)
        history = stock.history(start=start_date, end=end_date, interval='1d', actions=True)
        # Se vuota, lo comunico all'utente
        if history.empty:
            print(f"No data was found for Ticker: {ticker}")
            return None
        
        # Setto correttamente gli indici, converto la data e cancello le colonne non utili
        history.reset_index(inplace=True)
        history['Date'] = pd.to_datetime(history['Date']).dt.date
        history.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
        
        # Salvo il file con formato csv
        filename = f"datasets/stock_data/{ticker}.csv"
        history.to_csv(filename, index=False)
    # Se si verifica un errore, lo comunico all'utente
    except Exception as e:
        print(f"There was an error downloading Ticker {ticker}: {str(e)}")
        return None

    #############################################################################################################

if __name__ == "__main__":
    start = time.time()
    # Se la cartella che utilizzerà FinDash non esiste, la creo
    if not os.path.exists("datasets/stock_data"): 
        os.makedirs("datasets/stock_data") 
    
    print("Gathering historical data...\n")
    printProgressBar(0, len(STOCKS_TO_ANALYZE), prefix = 'Progress:', suffix = 'Completed', length = 100)
    
    progressBarCount = 0
    for ticker in STOCKS_TO_ANALYZE:
        retrieve_stock_history(ticker)
        printProgressBar(progressBarCount + 1, len(STOCKS_TO_ANALYZE), prefix = 'Progress:', suffix = 'Completed', length = 100)
        progressBarCount += 1
    
    
    print("\nData was successfully extracted!")
    print(f"The operation was completed in {round(time.time()-start,2)} seconds")    