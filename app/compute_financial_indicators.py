
    #############################################################################################################
    #   Questo script calcola gli indici finanziari rilevanti sull'intero periodo di ciascuna stock             #
    #   Per ridurre il carico computazionale in esecuzione                                                      #
    #   utilizza il multi-processing per velocizzare il processo                                                #
    #                                                                                                           #
    #   Gli indicatori calcolati sono:                                                                          #
    #       -WMA10 (Weighted Moving Average, 10 day period)                                                     #
    #           https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average                            #
    #                                                                                                           #
    #       - EMA12,26,9 (Exponential Moving Average, 12-26-9 day pedior)                                       #
    #           https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average                         #
    #                                                                                                           #
    #       -MACD_LINE (Moving Average Convergence/Divergence)                                                  #
    #           https://en.wikipedia.org/wiki/MACD                                                              #
    #                                                                                                           #
    #       -Signal line                                                                                        #
    #           vedi precedente...                                                                              #
    #                                                                                                           #
    #       -RSI (Moving Average Convergence/Divergence)                                                        #
    #           https://en.wikipedia.org/wiki/Relative_strength_index                                           #
    #                                                                                                           #
    #       -TWAP10 (Time-weighted average price, 10 day period)                                                #
    #           https://en.wikipedia.org/wiki/Time-weighted_average_price                                       #
    #                                                                                                           #
    #       -ATR (Average True Range)                                                                           #
    #           https://en.wikipedia.org/wiki/Average_true_range                                                #
    #                                                                                                           #
    #       -SVMA (Simple Volume Moving Average)                                                                #
    #           https://www.marketvolume.com/analysis/volume_ma.asp                                             #
    #############################################################################################################

# Python modules
import pandas as pd
import os
import numpy as np
import configparser
import matplotlib.pyplot as plt
from multiprocessing import Process, Value, Lock
import time
from datetime import datetime
# My libs
import compute as cpt
from progress_bar import printProgressBar

    #############################################################################################################

# Faccio il parse del file di configurazione per estrarre i dati rilevanti
config = configparser.ConfigParser()
config.read('config.ini')
STOCKS_TO_ANALYZE = tickers_list = [ticker.strip() for ticker in config.get('DATASET', 'tickers').split(',')]
PROCESS_NUM = config.getint('SCRIPTS_ANALYSIS', 'process_count')

    #############################################################################################################

# Funzione per multi-processing
def process(STOCKS, progressBarCount, lock):
    # Itero sul sottoinsieme di stock passato e calcolo tutti gli indicatori
    for ticker in STOCKS:    
        
        # Carico il dataset storico della stock in analisi al momento
        filename = f"datasets/stock_data/{ticker}.csv"
        data = pd.read_csv(filename, usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Calcolo tutti gli indicatori finanziari con i metodi della mia libreria
        cpt.compute_WMA(data)
        cpt.compute_EMA(data, 12)
        cpt.compute_EMA(data, 26)
        cpt.compute_MACD_LINE(data)
        cpt.compute_EMA(data, 26+9, 'MACD_LINE')
        cpt.compute_MACD_HISTOGRAMS(data)
        cpt.compute_RSI(data)
        cpt.compute_TWAP(data,10)
        cpt.compute_ATR(data,14)
        cpt.compute_SVMA(data, 20)
        cpt.compute_profits(data)
        
        # Salvo il file in CSV rimpiazzando il precedente
        data.to_csv(filename, index=False)
        # Aggiorno la progress bar
        with lock:
            printProgressBar(progressBarCount.value + 1, len(STOCKS_TO_ANALYZE), prefix = 'Progress:', suffix = 'Completed', length = 100)
            progressBarCount.value += 1

    #############################################################################################################

if __name__ == "__main__":
    start = time.time()
    
    print("Calculating financial indicators...\n")
    printProgressBar(0, len(STOCKS_TO_ANALYZE), prefix = 'Progress:', suffix = 'Completed', length = 100)

    # Inizializzo il necessario per la condivisione della variabile che gestisce la barra    
    progressBarCount = Value('i', 0) 
    lock = Lock()
    processes = []
    
    # Divido la lista delle stock in parti approssimativamente uguali
    split_stocks = [i.tolist() for i in np.array_split(STOCKS_TO_ANALYZE, PROCESS_NUM)]
    
    # Creo tutti i processi e gli passo le variabili necessarie
    for i in range(PROCESS_NUM):
        p = Process(target=process, args=(split_stocks[i], progressBarCount, lock))
        p.start()
        processes.append(p)
        
    # Aspetto che tutti i processi terminino    
    for p in processes:
        p.join() 
         
    print("\nThe operation was completed succesfully, datasets are now up to date!")
    print(f"The operation was completed in {round(time.time()-start,2)} seconds with {PROCESS_NUM} processes")    
        
    today = datetime.now().strftime('%Y-%m-%d')
    config.set('DATASET', 'latest_update', today)

    # Scrivi le modifiche sul file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)