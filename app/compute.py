
    #############################################################################################################
    #   Questo modulo contiene tutte le funzioni il calcolo degli indicatori finanziari ritenuti rilevanti      #
    #   Per ciascuno di essi è riportato, sia nella relazione che all'interno del codice, la sorgente           #
    #   da cui sono state prese le informazioni per il loro calcolo e la motivazione.                           #
    #                                                                                                           #
    #############################################################################################################

# Funzione per il calcolo della Weighted Moving Average
def compute_WMA(dataset, period=10):
    # Preparo i pesi per la media ponderata
    weights = list(range(1, period+1)) 
    total_weight = sum(weights)
    wma_values = []
    # Itero su tutto il dataset
    for i in range(len(dataset)):
        # Se non ho abbastanza dati, aggiungo None
        if i < period - 1:
            wma_values.append(None)
        else:
            # Creo il subset della finestra corrente e applico la formula per il calcolo della WMA
            window = dataset['Close'][i-period+1:i+1]
            wma = (window * weights).sum() / total_weight
            wma_values.append(wma)
    # Aggiungo la WMA al dataset originale
    dataset['WMA'] = wma_values
 
#############################################################################################################

# Funzione per il calcolo della Exponential Moving Average
def compute_EMA(dataset, period, column='Close'):
    # Calcolo i pesi da assegnare alle due parti della formula
    alpha=2/(period+1)
    beta = 1 - alpha
        
    ema_values = []
    # Itero su tutto il dataset
    for i in range(len(dataset)):
        # Valori base
        if i < period - 1:
            ema_values.append(None)
        elif i == period-1:
            ema_values.append(dataset[column].values[i])
        else:
            # Applico la formula
            ema_values.append(dataset[column].values[i]*alpha+ema_values[i-1]*beta) 
    # Aggiungo la EMA al dataset originale, la chiamo SIGNAL_LINE se specificato nella chiamata
    dataset[f'EMA{period}' if column=='Close' else f'SIGNAL_LINE'] = ema_values

#############################################################################################################

# Funzione per il calcolo della MACD_LINE
def compute_MACD_LINE(dataset):    
    dataset['MACD_LINE'] = dataset["EMA12"] - dataset["EMA26"]
    
#############################################################################################################

def compute_MACD_HISTOGRAMS(dataset):
    dataset['MACD_HISTOGRAMS'] = dataset["MACD_LINE"] - dataset["SIGNAL_LINE"]

#############################################################################################################

# FUnzione per il calcolo dei profitti
def compute_profits(dataset):
    returns = [None]
    for i in range (1,len(dataset)):
        returns.append((dataset['Close'].values[i]-dataset['Close'].values[i-1])/dataset['Close'].values[i-1])
    dataset['returns'] = returns
    
#############################################################################################################

# Funzione per il calcolo della Relative Strength Index
def compute_RSI(dataset):
    rsi_values = []
    # Itero su tutto il dataset
    for i in range(len(dataset)):
        if i < 10:
            rsi_values.append(None)
        else:
            # Creo il subset della finestra e calcolo i valori di UP e DOWN specificati dalla formula
            window = dataset['Close'][i-10:i+1].reset_index(drop=True)
            U = 0
            D = 0
            for j in range(1,11):
                if(window[j]>window[j-1]):
                    U += window[j]-window[j-1]
                else:
                    D += window[j-1]-window[j]
            if(D==0):
                rsi_values.append(100)
            else:
                rsi_values.append(100-(100/(1+(U/D))))
    # Aggiungo l'RSI al dataset originale
    dataset['RSI'] = rsi_values

#############################################################################################################

# Formula per il calcolo della Time Weighted Average Price
def compute_TWAP(dataset, period):
    # Calcolo la media dei prezzi
    twap = []
    avg = (dataset['Open']+dataset['Close']+dataset['High']+dataset['Low'])/4
    # Itero su tutto il dataset
    for i in range(len(dataset)):
        if i < period-1:
            twap.append(None)
        else:
            # Faccio la somma sulla finestra e applico la formula
            sum = avg.iloc[i-period+1:i+1].sum()
            twap.append(sum/period)
    # Aggiungo il TWAP al dataset originale     
    dataset[f"TWAP_{period}"] = twap

#############################################################################################################

# Formula per il calcolo del True Range
def compute_AR(dataset):
    ar = [None]
    # Itero su tutto il dataset e applico la formula
    for i in range(1,len(dataset)):
        ar.append(max((dataset.at[i,'High']-dataset.at[i,'Low']), abs(dataset.at[i,'High']-dataset.at[i-1,'Close']), abs(dataset.at[i,'Low']-dataset.at[i-1,'Close'])))    
    # Aggiungo il AR al dataset originale     
    dataset["AR"] = ar
    
#############################################################################################################

# Formula per il calcolo del Average True Range
def compute_ATR(dataset, period):
    # Calcolo prima il AR
    compute_AR(dataset)    
    # Valore di default
    atr = [None, dataset.at[1,'AR']]
    # Itero su tutto il dataset e applico la formula
    for i in range(2,len(dataset)):
        atr.append((dataset.at[i-1,'AR']*(period-1)+dataset.at[i,'AR'])/period)
    # Aggiungo il ATR al dataset originale      
    dataset["ATR"] = atr
    
#############################################################################################################

# Formula per il calcolo del Simple Volume Moving Average
def compute_SVMA(dataset, period):
    svma_values = []
    # Itero su tutto il dataset 
    for i in range(len(dataset)):
        if i < period-1:
            svma_values.append(None)
        else:
            # Applico la formula sulla finestra individuata
            window = dataset['Volume'][i-period+1:i+1]
            sum = 0
            for el in window:
                sum += el/period
            svma_values.append(sum)   
    # Aggiungo il SVMA al dataset originale        
    dataset[f"SVMA_{period}"] = svma_values