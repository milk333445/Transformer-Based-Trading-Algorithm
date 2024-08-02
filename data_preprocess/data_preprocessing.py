import yfinance as yf
import pandas as pd
import os
import numpy as np
import talib
import torch
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED, ALL_COMPLETED
import time
import multiprocessing 

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def fetch_and_adjust_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    stock_data = stock_data.reset_index()

    stock_data['adj_ratio'] = stock_data['Close'] / stock_data['Adj Close']
    
    def adjust(df):
        df['Open'] = df['Open'] / df['adj_ratio']
        df['High'] = df['High'] / df['adj_ratio']
        df['Low'] = df['Low'] / df['adj_ratio']
        df['Close'] = df['Close'] / df['adj_ratio']
        df['Volume'] = df['Volume'] * df['adj_ratio']
        return df
    stock_data = adjust(stock_data)

    stock_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    print('Stock dataset download successfully')
    stock_data = calculate_technical_indicators(stock_data)
    print('Indicator caculate done')

    return stock_data

def calculate_technical_indicators(df):
    # RSI
    rsi = talib.RSI(df['Close'], timeperiod=14)
    df['RSI'] = rsi
    
    # MACD
    dif, dem, histogram = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = histogram
    
    # WILLR
    willr = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['WILLR'] = willr
    
    # CCI
    cci = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['CCI'] = cci
    
    # CMO
    cmo = talib.CMO(df['Close'], timeperiod=14)
    df['CMO'] = cmo
    
    # ROC
    roc = talib.ROC(df['Close'], timeperiod=10)
    df['ROC'] = roc
    
    # ema
    ema = talib.EMA(df['Close'], timeperiod=30)
    df['EMA'] = ema
    
    # SMA
    sma = talib.SMA(df['Close'], timeperiod=30)
    df['SMA'] = sma
    
    # TEMA
    tema = talib.TEMA(df['Close'], timeperiod=30)
    df['TEMA'] = tema
    
    # WMA
    wma = talib.WMA(df['Close'], timeperiod=30)
    df['WMA'] = wma
    
    # HT_TRENDLINE
    ht_trendline = talib.HT_TRENDLINE(df['Close'])
    df['HT_TRENDLINE'] = ht_trendline
    
    # SAR
    sar = talib.SAR(df['High'], df['Low'], acceleration=0, maximum=0)
    df['SAR'] = sar
    
    # ATR
    atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ATR'] = atr
    
    # TRANGE
    trange = talib.TRANGE(df['High'], df['Low'], df['Close'])
    df['TRANGE'] = trange
    
    # AD
    ad = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
    df['AD'] = ad
    df = df.dropna()
    df['Year'] = df['Date'].dt.year
    df = df[df['Year'] >= 2000]
    df = df.drop(columns=['Year'])
    
    return df


def create_train_val_ofs_dataset(df, start_year, dataset_number, output_dir):
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    else:
        pass
    df_copy = df.copy()
    df_copy = df_copy[(df_copy['Year'] >= start_year) & (df_copy['Year'] <= start_year + 7)]
    df_copy.loc[(df_copy['Year'] >= start_year) & (df_copy['Year'] <= start_year + 5), 'Dataset Type'] = 'train'
    df_copy.loc[df_copy['Year'] == start_year + 6, 'Dataset Type'] = 'validation'
    df_copy.loc[df_copy['Year'] == start_year + 7, 'Dataset Type'] = 'out_of_sample'
    
    dataset_name = f'Dataset{dataset_number}.csv'
    dataset_dir = os.path.join(output_dir, dataset_name)
    df_copy.to_csv(dataset_dir, index=False)
    
    
def create_and_save_datasets(df, output_dir):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Dataset Type'] = 'Unknown'
    
    start_year = 2000
    dataset_number = 1
    
    while start_year + 7 <= 2022:
        create_train_val_ofs_dataset(df, start_year, dataset_number, output_dir)
        start_year += 1
        dataset_number += 1
    print('Create train&test dataset')
        



def generate_extreme_value_labels(df, window_size, threshold=3):
    df['Label'] = 'HOLD' 
    for i in range(len(df) - window_size):
        current_price = df.iloc[i]['Close']
        future_window = df.iloc[i:i+window_size]
        min_price = future_window['Close'].min()
        max_price = future_window['Close'].max()
        
        min_index = future_window['Close'].idxmin()
        max_index = future_window['Close'].idxmax()
        
        increase_percent = ((max_price - current_price) / current_price) * 100
        decrease_percent = ((current_price - min_price) / current_price) * 100
        
        if i == min_index and increase_percent > threshold:
            df.loc[i, 'Label'] = 'BUY'
        elif i == max_index and decrease_percent > threshold:
            df.loc[i, 'Label'] = 'SELL'
        
    total_return = calculate_sharpe_ratio(df)
        
    return total_return, df      
       
       
def generate_labels(file_path, df, window_sizes, threshold=3):
    best_df = None
    best_window_size = None
    best_return = -np.inf
    for windowSize in window_sizes:
        return_value, df = generate_extreme_value_labels(df, windowSize, threshold)
        if return_value > best_return:
            best_return = return_value
            best_df = df.copy()
            best_window_size = windowSize
            
    print(f'best return : {best_return}')
    print(f'best window size : {best_window_size}')
    print('-' * 50)
    return best_df, best_window_size
        
       
def process_csv_file(args):
    file_path, windowSizes, threshold = args
    df = pd.read_csv(file_path)
    df, best_window_size = generate_labels(file_path, df, windowSizes, threshold)
    df.to_csv(file_path, index=False)
    print(f'Sucessfully save the csv file : {file_path}')
    return best_window_size

def generate_labels_for_folder(input_folder, windowSizes, threshold=3, core_for_other_jobs=5):
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    best_window_size_list = []
    
    num_processes = multiprocessing.cpu_count() - core_for_other_jobs
    pool = multiprocessing.Pool(processes=num_processes)
    
    args_list = [(os.path.join(input_folder, csv_file), windowSizes, threshold) for csv_file in csv_files]
    
    best_window_sizes = pool.map(process_csv_file, args_list)
    
    pool.close()
    pool.join()
    
    for best_window_size in best_window_sizes:
        best_window_size_list.append(best_window_size)
    
    print('Create labels dataset sucessfully')
    
    txt_file_name = 'Best_window_sizes.txt'
    txt_file_path = os.path.join(input_folder, txt_file_name)
    with open(txt_file_path, 'w') as f:
        for best_window_size in best_window_size_list:
            f.write(str(best_window_size) + '\n')
        


def calculate_sharpe_ratio(df):
    fund = 100000
    money = 100000
    BS = None
    buy = []
    sell = []
    profit_list = [0]
    profit_list_realized = []
    
    for i in range(len(df)):
        if i == len(df) - 1:
            break
        ## entry the market
        entryLong = df['Label'][i] == 'BUY'
        ## exit the market
        exitShort = df['Label'][i] == 'SELL'

        if BS == None:
            profit_list.append(0)
            if entryLong:
                tempSize = money // df['Open'][i+1]
                BS = 'B'
                t = i+1
                buy.append(t+1)

        elif BS == 'B':
            profit=tempSize*(df['Open'][i+1]-df['Open'][i])
            profit_list.append(profit)

            if exitShort:
                pl_round=tempSize*(df['Open'][i+1]-df['Open'][t])
                sell.append(i+1)
                BS=None

                profit_realized = pl_round
                profit_list_realized.append(profit_realized)
    
    equity = pd.DataFrame({'profit': np.cumsum(profit_list)}, index=df.index)
    total_return = (equity['profit'].iloc[-1]) / fund

    return total_return

def get_date_inference(dataset_path):
    dataset = pd.read_csv(dataset_path)
    val_data_corrected = dataset[dataset['Dataset Type'] == 'validation']
    train_data_corrected = dataset[dataset['Dataset Type'] == 'train']
    train_date = train_data_corrected['Date'].to_list()
    date = val_data_corrected['Date'].to_list()
    return date, train_date




def preprocess_dataset_inferece(dataset_path):
    df = pd.read_csv(dataset_path)
    
    label_mapping = {
        'HOLD': 0,
        'BUY': 1,
        'SELL': 2
    }
    df['Label'] = df['Label'].apply(lambda x: label_mapping[x])
    
    tech_indicators = ['RSI', 'MACD', 'WILLR', 'CCI', 'CMO', 'ROC', 'EMA', 'SMA', 'TEMA', 'WMA', 'HT_TRENDLINE', 'SAR', 'ATR', 'TRANGE', 'AD']
    scaler = StandardScaler()
    df[tech_indicators] = scaler.fit_transform(df[tech_indicators])
    columns_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'Year']
    df = df.drop(columns=columns_to_drop)
    return df


# create train, validation, out_of_sample(not used) dataset
def preprocess_dataset_to_train(dataset_path):
    df = pd.read_csv(dataset_path)
    
    label_mapping = {
        'HOLD': 0,
        'BUY': 1,
        'SELL': 2
    }
    df['Label'] = df['Label'].apply(lambda x: label_mapping[x])
    
    tech_indicators = ['RSI', 'MACD', 'WILLR', 'CCI', 'CMO', 'ROC', 'EMA', 'SMA', 'TEMA', 'WMA', 'HT_TRENDLINE', 'SAR', 'ATR', 'TRANGE', 'AD']
    scaler = StandardScaler()
    df[tech_indicators] = scaler.fit_transform(df[tech_indicators])
    
    df_train = df[df['Dataset Type'] == 'train'].drop(columns=['Dataset Type'])
    df_validation = df[df['Dataset Type'] == 'validation'].drop(columns=['Dataset Type'])
    df_out_of_sample = df[df['Dataset Type'] == 'out_of_sample'].drop(columns=['Dataset Type'])
    df_train = df_train.reset_index(drop=True)
    df_validation = df_validation.reset_index(drop=True)
    df_out_of_sample = df_out_of_sample.reset_index(drop=True)
    
    columns_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'Year', 'Date']
    df_train = df_train.drop(columns=columns_to_drop)
    df_validation = df_validation.drop(columns=columns_to_drop)
    df_out_of_sample = df_out_of_sample.drop(columns=columns_to_drop)
    
    return df_train, df_validation, df_out_of_sample

    
def preprocess_dataframe(df, sequence_length=15):
    tech_indicators = ['RSI', 'MACD', 'WILLR', 'CCI', 'CMO', 'ROC', 'EMA', 'SMA', 'TEMA', 'WMA', 'HT_TRENDLINE', 'SAR', 'ATR', 'TRANGE', 'AD']
    tech_data = df[tech_indicators]
    
    # create 2D tensor
    tensor_list = []
    for i in range(len(tech_data) - sequence_length + 1):
        tensor = tech_data[i:i+sequence_length].values
        tensor_list.append(tensor)
    
    df_2d_tensor = np.array(tensor_list).reshape(-1, 1, sequence_length, len(tech_indicators))
    df_2d_tensor = np.swapaxes(df_2d_tensor, 2, 3)
    df_2d_tensor = torch.tensor(df_2d_tensor, dtype=torch.float32)
    
    # create label
    label_data_corrected = []
    train_labels = df['Label']
    for i in range(len(train_labels) - sequence_length + 1):
        label_data_corrected.append(train_labels[i+sequence_length-1])
    label_data_corrected = np.array(label_data_corrected)
    label_data_corrected = label_data_corrected.reshape(-1, 1)
    label_data_corrected_tensor = torch.tensor(label_data_corrected, dtype=torch.float32)
    
    
    return df_2d_tensor, label_data_corrected_tensor
