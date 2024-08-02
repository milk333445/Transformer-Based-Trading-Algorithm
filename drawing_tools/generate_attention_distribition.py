import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import argparse

def analyze_trend(data, results, threshold=0.2, window_size=10, change_threshold=1, n_clusters=5, selected_label='all'):
    attention_columns = [col for col in data.columns if 'attention' in col]

    def determine_future_trend_new(row, index, close_prices):
        if index + window_size < len(close_prices):
            future_price = close_prices[index + window_size]
            current_price = row['Close']
            change_percent = (future_price - current_price) / current_price * 100
            if change_percent > change_threshold:
                return 1  # buy
            elif change_percent < -change_threshold:
                return 2  # sell
            else:
                return 0 # empty
        return 0

    data['future_trend'] = [determine_future_trend_new(row, idx, data['Close']) for idx, row in data.iterrows()]
    
    high_accuracy_data = data[data['future_trend'] == data['pred_label']]
    
    # filter selected label
    if selected_label == 'buy':
        high_accuracy_data = high_accuracy_data[high_accuracy_data['pred_label'] == 1]
    elif selected_label == 'sell':
        high_accuracy_data = high_accuracy_data[high_accuracy_data['pred_label'] == 2]
    elif selected_label == 'hold':
        high_accuracy_data = high_accuracy_data[high_accuracy_data['pred_label'] == 0]
    else:
        pass
    
    
    if high_accuracy_data.isnull().values.any():
        high_accuracy_data = high_accuracy_data.dropna().reset_index(drop=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(high_accuracy_data[attention_columns])
    high_accuracy_data['cluster'] = clusters
    cluster_summaries = high_accuracy_data.groupby('cluster')[attention_columns].mean()
    # filter attention value(0.1)
    cluster_summaries[cluster_summaries < 0.1] = 0
    cluster_columns = cluster_summaries.columns
    # count
    for i in range(len(cluster_summaries)):
        data = cluster_summaries.iloc[i]
        for col in cluster_columns:
            if col not in results:
                results[col] = 0
            if data[col] > 0:
                results[col] += 1

    return results


if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./final_stock_attention_data/stock_attention_combined_results.xlsx')
    parser.add_argument('--start_date', type=str, choices=['financil_crisis', 'covid', 'others', 'all'], default='others')
    parser.add_argument('--selected_label', type=str, choices=['buy', 'sell', 'hold', 'all'], default='hold')

    args = parser.parse_args()
    data = pd.ExcelFile(args.data_path)
    sheet_names = data.sheet_names
    
    results_index = {}
    results_stock = {}
    
    index_columns = ["^DJI", "^IXIC", "^N225", "^SOX", "^TWII"]
    stock_columns = ["AAPL", "AMZN", "ASML", "HPQ", "IBM", "INTC", "MSFT", "MU", "NFLX", "ORCL", "T", "TXN", "VZ", "GOOGL"]
    
    for sheet_name in sheet_names:
        data_sheet = pd.read_excel(args.data_path, sheet_name)
        data_sheet['Date'] = pd.to_datetime(data_sheet['Date'])

        if args.start_date == 'financial_crisis':
            data_sheet = data_sheet[(data_sheet['Date'] >= '2007-01-01') & (data_sheet['Date'] <= '2010-12-31')]
        elif args.start_date == 'covid':
            data_sheet = data_sheet[(data_sheet['Date'] >= '2020-01-01') & (data_sheet['Date'] <= '2021-12-31')]
        elif args.start_date == 'others':
            data_sheet = data_sheet[~((data_sheet['Date'] >= '2020-01-01') & (data_sheet['Date'] <= '2021-12-31') | (data_sheet['Date'] >= '2007-01-01') & (data_sheet['Date'] <= '2010-12-31'))]
        else:
            pass
        
        
        data_sheet = data_sheet.reset_index(drop=True)
        
        if data_sheet.empty:
            continue    
        
        if sheet_name in index_columns:
            results_index = analyze_trend(data_sheet, results_index, selected_label=args.selected_label)
        elif sheet_name in stock_columns:
            results_stock = analyze_trend(data_sheet, results_stock, selected_label=args.selected_label)
            
    results_index = dict(sorted(results_index.items(), key=lambda item: item[1], reverse=True))
    results_stock = dict(sorted(results_stock.items(), key=lambda item: item[1], reverse=True))
    
    # normalize
    total_index = sum(results_index.values())
    total_stock = sum(results_stock.values())
    
    for key in results_index:
        results_index[key] = results_index[key] / total_index
    for key in results_stock:
        results_stock[key] = results_stock[key] / total_stock
    
    print('Index results:', results_index)
    print('Stock results:', results_stock)
    





