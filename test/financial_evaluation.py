import pandas as pd
import numpy as np
import os
from ..data_preprocess.data_preprocessing import *
# from data_preprocessing import *
import matplotlib.pyplot as plt
def calaulate_return_and_more(df):
    label_mapping = {
        0: 'HOLD',
        1: 'BUY',
        2: 'SELL'
    }
    df['pred_label'] = df['pred_label'].map(lambda x: label_mapping[x])
    
    fund = 100000
    money = 100000
    BS = None
    buy = []
    sell = []
    profit_list = [0]
    profit_list_realized = []
    trade_sucess_count = 0
    total_trading_days = 0
    total_non_trading_days = 0
    
    for i in range(len(df)):
        if i == len(df) - 1:
            if BS == 'B':
                pl_round = tempSize * (df['Open'][i] - df['Open'][t])
                sell.append(i)
                BS = None
                
                profit_realized = pl_round
                profit_list_realized.append(profit_realized)
                if profit_realized > 0:
                    trade_sucess_count += 1
            break
        
        entryLong = df['pred_label'][i] == 'BUY'
        exitShort = df['pred_label'][i] == 'SELL'
        
        if BS is None:
            profit_list.append(0)
            if entryLong:
                tempSize = money // df['Open'][i+1]
                BS = 'B'
                t = i+1
                buy.append(t+1)
                
        elif BS == 'B':
            profit = tempSize * (df['Open'][i+1] - df['Open'][i])
            profit_list.append(profit)
            total_trading_days += 1
            
            if exitShort:
                pl_round = tempSize * (df['Open'][i+1] - df['Open'][t])
                sell.append(i+1)
                BS = None
                
                profit_realized = pl_round
                profit_list_realized.append(profit_realized)
                
                if profit_realized > 0:
                    trade_sucess_count += 1
    equity = pd.DataFrame({'profit': np.cumsum(profit_list)}, index=df.index)
    final_value = equity['profit'].iloc[-1]
    total_return = (equity['profit'].iloc[-1]) / fund
    
    total_trade = len(buy)
    average_profit_per_trade = (final_value / total_trade) if total_trade != 0 else 0
    sucess_trade_rate = (trade_sucess_count / total_trade) * 100 if total_trade != 0 else 0
    average_trade_duration = (total_trading_days / total_trade) if total_trade != 0 else 0
    total_non_trading_days = len(df) - total_trading_days
    non_trading_days_rate = (total_non_trading_days / len(df)) * 100
    
    # annualized return
    years = len(df) / 252
    annualized_return = (((total_return + 1) ** (1 / years)) - 1) * 100
    
    return {
        'final_value': final_value,
        'annualized_return': annualized_return,
        'total_trade': total_trade,
        'average_profit_per_trade': average_profit_per_trade,
        'sucess_trade_rate': sucess_trade_rate,
        'average_trade_duration': average_trade_duration,
        'total_non_trading_days': total_non_trading_days,
        'non_trading_days_rate': non_trading_days_rate,
        
    }
                     
def print_results(results):
    print('Final Value: {}'.format(results['final_value']))
    print('Annualized Return: {}%'.format(results['annualized_return']))
    print('Total Trades: {}'.format(results['total_trade']))
    print('Average Profit per Trade: {}'.format(results['average_profit_per_trade']))
    print('Success Trade Rate: {}'.format(results['sucess_trade_rate']))
    print('Average Trade Duration: {} days'.format(results['average_trade_duration']))
    print('Total Non-Trading Days: {}'.format(results['total_non_trading_days']))
    print('Non-Trading Days Rate: {}%'.format(results['non_trading_days_rate']))

if __name__ == "__main__":
    test_data = pd.read_csv(r'C:\Users\User\碩士論文資料處理\thesis_data_processing\資料處理\stock_price_data\^GSPC\^GSPC_results.csv')
    print(test_data['pred_label'].value_counts())
    results = calaulate_return_and_more(test_data)
    print_results(results)
    
    