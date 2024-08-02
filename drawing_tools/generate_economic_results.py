import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils import *


    
def calculate_specific_interval_return(foldername, interval, savefoldername):
    if not os.path.exists(savefoldername):
        os.makedirs(savefoldername)
        
    savename = 'combined_results.xlsx'
    savepath = os.path.join(savefoldername, savename)
    writer = pd.ExcelWriter(savepath, engine='xlsxwriter')
        
    file_names = os.listdir(foldername)
    for filename in file_names:
        if filename.endswith('.csv'):
            stock_symbol = filename[:-4].split('_')[0]
            df = pd.read_csv(os.path.join(foldername, filename), index_col=0)
            df['Year'] = pd.to_datetime(df['Date']).dt.year
            specific_df = df[df['Year'].isin(interval)].reset_index(drop=True)
            try:
                result = calaulate_return_and_visulize(stock_symbol=stock_symbol, df=specific_df, visulize=False)
                result_df = pd.DataFrame([result])
                
                result_df.to_excel(writer, sheet_name=stock_symbol)
            except:
                print(f"Error: {stock_symbol}")
    writer._save()


def calaulate_return_and_visulize(stock_symbol, df, visulize=True, save_folder=None):

    # 轉換資料
    label_mapping = {
        0: 'HOLD',
        1: 'BUY',
        2: 'SELL'
    }
    df['pred_label'] = df['pred_label'].map(lambda x: label_mapping[x])
    # 計算報酬率
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
    max_profit_pct = 0
    max_loss_pct = 0
    
    
    # 計算買入持有的報酬率
    buy_part = money/df['Open'].iloc[0]
    sell_price = df['Open'].iloc[-1]
    hold_profit = buy_part * sell_price  - money
    hold_return = hold_profit / fund
    
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
        ## 進場
        entryLong = df['pred_label'][i] == 'BUY'
        ## 出場
        exitShort = df['pred_label'][i] == 'SELL'
        
        if BS is None:
            profit_list.append(0)
            if entryLong:
                tempSize = money // df['Open'][i+1]
                BS = 'B'
                t = i+1
                if t == len(df) - 1:
                    break
                buy.append(t+1)
                
        elif BS == 'B':
            profit = tempSize * (df['Open'][i+1] - df['Open'][i])
            profit_list.append(profit)
            total_trading_days += 1
            
            if exitShort:
                pl_round = tempSize * (df['Open'][i+1] - df['Open'][t])
                t = i+1
                if t == len(df) - 1:
                    break
                sell.append(i+1)
                BS = None
                
                profit_realized = pl_round
                profit_list_realized.append(profit_realized)
                
                if profit_realized > 0:
                    trade_sucess_count += 1
                    
    # 計算盈虧百分比             
    
    for i in range(len(profit_list_realized)):
        if profit_list_realized[i] > 0:
            profit_pct = (profit_list_realized[i] / money) * 100
            max_profit_pct = max(max_profit_pct, profit_pct)
        else:
            loss_pct = (profit_list_realized[i] / money) * 100
            max_loss_pct = min(max_loss_pct, loss_pct)
    
    
    equity = pd.DataFrame({'profit': np.cumsum(profit_list)}, index=df.index)
    final_value = equity['profit'].iloc[-1]
    total_return = (equity['profit'].iloc[-1]) / fund
    
    # more
    total_trade = len(buy)
    average_profit_per_trade = (final_value / total_trade) if total_trade != 0 else 0
    sucess_trade_rate = (trade_sucess_count / total_trade) * 100 if total_trade != 0 else 0
    average_trade_duration = (total_trading_days / total_trade) if total_trade != 0 else 0
    total_non_trading_days = len(df) - total_trading_days
    non_trading_days_rate = (total_non_trading_days / len(df)) * 100
    
    # annualized return
    years = len(df) / 252
    annualized_return = (((total_return + 1) ** (1 / years)) - 1) * 100
    buy_and_hold_annualized_return = (((hold_return + 1) ** (1 / years)) - 1) * 100
    
    # rsi sma
    rsi_annualized_return = calaulate_rsi_resturn(df, window=14)
    sma_annualized_return = calaulate_sma_resturn(df,  window=14)
    macd_annualized_return = calaulate_macd_resturn(df)
    bollinger_annualized_return = calaulate_bollinger_bands_resturn(df)
    
    
    year = df['Date'][0].split('-')[0]
    if visulize:
        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Open Price', color=color)
        ax1.plot(df['Open'], label='Open Price', color=color)
        ax1.scatter(buy, df['Open'][buy], label='Buy', color='green', marker='^', s=100)
        ax1.scatter(sell, df['Open'][sell], label='Sell', color='red', marker='v', s=100)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # 第二個y軸
        color = 'tab:red'
        ax2.set_ylabel('Cumulative Profit', color=color)
        ax2.plot(equity['profit'], label='Cumulative Profit', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        textstr = '\n'.join((
        f"Total Trades: {total_trade}",
        f"Success Trade Rate: {sucess_trade_rate:.2f}%",
        f"Annualized Return: {annualized_return:.2f}%",
        f"Average Profit per Trade: {average_profit_per_trade:.2f}"))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        #year = df['Date'][0].split('-')[0]
        
        plt.title(f'{stock_symbol} {year} Cumulative Profit')
        fig.tight_layout()
        #plt.show()
        if save_folder is not None:
            plt.savefig(os.path.join(save_folder, f'{stock_symbol}_{year}_cumulative_profit.png'))
        #plt.close('all')
        
    return {
        'year': year,
        'final_value': final_value,
        'annualized_return': annualized_return,
        "buy_and_hold_annualized_return": buy_and_hold_annualized_return,
        'rsi_annualized_return': rsi_annualized_return,
        'sma_annualized_return': sma_annualized_return,
        "macd_annualized_return": macd_annualized_return,
        "bollinger_annualized_return": bollinger_annualized_return,
        'total_trade': total_trade,
        'average_profit_per_trade': average_profit_per_trade,
        'sucess_trade_rate': sucess_trade_rate,
        'average_trade_duration': average_trade_duration,
        'total_non_trading_days': total_non_trading_days,
        'non_trading_days_rate': non_trading_days_rate, 
        'max_profit_pct': max_profit_pct,
        'max_loss_pct': max_loss_pct
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', type=str, default='final_raw_data')
    parser.add_argument('--save_folder_names', type=list, default=['economic_cycle_up_test', 'economic_cycle_down_test', 'economic_cycle_normal_test'])
    args = parser.parse_args()
    interval_periods = [
        [2006, 2009, 2010, 2012, 2013, 2014, 2017, 2019, 2020, 2021],
        [2008, 2022],
        [2007, 2011, 2015]
    ]
    for save_folder_name, interval in zip(args.save_folder_names, interval_periods):
        print(f"Calculating {save_folder_name}")
        calculate_specific_interval_return(args.folder_name, interval, save_folder_name)
        print(f"Finish {save_folder_name}")