from visualizer import get_local
get_local.activate()
import torch
import torchvision.transforms as T
from timm.models.vision_transformer import vit_small_patch32_224
#from timm.models.vision_transformer import vit_small_patch16_224
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from train_engine import *
import talib
from data_preprocessing import *
from utils import *
from matplotlib.patches import Rectangle
import argparse


def predict_stock_data(stock_symbol, period, checkpoint, sequence_length=15, mode='val', visulize_attention=True, visulize_final_result=True):
    print('模式 : ', mode)
    all_results = []
    output_directory = '../../stock_dataset/'
    output_directory = os.path.join(output_directory, stock_symbol)
    if not os.path.exists(output_directory):
        print('請先執行執行前處理程式')
        return
    csv_files = os.listdir(output_directory)
    csv_files = csv_files[1:]
    csv_files.sort(key=extract_and_convert_to_int)
    csv_files = csv_files[1:]
    checkpoint_directory = os.path.join(output_directory, 'model', checkpoint)
    model_files = os.listdir(checkpoint_directory)
    model_files.sort(key=extract_and_convert_to_int)
    
    # 建立績效資料夾
    result_folder = os.path.join(output_directory, "model", "result")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)    
    # 建立圖片資料夾
    visulize_folder = os.path.join(output_directory, "model", "result", "visulize")
    if not os.path.exists(visulize_folder):
        os.makedirs(visulize_folder)
        
        
    # 建立空的dataframe
    final_pred = pd.DataFrame()
    
    for i in range(len(csv_files)):
        if stock_symbol == 'GOOGL':
            if i < 5:
                continue
        dataset_path = os.path.join(output_directory, csv_files[i])
        print('目前處理資料集 : ', dataset_path)
        if stock_symbol == 'GOOGL':
            print('目前處理模型 : ', model_files[i-5])
            model_path = os.path.join(checkpoint_directory, model_files[i-5])
        else:
            print('目前處理模型 : ', model_files[i])
            model_path = os.path.join(checkpoint_directory, model_files[i])
        print('加載模型路徑 : ', model_path)
        original_df = pd.read_csv(dataset_path)
        dataset = preprocess_dataset_inferece(dataset_path)
        df_2d_tensor, label_data_corrected_tensor = preprocess_dataframe(dataset, sequence_length=sequence_length)
        val_date, train_date = get_date_inference(dataset_path)
        
        # 抓出原始資料
        len_of_train_date = len(train_date)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if mode == 'val':
            val_df = original_df[original_df['Date'].isin(val_date)]
            val_df = val_df.reset_index(drop=True)
            val_df = val_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Label']]
            len_of_val_date = len(val_date)
            val_of_dataset = df_2d_tensor[len_of_train_date-14:len_of_train_date-14+len_of_val_date]
            val_of_dataset = val_of_dataset.to(device)
        elif mode == 'train':
            train_of_dataset = df_2d_tensor[0:len_of_train_date-14] # 這邊要減14是因為要預測的日期是從14天後開始
            train_date = train_date[14:]
            train_df = original_df[original_df['Date'].isin(train_date)]
            train_df = train_df.reset_index(drop=True)
            train_df = train_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Label']]
        
        # 載入模型
        
        model = torch.load(model_path)
        #print('模型載入成功')
        model.to(device)
        model.eval()
        
        
        if mode == 'val':
            get_local.clear()
            pred = model(val_of_dataset)
            cache = get_local.cache
            pred_labels = torch.argmax(torch.softmax(pred, dim=1), dim=1)
            pred_labels = pred_labels.cpu().numpy()
            
            
            # 視覺化
            
            #attention_maps = cache['Attention.forward']
            #attention_maps = attention_maps[5][5]
            #attention_maps = np.expand_dims(attention_maps, axis=0)
            #visualize_heads(attention_maps, cols=4)
            
            
            val_df['pred_label'] = pred_labels
            final_pred = pd.concat([final_pred, val_df], axis=0) # 垂直合併
            # reset index
            final_pred = final_pred.reset_index(drop=True)
            val_of_dataset = val_of_dataset.cpu()
            if visulize_attention:
                visulize_attention_map(
                    stock_symbol,
                    val_df['Label'],
                    val_date,
                    val_of_dataset, 
                    cache, 
                    layer_index=5, 
                    head_index=5, 
                    grid_index=0, 
                    grid_size=15, 
                    alpha=0.6,
                    mode=mode,
                    save=False
                )
            del model, pred, val_of_dataset
        elif mode == 'train':
            batch_size = 100
            n = len(train_of_dataset)
            all_pred = []
            print('開使分批預測')
            for i in range(0, n, batch_size):
                print(f'目前處理第{i+1}筆batch')
                batch = train_of_dataset[i:i+batch_size].to(device)
                get_local.clear()
                pred = model(batch)
                pred = pred.detach().cpu()
                all_pred.append(pred)
                
                if visulize_attention:
                    visulize_attention_map(
                        stock_symbol,
                        train_df['Label'][i:i+batch_size],
                        train_date[i:i+batch_size],
                        batch.cpu(),
                        get_local.cache,
                        layer_index=5, 
                        head_index=5, 
                        grid_index=0, 
                        grid_size=15, 
                        alpha=0.6,
                        mode=mode
                    )
                del batch, pred
                torch.cuda.empty_cache()
            pred = torch.cat(all_pred, dim=0)
            pred_labels = torch.argmax(torch.softmax(pred, dim=1), dim=1)
            pred_labels = pred_labels.cpu().numpy()
            train_df['pred_label'] = pred_labels
            del train_of_dataset, model
        torch.cuda.empty_cache()
        print(f'第{i+1}個資料集預測完成')  
        
        # 視覺化財務績效
        if mode == 'val':
            results = calaulate_return_and_visulize(stock_symbol, val_df, visulize=visulize_final_result, save_folder=visulize_folder)
            all_results.append(results)
        elif mode == 'train':
            pass

    if period == '2007-2010':
        final_pred = final_pred.loc[(final_pred['Date'].str.startswith('2007')) | (final_pred['Date'].str.startswith('2008')) | (final_pred['Date'].str.startswith('2009')) | (final_pred['Date'].str.startswith('2010'))]
    elif period == '2020-2021':
        final_pred = final_pred.loc[(final_pred['Date'].str.startswith('2020')) | (final_pred['Date'].str.startswith('2021'))]
    elif period == 'All':
        pass
    #final_pred = final_pred.loc[(final_pred['Date'].str.startswith('2007')) | (final_pred['Date'].str.startswith('2008')) | (final_pred['Date'].str.startswith('2009')) | (final_pred['Date'].str.startswith('2010'))] 
    final_pred = final_pred.reset_index(drop=True)
    #print(final_pred)
    results_all = calaulate_return_and_visulize(stock_symbol, final_pred, visulize=visulize_final_result, save_folder=visulize_folder)
    #print('總體財務績效 : ', results_all)
    #final_pred.to_csv(f'{stock_symbol}_final_pred.csv')
    #result_df = pd.DataFrame(all_results)
    result_df = pd.DataFrame([results_all])
    desired_columns = ['year', 'final_value', 'annualized_return', "buy_and_hold_annualized_return", 'rsi_annualized_return', 'sma_annualized_return', "macd_annualized_return", "bollinger_annualized_return", 'total_trade', 
                   'average_profit_per_trade', 'sucess_trade_rate', 
                   'average_trade_duration', 'total_non_trading_days', 
                   'non_trading_days_rate', 'max_profit_pct', 'max_loss_pct']
    result_df = result_df[desired_columns]
    result_df.set_index('year', inplace=True)
    average_values = result_df.mean()
    average_df =pd.DataFrame([average_values], index=['average'])
    result_df = pd.concat([result_df, average_df])
    
    average_2007_to_2010 = result_df.loc["2007":"2010"].mean()
    average_2020_to_2021 = result_df.loc["2020":"2021"].mean()
    average_df = pd.DataFrame([average_2007_to_2010, average_2020_to_2021], index=['average(2007-2010)', 'average(2020-2021)'])
    result_df = pd.concat([result_df, average_df])
    
    # 儲存csv
    #result_df.to_csv(os.path.join(result_folder, f'{stock_symbol}_result.csv'))
    return result_df, cache

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
    all_stocks = ["^DJI", "^IXIC", "^N225", "^SOX", "^TWII", "AAPL", "AMZN", "ASML", "GOOGL", "HPQ", "IBM", "INTC", "MSFT", "MU", "NFLX", "ORCL", "T", "TXN", "VZ"]
    all_stocks = ['^DJI']
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='vit_checkpoints')
    parser.add_argument('--save_folder_name', type=str, default='./specific_period_results')
    parser.add_argument('--period', type=str, choices=['2007-2010', '2020-2021', 'All'], default='2007-2010')
    args = parser.parse_args()

    if not os.path.exists(args.save_folder_name):
        os.makedirs(args.save_folder_name)
    save_path = os.path.join(args.save_folder_name, f'{args.period}_results.xlsx')
    with pd.ExcelWriter(save_path) as writer:
        for symbol in all_stocks:
            print('目前處理股票代號 : ', symbol)
            result_df, cache = predict_stock_data(symbol, args.period, args.checkpoint, mode='val', visulize_attention=False, visulize_final_result=False)
            result_df.to_excel(writer, sheet_name=symbol)

