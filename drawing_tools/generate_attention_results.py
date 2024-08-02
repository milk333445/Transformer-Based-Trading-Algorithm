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
import argparse


def stock_attention_visulize(stock_symbol, checkpoint, sequence_length=15, mode='val', visulize_attention=True):
    print('模式 : ', mode)
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
    attention_sum_df = pd.DataFrame()
    
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
            val_df = val_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'WILLR', 'CCI', 'CMO', 'ROC', 'EMA', 'SMA', 'TEMA', 'WMA', 'HT_TRENDLINE', 'SAR', 'ATR', 'TRANGE', 'AD', 'Label']]
            len_of_val_date = len(val_date)
            val_of_dataset = df_2d_tensor[len_of_train_date-14:len_of_train_date-14+len_of_val_date]
            val_of_dataset = val_of_dataset.to(device)
        elif mode == 'train':
            train_of_dataset = df_2d_tensor[0:len_of_train_date-14] # 這邊要減14是因為要預測的日期是從14天後開始
            train_date = train_date[14:]
            train_df = original_df[original_df['Date'].isin(train_date)]
            train_df = train_df.reset_index(drop=True)
            train_df = train_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'WILLR', 'CCI', 'CMO', 'ROC', 'EMA', 'SMA', 'TEMA', 'WMA', 'HT_TRENDLINE', 'SAR', 'ATR', 'TRANGE', 'AD', 'Label']]
        
        # 載入模型
        
        model = torch.load(model_path)
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
                indicator_sum_attention_df = visulize_attention_layer_map(
                                                            stock_symbol,
                                                            val_df['Label'],
                                                            val_date,
                                                            val_of_dataset, 
                                                            cache, 
                                                            layer_index=5, 
                                                            grid_index=0, 
                                                            grid_size=15, 
                                                            alpha=0.6,
                                                            mode=mode,
                                                            save=True,
                                                            show=False
                                                        )
                attention_sum_df = pd.concat([attention_sum_df, indicator_sum_attention_df], axis=0)
                attention_sum_df = attention_sum_df.reset_index(drop=True)
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
                        mode=mode,
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
    final_pred = pd.concat([final_pred, attention_sum_df], axis=1)
    final_pred = final_pred.reset_index(drop=True)
    return cache, final_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='vit_checkpoints')
    parser.add_argument('--save_folder_name', type=str, default='final_stock_attention_data_test')
    parser.add_argument('--save_name', type=str, default='stock_attention_combined_results.xlsx')
    args = parser.parse_args()
    if not os.path.exists(args.save_folder_name):
        os.makedirs(args.save_folder_name)
    save_path = os.path.join(args.save_folder_name, args.save_name)
    writer = pd.ExcelWriter(save_path, engine='xlsxwriter')
    all_stocks = ["^DJI", "^IXIC", "^N225", "^SOX", "^TWII", "AAPL", "AMZN", "ASML", "HPQ", "IBM", "INTC", "MSFT", "MU", "NFLX", "ORCL", "T", "TXN", "VZ", "GOOGL"]
    all_stocks = ['^DJI']
    for symbol in all_stocks:
        print('目前處理股票代號 : ', symbol)
        cache,final_pred = stock_attention_visulize(symbol, args.checkpoint, mode='val', visulize_attention=True)
        final_pred.to_excel(writer, sheet_name=symbol, index=False)
    writer._save()
