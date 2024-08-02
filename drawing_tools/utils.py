import torch
import torchvision.transforms as T
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import talib
from matplotlib.patches import Rectangle
import os
import pandas as pd
def grid_show(to_shows, cols):
    rows = (len(to_shows)-1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows*8.5, cols*2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()

def visualize_head(att_map):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()
    
def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)

def gray2rgb(image):
    return np.repeat(image[...,np.newaxis],3,2)
    
def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
        
    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H/grid_size[0])
    delta_W = int(W/grid_size[1])
    
    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]
    
    padded_image = np.hstack((padding,image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W/4),int(delta_H/4)),'CLS', fill=(0,0,0)) # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask),cls_weight)
    cls_weight = cls_weight / max(np.max(mask),cls_weight)
    
    if len(padding.shape) == 3:
        padding = padding[:,:,0]
        padding[:,:] = np.min(mask)
    mask_to_pad = np.ones((1,1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H,  :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask
    
    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1],4))
    meta_mask[delta_H:,0: delta_W, :] = 1 
    
    return padded_image, padded_mask, meta_mask

def cls_padding_for_stock(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
        
    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H/grid_size[0])
    delta_W = int(W/grid_size[1])
    
    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image)
    padding = padding[:padding_h, :padding_w]
    
    padded_image = np.hstack((padding,image))
    #padded_image = Image.fromarray(padded_image)
    #draw = ImageDraw.Draw(padded_image)
    #draw.text((int(delta_W/4),int(delta_H/4)),'CLS', fill=(0,0,0)) # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask),cls_weight)
    cls_weight = cls_weight / max(np.max(mask),cls_weight)
    
    if len(padding.shape) == 3:
        padding = padding[:,:,0]
        padding[:,:] = np.min(mask)
    mask_to_pad = np.ones((1,1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H,  :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask
    
    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1],4))
    meta_mask[delta_H:,0: delta_W, :] = 1 
    
    return padded_image, padded_mask, meta_mask
    

def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]
    
    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    padded_image ,padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)
    
    if grid_index != 0: # adjust grid_index since we pad our image
        grid_index = grid_index + (grid_index-1) // grid_size[1]
        
    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1]+1))
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(grid_image)
    ax[0].axis('off')
    
    ax[1].imshow(grid_image)
    ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    ax[1].imshow(meta_mask)
    ax[1].axis('off')
    

def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    H,W = att_map.shape
    with_cls_token = False
      
    grid_image = highlight_grid(image, [grid_index], grid_size)
    
    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(grid_image)
    ax[0].axis('off')
    
    ax[1].imshow(grid_image)
    ax[1].imshow(mask/np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')
    plt.show()
    
def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a= ImageDraw.ImageDraw(image)
        a.rectangle([(y*w,x*h),(y*w+w,x*h+h)],fill =None,outline ='red',width =2)
    return image



def visulize_attention_map(stock_symbol, Labels, val_date, val_of_dataset, cache, layer_index, head_index, grid_index=0, grid_size=15, alpha=0.6, save=False, mode='val'):
    days_before = [f"before {i} days" for i in range(14, -1, -1)]
    days_before.insert(0, 'padding')
    indicators = ['RSI', 'MACD', 'WILLR', 'CCI', 'CMO', 'ROC', 'EMA', 'SMA', 'TEMA', 'WMA', 'HT_TRENDLINE', 'SAR', 'ATR', 'TRANGE', 'AD']
    for i in range(len(val_of_dataset)):
        Label = Labels[i]
        date = val_date[i]
        numpy_array = val_of_dataset[i].numpy()  # Assume val_of_dataset[i] is a tensor
        image = numpy_array.reshape(15, 15, 1)  # Reshape to 15 x 15 x 1

        # Extracting the attention map
        attention_maps = cache['Attention.forward'][layer_index][i]  # 第一layer的attention map
        attention_maps = np.expand_dims(attention_maps, axis=0)  # (1, 6, 226, 226)
        att_map = attention_maps[0, head_index, :, :]
        
        if not isinstance(grid_size, tuple):
                grid_size = (grid_size, grid_size)
        attention_map = att_map[grid_index]
        cls_weight = attention_map[0]
        mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
        mask = Image.fromarray(mask).resize((image.shape[0], image.shape[1]))
        H, W = image.shape[:2]
        delta_H = int(H/grid_size[0])
        delta_W = int(W/grid_size[1])
        padding_w = delta_W
        padding_h = H
        padding = np.ones_like(image)  # 生出一個跟image一樣大小的array，裡面的值都是1
        padding = padding[:padding_h, :padding_w] # 取出padding的部分
        padded_image = np.hstack((padding,image)) # 將padding跟image水平合併
        
        mask = mask / max(np.max(mask),cls_weight)
        cls_weight = cls_weight / max(np.max(mask),cls_weight)
        
        if len(padding.shape) == 3:
            padding = padding[:,:,0]
            padding[:,:] = np.min(mask)
        mask_to_pad = np.ones((1,1)) * cls_weight
        mask_to_pad = Image.fromarray(mask_to_pad)
        mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
        mask_to_pad = np.array(mask_to_pad)
        
        padding[:delta_H,  :delta_W] = mask_to_pad
        padded_mask = np.hstack((padding, mask))
        meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1],4))
        meta_mask[delta_H:,0: delta_W, :] = 1
        
        if grid_index != 0:
            grid_index = grid_index + (grid_index-1) // grid_size[1]
        row = grid_index // grid_size[1]
        col = grid_index % grid_size[1]
        x = col - 0.5
        y = row - 0.5
        fig, ax = plt.subplots(1, 2, figsize=(10, 7))

        # First subplot
        im0 = ax[0].imshow(padded_image)
        fig.colorbar(im0, ax=ax[0])
        ax[0].imshow(meta_mask)
        rect = Rectangle((x, y), 1, 1, fill=False, edgecolor='red', lw=2)
        ax[0].add_patch(rect)
        ax[0].text(0, 0, 'cls', color='black', ha='center', va='center')
        ax[0].set_yticks(ticks=np.arange(len(indicators)), labels=indicators)
        ax[0].set_xticks(ticks=np.arange(len(days_before)), labels=days_before, rotation=90)
        ax[0].set_title(f'Attention map of {date}')
        # Second subplot
        im1 = ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
        fig.colorbar(im1, ax=ax[1])
        ax[1].imshow(meta_mask)
        rect = Rectangle((x, y), 1, 1, fill=False, edgecolor='red', lw=2)
        ax[1].add_patch(rect)
        ax[1].text(0, 0, 'cls', color='black', ha='center', va='center')
        ax[1].set_yticks(ticks=np.arange(len(indicators)), labels=indicators)
        ax[1].set_xticks(ticks=np.arange(len(days_before)), labels=days_before, rotation=90)
        ax[1].set_title(f'Attention map of {date}')
        
        
        suptitle = f'Layer: {layer_index}, Head: {head_index}, Grid: {grid_index}, Label: {Label}'
        plt.suptitle(suptitle)
        plt.tight_layout()
        
        # Save the figure
        if save:
            save_folder = r'C:\Users\User\碩士論文資料處理\thesis_data_processing\資料處理\results'
            img_folder = os.path.join(save_folder, stock_symbol, mode)
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
            plt.savefig(os.path.join(img_folder, f'{date}.png'))
        
        plt.show()
        # 關閉圖片
        plt.close('all')
        
def process_attention_map(att_map, image, grid_size, grid_index=0):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]
    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.shape[0], image.shape[1]))
    
    H, W = image.shape[:2]
    delta_H = int(H/grid_size[0])
    delta_W = int(W/grid_size[1])
    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image)  # 生出一個跟image一樣大小的array，裡面的值都是1
    padding = padding[:padding_h, :padding_w] # 取出padding的部分
    padded_image = np.hstack((padding,image)) # 將padding跟image水平合併
    
    mask = mask / max(np.max(mask),cls_weight)
    cls_weight = cls_weight / max(np.max(mask),cls_weight)
    
    if len(padding.shape) == 3:
        padding = padding[:,:,0]
        padding[:,:] = np.min(mask)
    mask_to_pad = np.ones((1,1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)
    
    padding[:delta_H,  :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1],4))
    meta_mask[delta_H:,0: delta_W, :] = 1
    
    if grid_index != 0:
        grid_index = grid_index + (grid_index-1) // grid_size[1]
    row = grid_index // grid_size[1]
    col = grid_index % grid_size[1]
    x = col - 0.5
    y = row - 0.5
    
    return padded_image, padded_mask, meta_mask, x, y, mask



def visulize_attention_layer_map(stock_symbol, Labels, val_date, val_of_dataset, cache, layer_index, grid_index=0, grid_size=15, alpha=0.6, save=False, mode='val' , show=False, search_date=None):
    days_before = [f"before {i} days" for i in range(14, -1, -1)]
    days_before.insert(0, 'padding')
    indicators = ['RSI', 'MACD', 'WILLR', 'CCI', 'CMO', 'ROC', 'EMA', 'SMA', 'TEMA', 'WMA', 'HT_TRENDLINE', 'SAR', 'ATR', 'TRANGE', 'AD']
    attention_indicators = [indicator + '_attention' for indicator in indicators]
    all_data = []
    
    for i in range(len(val_of_dataset)):
        Label = Labels[i]
        date = val_date[i]
        if search_date is not None:
            if date not in search_date:
                continue
            else:
                print(f'Found {date}')
    
        numpy_array = val_of_dataset[i].numpy()  # Assume val_of_dataset[i] is a tensor
        image = numpy_array.reshape(15, 15, 1)  # Reshape to 15 x 15 x 1

        # Extracting the attention map
        attention_maps = cache['Attention.forward'][layer_index][i]  # 第一layer的attention map
        attention_maps = np.expand_dims(attention_maps, axis=0)  # (1, 6, 226, 226)
        # if show:
        #     num_heads = attention_maps.shape[1]
        #     fig, axs = plt.subplots(1, num_heads+1, figsize=(20, 3))
        #     for head_index in range(num_heads):
        #         att_map = attention_maps[0, head_index, :, :]
        #         padded_image, padded_mask, meta_mask, x, y, mask = process_attention_map(att_map, image, grid_size, grid_index)
        #         im1 = axs[head_index].imshow(padded_mask, alpha=alpha, cmap='rainbow')
        #         fig.colorbar(im1, ax=axs[head_index])
        #         axs[head_index].imshow(meta_mask)
        #         rect = Rectangle((x, y), 1, 1, fill=False, edgecolor='red', lw=2)
        #         axs[head_index].add_patch(rect)
        #         axs[head_index].text(0, 0, 'cls', color='black', ha='center', va='center')
        #         axs[head_index].set_yticks(ticks=np.arange(len(indicators)), labels=indicators)
        #         axs[head_index].set_xticks(ticks=np.arange(len(days_before)), labels=days_before, rotation=90)
        #         axs[head_index].set_title(f'Attention map of head {head_index}')
                
        # average attention map
        att_map = attention_maps.mean(axis=1)
        att_map = att_map[0]
        padded_image, padded_mask, meta_mask, x, y, mask = process_attention_map(att_map, image, grid_size, grid_index)
        indicator_sum_attention = np.sum(mask, axis=1)
        
        all_data.append([*indicator_sum_attention])
        if show:
            # im1 = axs[-1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
            # fig.colorbar(im1, ax=axs[-1])
            # axs[-1].imshow(meta_mask)
            # rect = Rectangle((x, y), 1, 1, fill=False, edgecolor='red', lw=2)
            # axs[-1].add_patch(rect)
            # axs[-1].text(0, 0, 'cls', color='black', ha='center', va='center')
            # axs[-1].set_yticks(ticks=np.arange(len(indicators)), labels=indicators)
            # axs[-1].set_xticks(ticks=np.arange(len(days_before)), labels=days_before, rotation=90)
            # axs[-1].set_title(f'Attention map of average head')
            im1 = plt.imshow(padded_mask, alpha=alpha, cmap='rainbow')
            plt.colorbar(im1)
            plt.imshow(meta_mask)
            rect = Rectangle((x, y), 1, 1, fill=False, edgecolor='red', lw=2)
            plt.gca().add_patch(rect)
            plt.text(0, 0, 'cls', color='black', ha='center', va='center')
            plt.yticks(ticks=np.arange(len(indicators)), labels=indicators)
            plt.xticks(ticks=np.arange(len(days_before)), labels=days_before, rotation=90)
            plt.title(f'Attention map of average head')
            plt.suptitle(f'Layer: {layer_index}, Grid: {grid_index}, Label: {Label}')
            #plt.suptitle(f'Date: {date}')
            plt.tight_layout()
            if save:
                save_folder = './tmp'
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                img_folder = os.path.join(save_folder, stock_symbol, mode)
                if not os.path.exists(img_folder):
                    os.makedirs(img_folder)
                plt.savefig(os.path.join(img_folder, f'{date}.png'))
            plt.show()
            plt.close('all')
    indicator_sum_attention_df = pd.DataFrame(all_data, columns=attention_indicators)
    indicator_sum_attention_df_normalized = indicator_sum_attention_df.apply(lambda x: x / x.sum(), axis=1)
    return indicator_sum_attention_df_normalized
        
            
            
            
            
        
        
        
def calaulate_rsi_resturn(df, window=14):
    rsi = talib.RSI(df['Close'], timeperiod=window)
    df['RSI'] = rsi
    
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
        entryLong = df['RSI'][i] < 30
        ## 出場
        exitShort = df['RSI'][i] > 70
        
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
    equity = pd.DataFrame({'profit': np.cumsum(profit_list)}, index=df.index)
    final_value = equity['profit'].iloc[-1]
    total_return = (equity['profit'].iloc[-1]) / fund
    years = len(df) / 252
    annualized_return = (((total_return + 1) ** (1 / years)) - 1) * 100
    return annualized_return
    
def calaulate_sma_resturn(df, window=14):
    sma = talib.SMA(df['Close'], timeperiod=window)
    df['SMA'] = sma
    
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
        entryLong = df['Open'][i] > df['SMA'][i-1] and df['Open'][i-1] < df['SMA'][i-2] if i > 1 else False
        ## 出場
        exitShort = df['Open'][i] < df['SMA'][i-1] and df['Open'][i-1] > df['SMA'][i-2] if i > 1 else False
        
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
    equity = pd.DataFrame({'profit': np.cumsum(profit_list)}, index=df.index)
    final_value = equity['profit'].iloc[-1]
    total_return = (equity['profit'].iloc[-1]) / fund
    years = len(df) / 252
    annualized_return = (((total_return + 1) ** (1 / years)) - 1) * 100
    return annualized_return
    

def calaulate_macd_resturn(df):
    dif, dem, histogram = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = histogram
    
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
        entryLong = df['MACD'][i] > 0 and df['MACD'][i-1] < 0 if i > 1 else False
        ## 出場
        exitShort = df['MACD'][i] < 0 and df['MACD'][i-1] > 0 if i > 1 else False
        
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
    equity = pd.DataFrame({'profit': np.cumsum(profit_list)}, index=df.index)
    final_value = equity['profit'].iloc[-1]
    total_return = (equity['profit'].iloc[-1]) / fund
    years = len(df) / 252
    annualized_return = (((total_return + 1) ** (1 / years)) - 1) * 100
    return annualized_return

def calaulate_bollinger_bands_resturn(df):
    upper_band, middle_band, lower_band = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['UpperBand'] = upper_band
    df['LowerBand'] = lower_band
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
        entryLong = df['Open'][i] < df['LowerBand'][i-1] and df['Open'][i-1] > df['LowerBand'][i-2] if i > 1 else False
        ## 出場
        exitShort = df['Open'][i] > df['UpperBand'][i-1] and df['Open'][i-1] < df['UpperBand'][i-2] if i > 1 else False
        
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
    equity = pd.DataFrame({'profit': np.cumsum(profit_list)}, index=df.index)
    final_value = equity['profit'].iloc[-1]
    total_return = (equity['profit'].iloc[-1]) / fund
    years = len(df) / 252
    annualized_return = (((total_return + 1) ** (1 / years)) - 1) * 100
    return annualized_return