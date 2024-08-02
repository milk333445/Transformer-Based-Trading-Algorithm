import yfinance as yf
import pandas as pd
import os
import numpy as np
import talib
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_preprocessing import *
from dataloader import *
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def best_sampling_strategy(data):
    # Split the data into features and labels
    X = data.drop('Label', axis=1)
    y = data['Label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train and evaluate a model on the original data
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred_original = clf.predict(X_test)
    report_original = classification_report(y_test, y_pred_original, output_dict=True)
    baseline_f1 = (report_original['1.0']['f1-score'] + report_original['2.0']['f1-score']) / 2
    #print('baseline的平均F1:', baseline_f1)

    # Define possible sampling strategies
    label_distribution = y_train.value_counts()
    label_0_count = label_distribution[0]
    strategies = [
        {1.0: int(label_0_count * 0.1), 2.0: int(label_0_count * 0.1)},
        {1.0: int(label_0_count * 0.2), 2.0: int(label_0_count * 0.2)},
        {1.0: int(label_0_count * 0.3), 2.0: int(label_0_count * 0.3)},
        {1.0: int(label_0_count * 0.4), 2.0: int(label_0_count * 0.4)},
        {1.0: int(label_0_count * 0.5), 2.0: int(label_0_count * 0.5)}
    ]

    best_f1 = 0
    best_resampled_data = None

    # Try each sampling strategy and keep the best one
    for strategy in strategies:
        #print('目前策略:', strategy)
        smote = SMOTE(sampling_strategy=strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        clf.fit(X_resampled, y_resampled)
        y_pred_resampled = clf.predict(X_test)
        report_resampled = classification_report(y_test, y_pred_resampled, output_dict=True)
        current_f1 = (report_resampled['1.0']['f1-score'] + report_resampled['2.0']['f1-score']) / 2
        #print('目前平均F1:', current_f1)
        #print('---------------------------------------')

        if current_f1 > best_f1:
            best_strategy = strategy
            best_f1 = current_f1
            best_X_resampled, best_y_resampled = smote.fit_resample(X, y)
            best_resampled_data = pd.concat([best_X_resampled, best_y_resampled], axis=1)

    # Return the original data if no oversampling strategy improves the F1 score
    if best_f1 <= baseline_f1:
        return data, "原本的資料集"
    else:
        #print('最佳策略:', best_strategy)
        return best_resampled_data, best_strategy
    
    
def dataframe_to_tensors(df):
    # 提取特徵數據和標籤數據
    x_data = df.iloc[:, :-1].values
    label_data = df['Label'].values
    
    # 將 x_data 轉換為張量 [n, 1, 225]
    x_tensor = torch.tensor(x_data, dtype=torch.float32).view(-1, 1, 225)
    
    x_tensor = x_tensor.view(-1, 1, 15, 15)
    
    # 將標籤轉換為張量 [n, 1]
    label_tensor = torch.tensor(label_data, dtype=torch.long).view(-1, 1)
    
    return x_tensor, label_tensor

def tensor_to_dataframe_and_oversampling(df_2d_tensor, label_data_corrected_tensor, sampling_ratio=0.1):
    print('進入tensor_to_dataframe_and_oversampling')
    print('---------------------------------------')
    print('轉換前的資料集:')
    print(df_2d_tensor.shape)
    print(label_data_corrected_tensor.shape)
    new_tensor = df_2d_tensor.reshape(df_2d_tensor.shape[0], 1, 225)
    new_tensor = new_tensor.view(new_tensor.shape[0], 225)
    
    numpy_array = new_tensor.numpy()
    
    df = pd.DataFrame(numpy_array)
    df.columns = [f'col_{i}' for i in range(df.shape[1])]
    label = label_data_corrected_tensor.numpy()
    df['Label'] = label
    
    # 進行oversampling
    X = df.drop('Label', axis=1)
    y = df['Label']
    label_distribution = y.value_counts()
    label0_count = label_distribution[0]
    strategy = {1.0: int(label0_count * sampling_ratio), 2.0: int(label0_count * sampling_ratio)}
    X_resampled, y_resampled = RandomOverSampler(sampling_strategy=strategy, random_state=42).fit_resample(X, y)
    
    x_tensor, label_tensor = dataframe_to_tensors(pd.concat([X_resampled, y_resampled], axis=1))
    print('轉換後的資料集:')
    print(y_resampled.value_counts())
    print(x_tensor.shape)
    print(label_tensor.shape)
    return x_tensor, label_tensor



def tensor_to_dataframe_and_SMOTE(df_2d_tensor, label_data_corrected_tensor):
    print('進入tensor_to_dataframe_and_SMOTE')
    print('---------------------------------------')
    print('轉換前的資料集:')
    print(df_2d_tensor.shape)
    print(label_data_corrected_tensor.shape)
    # 將輸入型態由 PyTorch 張量轉換為 NumPy 數組且將其形狀從 [n, 1, 15, 15] 轉換為 [n, 225]
    new_tensor = df_2d_tensor.reshape(df_2d_tensor.shape[0], 1, 225)
    new_tensor = new_tensor.view(new_tensor.shape[0], 225)
    
    # 將張量轉換為 NumPy 數組
    numpy_array = new_tensor.numpy()
    
    # 使用 NumPy 數組創建 DataFrame
    df = pd.DataFrame(numpy_array)
    
    # 將 DataFrame 的列名設置為 col_0, col_1, ..., col_224
    df.columns = [f'col_{i}' for i in range(df.shape[1])]
    
    # 將標籤數據轉換為 NumPy 數組
    label = label_data_corrected_tensor.numpy()
    df['Label'] = label
    
    # 進行SMOTE
    best_resampled_data, best_strategy = best_sampling_strategy(df)
    
    # 將資料轉換回tensor
    x_tensor, label_tensor = dataframe_to_tensors(best_resampled_data)
    print('轉換後的資料集:')
    print(x_tensor.shape)
    print(label_tensor.shape)
    
    return x_tensor, label_tensor



