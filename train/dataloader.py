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
from ..data_preprocess.data_preprocessing import *
from ..data_preprocess.smote_processing import *
# from data_preprocessing import *
# from smote_processing import *

class CustomerTrainDataset(Dataset):
    def __init__(self, dataset_path, sequence_length=15, smote=False, oversampling=None):
        super(CustomerTrainDataset, self).__init__()
        # Create dataset
        df_train, df_validation, df_out_of_sample = preprocess_dataset_to_train(dataset_path)
        # Convert to 2D Tensor(for training set)
        df_train_2d_tensor, train_label_data_corrected_tensor = preprocess_dataframe(df_train, sequence_length=sequence_length)
        # If smote is True, then apply SMOTE
        if smote:
            df_train_2d_tensor, train_label_data_corrected_tensor = tensor_to_dataframe_and_SMOTE(df_train_2d_tensor, train_label_data_corrected_tensor)
            self.df_2d_tensor = df_train_2d_tensor
            # Convert label to one-dimensional tensor
            train_label_data_corrected_tensor = torch.squeeze(train_label_data_corrected_tensor)
            self.label_data_corrected_tensor = train_label_data_corrected_tensor
        
        elif oversampling is not None:
            df_train_2d_tensor, train_label_data_corrected_tensor = tensor_to_dataframe_and_oversampling(df_train_2d_tensor, train_label_data_corrected_tensor, oversampling)
            self.df_2d_tensor = df_train_2d_tensor
            # Convert label to one-dimensional tensor
            train_label_data_corrected_tensor = torch.squeeze(train_label_data_corrected_tensor)
            self.label_data_corrected_tensor = train_label_data_corrected_tensor
        
        else:
            self.df_2d_tensor = df_train_2d_tensor
            # Convert label to integer
            train_label_data_corrected_tensor = train_label_data_corrected_tensor.to(torch.long)
            # Convert label to one-dimensional tensor
            train_label_data_corrected_tensor = torch.squeeze(train_label_data_corrected_tensor)
            self.label_data_corrected_tensor = train_label_data_corrected_tensor
    
    def __getitem__(self, index):
        return self.df_2d_tensor[index], self.label_data_corrected_tensor[index]
    
    def __len__(self):
        return len(self.df_2d_tensor)
    
    
class CustomerValidationDataset(Dataset):
    def __init__(self, dataset_path, sequence_length=15):
        super(CustomerValidationDataset, self).__init__()
        # Create dataset
        df_train, df_validation, df_out_of_sample = preprocess_dataset_to_train(dataset_path)
        # Convert to 2D Tensor(for validation set)
        df_validation_2d_tensor, validation_label_data_corrected_tensor = preprocess_dataframe(df_validation, sequence_length=sequence_length)
        self.df_2d_tensor = df_validation_2d_tensor
        # Convert label to integer
        validation_label_data_corrected_tensor = validation_label_data_corrected_tensor.to(torch.long)
        # Convert label to one-dimensional tensor
        validation_label_data_corrected_tensor = torch.squeeze(validation_label_data_corrected_tensor)
        self.label_data_corrected_tensor = validation_label_data_corrected_tensor
    
    def __getitem__(self, index):
        return self.df_2d_tensor[index], self.label_data_corrected_tensor[index]
    
    def __len__(self):
        return len(self.df_2d_tensor)
    
class CustomerOutOfSampleDataset(Dataset):
    def __init__(self, dataset_path, sequence_length=15):
        super(CustomerOutOfSampleDataset, self).__init__()
        # Create dataset
        df_train, df_validation, df_out_of_sample = preprocess_dataset_to_train(dataset_path)
        # Convert to 2D Tensor(for out-of-sample set)
        df_out_of_sample_2d_tensor, out_of_sample_label_data_corrected_tensor = preprocess_dataframe(df_out_of_sample, sequence_length=sequence_length)
        self.df_2d_tensor = df_out_of_sample_2d_tensor
        # Convert label to integer
        out_of_sample_label_data_corrected_tensor = out_of_sample_label_data_corrected_tensor.to(torch.long)
        # Convert label to one-dimensional tensor
        out_of_sample_label_data_corrected_tensor = torch.squeeze(out_of_sample_label_data_corrected_tensor)
        self.label_data_corrected_tensor = out_of_sample_label_data_corrected_tensor
    
    def __getitem__(self, index):
        return self.df_2d_tensor[index], self.label_data_corrected_tensor[index]
    
    def __len__(self):
        return len(self.df_2d_tensor)