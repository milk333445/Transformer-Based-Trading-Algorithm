
from dataloader import *
import numpy as np
import torch
import os
from smote_processing import *
from data_preprocessing import *
import os
from dataloader import *
from smote_processing import *
from data_preprocessing import *
import time
import argparse



def main(args):
    
    df = fetch_and_adjust_stock_data(args.stock_symbol, args.start_date, args.end_date)
    
    current_directory = os.getcwd()
    output_directory = os.path.join(current_directory, args.output_directory, args.stock_symbol)
    
    # multi-process
    start_time = time.time()
    create_and_save_datasets(df, output_directory)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time of create dataset：{elapsed_time} 秒")
    
    start_time = time.time()
    generate_labels_for_folder(output_directory, [args.window_size], args.profit_threshold)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time of caculate labels：{elapsed_time} 秒")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create thesis origin dataset')
    parser.add_argument('--stock_symbol', type=str, default='^TWII')
    parser.add_argument('--start_date', type=str, default='1999-06-01')
    parser.add_argument('--end_date', type=str, default='2022-12-31')
    parser.add_argument('--output_directory', type=str, default='stock_price_data')
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--profit_threshold', type=int, default=1)
    args = parser.parse_args()


    main(args)
