import torch
import torchvision.transforms as T
from timm.models.vision_transformer import vit_small_patch32_224
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from ..train.train_engine import *
from ..data_preprocess.data_preprocessing import *
# from train_engine import *
# from dataloader import *
import numpy as np
# from smote_processing import *
# from data_preprocessing import *
# from train_engine import *
from financial_evaluation import *


def predict_stock_data(stock_symbol, checkpoint, sequence_length=15):
    current_directory = os.getcwd()
    output_directory = os.path.join(current_directory, 'stock_price_data', stock_symbol)
    if not os.path.exists(output_directory):
        print('Please run the training script first')
        return
    csv_files = os.listdir(output_directory)
    csv_files = csv_files[1:]
    csv_files.sort(key=extract_and_convert_to_int)
    csv_files = csv_files[1:]
    checkpoint_directory = os.path.join(output_directory, 'model', checkpoint)
    model_files = os.listdir(checkpoint_directory)

    final_pred = pd.DataFrame()
    
    for i in range(len(csv_files)):
        dataset_path = os.path.join(output_directory, csv_files[i])
        print('Current dataset path : ', dataset_path)
        model_path = os.path.join(checkpoint_directory, model_files[i])
        print('Loading model : ', model_path)
        original_df = pd.read_csv(dataset_path)
        dataset = preprocess_dataset_inferece(dataset_path)
        df_2d_tensor, label_data_corrected_tensor = preprocess_dataframe(dataset, sequence_length=sequence_length)
        val_date, train_date = get_date_inference(dataset_path)
        
        val_df = original_df[original_df['Date'].isin(val_date)]
        val_df = val_df.reset_index(drop=True)
        val_df = val_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Label']]
        len_of_val_date = len(val_date)
        len_of_train_date = len(train_date)
        val_of_dataset = df_2d_tensor[len_of_train_date-14:len_of_train_date-14+len_of_val_date]
        
        for i in range(len(val_of_dataset)):
            numpy_array = val_of_dataset[i][0].numpy()
            image = Image.fromarray((numpy_array * 255).astype(np.uint8))
            img_folder = os.path.join(current_directory, 'results', stock_symbol)
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
            image_name = val_df['Date'][i]
            image.save(os.path.join(img_folder, f'{image_name}.png'))
        
        model = torch.load(model_path)
        print('Model loaded')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        val_of_dataset = val_of_dataset.to(device)
        pred = model(val_of_dataset)
        pred_labels = torch.argmax(torch.softmax(pred, dim=1), dim=1)
        pred_labels = pred_labels.cpu().numpy()
    
        val_df['pred_label'] = pred_labels
        final_pred = pd.concat([final_pred, val_df], axis=0)
        del model
        del val_of_dataset
        torch.cuda.empty_cache()
        print(f'{i+1} csv file has been predicted')
        break
    return final_pred    

def calaulate_return_for_training(data_path, model, device, sequence_length=15):
    original_df = pd.read_csv(data_path)
    dataset = preprocess_dataset_inferece(data_path)
    
    df_2d_tensor, label_data_corrected_tensor = preprocess_dataframe(dataset, sequence_length=sequence_length)
    val_date, train_date = get_date_inference(data_path)
    val_df = original_df[original_df['Date'].isin(val_date)]
    val_df = val_df.reset_index(drop=True)
    val_df = val_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Label']]
    
    len_of_val_date = len(val_date)
    lan_of_train_date = len(train_date)
    val_of_dataset = df_2d_tensor[lan_of_train_date-14:lan_of_train_date-14+len_of_val_date]
    val_of_dataset = val_of_dataset.to(device)
    pred = model(val_of_dataset)
    pred_labels = torch.argmax(torch.softmax(pred, dim=1), dim=1)
    pred_labels = pred_labels.cpu().numpy()
    del val_of_dataset
    val_df['pred_label'] = pred_labels
    results = calaulate_return_and_more(val_df)
    return results
    
      
if __name__ == "__main__":
    stock_symbol = '^GSPC'
    checkpoint = 'vit_checkpoints_on_train_loss'
    final_pred = predict_stock_data(stock_symbol, checkpoint)
        
    
    
    
    
    
    
    