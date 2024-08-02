import torch
import torchvision.transforms as T
from timm.models.vision_transformer import vit_small_patch32_224
import json
import numpy as np
import matplotlib.pyplot as plt
from train_engine import *
from dataloader import *
from ..data_preprocess.data_preprocessing import *
from ..data_preprocess.smote_processing import *
# from smote_processing import *
# from data_preprocessing import *
from torch.utils.tensorboard import SummaryWriter


def run_training_with_pretrain_model(pretrain_model_path, stock_symbol, csv_files, output_directory, epochs=10, learning_rate=1e-3, batch_size=16, smote=False, writer=None, class_weights=None, oversampling=0.5):
    # load pretrain model
    pretrain_vit = torch.load(pretrain_model_path)
    
    dataset_path = os.path.join(output_directory, csv_files)
    TrainDataloader = DataLoader(CustomerTrainDataset(dataset_path, smote=smote, oversampling=oversampling), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    ValDataloader = DataLoader(CustomerValidationDataset(dataset_path), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    optimizer = torch.optim.Adam(params=pretrain_vit.parameters(), lr=learning_rate)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        
    vit_result = train(
        model=pretrain_vit,
        train_loader=TrainDataloader,
        val_loader=ValDataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
        stock_symbol=stock_symbol,
        output_directory=output_directory,
        csv_files=csv_files,
        writer=writer,
        data_path=dataset_path
    )
    

if __name__ == "__main__":
    writer = SummaryWriter('runs/vit')
    
    pretrain_stock_symbol = "^GSPC"
    pretrain_model_directory = os.path.join(os.getcwd(), 'stock_price_data', pretrain_stock_symbol, 'model\\vit_checkpoints')
    pretrain_model_files = os.listdir(pretrain_model_directory)
    pretrain_model_files.sort(key=extract_and_convert_to_int)
    
    
    stock_symbol = "^TWII"
    output_directory = os.path.join(os.getcwd(), 'stock_price_data', stock_symbol)
    csv_files = [f for f in os.listdir(output_directory) if f.endswith('.csv')]
    csv_files.sort(key=extract_and_convert_to_int)
    
    set_seeds()
    for i in range(len(csv_files)):
        if i == 0:
            continue
        print('Use pretrained model : ', pretrain_model_files[i])
        pretrain_model_path = os.path.join(pretrain_model_directory, pretrain_model_files[i])
        print('Ready to train stock symbol : ', stock_symbol)
        print('Ready to train csv file : ', csv_files[i])
        run_training_with_pretrain_model(pretrain_model_path, stock_symbol, csv_files[i], output_directory, epochs=150, learning_rate=1e-3, batch_size=8, smote=False, writer=writer, class_weights=None, oversampling=0.9)
        print(i+1, 'csv file has been trained')
        print('------------------------------------')
        torch.cuda.empty_cache()
        