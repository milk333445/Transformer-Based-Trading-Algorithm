import torch
import torchvision.transforms as T
from timm.models.vision_transformer import vit_small_patch32_224
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from train_engine import *
from dataloader import *
import numpy as np
import torch
from ..data_preprocess.smote_processing import *
from ..data_preprocess.data_preprocessing import * 
# from smote_processing import *
# from data_preprocessing import *
from torch.utils.tensorboard import SummaryWriter

def run_training(stock_symbol, csv_files, output_directory, epochs=10, learning_rate=1e-3, batch_size=16, smote=False, writer=None, class_weights=None, oversampling=0.1):
    vit = vit_small_patch32_224(pretrained=False)
    vit.head = torch.nn.Linear(384, 3)
 
    dataset_path = os.path.join(output_directory, csv_files)
    TrainDataloader = DataLoader(CustomerTrainDataset(dataset_path, smote=smote, oversampling=oversampling), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    ValDataloader = DataLoader(CustomerValidationDataset(dataset_path), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    OutOfSampleDataloader = DataLoader(CustomerOutOfSampleDataset(dataset_path), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.Adam(params=vit.parameters(), lr=learning_rate)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    vit_result = train(
        model=vit,
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
    # open tensorboard
    writer = SummaryWriter('runs/vit')
    
    stock_symbol = "^GSPC"
    output_directory = os.path.join(os.getcwd(), 'stock_price_data', stock_symbol)
    csv_files = [f for f in os.listdir(output_directory) if f.endswith('.csv')]
    csv_files.sort(key=extract_and_convert_to_int)
    run_training(stock_symbol, csv_files[0], output_directory, epochs=500, learning_rate=1e-3, batch_size=8, smote=False, writer=writer, oversampling=0.5)
