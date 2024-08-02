from tqdm.auto import tqdm
from typing import List, Tuple, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
from pathlib import Path
import torch
import re
from ..test.valuation import calaulate_return_and_more
# from valuation import calaulate_return_for_training

def extract_and_convert_to_int(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return 0

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
):
    # switch to train mode
    model.train()
    train_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        del X, y, y_pred
    train_loss = train_loss / len(dataloader)
    
    return train_loss


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    data_path = None
):
    # switch to test mode
    model.eval()
    test_loss, test_acc, weighted_f1 = 0.0, 0.0, 0.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        test_pred_logits = model(X)
        loss = loss_fn(test_pred_logits, y)
        test_loss += loss.item()
        # calculate measurements
        test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
        test_acc += torch.sum(test_pred_labels == y).item()/len(test_pred_labels)
        y_np = y.cpu().numpy()
        test_pred_labels_np = test_pred_labels.cpu().numpy()
        weighted_f1 += f1_score(y_np, test_pred_labels_np, average='weighted')

        del X, y, test_pred_logits, test_pred_labels
    
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    weighted_f1 = weighted_f1 / len(dataloader)
    
    finanacial_results = calaulate_return_for_training(data_path, model, device)
    
    return test_loss, test_acc, weighted_f1, finanacial_results
    
    
def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    stock_symbol: str,
    output_directory: str,
    csv_files: str,
    writer,
    data_path: None
):
    # store last epoch model
    model_dir_last_epoch = os.path.join(output_directory, 'model\\vit_checkpoints_on_last_epoch')
    if not os.path.exists(model_dir_last_epoch):
        os.makedirs(model_dir_last_epoch)
        print("Successfully created the directory: ", model_dir_last_epoch)
    else:
        print("Directory already exists: ", model_dir_last_epoch)
    
    # store best train loss model
    model_dir_train_loss = os.path.join(output_directory, 'model\\vit_checkpoints_on_train_loss')
    if not os.path.exists(model_dir_train_loss):
        os.makedirs(model_dir_train_loss)
        print("Successfully created the directory: ", model_dir_train_loss)
    else:
        print("Directory already exists: ", model_dir_train_loss)
        
    results = {
        'train_loss':[],
        'val_loss':[],
        'val_acc':[],
        'weighted_f1':[]
    }
    best_train_loss = 9999
    model.to(device)
    
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model, train_loader, loss_fn, optimizer, device)
        writer.add_scalar(f'{csv_files}/Loss/train', train_loss, epoch)
        test_loss, test_acc, weighted_f1, _ = test_step(model, val_loader, loss_fn, device, data_path)
        writer.add_scalar(f'{csv_files}/Loss/test', test_loss, epoch)
    
        if epoch == len(range(epochs))-1:
            torch.save(model, os.path.join(model_dir_last_epoch, f'best_vit_{stock_symbol}_{csv_files}.pth'))
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model, os.path.join(model_dir_train_loss, f'best_vit_{stock_symbol}_{csv_files}.pth'))
        
        print(
            f'Epoch:{epoch+1}/{epochs} |'
            f'train_loss:{train_loss:.4f} |'
            f'val_loss:{test_loss:.4f} |'
        )
        results['train_loss'].append(train_loss)
        results['val_loss'].append(test_loss)
        results['val_acc'].append(test_acc)
        results['weighted_f1'].append(weighted_f1)
    
    return results
    
