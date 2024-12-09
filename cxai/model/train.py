import os
import random
from datetime import datetime
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from cxai.model.dataloader.gtzan_dataset import get_data_loaders
from cxai.model.create_model import VGGType


def fit(
    model: nn.Module, 
    loss_func: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    dataloaders: Dict[str,torch.utils.data.DataLoader], 
    num_epochs: int = 100, 
    device: str | torch.device = torch.device('cpu'), 
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None, 
    from_epoch: int = 0, 
    model_path: str | None = None,
    save_step: int = 100,
    is_gtzan: bool = True,
) -> None:
    """Performs training of a neural network.
    
    Args:
        model (nn.Module): Neural network model.
        loss_func (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        dataloaders (Dict[str,torch.utils.data.DataLoader]): Dataloader for different splits ['train', 'valid'].
        num_epochs (int, optional): Number of epochs to train the network.
        device (str | torch.device, optional): Device.
        scheduler (torch.optim.lr_scheduler._LRScheduler | None, optional): Learning rate scheduler.
        from_epoch (int, optional): Nessecary if a pretrained model should be fine tuned.
        model_path (str | None, optional): Path to the pretrained model.
        save_step (int, optional): Number of training epochs after model checkpoint should be saved.
        is_gtzan (bool, optional): 
    """
    device = torch.device(device) if isinstance(device, str) else device
    model.to(device)
    
    print('Starting training...')

    # if we train new model, create model path and folder
    if from_epoch == 0:
        time = datetime.now()
        model_path = model_path if model_path else os.path.join('../models', str(time))
        if not os.path.exists(model_path):
            os.makedirs(model_path)

    train_losses, valid_losses, train_acc, valid_acc = [], [], [], []
    tbar = tqdm(range(1, num_epochs+1))

    # start main training loop
    for epoch in tbar:
        # iterate train and validation pahse in each epoch
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval() 

            # track loss
            running_loss = 0.0
            running_acc = 0.0
            for xb, yb in dataloaders[phase]:
                xb, yb = xb.to(device), yb.to(device)
                
                if phase == 'valid':
                    if is_gtzan:
                        # case GTZAN, test data is in the form (batch x chunks x channel x mel-bins x time-bins)
                        b, chunks, c, f, t = xb.size()
                        yb = yb.view(-1, chunks*b)
                    else:
                        # case toy data
                        _, c, f, t = xb.size()
                    xb = xb.view(-1, c, f, t)
                    
                with torch.set_grad_enabled(phase == 'train'):
                    loss, acc = loss_batch(
                        model, 
                        loss_func, 
                        xb, 
                        yb, 
                        opt=optimizer, 
                        phase=phase
                    )
                # log stats
                running_loss += loss
                running_acc += acc
            
            # take the average loss per epoch from the sum of losses
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_acc / len(dataloaders[phase])
            train_losses.append(epoch_loss) if phase=='train' else valid_losses.append(epoch_loss)
            train_acc.append(epoch_acc) if phase=='train' else valid_acc.append(epoch_acc)

            if phase == 'train':
                # update learning rate scheduler
                if scheduler:
                    scheduler.step()
                train_loss = epoch_loss
                train_acc_ = epoch_acc
        
        # Create tbar description
        descr = f'TRAIN acc: {train_acc_*100:.2f}% - loss: {train_loss:.4f}'
        descr += f' || VALID acc: {epoch_acc*100:.2f}% - loss: {epoch_loss:.4f}'
        tbar.set_description(descr)
        #current_lr = scheduler.get_last_lr()[0]
    
        # deep copy the model
        if epoch % save_step == 0:
            save_checkpoint(
                model_path, 
                model.state_dict(), 
                optimizer.state_dict(), 
                epoch=epoch+from_epoch, 
            )
            save_train_stats(
                model_path, 
                train_losses, 
                train_acc, 
                valid_losses, 
                valid_acc,
                from_epoch
            )


def loss_batch(
    model: nn.Module, 
    loss_func: callable, 
    xb: torch.Tensor, 
    yb: torch.Tensor, 
    optimizer: torch.optim.Optimizer, 
    phase: str = 'train'
) -> float:
    """Computes loss of batch and backpropagates if phase=='train'.
    
    Args:
        model (nn.Module): Neural network model.
        loss_func (nn.Module): Loss function.
        xb (torch.Tensor): Batch input samples. 
        yb (torch.Tensor): Batch labels.
        optimizer (torch.optim.Optimizer): Optimizer.
        phase (str): Phase ('train' or 'valid').

    Returns:
        tuple: A tuple containing:
            - loss (float): Loss.
            - acc (float): Accuracy.
    """
    # propagate batch through model
    out = model(xb)
    # compute loss
    loss = loss_func(out, yb)
    
    if phase == 'train':
        loss.backward()
        #clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    # get predicted labels
    _, pred = torch.max(out, dim=1)
    # calc accuracy
    acc = torch.sum(pred==yb.data) / yb.size(0)
    return float(loss.item()), float(acc.detach().cpu())


def save_checkpoint(
    model_path: str, 
    model_state, 
    opt_state, 
    epoch: int
) -> None:
    """Save model .pth file and train stats every 50 epochs."""
    torch.save({
        'model_state_dict': model_state,
        'opt_state_dict': opt_state,
        'random_rng_state': random.getstate(),
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
    }, os.path.join(model_path, 'best_model_%s.pth' % epoch))


def save_train_stats(
        path_to_model: str, 
        train_losses: List[float], 
        train_acc: List[float], 
        valid_losses: List[float], 
        valid_acc: List[float], 
        from_epoch: int
    ) -> None:
    """Log train stats."""
    # just overwrite the train stats csv with the new train stats
    df_train_stats = pd.DataFrame({
        'train_loss': train_losses, 
        'train_acc': train_acc, 
        'valid_losses': valid_losses, 
        'valid_acc': valid_acc
    })
    df_train_stats.to_csv(os.path.join(path_to_model, 'train_stats_%d.csv' % from_epoch))


# for training in cluster with CUDA
def main(args):
    # grid search
    device = torch.device('cuda')
    
    # define cluster paths
    path_to_data = '/input-data'
    path_to_models = '/home/sharck/models/'


    # Audio params
    mel_values = {"n_mels": 128,
                "n_fft": 1024,
                "hop_length": 512,
                "slice_length": 6,
                "num_chunks": 4}
    
    """mel_values = {"n_mels": 128,
              "n_fft": 800,
              "hop_length": 360,
              "slice_length": 3,
              "num_chunks": 8}"""

    # input shape of mel-specs
    input_size   = (128,256)
    decrease_factor = 0.1
    decrease_epoch = 100
    lambda_lr = lambda epoch: decrease_factor if epoch >= decrease_epoch else 1.0
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    scheduler=None

    num_epochs        = 500
    num_workers       = 4
    #validation_fold   = None

    if args.conf == 1:
        batch_sizes = [16]
        learning_rates =  [1e-4]
    elif args.conf == 2:
        batch_sizes = [16]
        learning_rates =  [5e-5, 1e-5]
    elif args.conf == 3:
        batch_sizes = [32]
        learning_rates =  [1e-3, 1e-4]
    elif args.conf == 4:
        batch_sizes = [32]
        learning_rates =  [5e-5, 1e-5]
    elif args.conf == 5:
        batch_sizes = [64]
        learning_rates =  [1e-3, 1e-4]
    elif args.conf == 5:
        batch_sizes = [64]
        learning_rates =  [5e-5, 1e-5]


    #batch_sizes        = [16, 32, 64]
    wav_normalization = False
    mel_normalization = True

    # model configuration and grid
    #confs = [((32,32,64,64), 256, ((4,4), (2,4), (2,2), (2,2)))]
    confs = [((64,64,100,128,128), 100, ((2,4), (2,2), (2,2), (2,2), (2,2)))]
    #confs = [((32,32,64,100,128), 1024, ((2,2), (2,2), (2,2), (2,2), (2,2)))]
    
    dropout = 0.3
    weight_decays = [1e-4]
    #weight_decay = weight_decays[0]
    momentums = [0.99]
    #momentum = momentums[0]
    #validation_folds = [2,3,4,5]
    #validation_fold = validation_folds[0]
    validation_fold = 1
    conv_bn = True


    model_performances = {}
    for conf in confs:
        for lr in learning_rates:
            for momentum in momentums:
            #for weight_decay in weight_decays:
                for batch_size in batch_sizes:
                    for weight_decay in weight_decays:                    
                        #random.seed(42)
                        #torch.manual_seed(42)
                        # create model
                        model_path = os.path.join(
                            path_to_models, 
                            '6s_gtzan', 
                            f'{str(conf[:2])}_BN', 
                            f'dr{dropout}_lr{lr}_bs{batch_size}_wd{weight_decay}_mm{momentum}'
                        )
                        # create model
                        model = VGGType(
                            n_filters=conf[0], 
                            n_dense=conf[1], 
                            pool_kernels=conf[2], 
                            dropout=dropout, 
                            input_size=input_size, 
                            n_classes=10
                        )
                        # Load states from checkpoint
                        checkpoint = torch.load(os.path.join(model_path, 'best_model_1500.pth'), map_location=torch.device('cuda'))
                        model.load_state_dict(checkpoint['model_state_dict'])
                        torch.set_rng_state(checkpoint['torch_rng_state'].type(torch.ByteTensor))
                        np.random.set_state(checkpoint['numpy_rng_state'])
                        random.setstate(checkpoint['random_rng_state'])
                        model.to(device)

                        # get data loaders
                        trainloader, validloader  = get_data_loaders(
                            data_path=path_to_data, 
                            batch_size=batch_size, 
                            valid_fold=validation_fold,
                            num_workers=num_workers
                        )
                        dataloaders = {'train': trainloader, 'valid': validloader}

                        # Define opt etc
                        loss_fn   = torch.nn.CrossEntropyLoss()
                        params    = [p for p in model.parameters() if p.requires_grad]
                        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
                        #optimizer = torch.optim.Adam(params, lr = lr, weight_decay=weight_decay)

                        # create path
                        #model_path = os.path.join(path_to_models, '6s_gtzan', f'{str(conf[:2])}_BN' if conv_bn else f'{str(conf[:2])}_noBN', 
                        #                        f'dr{dropout}_lr{lr}_bs{batch_size}_wd{weight_decay}_mm{momentum}')

                        print('Starting to fit model...')
                        print('-'*5)
                        print(f'Model configuration: {conf[:2]}') 
                        print(f'Validation fold: {validation_fold}')
                        print(f'Traing parameters: dr={dropout}, lr={lr}, bs={batch_size}, wd={weight_decay}, mm={momentum}')
                        print('-'*5)

                        # Train model
                        fit(
                            model=model, 
                            loss_func=loss_fn, 
                            optimizer=optimizer, 
                            dataloaders=dataloaders, 
                            num_epochs=num_epochs, 
                            device=device, 
                            scheduler=scheduler, 
                            model_path=model_path, 
                            save_step=100, 
                            from_epoch=1500
                        )
                        
                        # get test accuracy (validation accuracy in case of k-fold)
                        #acc, _, _ = get_acc(model, testloader=validloader, is_toy=False, device=torch.device('mps'))
                        #model_performances[str.split(model_path, '/')[-2]] = acc
                        #print(str.split(model_path, '/')[-2], 'Accuracy:', acc)


import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Ddifferent calls of training')
    parser.add_argument('--conf', type=int, required=False)
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Call main() with the parsed arguments
    main(args)


    