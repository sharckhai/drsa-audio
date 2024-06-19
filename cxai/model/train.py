
import os
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np
from datetime import datetime

from cxai.model.dataloader.gtzan_dataset import get_data_loaders
from cxai.model.create_model import VGGType


def fit(model: nn.Module, 
        loss_func, 
        optimizer: torch.optim, 
        dataloaders: dict, 
        num_epochs: int = 10, 
        device = torch.device('cpu'), 
        scheduler = None, 
        from_epoch: int = 0, 
        model_path: str = None,
        save_step: int = 100,
        is_gtzan: bool = True,
        is_gtzan_vggish: bool = False,
        ):
    
    print('starting fit')
    
    # model to device
    model.to(device)
    # track smoothed accuracy
    #smoothed_train_acc = SmoothedMetricHistory(window_size=5)
    #smoothed_valid_acc = SmoothedMetricHistory(window_size=5)
    #smoothing_metrics = {'train': smoothed_train_acc, 'valid': smoothed_valid_acc}
    train_losses = []
    valid_losses = []
    train_acc = []
    valid_acc = []
    
    if from_epoch == 0:
        time = datetime.now()
        model_path = model_path if model_path else os.path.join('../models', str(time))
        if not os.path.exists(model_path):
            os.makedirs(model_path)


    tbar = tqdm(range(1, num_epochs+1))

    for epoch in tbar:

        for phase in ['train', 'valid']:

            if phase == 'train':
                model.train()
            else:
                model.eval() 

            running_loss = 0.0
            running_acc = 0.0

            for xb, yb in dataloaders[phase]:
                xb, yb = xb.to(device), yb.to(device)
                
                if is_gtzan:
                    # reshape validation chunk level predictions (see dataset)
                    if phase == 'valid':
                        if xb.dim() == 5:
                            # case GTZAN, test data is of the form (batch x chunks x channel x mel-bin x time-bin), this is because for testing 
                            # we dont extract one random slice of an audio but slice each audio in num_chunks chunks
                            _, chunks, c, f, t = xb.size() # b=batch
                            yb = yb.repeat_interleave(chunks)
                        else:
                            # case toy data
                            _, c, f, t = xb.size()

                        xb = xb.view(-1, c, f, t)
                
                if is_gtzan_vggish:
                    # reshape validation chunk level predictions (see dataset)
                    if xb.dim() == 5:
                        # case GTZAN, test data is of the form (batch x chunks x channel x mel-bin x time-bin), this is because for testing 
                        # we dont extract one random slice of an audio but slice each audio in num_chunks chunks
                        _, chunks, c, f, t = xb.size() # b=batch
                        yb = yb.repeat_interleave(chunks)
                    else:
                        # case toy data
                        _, c, f, t = xb.size()

                    xb = xb.view(-1, c, f, t)


                with torch.set_grad_enabled(phase == 'train'):
                    loss, acc = loss_batch(model, loss_func, xb, yb, opt=optimizer, phase=phase)

                # stats
                running_loss += loss
                running_acc += acc
            
            # take the average loss per epoch from the sum of losses
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_acc / len(dataloaders[phase])
            train_losses.append(epoch_loss) if phase=='train' else valid_losses.append(epoch_loss)
            train_acc.append(epoch_acc) if phase=='train' else valid_acc.append(epoch_acc)

            # update smoothed accuarcy
            #smoothing_metrics[phase].update(epoch_acc)
            #smoothed_acc = smoothing_metrics[phase].average()

            if phase == 'train':
                # update learning rate scheduler
                if scheduler:
                    scheduler.step()
                train_loss = epoch_loss
                train_acc_ = epoch_acc
        
        #if epoch%50 == 0:
        tbar.set_description(f'TRAIN acc: {train_acc_*100:.2f}% - loss: {train_loss:.4f}  || VALID acc: {epoch_acc*100:.2f}% - loss: {epoch_loss:.4f}')

        #current_lr = scheduler.get_last_lr()[0]
    
        # deep copy the model
        if epoch % save_step == 0:
            save_checkpoint(model_path, model.state_dict(), optimizer.state_dict(), epoch=epoch+from_epoch, from_epoch=from_epoch)
            save_train_stats(model_path, train_losses, train_acc, valid_losses, valid_acc, from_epoch)

    #return train_stats, model_path


def loss_batch(model: nn.Module, loss_func, xb, yb, opt, phase='train'):

    out = model(xb)
    loss = loss_func(out, yb)
    
    if phase == 'train':
        loss.backward()
        #clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad()

    _, pred = torch.max(out, dim=1)

    acc = torch.sum(pred==yb.data) / yb.size(0)

    return float(loss.item()), float(acc.detach().cpu())



def save_checkpoint(model_path, model_state, opt_state, epoch, from_epoch):
    """
    Save model .pth file and train stats every 50 epochs.

    """

    # save model
    torch.save({
            'model_state_dict': model_state,
            'opt_state_dict': opt_state,
            'random_rng_state': random.getstate(),
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
        }, os.path.join(model_path, 'best_model_%s.pth' % epoch))




def save_train_stats(path_to_model, train_losses, train_acc, valid_losses, valid_acc, from_epoch):
    # just overwrite the train stats csv with the new train stats
    df_train_stats = pd.DataFrame({'train_loss': train_losses, 'train_acc': train_acc, 'valid_losses': valid_losses, 'valid_acc': valid_acc})
    df_train_stats.to_csv(os.path.join(path_to_model, 'train_stats_%d.csv' % from_epoch))



#################################################### Müll ####################################################

## no need for this function
def reset_weights(model):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()



class SmoothedMetricHistory:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def average(self):
        return sum(self.values) / len(self.values) if self.values else 0




"""def fit_baseline(model: nn.Module, 
        loss_func, 
        optimizer: torch.optim, 
        dataloaders: dict, 
        num_epochs: int=10, 
        device = torch.device('cpu'), 
        scheduler = None, 
        from_epoch: int = 0, 
        model_path=None,
        ):
    
    # model to device
    model.to(device)
    
    if from_epoch == 0:
        time = datetime.now()
        model_path = model_path if model_path else os.path.join('../models', str(time))
        os.makedirs(model_path)

    train_stats = {'train_loss': [], 'train_acc': [], 'validation_loss': [], 'validation_accuracy': []}

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 30)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval() 

            running_loss = 0.0
            running_acc = 0.0

            for xb, yb in dataloaders[phase]:
                xb, yb = xb.to(device), yb.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    loss, acc = loss_batch(model, loss_func, xb, yb, opt=optimizer, phase=phase, device=device)

                # stats
                running_loss += loss
                running_acc += acc

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_acc / len(dataloaders[phase])

            if phase == 'train':
                #scheduler.step()
                train_loss = epoch_loss
                train_acc = epoch_acc

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()
        train_stats = log_dict(train_stats, train_loss, train_acc, epoch_loss, epoch_acc)

        # deep copy the model
        if epoch % 50==0:
            torch.save(model.state_dict(), os.path.join(model_path, 'best_model_%s.pth' % (epoch+from_epoch)))
            save_train_stats(model_path, train_stats, (epoch+from_epoch))

    return train_stats, model_path"""



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
                        #model = VGGType(n_filters=conf[0], n_dense=conf[1], pool_kernels=conf[2], dropout=dropout, input_size=input_size, conv_bn=conv_bn)
                        model_path = os.path.join(path_to_models, '6s_gtzan', f'{str(conf[:2])}_BN', f'dr{dropout}_lr{lr}_bs{batch_size}_wd{weight_decay}_mm{momentum}')
                        # create model
                        model = VGGType(n_filters=conf[0], n_dense=conf[1], pool_kernels=conf[2], dropout=dropout, input_size=input_size, n_classes=10)
                        # Load states from checkpoint
                        checkpoint = torch.load(os.path.join(model_path, 'best_model_1500.pth'), map_location=torch.device('cuda'))
                        model.load_state_dict(checkpoint['model_state_dict'])
                        torch.set_rng_state(checkpoint['torch_rng_state'].type(torch.ByteTensor))
                        np.random.set_state(checkpoint['numpy_rng_state'])
                        random.setstate(checkpoint['random_rng_state'])
                        model.to(device)

                        # get data loaders
                        trainloader, validloader  = get_data_loaders(data_path=path_to_data, batch_size=batch_size, n_mels=mel_values["n_mels"], 
                                                                    n_fft=mel_values["n_fft"], hop_length=mel_values['hop_length'], slice_length=mel_values["slice_length"],
                                                                    num_chunks=mel_values["num_chunks"], valid_fold=validation_fold, wav_normalization=wav_normalization,
                                                                    mel_normalization=mel_normalization, num_workers=num_workers)
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
                        fit(model=model, loss_func=loss_fn, optimizer=optimizer, dataloaders=dataloaders, 
                            num_epochs=num_epochs, device=device, scheduler=scheduler, model_path=model_path, save_step=100, from_epoch=1500)
                        
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


    