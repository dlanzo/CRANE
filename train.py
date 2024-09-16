# <<< import external stuff <<<
import torch
import torch.nn as nn
from torchvision import utils, datasets, transforms

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import os
import sys

import numpy as np

import PIL
from PIL import Image

import time
# --- import external stuff ---

# <<< import my stuff <<<
from src.classes import *
from src.utils import *
from src.dataloaders import give_dataloaders, give_3D_dataloaders
from src.parser import TrainingParser
# --- import my stuff ---

# <<< training function <<<
def train(model, loss_fn, optimizer, loaders, args):
    '''
    This function trains the model given selected loss function
    '''
    valid_losses = []
    train_losses = []
    
    train_loader, valid_loader = loaders
    
    len_train_loader = len(train_loader)
    len_valid_loader = len(valid_loader)
    
    for epoch in range(args.epochs):
        
        start_epoch = time.time()
        log_epoch_start_info(epoch, args)
        
        optimizer.zero_grad()
        
        epoch_train_losses = []
        
        model.train()
        
        # <<< training loop <<<
        for j, series_with_params in enumerate(train_loader):

            if args.num_params != 0:
                series = series_with_params[0]
                params = series_with_params[1]
            else:
                series = series_with_params
                params = None
            
            # breaking if in debug mode
            if j >= 1 and args.debug:
                print('Breaking because of DEBUG mode.')
                break
            
            # first epoch in reloading has a ramping lr (this way Adam can re-recongnize slow and fast modes in loss function landscape)
            if epoch == 0 and args.reload: 
                for g in optimizer.param_groups:
                    temp_lr = ((j+1)/(len_train_loader+1))*args.lr
                    g['lr'] = temp_lr
                    print(f'Learning rate updated to: {temp_lr:.4e}')
            elif epoch == 1 and args.reload:
                for g in optimizer.param_groups:
                    g['lr'] = args.lr
            
            if args.ramp:
                in_seq_length = int( args.subseq_max*(1-(epoch+args.start_ramp)/args.ramp_length) )
                in_seq_length = min(series.shape[1]-1, in_seq_length)
                in_seq_length = max(args.subseq_min, in_seq_length)
            else:
                in_seq_length   = np.random.randint( args.subseq_min, args.subseq_max+1 ) 
            
            future          = series.shape[1]-in_seq_length-1
            
            if j%args.logfreq == 0 and not args.extract_param: # <- print sub-epoch infos
                print(f'Passing example[{j}/{len_train_loader-1}] in epoch {epoch} with {future} f-frames')
            elif j%args.logfreq == 0:
                print(f'Passing example[{j}/{len_train_loader-1}] in epoch {epoch}')
            
            input_data  = clip_series(series, in_seq_length).to(args.device)
            
            if not args.extract_param:
                target_data = series[:,1:,:,:,:].to(args.device)
            
                if args.dual:
                    input_data  = withdual(input_data)
                    target_data = withdual(target_data)
                    if params is not None:
                        for pp, param in enumerate(params):
                            params[pp] = torch.cat([params[pp], params[pp]])
                        
            else:
                if args.dual:
                    input_data  = withdual(input_data)
                    if params is not None:
                        for pp, param in enumerate(params):
                            params[pp] = torch.cat([params[pp], params[pp]])
                            
                target_data = torch.cat([p.unsqueeze(1) for p in params], dim=1).to(args.device)
                target_data = target_data.float()
                
            if not args.extract_param:
                y_pred = model(input_data, future=future, params=params, noise_reg=args.noise_reg, approx_inference=False)
            else:
                y_pred = model(input_data, noise_reg=args.noise_reg, approx_inference=False)
                
            
            loss = loss_fn(y_pred, target_data)
            loss.backward()
            
            if j%args.superbatch == 0 or j==len_train_loader-1:
                optimizer.step()
                optimizer.zero_grad()
            
            loss4print = loss.item()
            
            epoch_train_losses.append( loss4print )
            
            if j%args.logfreq == 0:
                print(f'Loss: {loss4print:.4e} \t Running mean loss: {np.mean(epoch_train_losses):.4e}')
                
        train_losses.append( np.mean(epoch_train_losses) )
        with open( f'{args.paths["trainloss"]}', 'a+') as train_loss_file:
            train_loss_file.write(f'{epoch_train_losses[-1]}\n')
        # --- training loop ---
        
        # <<< validation loop <<<
        with torch.no_grad():
            
            model.eval()
            
            epoch_valid_losses = []
            
            y_preds = []
            y_trues = []
            
            for j, series_with_params in enumerate(valid_loader):
                
                if args.num_params != 0:
                    series = series_with_params[0]
                    params = series_with_params[1]
                else:
                    params = None
                
                if j >= 3 and args.debug:
                    print('Breaking because of DEBUG mode.')
                    break
                
                in_seq_length   = args.subseq_min # this should make validation always as hard as possible
                future          = series.shape[1]-in_seq_length-1
                    
                
                if not args.extract_param:
                    input_data  = clip_series(series, in_seq_length).to(args.device)
                    target_data = series[:,1:,:,:,:].to(args.device)
                
                    if args.dual:
                        input_data  = withdual(input_data)
                        target_data = withdual(target_data)
                        if params is not None:
                            for pp, param in enumerate(params):
                                params[pp] = torch.cat([params[pp], params[pp]])
                        
                else:
                    input_data  = series.to(args.device)
                    
                    if args.dual:
                        input_data  = withdual(input_data)
                        if params is not None:
                            for pp, param in enumerate(params):
                                params[pp] = torch.cat([params[pp], params[pp]])
                                
                    target_data = torch.cat([p.unsqueeze(1) for p in params], dim=1).to(args.device)
                    target_data = target_data.float()
                    
                    
                if not args.extract_param:
                    y_pred  = model(input_data, future=future, params=params)
                else:
                    y_pred  = model(input_data)
                    y_preds.append(y_pred.detach().cpu())
                    y_trues.append(target_data.detach().cpu())
                
                loss = loss_fn(y_pred, target_data)
                loss4print = loss.item()
                
                epoch_valid_losses.append( loss4print )
                
            valid_losses.append( np.mean(epoch_valid_losses) )
            
            with open( f'{args.paths["validloss"]}', 'a+') as valid_loss_file:
                valid_loss_file.write(f'{epoch_valid_losses[-1]}\n')
            
        optimizer.zero_grad() # <- better safe than sorry
        # --- validation loop ---
        
        # <<< graphic output <<<
        if args.graphics and not args.threeD and not args.extract_param:
            
            y_pred_cpu = y_pred.detach().cpu()
            target_data_cpu = target_data.detach().cpu()
            
            im_last_pred    = y_pred_cpu[0,-1,:,:,:].permute(1,2,0)
            im_last_target  = target_data_cpu[0,-1,:,:,:].permute(1,2,0)
            
            out_png(
                im_pred     = im_last_pred,
                im_target   = im_last_target,
                path        = f'{args.paths["png"]}/epoch_{epoch}.png',
                cmap        = args.cmap
                )
            
            make_lossplot(train_losses, valid_losses, args)
            
            if args.gifs:
                out_gifs(
                    y_pred      = y_pred_cpu,
                    path        = f'{args.paths["gif"]}/epoch_{epoch}.gif'
                    )
                
                out_gifs(
                    y_pred      = target_data_cpu,
                    path        = f'{args.paths["gif"]}/epoch_{epoch}_TRUE.gif'
                    )
                
            if args.vtk:
                
                epoch_path = f'{args.paths["vtk"]}/epoch_{epoch}'
                epoch_path_TRUE = f'{args.paths["vtk"]}/epoch_{epoch}_TRUE'
                
                os.mkdir(epoch_path)
                os.mkdir(epoch_path_TRUE)
                
                seq2vtk(
                    y_pred  = y_pred_cpu,
                    path    = epoch_path
                    )
                
                seq2vtk(
                    y_pred  = target_data_cpu,
                    path    = epoch_path_TRUE
                    )
                
                del epoch_path, epoch_path_TRUE
                
            if args.npy:
                
                epoch_path = f'{args.paths["npy"]}/epoch_{epoch}'
                epoch_path_TRUE = f'{args.paths["npy"]}/epoch_{epoch}_TRUE'
                
                os.mkdir(epoch_path)
                os.mkdir(epoch_path_TRUE)
                
                seq2npy(
                    y_pred  = y_pred_cpu,
                    path    = epoch_path
                    )
                
                seq2npy(
                    y_pred  = target_data_cpu,
                    path    = epoch_path_TRUE
                    )
                
                del epoch_path, epoch_path_TRUE
                
                
        elif args.graphics and args.threeD and args.vtk:
            
            make_lossplot(train_losses, valid_losses, args)
            
            y_pred_cpu = y_pred.detach().cpu()
            target_data_cpu = target_data.detach().cpu()
            
            epoch_path = f'{args.paths["gif"]}/epoch_{epoch}'
            epoch_path_TRUE = f'{args.paths["gif"]}/epoch_{epoch}_TRUE'
            
            os.mkdir( epoch_path )
            os.mkdir( epoch_path_TRUE )
            
            seq2vtk(
                y_pred      = y_pred_cpu,
                path        = epoch_path
                )
            
            seq2vtk(
                y_pred      = target_data_cpu,
                path        = epoch_path_TRUE
                )
            
        elif args.graphics and args.extract_param:
            
            make_lossplot(train_losses, valid_losses, args)
            
            for kk in range(args.num_params):
                
                preds = []
                trues = []
                
                for y_pred, target_data in zip(y_preds, y_trues):
                    for bb in range(y_pred.shape[0]):
                        preds.append( y_pred[bb,kk] )
                        trues.append( target_data[bb,kk] )
                        
                target_data_cpu = np.array(trues)
                y_pred_cpu = np.array(preds)
                
                plt.scatter(target_data_cpu, y_pred_cpu)
                plt.plot(
                    [np.min(target_data_cpu), np.max(target_data_cpu)],
                    [np.min(target_data_cpu), np.max(target_data_cpu)]
                    )
                plt.title(f'Regression plot param {kk}')
                plt.xlabel('True value')
                plt.ylabel('Predicted value')
                plt.savefig(f'{args.paths["png"]}/epoch_{epoch}_param{kk}.png')
                plt.close()
            
            
        # --- graphic output ---
        
        # <<< epoch end logging <<<
        end_epoch = time.time()
        epoch_time = end_epoch-start_epoch
        
        if not args.extract_param:
            log_epoch_end_info(epoch, epoch_time, (y_pred, target_data), train_losses[-1], valid_losses[-1], args)
        
        save_model(
            model   = model,
            path    = f'{args.paths["model"]}/epoch_{epoch}.pt'
            )
        # --- epoch end logging ---

# <<< main function <<<
def main():
    '''
    Main function: istantiation of models and dataloaders and launcing of training function
    '''
    
    #Parse arguments
    parser  = TrainingParser()
    args    = parser.parse_args()
    
    # crate folder structure
    args = build_train_logs_dir_tree(args)
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Instantiate dataloaders
    if not args.threeD and not args.extract_param:
        dataloaders = give_dataloaders(args)
        model_class = ConvGRU
    elif args.extract_param:
        dataloaders = give_dataloaders(args)
        model_class = ConvGRUClassifier
    elif args.threeD and not args.extract_param:
        dataloaders = give_3D_dataloaders(args)
        model_class = ConvGRU3D
    elif args.threeD:
        raise NotImplementedError('train.py was not able to recognize the training mode. Aborting.')
    
    train_loader = dataloaders["train_set"]
    valid_loader = dataloaders["valid_set"]
    
    # Define model and put to device
    model = model_class(
        hidden_units        = args.hidden,
        input_channels      = 1, # this is hardcoded for the moment... waiting for multidimensional data!
        output_channels     = None if not args.extract_param else args.num_params,
        hidden_channels     = args.channels,
        kernel_size         = args.kernel_size,
        padding_mode        = args.padding,
        separable           = False,
        bias                = args.bias,
        divergence          = args.divergence,
        num_params          = args.num_params if not args.extract_param else 0,
        dropout             = args.dropout,
        dropout_prob        = args.dropout_prob
        )
    
    print_model_info(model)
    
    if args.divergence:
        model.make_div_filters( torch.zeros(1, device=args.device) )
    
    #model = torch.compile(model)
    
    # Reload operation
    if args.reload:
        model = import_model(model, args)
        
    if args.symm_kernel:
        model.symmetrize()
        
    model.to(args.device)
    
    # save inputs
    save_args(args)
    
    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr              = args.lr,
        weight_decay    = args.weightd
        )
    
    # define loss function
    if args.extract_param:
        loss_fn = nn.MSELoss()
    elif not args.threeD:
        loss_fn = lambda x,y: \
            nn.MSELoss()(x,y) + args.massW*nn.MSELoss()(
                torch.mean( x, axis=(-1,-2) ),
                torch.mean( y, axis=(-1,-2) )
                )
    else:
        loss_fn = lambda x,y: \
            nn.MSELoss()(x,y) + args.massW*nn.MSELoss()(
                torch.mean( x, axis=(-1,-2,-3) ),
                torch.mean( y, axis=(-1,-2,-3) )
                )
    
    # training loop
    train(model, loss_fn, optimizer, (train_loader, valid_loader), args)
# --- main function ---

# <<< main calling <<<
if __name__ == '__main__':    
    main()
# --- main calling ---
