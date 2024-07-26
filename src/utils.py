# <<< importing external stuff <<<
import os
import sys

import torch
import torch.nn as nn
from torchvision import utils, datasets, transforms

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import PIL
from PIL import Image

from warnings import warn

from multiprocessing import Process

import pyvista as pv
# --- importing external stuff ---

# <<< import numba <<<
try:
    from numba import njit, prange
except ImportError:
    print('It seems that "numba" is not installed on this machine or there are problems in importing it. Falling back on non-jitted versions of scripts. Some operations will be slower. Consider installing it (e.g. by "pip install numba").')
    
    def njit(fun): # <- alternative definition of njit
        return fun
# --- import numba ---

def build_train_logs_dir_tree(args):
    '''
    This function builds the output folder structure and adds appropriate paths to dictionary in args
    '''
    
    master = f'train_logs/{args.id}' # <- name of master folder
    
    if os.path.isdir(master):
        # in this case we have a naming conflict: id will be replaced with id_N with N=number of folders with the same id
        print(f'Naming conflict found with id "{args.id}".')
        num_existing_folders = len([subfolder for subfolder in os.listdir('train_logs') if args.id in subfolder])
        args.id = f'{args.id}_{num_existing_folders}'
        
        master = f'train_logs/{args.id}'
        print(f'Naming conflict handled. Training logs will be saved in {master}')
        
    os.mkdir(master)
    
    model_path = f'{master}/model'
    
    if args.graphics:
        pngs_path = f'{master}/pngs'
        lossplot_path = f'{master}/lossplot.png'
        
        if args.vtk:
            vtk_path  = f'{master}/vtk'
        else:
            vtk_path  = None
        
        if args.npy:
            npy_path  = f'{master}/npy'
        else:
            npy_path  = None
        
        if args.gifs:
            gifs_path = f'{master}/gifs'
        else:
            gifs_path = None
    else:
        pngs_path       = None
        lossplot_path   = None
        gifs_path       = None
        
    trainloss_path = f'{master}/train_loss.txt'
    validloss_path = f'{master}/valid_loss.txt'
    
    args.paths = {
        'master'    :   master,
        'model'     :   model_path,
        'png'       :   pngs_path,
        'gif'       :   gifs_path,
        'lossplot'  :   lossplot_path,
        'trainloss' :   trainloss_path,
        'validloss' :   validloss_path,
        'vtk'       :   vtk_path,
        'npy'       :   npy_path
        }
    
    for name in ['model','png', 'gif', 'vtk', 'npy']:
        if args.paths[name] is not None:
            os.mkdir(args.paths[name])
    
    return args


def build_test_dir_tree(args):
    '''
    This function builds the output folder for testing procedure, given the id in args
    '''
    master = f'test_outputs/{args.id}' # <- name of the master folder

    if os.path.isdir(master):
            
        print(f'Naming conflict found with id "{args.id}".')
        
        num_existing_folders = len([subfolder for subfolder in os.listdir('test_outputs') if args.id in subfolder])
        
        args.id = f'{args.id}_{num_existing_folders}'
        master = f'test_outputs/{args.id}'
        
        print(f'Naming conflict handled. Testing outputs will be saved in {master}')
            
    os.mkdir(master)
    
    if args.graphics:
        png_path    = f'{master}/png'
        if args.gifs:
            gifs_path = f'{master}/gifs'
        else:
            gifs_path = None
    
    else:
        warn('Graphics was disabled for testing procedure. Little information for many calculations will be produced.')
        png_path    = None
        gifs_path   = None
        
    area_path       = f'{master}/area'
    progloss_path   = f'{master}/progloss'
    
    AR_path     = f'{master}/AR' if args.AR else None
    
    args.paths = {
        'master'    :   master,
        'png'       :   png_path,
        'gif'       :   gifs_path,
        'area'      :   area_path,
        'progloss'  :   progloss_path,
        'AR'        :   AR_path
        }
    
    for name in ['png', 'gif','area','progloss','AR']:
        if args.paths[name] is not None:
            os.mkdir(args.paths[name])
    
    return args


def build_predict_dir_tree(args):
    '''
    This function builds the directory structure for the output of a prediction run. It returns paths in addition to args.paths
    '''
    master  = f'out/{args.id}'
    if os.path.isdir(master):
        print(f'Naming conflict found with id "{args.id}".')
        num_existing_folders = len([subfolder for subfolder in os.listdir('out') if args.id in subfolder])
        args.id = f'{args.id}_{num_existing_folders}'
        
        master = f'out/{args.id}'
        print(f'Naming conflict handled. Prediction outputs will be saved in {master}')
        
    os.mkdir(master)
    
    pngs_path       = f'{master}/pngs' if args.graphics                 else None
    AR_path         = f'{master}/AR'   if args.AR                else None
    gif_path        = f'{master}/gifs' if args.gifs and args.graphics   else None
    
    phi_0_path  = f'{master}/initial_condition.png'
    area_path   = f'{master}/area'
    
    init_geo_source_path = \
        f'{master}/init_geo.py' if args.gengeo else None
    
    args.paths = {
        'png'               :   pngs_path,
        'AR'                :   AR_path,
        'phi_0'             :   phi_0_path,
        'geo_source'        :   init_geo_source_path,
        'gifs'              :   gif_path,
        'area'              :   area_path
        }
    
    for name in ['png', 'gifs']:
        if args.paths[name] is not None:
            os.mkdir(args.paths[name])
        
    return args


def save_args(args):
    '''
    This function saves the argument list in the log master folder
    '''
    with open(f'{args.paths["master"]}/args.txt', 'w+') as args_file:
        for arg in dir(args):
            if arg[0] != '_':
                args_file.write(f'{arg} \t : \t {vars(args)[arg]} \n')
                

def print_model_info(model):
    '''
    This function prints some infos about the model
    '''
    if not hasattr(model, 'model_list'):
        # model is a simple model: just print the number of parameters
        params_nums = sum( [p.numel() for p in model.parameters()] )
        print()
        print('<<< model infos <<<')
        print(f'The number of parameters in the model is: {params_nums}')
        print('--- model infos ---')
        print()
    elif hasattr(model, 'model_list'):
        # model is a committee model; print some additional infos
        params_nums = sum( [p.numel() for p in model.model_list[0].parameters()] )
        num_models  = len(model)
        print()
        print('<<< model infos <<<')
        print(f'model is a CommitteeModel combining inferences from {num_models} models.')
        print(f'Each model has a number of parameters: {params_nums}, totalling {num_models*params_nums} parameters')
        print()
    else:
        raise RuntimeError('It seems that model is neither a torch.nn.Module nor a CommitteeModel.')
    
    
def import_model(model, args):
    '''
    This function loads a .pt file model in provided model.
    '''
    if not os.path.isfile(args.reload_model):
        raise FileNotFoundError(
            f'No model found at path "{args.reload_model}".'
            )
    
    print('<<< Loading model ... <<<')
    model.load_state_dict(
        torch.load(
            args.reload_model,
            map_location = 'cpu'
            )
        )
    print('--- Loading done! ---')
    
    return model
    
    
def log_epoch_start_info(epoch, args):
    '''
    This function prints some infos at the begining of the epoch
    '''
    print()
    print('<<< Epoch starting... <<<')
    print(f'Starting epoch {epoch} with subseq in range [{args.subseq_min}/{args.subseq_max}]')
    print(f'master folder for outputs is {args.paths["master"]}')
    print()
    

def log_epoch_end_info(epoch, epoch_time, vals, train_loss, valid_loss, args):
    '''
    This function prints some infos at the endo of each epoch and, if proper args, produces gifs and pngs
    '''
    pred, true = vals
    with torch.no_grad():
        min_pred = torch.min(pred[0,-1,:,:,:]).item()
        max_pred = torch.max(pred[0,-1,:,:,:]).item()
        min_true = torch.min(true[0,-1,:,:,:]).item()
        max_true = torch.max(true[0,-1,:,:,:]).item()
        
        max_deviation = torch.max(
            torch.abs( true[0,-1,:,:,:] - pred[0,-1,:,:,:] )
            ).item()
        
        area_pred = torch.mean(pred[0,-1,:,:,:])
        area_pred_init = torch.mean(pred[0,0,:,:,:])
        area_true = torch.mean(true[0,-1,:,:,:])
        
        area_delta_last = 100*(area_pred-area_true)/(area_true+1e-6)
        area_delta_start_end = 100*(-area_pred+area_pred_init)/(area_pred_init+1e-6)
    
    print()
    print('<<< Epoch ended <<<')
    print(f'Endend epoch {epoch} in {epoch_time:.2f} s')
    print(f'Mean epoch training loss is: {train_loss:.2e}')
    print(f'Mean epoch validation loss is: {valid_loss:.2e}')
    print()
    print('<<< Training stats <<<')
    print(f'Predicted min: {min_pred:.2e}')
    print(f'True min: {min_true:.2e}')
    print(f'Predicted max: {max_pred:.2e}')
    print(f'True max: {max_true:.2e}')
    print(f'Absolute max deviation: {max_deviation:.2e}')
    print(f'Relative variation in mass in final state is: {area_delta_last:.2} %')
    print(f'Relative variation from initial state is: {area_delta_start_end:.2} %')
    print('--- Training stats ---')
    print()
    
    
def withdual(tensor):
    '''
    This function returns the original tensor and its dual concatenated
    '''
    return torch.cat( [tensor, -tensor+1] )


def clip_series(series, in_seq_length):
    '''
    This funciton clips the provided series at the in_seq_length length
    '''
    clipped_series  = series[:,:in_seq_length,:,:,:]
        
    return clipped_series


def make_lossplot(train_losses, valid_losses, args):
    '''
    This function outputs the training/validation loss as a function of the number of epochs
    '''
    matplotlib.use('Agg')
    
    print('Outputting train/valid loss plot...', end='')
    
    plt.plot(
        np.arange(len(train_losses)),
        np.array(train_losses)
        )
    plt.plot(
        np.arange(len(valid_losses)),
        np.array(valid_losses)
        )
    
    plt.legend(['Training loss', 'Validation loss'])
    plt.savefig(args.paths['lossplot'])
    plt.close()
    
    print('done!')
    

def out_png(im_pred, im_target, path, cmap, var=None):
    '''
    This function outputs image for visual inspection of the quality of the model
    '''
    
    matplotlib.use('Agg')
    
    with torch.no_grad():
        
        if var is None:
            f, axarr = plt.subplots( 1,3 )
            axarr[0].imshow(
                im_pred, cmap = cmap, vmin=0, vmax=1
                )
            axarr[1].imshow(
                im_target, cmap = cmap, vmin=0, vmax=1
                )
            axarr[2].imshow(
                torch.abs(im_target-im_pred), cmap = cmap, vmin=0, vmax=1)
            
            axarr[0].title.set_text('Predicted')
            axarr[1].title.set_text('True')
            axarr[2].title.set_text('Error')
        
        else:
            f, axarr = plt.subplots( 2,2 )
            axarr[0,0].imshow(
                im_pred, cmap=cmap, vmin=0, vmax=1
                )
            axarr[0,1].imshow(
                im_target, cmap=cmap, vmin=0, vmax=1
                )
            axarr[1,0].imshow(
                torch.abs(im_target-im_pred), cmap=cmap, vmin=0, vmax=1
                )
            axarr[1,1].imshow(
                var, cmap=cmap, vmin=0, vmax=1
                )
            
            axarr[0,0].title.set_text('Predicted')
            axarr[0,1].title.set_text('True')
            axarr[1,0].title.set_text('Error')
            axarr[1,1].title.set_text('Committee variance')
    
    plt.tight_layout()
    plt.savefig(path)
    
    plt.close()
    del f
    del axarr
    

def out_gifs(y_pred, path):
    '''
    This function outputs gifs in the relative folder
    '''
    matplotlib.use('Agg')
    
    imagelist = []
    
    for ii in range(y_pred.shape[1]):
        imagelist.append( 255*y_pred[0,ii,:,:,:].permute(1,2,0).squeeze(-1).numpy() )
        
    imagelist = [Image.fromarray(img) for img in imagelist]
    imagelist[0].save(
        path,
        save_all=True,
        append_images=imagelist[1:],
        duration=100,
        loop=20
        )
    
    del imagelist
    

def out_area_deviation(name, seq_pred, seq_target, graphic=True):
    '''
    This function outputs the fractional area deviation (phi integral) between seq_pred and seq_target, calculated as (A_target-A_true)/A_target. Output is as float in txt <name>.txt and an optional (controlled by graphic) png.
    '''
    
    print('Outputting area deviation analysis...')
    
    matplotlib.use('Agg')
    with torch.no_grad():
        area_deviations = []
        
        if seq_target.shape[1] == 1:
            # in the case seq_target is a single frame (initial condition), all areas are compared to it
            area_init = torch.sum( seq_pred[0,0,:,:,:] ).item()
            two_sequences = False
        else:
            two_sequences = True
            
        for kk in range(seq_pred.shape[1]):
            area_pred   = torch.sum( seq_pred[0,kk,:,:,:] ).item()
            if two_sequences:
                area_true   = torch.sum( seq_target[0,kk,:,:,:] ).item()
            else:
                area_true   = area_init
            
            area_deviation = (area_pred-area_true)/area_true
            
            area_deviations.append(area_deviation)
        
        with open(f'{name}.txt','w') as area_deviation_file:
            for val in area_deviations:
                area_deviation_file.write(f'{val} ')
        
        if graphic:
            plt.plot(
                np.arange( len(area_deviations) )+1,
                area_deviations
                )
            #plt.ylim( (-2, 2) )
            plt.xlabel('Frame number')
            plt.ylabel('Relative area error')
            plt.savefig(f'{name}.png')
            plt.close()
            
        print(f'Last area deviation is {100*area_deviations[-1]:.2f} %')
            

def out_progloss(seq_pred, seq_target, name, graphic=True):
    '''
    This function outputs the MSELoss difference between the given sequence (seq_pred) and the true one (seq_target). Output is as float in txt file (<name>.txt) and an optional (controlled by graphic) png.
    '''
    matplotlib.use('Agg')
    with torch.no_grad():
        progloss    = []
        loss_fn     = nn.MSELoss()
        for kk in range( seq_pred.shape[1] ):
            loss = loss_fn(
                seq_pred[0,kk,:,:,:], seq_target[0,kk,:,:,:]
                ).item()
            progloss.append(loss)
        
        with open(f'{name}.txt','w') as progloss_file:
            for val in progloss:
                progloss_file.write(f'{val} ')
                
        if graphic:
            plt.semilogy(
                np.arange( len(progloss) )+1,
                progloss
                )
            plt.ylim( 1e-6, 1e0 )
            plt.xlabel('Frame number')
            plt.ylabel('MSE Loss')
            plt.savefig(f'{name}.png')
            plt.close()
            
        print(f'Last MSELoss value is {progloss[-1]:.2e}')
    

def save_model(model, path):
    '''
    This function saves model in specified path
    '''
    torch.save(
        model.state_dict(),
        path
        )


def give_model_paths(master_path):
    '''
    This function returns a list containing master_path if it points to a .pt file. If it is a folder, it returns a list containing paths of all .pt files contained in the direct subfolder.
    '''
    if master_path.endswith('.pt'):
        return [master_path]
    
    else:
        if master_path.endswith('/'): master_path = master_path[:-1]
        if not os.path.isdir(master_path):
            raise RuntimeError(f'Provided path "{master_path}" does not point to a directory or a .pt file.')
        else:
            content_list = os.listdir(master_path)
            models_list = [model for model in content_list if model.endswith('.pt')]
            if len(models_list) == 0:
                raise RuntimeError(f'Provided path "{master_path}" points to a directory, but it seems that it does not contain any .pt file.')
            else:
                model_paths = []
                for model in models_list:
                    model_paths.append(f'{master_path}/{model}')
                
                return model_paths


def make_square(im, min_size=1, fill_color=(0, 0, 0, 0), cropkey=True, crop_lim=(0.25,0.75)):
    '''
    Utility function. Image im will be padded with black pixels to have a square aspect ratio. Optionally they will be cropped
    '''
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    if cropkey:
        crop_low, crop_high = crop_lim
        new_im = new_im.crop((int(crop_low*size),int(crop_low*size),int(crop_high*size),int(crop_high*size)))
    return new_im


def copy_init_geo_source(args):
    '''
    This function copies the source code for the generation of the initial configuration used in the appropriate path.
    '''
    os.cp(args.init_geo, args.paths['geo_source'])


def seq2png(seq, name, args, start_num):
    '''
    This functions takes a tensor containing a sequence and outputs it as a series of png append_images
    '''
    matplotlib.use('Agg')
    for kk in range(seq.shape[1]):
        frame = seq[0,kk,0,:,:].numpy()
        plt.imshow(frame, cmap=args.cmap, vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(
            f'{args.paths["png"]}/{name}_frame_{kk+start_num}.png',
            bbox_inches     = 'tight',
            pad_inches      = 0
            )


def seq2png_treaded(seq, name, args):
    '''
    This function uses multiprocessing to output sequences into pngs.
    '''
    # chunk sequence
    mp_args         = []
    seq_length      = seq.shape[1]
    proc2use        = min( args.nproc, os.cpu_count() )
    chunk_length    = seq_length//proc2use
    
    if seq_length == 0:
        raise RuntimeError('There was an attempt to output a sequence with legth 0.')
    
    for kk in range(args.nproc):
        if kk != args.nproc-1:
            mp_args.append(
                (seq[:,kk*chunk_length:(kk+1)*chunk_length,:,:,:],
                 name,
                 args,
                 kk*chunk_length)
                )
        else:
            mp_args.append(
                (seq[:,kk*chunk_length:,:,:,:],
                 name,
                 args,
                 kk*chunk_length)
                )
    # launch threads and wait for all to complete
    processes = []
    for arg in mp_args:
        process = Process(target=seq2png, args=arg)
        processes.append(process)
        process.start()
    for process in processes:
        process.join()


@njit
def estimate_AR(image):
    '''
    This function estimates the AR of shapes along x and y directions (shape is supposed to be in the middle).
    '''
    x_search = range(int(0.5*image.shape[0])-1, int(0.5*image.shape[0])+1)
    y_axis = 0
    for x in x_search:
        y_slice = image[x,:]
        total = y_slice.sum()
        if total >= y_axis:
            y_axis = total 
    
    y_search = range(int(0.5*image.shape[1])-1, int(0.5*image.shape[1])+1)
    x_axis = 0
    for y in y_search:
        x_slice = image[:,y]
        total = x_slice.sum()
        if total >= x_axis:
            x_axis = total
            
    AR = x_axis/y_axis
    if AR < 1: AR **= -1
    
    return AR


def out_AR(name, seq, graphic, true_seq=None):
    '''
    Given a sequence, this function returns the AR estimation for each frame
    '''
    print('Estimating sequence AR... ', end='')
    matplotlib.use('Agg')
    AR_list = []
    for kk in range(seq.shape[1]):
        im = seq[0,kk,0,:,:].numpy()
        AR = estimate_AR(im)
        AR_list.append(AR)
        
    ARs = np.array(AR_list)
    
    if true_seq is not None:
        AR_true_list = []
        for kk in range(true_seq.shape[1]):
            im = true_seq[0,kk,0,:,:].numpy()
            AR = estimate_AR(im)
            AR_true_list.append(AR)
        ARs_true = np.array(AR_true_list)
    
    with open(f'{name}.txt', 'w') as AR_file:
        for AR in ARs:
            AR_file.write(f'{AR} ')
            
        if true_seq is not None:
            AR_file.write('\n')
            for AR in ARs_true:
                AR_file.write(f'{AR} ')
        
    if graphic:
        plt.plot(
            np.arange( len(ARs) )+1,
            ARs,
            label = 'Predicted AR'
            )
        
        if true_seq is not None:
            plt.plot(
                np.arange( len(ARs_true) )+1,
                ARs_true,
                label = 'True AR'
                )
            plt.legend()
        
        plt.ylim( (np.min(ARs)-0.2, np.max(ARs)+0.2) )
        plt.xlabel('Frame number')
        plt.ylabel('Estimated Aspect Ratio')
        plt.savefig(f'{name}.png')
        plt.close()
    print('done!')
    

def draw_config(phi, list_add, list_remove):
    '''
    This function takes a Phi object "phi" and two lists of shapes and draws them into phi.val.
    '''
    print('Drawing configuration from list definition... ', end='')
    for shape in list_add:
        phi.paint_shape(shape, filler_value=1.0)
    for shape in list_remove:
        phi.paint_shape(shape, filler_value=0.0)
    print('done!')
    
    
def save_vtk(array, path):
    data = pv.wrap(array)
    data.save(path, binary=True)
    
    
def seq2vtk(y_pred, path, start_snap=0):
    '''
    This function saves the whole sequence y_pred as vtk snapshots
    '''
    if y_pred.ndim == 6: # we are 3D, use the usual method
        for kk in range(y_pred.shape[1]):
            save_vtk(y_pred.numpy()[0,kk,0,:,:,:], path=f'{path}/snap_{kk+start_snap}.vtk')
    elif y_pred.ndim == 5: # a 2D evolution is passed: we need to add a fake spatial dimension
        for kk in range(y_pred.shape[1]): # we are still looping over time
            snap = y_pred[0,kk,0,...].numpy()
            snap = np.expand_dims(snap, 0)
            save_vtk(snap, path=f'{path}/snap_{kk+start_snap}.vtk')
    else: # data is not in the correct format
        raise ValueError(f'{y_pred.ndim} dimensional data has been passed for vtk output. Only 5D (area/time) and 6D (volumentric/time) data should be considered.')
        
        
        
def seq2npy(y_pred, path, start_snap=0):
    '''
    This function saves the whole sequence as a npy file
    '''
    if y_pred.ndim == 6 or y_pred.ndim == 5:
        for kk in range(y_pred.shape[1]):
            np.save(f'{path}/snap_{kk+start_snap}.npy', y_pred.numpy()[0,kk,0,...]) # here ellipses does the trick
    else:
        raise ValueError(f'{y_pred.ndim} dimensional data has been passed for npy output. Only 5D (area/time) and 6D (volumentric/time) data should be considered.')
    
    
