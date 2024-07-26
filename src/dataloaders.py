# <<< import external stuff <<<
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, utils

import os

import numpy as np

import PIL
from PIL import Image

import time
# --- import external stuff ---

# <<< import my stuff <<<
from src.utils import make_square
# --- import my stuff ---

class TabulatedSeries(torch.utils.data.Dataset):
    '''
    Create a dataset. Data are indexed in a text file containing images paths.
    init arguments are as follows:
    - table_path        ->  path referencing to the text file containing examples' paths
    - transform         ->  transforms to be applied to images
    - rotation          ->  toggles continuous rotation of the image
    - reflection        ->  toggles reflection of the images
    - translation       ->  toggles translation of the image (PBCs will be applied)
    - rotation_90       ->  toggles 90° rotations of the input
    '''
    
    def __init__(self, table_path, params_num=0, transform=None, rotation=True, reflections = (False, False), translation=True, rotation_90=False, rotation_order=0, cropkey=True, crop_lim=(0.25,0.75), bootstrap_loader=False, twin_image=False):
        
        super(TabulatedSeries, self).__init__()
        
        self.params_num     = params_num
        self.table_path     = table_path
        self.transform      = transform
        
        self.rotation       = rotation
        self.rotation_90    = rotation_90
        self.rotation_order = rotation_order
        
        self.reflectionX    = reflections[0]
        self.reflectionY    = reflections[1]
        self.translation    = translation
        
        self.cropkey        = cropkey
        self.crop_lim       = crop_lim
        
        self.bootstrap_loader = bootstrap_loader
        
        self.twin_image     = twin_image
        
        with open(self.table_path,'r') as table_file:
            self.table  = table_file.readlines()
            self.length = len(self.table)
            
        if self.bootstrap_loader:
            table_old   = self.table
            self.table  = []
            id_set      = set()
            for line in table_old:
                id_set.add( line.split()[-1] )
            id_list = list(id_set)
            del id_set # <- free memory
            for ii in range( len(id_list) ):
                index = torch.randint( 0, len(id_list), (1,) )
                lines2append = [
                    line for line in table_old if line.split()[-1]==id_list[index]
                    ]
                for line in lines2append:
                    self.table.append( line )
            print(f'Table length is {len(self.table)}; __len__ is {self.length}')
            self.length = len(self.table)
            
        print('Dataloader is checking data extension...', end='')
        
        first_ext = self.table[0].split()[0][-3:]
        for line in self.table: # check all extensions are the same
            ext = line.split()[0][-3:]
            if ext != first_ext:
                raise NotImplementedError(f'Extension {ext} found which is not consistent with previous {first_ext}... Aborting.')
            
        if first_ext == 'png':
            self.extension = 'png'
        elif first_ext == 'npy':
            self.extension = 'npy'
        else:
            raise NotImplementedError( f'Data format {first_ext} not implemented... Aborting.' )
        
        print('DONE!')
            
        
    def __len__(self): return self.length

    
    def pick_image(self, idx):
        '''
        This function loads and transforms an image
        '''
            
        table_line = self.table[idx]
        splitted_line = table_line.split()
        without_idx = splitted_line # <- the [:-1] is to remove id
        
        if self.params_num != 0:
            without_idx = table_line.split()[:(-self.params_num)]
            params = splitted_line[-self.params_num:]
        else:
            params = []
        
        paths = without_idx
        
        out_list = []
        
        for path in paths:
            
            image = make_square(Image.open(path), cropkey=self.cropkey, crop_lim=self.crop_lim)
            out_list.append(image)
            
        # rotation management block
        if self.rotation: # continuous rotation
            rotation_angle = 360*torch.rand(1).item()
            for ii in range(len(out_list)):
                out_list[ii] = out_list[ii].rotate(rotation_angle, PIL.Image.NEAREST, fillcolor=(0,0,0))
        elif self.rotation_90: # special case of square symmetry (also for PBC case)
            coin = torch.rand(1).item()
            if 0.0 <= coin < 0.25:
                rotation_key = PIL.Image.ROTATE_90
            elif 0.25 <= coin < 0.5:
                rotation_key = PIL.Image.ROTATE_180
            elif 0.5 <= coin < 0.75:
                rotation_key = PIL.Image.ROTATE_270
            if coin < 0.75:
                for ii in range(len(out_list)):
                    out_list[ii] = out_list[ii].transpose(rotation_key)
        elif self.rotation_order != 0: #prescribed symmetry (basically same as rotation but on integers)
            if type(self.rotation_order) is not int:
                raise ValueError('A non integer order of rotation was used for dataloaders')
            rotation_angle = 360/self.rotation_order*torch.randint(self.rotation_order).item()
            for ii in range(len(out_list)):
                out_list[ii] = out_list[ii].rotate(rotation_angle, PIL.Image.NEAREST, fillcolor=(0,0,0))
                    
                    
        # reflect frames if reflection is enabled
        if self.reflectionX:
            hor_flip = torch.rand(1) >= 0.5
            for ii in range(len(out_list)):
                if hor_flip:
                    out_list[ii] = out_list[ii].transpose(PIL.Image.FLIP_LEFT_RIGHT)
        if self.reflectionY:
            ver_flip = torch.rand(1) >= 0.5
            for ii in range(len(out_list)):
                if ver_flip:
                    out_list[ii] = out_list[ii].transpose(PIL.Image.FLIP_TOP_BOTTOM)
        
        # apply transforms
        if self.transform:
            for ii in range(len(out_list)):
                out_list[ii] = self.transform(out_list[ii])
            
        # apply translations (should not be necessary for CNN)
        if self.translation:
            vertical    = torch.randint(0, out_list[0].shape[-1], (1,) ).item()
            horizontal  = torch.randint(0, out_list[0].shape[-2], (1,) ).item()
            for ii in range(len(out_list)):
                out_list[ii] = torch.roll( out_list[ii], (vertical, horizontal), dims=(-1,-2) )
                
        for ii in range(len(out_list)):
            out_list[ii] = out_list[ii].unsqueeze(1)
            
            
        out_tensor = torch.cat(out_list, dim=0)
                
        return out_tensor, params
    
    
    def pick_npy(self, idx):
        '''
        Same as pick image, but for npy files
        '''
        table_line = self.table[idx]
        splitted_line = table_line.split()
        without_idx = splitted_line # <- the [:-1] is to remove id
        
        if self.params_num != 0:
            without_idx = table_line.split()[:(-self.params_num)]
            params = splitted_line[-self.params_num:]
        else:
            params = []
        
        paths = without_idx
        
        out_list = []
        
        for path in paths:
            
            image = torch.from_numpy( np.load(path) ).float().unsqueeze(0).unsqueeze(0) # 1x res x res image
            out_list.append(image)
            
        # rotation management block
        if self.rotation: # continuous rotation
            raise NotImplementedError('Continuus rotation is not implemented yet for .npy datasets... Aborting.')
        elif self.rotation_90: # special case of square symmetry (also for PBC case)
            coin = torch.rand(1).item()
            if 0.0 <= coin < 0.25:
                rotation_key = 1
            elif 0.25 <= coin < 0.5:
                rotation_key = 2
            elif 0.5 <= coin < 0.75:
                rotation_key = 3
            if coin < 0.75:
                for ii in range(len(out_list)):
                    out_list[ii] = torch.rot90( out_list[ii], k=rotation_key, dims=(-1,-2) )
        elif self.rotation_order != 0: #prescribed symmetry (basically same as rotation but on integers)
            raise NotImplementedError('Rotations other than square is not implemented yet for .npy datasets... Aborting.')
                    
        # reflect frames if reflection is enabled
        if self.reflectionX:
            hor_flip = torch.rand(1) >= 0.5
            for ii in range(len(out_list)):
                if hor_flip:
                    out_list[ii] = torch.flip(out_list[ii], dims=(-2,))
        if self.reflectionY:
            ver_flip = torch.rand(1) >= 0.5
            for ii in range(len(out_list)):
                if ver_flip:
                    out_list[ii] = torch.flip(out_list[ii], dims=(-1,))
        
        # apply transforms
        #if self.transform:
            #for ii in range(len(out_list)):
                #out_list[ii] = self.transform(out_list[ii])
            
        out_tensor = torch.cat(out_list, dim=0)
                
        return out_tensor, params
    
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.extension == 'png':
            image1, params = self.pick_image(idx)
        elif self.extension == 'npy':
            image1, params = self.pick_npy(idx)
        else:
            raise NotImplementedError( f'Dataset format is not .png nor .npy... Aborting.' )
        
        if self.twin_image:
            
            good_image = False
            
            while not good_image:
                                
                image2, params2 = self.pick_image( torch.randint(self.length,(1,)).item() )
                
                vertical = torch.randint(image1.shape[-1], (1,)).item()
                horizontal = torch.randint(image1.shape[-2],(1,)).item()
                
                image2 = torch.roll( image2, (vertical, horizontal), dims=(-1,-2) )
                
                image = image1 + image2
                
                if torch.max(image)-1 <= 1e-3:
                    good_image = True
        
        else:
            image = image1
        
        if self.params_num == 0:    return image
        else:                       return image, [float(param) for param in params]
        



class TabulatedSeries3D(torch.utils.data.Dataset):
    '''
    Create a dataset. Data are indexed in a text file containing paths to .npy data
    '''
    
    def __init__(self, table_path, params_num=0, reflections = (False, False, False), rotation_90=False, bootstrap_loader=False, size=64):
        
        super(TabulatedSeries3D, self).__init__()
        
        self.params_num     = params_num
        self.table_path     = table_path
        
        self.rotation_90    = rotation_90
        
        self.reflectionX    = reflections[0]
        self.reflectionY    = reflections[1]
        self.reflectionZ    = reflections[2]
        
        self.bootstrap_loader = bootstrap_loader
        
        self.size           = size
        
        self.resizer = lambda x: torch.nn.functional.interpolate(x, size=(self.size, self.size, self.size))
        
        with open(self.table_path,'r') as table_file:
            self.table  = table_file.readlines()
            self.length = len(self.table)
            
        if self.bootstrap_loader:
            table_old   = self.table
            self.table  = []
            id_set      = set()
            for line in table_old:
                id_set.add( line.split()[-1] )
            id_list = list(id_set)
            del id_set # <- free memory
            for ii in range( len(id_list) ):
                index = torch.randint( 0, len(id_list), (1,) )
                lines2append = [
                    line for line in table_old if line.split()[-1]==id_list[index]
                    ]
                for line in lines2append:
                    self.table.append( line )
            print(f'Table length is {len(self.table)}; __len__ is {self.length}')
            self.length = len(self.table)
            
        
    def __len__(self): return self.length


    def load_to_tensor(self, path):
        '''
        This function transforms the given .npy raw data to a torch tensor
        '''
        out = torch.from_numpy( np.load(path) ).float().unsqueeze(0).unsqueeze(0)
        out = self.resizer(out)
        return out

    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        table_line = self.table[idx]
        splitted_line = table_line.split()
        without_idx = splitted_line # <- the [:-1] is to remove id
        
        if self.params_num != 0:
            without_idx = table_line.split()[:(-self.params_num)]
            params = splitted_line[-self.params_num:]
        else:
            params = []
        
        paths = without_idx
        out_list = []
        
        for path in paths: out_list.append( self.load_to_tensor(path) )
        
        # Transformation blocks
        
        if self.rotation_90:
            # rotation blocks
            kx, ky, kz = torch.randint(low=0, high=3, size=(3,)) #number of rotations for each axis
            for ii in range(len(out_list)):
                out_list[ii] = torch.rot90( out_list[ii], kx, [-1,-2] )
                out_list[ii] = torch.rot90( out_list[ii], ky, [-1,-3] )
                out_list[ii] = torch.rot90( out_list[ii], kz, [-2,-3] )
        
        if self.reflectionX:
            coin = torch.rand(1)
            if coin <= 0.5:
                for ii in range(len(out_list)):
                    out_list[ii] = torch.flip(out_list[ii], dims=(-3,))
        if self.reflectionY:
            coin = torch.rand(1)
            if coin <= 0.5:
                for ii in range(len(out_list)):
                    out_list[ii] = torch.flip(out_list[ii], dims=(-2,))
        if self.reflectionZ:
            coin = torch.rand(1)
            if coin <= 0.5:
                for ii in range(len(out_list)):
                    out_list[ii] = torch.flip(out_list[ii], dims=(-1,))
            
        out_list = torch.cat(out_list, dim=0)
            
        if self.params_num == 0:    return out_list
        else:                       return out_list, [float(param) for param in params]



    
def give_dataloaders(args):
    '''
    This function returns train, validation or testing dataloaders, depending on what variable is present in args. Return type is a dictionary.
    '''
    
    num_workers = args.nproc
    
    set_names = ['train_set', 'valid_set', 'test_set']
    
    has_sets = False
    
    for set_name in set_names:
        if hasattr(args, set_name):
            has_sets = True
        
    if not has_sets:
        raise RuntimeError('No dataset was detected in args. Check arguments are parsed correctly.')
    
    transform = transforms.Compose(
            [
                transforms.Grayscale( num_output_channels=1 ),
                transforms.Resize( args.size ),
                transforms.ToTensor()
            ]
        )
            
    dataloaders = {}
    
    master_path = args.paths['master']
            
    for set_name in set_names:
        
        if hasattr(args, set_name):
            
            set_path = getattr(args, set_name)
            
            bootstrap = args.bootstrap if hasattr(args, 'bootstrap') else False
            twin_image  = args.twin_image if hasattr(args, 'twin_image') else False
            params_num  = args.num_params if hasattr(args, 'num_params') else False
            
            dataset = TabulatedSeries(
                table_path          = set_path,
                params_num          = params_num,
                transform           = transform,
                translation         = args.translation,
                rotation            = args.rotation,
                rotation_90         = args.rotation90,
                reflections         = (args.reflectionX, args.reflectionY),
                cropkey             = args.crop,
                crop_lim            = args.croplims,
                bootstrap_loader    = bootstrap,
                twin_image          = twin_image
                )
            
            if bootstrap:
                with open(f'{master_path}/{set_name}_bootstrap.txt', 'w+') as bootstrap_file:
                    for line in dataset.table:
                        bootstrap_file.write(line)
            else:
                with open(f'{master_path}/{set_name}.txt', 'w+') as check_set_file:
                    for line in dataset.table:
                        check_set_file.write(line)
                        
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size      = args.batch,
                shuffle         = False if set_name == 'test_set' else True, # <- in the case of testing and validation, we want a fixed order for data
                num_workers     = num_workers,
                pin_memory      = True
                )
            
            dataloaders[set_name] = dataloader
    
    return dataloaders




def give_3D_dataloaders(args):
    '''
    This function returns train, validation or testing dataloaders, depending on what variable is present in args. Return type is a dictionary.
    '''
    
    num_workers = args.nproc
    
    set_names = ['train_set', 'valid_set', 'test_set']
    
    has_sets = False
    
    for set_name in set_names:
        if hasattr(args, set_name):
            has_sets = True
        
    if not has_sets:
        raise RuntimeError('No dataset was detected in args. Check arguments are parsed correctly.')
    
    dataloaders = {}
    
    master_path = args.paths['master']
            
    for set_name in set_names:
        
        if hasattr(args, set_name):
            
            set_path = getattr(args, set_name)
            
            bootstrap = args.bootstrap if hasattr(args, 'bootstrap') else False
            twin_image  = args.twin_image if hasattr(args, 'twin_image') else False
            params_num  = args.num_params if hasattr(args, 'num_params') else False
            
            dataset = TabulatedSeries3D(
                table_path          = set_path,
                params_num          = params_num,
                rotation_90         = args.rotation90,
                reflections         = (args.reflectionX, args.reflectionY, args.reflectionZ),
                bootstrap_loader    = bootstrap,
                size                = args.size
                )
            
            if bootstrap:
                with open(f'{master_path}/{set_name}_bootstrap.txt', 'w+') as bootstrap_file:
                    for line in dataset.table:
                        bootstrap_file.write(line)
            else:
                with open(f'{master_path}/{set_name}.txt', 'w+') as check_set_file:
                    for line in dataset.table:
                        check_set_file.write(line)
                        
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size      = args.batch,
                shuffle         = False if set_name == 'test_set' else True, # <- in the case of testing and validation, we want a fixed order for data
                num_workers     = num_workers,
                pin_memory      = True
                )
            
            dataloaders[set_name] = dataloader
    
    return dataloaders
