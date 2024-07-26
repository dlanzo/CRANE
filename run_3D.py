# <<< import external stuff <<<
import torch
import torch.nn as nn
from torchvision import utils, datasets, transforms

import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')

import os
import sys

import numpy as np

import PIL
from PIL import Image

import time
# --- import external stuff ---

# <<< import my stuff <<<
from PF.classes import *
from PF.utils import *
from PF.parser import EvaluationParser
# --- import my stuff ---


class PersistentConvGRU3D(ConvGRU3D):
    '''
    This is a class implementing a ConvGRU3D with persistent memory
    '''
    
    def __init__(self, *args, **kwargs):
        '''
        Constructor method
        '''
        super(PersistentConvGRU3D, self).__init__(*args, **kwargs)
        self.hidden_list = None
        
        
    #def divergence(self, x):
        #'''
        #This method calculates the divergence of the given field using finite differences approximation
        #'''
        
        #Jx, Jy, Jz = torch.split(x, 1, dim=1)
        
        #Jx = Jx - torch.mean(Jx, dim=(-1,-2,-3), keepdim=True)
        #Jy = Jy - torch.mean(Jy, dim=(-1,-2,-3), keepdim=True)
        #Jz = Jz - torch.mean(Jz, dim=(-1,-2,-3), keepdim=True)
        
        #gradx = self.divergence_filters[0](Jx)
        #grady = self.divergence_filters[1](Jy)
        #gradz = self.divergence_filters[2](Jz)
        
        #divergence = gradx+grady+gradz
        
        #return divergence
        
    
    def set_hidden(self, in_sequence):
        
        self.hidden_list = []
        
        device = in_sequence.device
        
        self.hidden_list_shape = list(in_sequence.shape)
        self.hidden_list_shape[2] = self.hidden_channels # tensor is (Batch, Time, Channels, X, Y, Z)
        self.hidden_list_shape.pop(1) # remove time dimension
        
        for ll in range(self.hidden_units):
            self.hidden_list.append(
                torch.zeros(self.hidden_list_shape, device=device, requires_grad=False)
                )
        
        
    def forward_old(self, in_sequence, future=0, params=None):
        '''
        This method is called from forward if you are not in divergence mode
        '''
        
        if self.hidden_list is None: self.set_hidden(in_sequence)
        
        outputs = []
        
        device = in_sequence.device #'cuda' if in_sequence.is_cuda else 'cpu'
        
        for input_t in in_sequence.split(1, dim=1):
            
            for kk in range(self.hidden_units):

                if kk==0:
                    self.hidden_list[kk] = self.GRU_list[kk](input_t.squeeze(1), self.hidden_list[kk])
                else: self.hidden_list[kk] = self.GRU_list[kk](self.hidden_list[kk-1], self.hidden_list[kk])
            
                output = self.toOut(self.hidden_list[-1])
                output = self.sigmoid(output)
                
            outputs += [output]
            

        for _ in range(future):
            for kk in range(self.hidden_units):
                
                if kk==0: self.hidden_list[kk] = self.GRU_list[kk](self.cat_params(output, params), self.hidden_list[kk])
                else: self.hidden_list[kk] = self.GRU_list[kk](self.hidden_list[kk-1], self.hidden_list[kk])
                
            output = self.toOut(self.hidden_list[-1])
            output = self.sigmoid(output)
            
            outputs += [output]
        
        outputs = torch.stack(outputs, dim=1)

        return outputs
    
    
    def forward_div(self, in_sequence, future=0, params=None):
        '''
        This method is called in divergence mode; BETA
        '''
        
        if self.hidden_list is None: self.set_hidden(in_sequence)
        
        outputs = []
        
        device = in_sequence.device#'cuda' if in_sequence.is_cuda else 'cpu'

            
        for input_t in in_sequence.split(1, dim=1):
            
            for kk in range(self.hidden_units):

                if kk==0:
                    self.hidden_list[kk] = self.GRU_list[kk](input_t.squeeze(1), self.hidden_list[kk])
                else: self.hidden_list[kk] = self.GRU_list[kk](self.hidden_list[kk-1], self.hidden_list[kk])
            
            output = self.toOut(self.hidden_list[-1])
            output = input_t.squeeze(1)+self.divergence(output)
                
            outputs += [output]
            
        for _ in range(future):
            
            output_old = output
            
            for kk in range(self.hidden_units):
                
                if kk==0: self.hidden_list[kk] = self.GRU_list[kk](output, self.hidden_list[kk])
                else: self.hidden_list[kk] = self.GRU_list[kk](hidden_list[kk-1], self.hidden_list[kk])
            
            output = self.toOut(self.hidden_list[-1])
            output = output_old+self.divergence(output)
            
            outputs += [output]
        
        outputs = torch.stack(outputs, dim=1)

        return outputs
    

    def forward(self, in_sequence, future=0, params=None):
        
        in_sequence = self.cat_params(in_sequence, params)
        
        if self.div_mode:
            return self.forward_div(in_sequence, future, params=params) # conservative dynamics
        else:
            return self.forward_old(in_sequence, future, params=params) # non conservative dynamics


def main():
    '''
    Main function definition
    '''
    with torch.no_grad(): # disable gradient tracking
        
        # parsing external parameters
        parser      = EvaluationParser()
        args        = parser.parse_args()
        
        # checking we have a 3D evolution
        if not args.threeD:
            raise ValueError('Script is not in 3D mode.')
        
        # create and load model
        args        = build_predict_dir_tree(args)
        print(args.paths)
        
        # setting random seeds
        np.random.seed( args.seed )
        torch.manual_seed( args.seed )
        
        # define model
        # Define model and put to device
        model = PersistentConvGRU3D(
            hidden_units        = args.hidden,
            input_channels      = 1,
            hidden_channels     = args.channels,
            kernel_size         = args.kernel_size,
            padding_mode        = args.padding,
            separable           = False,
            bias                = args.bias,
            divergence          = args.divergence,
            num_params          = args.num_params
            )
        
        reloaded_model = ConvGRU3D(
            hidden_units        = args.hidden,
            input_channels      = 1,
            hidden_channels     = args.channels,
            kernel_size         = args.kernel_size,
            padding_mode        = args.padding,
            separable           = False,
            bias                = args.bias,
            divergence          = args.divergence,
            num_params          = args.num_params
            )
        
        print_model_info(model)
        
        #model = torch.compile(model)
        #reloaded_model = torch.compile(reloaded_model)
        
        # Reload operation
        args.reload_model = args.model_name
        reloaded_model = import_model(reloaded_model, args)
        
        model.GRU_list = reloaded_model.GRU_list
        model.toOut = reloaded_model.toOut
            
        if args.symm_kernel:
            model.symmetrize()
            
        if args.divergence:
            model.make_div_filters( torch.zeros(1, device=args.device) )
            
            
        model.to(args.device)
        
        
        
        
        # resize initial state
        resizer = lambda x : torch.nn.functional.interpolate(x, size=(args.size, args.size, args.size))
        
        # time evolution parameters
        future = args.tot_frames-1 # need to remove 1 as we have an initial condition
        
        # time evolution
        print('Starting evolution...')
        
        start_evo = time.time()
        
        start_frame_num = args.load_image.split('_')
        start_frame_num = int(start_frame_num[-1][:-4])
        
        splitted_name = args.load_image.split('_')
        base_name = ''
        for chunk in splitted_name[:-1]:
            base_name += chunk
            base_name += '_'
        
        
        for kk in range(2*args.in_frames):
            
            print(f'Processing initial frame {kk}')
            
            if kk%2 == 0:
                frame_num = (kk//2)+start_frame_num
                
                load_name = f'{base_name}{frame_num}.npy'
                
                 # load initial condition
                initial_state = np.load( load_name )
                initial_state = torch.from_numpy( initial_state ).float()
                # add fake batch dimension
                initial_state = initial_state.to(args.device)
                
                for _ in range(2): # resize to have 6 indices
                    initial_state = initial_state.unsqueeze(0)
                
                initial_state = resizer(initial_state)
                initial_state = initial_state.unsqueeze(0)
            
            else:
                initial_state = y_pred
            
            y_pred_cpu = initial_state[:,-1,:,:,:,:].unsqueeze(0).cpu()
            
            # graphical output
            if args.graphics and kk%args.save_every==0:
                
                seq2vtk(
                    y_pred      = y_pred_cpu,
                    path        = f'{args.paths["gifs"]}',
                    start_snap  = kk//args.save_every
                    )
                
                seq2npy(
                    y_pred      = y_pred_cpu,
                    path        = f'{args.paths["gifs"]}',
                    start_snap  = kk//args.save_every
                    )
                
            y_pred = model( initial_state, future=0 )
            
            
        
        for kk in range(future-args.in_frames):
            
            print(f'Evolution frame {kk}')
            
            y_pred_cpu = initial_state[:,-1,:,:,:,:].unsqueeze(0).cpu()
            
            # graphical output
            if args.graphics and (kk%args.save_every==1 or args.save_every==1):

                seq2vtk(
                    y_pred      = y_pred_cpu,
                    path        = f'{args.paths["gifs"]}',
                    start_snap  = (kk+args.in_frames-1)//args.save_every
                    )
                
                seq2npy(
                    y_pred      = y_pred_cpu,
                    path        = f'{args.paths["gifs"]}',
                    start_snap  = (kk+args.in_frames-1)//args.save_every
                    )
                
            initial_state = model( initial_state, future=0 )
            
        end_evo = time.time()
        print(f'Time elapsed is {end_evo-start_evo}')


if __name__ == '__main__':
    main()
