# <<< importing stuff <<<
import os
from argparse import ArgumentParser
from warnings import warn
import time
# --- importing stuff ---

class GeneralParser():
    '''
    This is a class implementing a general parser, containing args required for all tasks (e.g. reshape dimension, padding mode, NN topology)
    '''
    
    def __init__(self):
        
        self.parser = ArgumentParser()
        
        self.parser.add_argument(
            '--padding',
            type        = str,
            default     = 'circular',
            choices     = ['circular', 'zeros', 'reflect'],
            help        = 'Padding mode for NN. In the case of PBCs, select "circular" (it is the default option)'
            )
        
        self.parser.add_argument(
            '--bias',
            action      = 'store_true',
            help        = 'Enable bias in convolutional layers.'
            )
        
        self.parser.add_argument(
            '--device',
            type        = str,
            default     = 'cuda',
            help        = 'Selects the device to be used ("cpu" or "cuda"). In the case of multiple GPUs, a specific "cuda:n" can be selected.'
            )
        
        self.parser.add_argument(
            '--size',
            type        = int,
            default     = -1,
            help        = 'Dimension at which images are to be rescaled (measured in pixels); currently working only for square images. Set to -1 to remove resizing.'
            )
        
        self.parser.add_argument(
            '--cmap',
            type        = str,
            default     = 'gray',
            help        = 'Colormap to be used in plotting and gifs.'
            )
        
        self.parser.add_argument(
            '--hidden',
            type        = int,
            default     = 2,
            help        = 'Number of hidden layers in ConvGRU.'
            )
        
        self.parser.add_argument(
            '--channels',
            type        = int,
            default     = 35,
            help        = 'Number of channels in ConvGRU.'
            )
        
        self.parser.add_argument(
            '--kernel_size',
            type        = int,
            default     = 3,
            help        = 'Size of convolution kernels in ConvGRU.'
            )
        
        self.parser.add_argument(
            '--symm_kernel',
            action      = 'store_true',
            help        = 'Enforce bisymmetric kernels (should provide local rotational equivariance).'
            )
        
        self.parser.add_argument(
            '--seed',
            type        = int,
            default     = 0,
            help        = 'Seed for RNGs.'
            )
        
        self.parser.add_argument(
            '--nographics',
            action      = 'store_true',
            help        = 'Disable graphical output (e.g. training loss plots, predictions pngs etc.). If selected, it will also disable gifs (diregarding --nogifs use).'
            )
        
        self.parser.add_argument(
            '--nogifs',
            action      = 'store_true',
            help        = 'Diable gifs output. If --nographics has not been used, png outputs will still be present'
            )
        
        self.parser.add_argument(
            '--nocrop',
            action      = 'store_true',
            help        = 'Disable image cropping'
            )
        
        self.parser.add_argument(
            '--croplims',
            type        = float,
            nargs       = 2,
            default     = (0.25, 0.75),
            help        = 'Cropping bounds (as a fraction of total image; lower/upper)'
            )
        
        self.parser.add_argument(
            '--debug',
            action      = 'store_true',
            help        = 'Launch program in debug mode. This will lead to partial evaluation of training/validation and test sets.'
            )
        
        self.parser.add_argument(
            '--id',
            type        = str,
            default     = '',
            help        = 'ID of the training procedure (will be used for model, training outputs, testing, inference...).'
            )
        
        self.parser.add_argument(
            '--nproc',
            type        = int,
            default     = 1,
            help        = 'Number of procs to be used in parallelized operations (e.g. num_workers in dataloaders or graphical output processes).'
            )
        
        self.parser.add_argument(
            '--divergence',
            action      = 'store_true',
            help        = 'Models are in divergence mode (the evolution follows a continuity law)'
            )
        
        self.parser.add_argument(
            '--num_params',
            type        = int,
            default     = 0,
            help        = 'Number of external parameters regulating the evolution. This value is set to 0 if there is no external parameter to be passed'
            )
        
        self.parser.add_argument(
            '--threeD',
            action      = 'store_true',
            help        = 'Enables 3D mode. This requires the dataset to be composed of .npy data (only supported format at the moment); some arguments will not work because of lack of implementation or sense. BETA'
            )
        
        self.parser.add_argument(
            '--dropout',
            action      = 'store_true',
            help        = 'Enable dropout of Convolutional Recurrent layers (dropout hidden state channels are selected at each new forward run).'
            )
        
        self.parser.add_argument(
            '--dropout_prob',
            type        = float,
            default     = 0.25,
            help        = 'Dropout probability.'
            )
        
        self.parser.add_argument(
            '--extract_param',
            action      = 'store_true',
            help        = 'Set training mode for extracting parameters'
            )
        
        self.parser.add_argument(
            '--rotation',
            action      = 'store_true',
            help        = 'Rotate images in training procedure (continuous angle; overrides other rotations).'
            )
        
        self.parser.add_argument(
            '--rotation90',
            action      = 'store_true',
            help        = 'Rotate images in training procedure (90 degrees and multiples; overrides rotation_order if set).'
            )
        
        self.parser.add_argument(
            '--rotation_order',
            type        = int,
            default     = 0,
            help        = 'Set a custom rotational symmetry (e.g. a value of 3 means triangular symmetry, an order of 6 means hexagonal symmetry)'
            )
        
        self.parser.add_argument(
            '--reflectionX',
            action      = 'store_true',
            help        = 'Enables reflection symmetry along x in data augmentation'
            )
        
        self.parser.add_argument(
            '--reflectionY',
            action      = 'store_true',
            help        = 'Enables reflection symmetry along y in data augmentation'
            )
        
        self.parser.add_argument(
            '--reflectionZ',
            action      = 'store_true',
            help        = 'Enables reflection symmetry along z in data augmentation (only for 3D cases)'
            )
        
        self.parser.add_argument(
            '--reflection',
            action      = 'store_true',
            help        = 'Enables reflection symmetry in both x and y (overrides reflectionX, reflectionY and reflectionZ)'
            )
        
        self.parser.add_argument(
            '--vtk',
            action      = 'store_true',
            help        = 'Toggles saving the output in vtk format'
            )
        
        self.parser.add_argument(
            '--npy',
            action      = 'store_true',
            help        = 'Toggles saving the output in npy format'
            )
        
        
    def parse_args(self):
        
        args = self.parser.parse_args()
        
        # add "derived" arguments
        if args.nographics:
            args.graphics   = False
            args.gifs       = False
            args.vtk        = False
            args.npy        = False
        elif args.nogifs:
            args.graphics   = True
            args.gifs       = False
        else:
            args.graphics   = True
            args.gifs       = True
        
        # set both reflections on X, Y and Z (will be ignored if 2D) if --reflection is used
        if args.reflection:
            args.reflectionX = True
            args.reflectionY = True
            args.reflectionZ = True
        
        if args.nocrop:
            args.crop = False
        else:
            args.crop = True
            
        return args


class TrainingParser( GeneralParser ):
    '''
    This class implements a parser specific for training tasks. Therefore, it will contain number of epochs, lr, ...
    '''
    def __init__(self):
        
        super().__init__()
        
        self.parser.add_argument(
            '--epochs',
            type        = int,
            default     = 1_000,
            help        = 'Number of training epochs.'
            )
        
        self.parser.add_argument(
            '--lr',
            type        = float,
            default     = 5e-4,
            help        = 'Learning rate to be used.'
            )
        
        self.parser.add_argument(
            '--batch',
            type        = int,
            default     = 1,
            help        = 'Training batch dimension.'
            )
        
        self.parser.add_argument(
            '--weightd',
            type        = float,
            default     = 0.0,
            help        = 'Weight decay argument in Adam opptimizer.'
            )
        
        self.parser.add_argument(
            '--massW',
            type        = float,
            default     = 2.0,
            help        = 'Weight of mass conservation term in loss function.'
            )
        
        self.parser.add_argument(
            '--translation',
            action      = 'store_true',
            help        = 'Translate images in training procedure (in principle should not be useful because of translational equivaruance in convolution).'
            )
        
        self.parser.add_argument(
            '--train_set',
            type        = str,
            default     = 'data/table_comb.txt',
            help        = 'File containing paths of images to be used in training.'
            )
        
        self.parser.add_argument(
            '--valid_set',
            type        = str,
            default     = 'data/table_comb_valid.txt',
            help        = 'File containing paths of images to be used in validation.'
            )
        
        self.parser.add_argument(
            '--subseq_min',
            type        = int,
            default     = 1,
            help        = 'Minumum length of the subsequence to be passed during training.'
            )
        
        self.parser.add_argument(
            '--subseq_max',
            type        = int,
            default     = 99,
            help        = 'Maximum length of the subsequence to be passed during training'
            )
        
        self.parser.add_argument(
            '--logfreq',
            type        = int,
            default     = 1,
            help        = 'Logging frequency on terminal.'
            )
        
        self.parser.add_argument(
            '--dual',
            action      = 'store_true',
            help        = 'Enable passing all images as a "normal" and mirrored dual (phi is reversed).'
            )
        
        self.parser.add_argument(
            '--superbatch',
            type        = int,
            default     = 1,
            help        = 'Model parameters update frequency (allows for simulating the effects on training of bigger batches).'
            )
        
        self.parser.add_argument(
            '--bootstrap',
            action      = 'store_true',
            help        = 'Use a bootstrap procedure to resample the training/validation sets.'
            )
        
        self.parser.add_argument(
            '--reload_model',
            type        = str,
            default     = '',
            help        = 'Specify path of a .pt model to reload to continue training'
            )
        
        self.parser.add_argument(
            '--twin_image',
            action      = 'store_true',
            help        = 'Enable twin image trainin mode. In this mode, a second image is superimosed to the first one (checking and avoiding overlap). This shoold make training more robust to domain interactions, but has a greater computational cost'
            )
        
        self.parser.add_argument(
            '--ramp',
            action      = 'store_true',
            help        = 'Use a linear "ramp" which decreses the number of examples passed in teacher loss mode.'
            )
        
        self.parser.add_argument(
            '--ramp_length',
            type        = int,
            default     = 100,
            help        = 'Number of epochs in the ramp to reach full backpropagation through time.'
            )
        
        self.parser.add_argument(
            '--start_ramp',
            type        = int,
            default     = 0,
            help        = 'Ramp offset (effective ramp length will be the difference between ramp_length and this value).'
            )
        
        self.parser.add_argument(
            '--noise_reg',
            type        = float,
            default     = 0.0,
            help        = 'Noise regularization to be used during training (standard deviation of gaussian noise corruption in series generation, should make the Net more resilient to wrong projections in next state)'
            )
        
        
    def parse_args(self):
        
        # rescale learning rate
        args = super().parse_args()
        
        # set both reflections on X and Y if --reflection is used
        if args.reflection:
            args.reflectionX = True
            args.reflectionY = True
            args.reflectionZ = True
        
        args.lr /= args.superbatch
        
        # create helper arg on reloading
        if args.reload_model != '':
            args.reload = True
        else:
            args.reload = False
        
        return args
        

class NonTrainParser( GeneralParser ):
    '''
    This class implements a parser for general non-training purpouses.
    '''
    def __init__(self):
        
        super().__init__()
        
        self.parser.add_argument(
            '--tot_frames',
            type        = int,
            default     = 100,
            help        = 'Total number of frames in the evolution.'
            )
        
        self.parser.add_argument(
            '--in_frames',
            type        = int,
            default     = 1,
            help        = 'Number of provided initial evolution frames.'
            )
        
        self.parser.add_argument(
            '--model_name',
            type        = str,
            default     = 'models/model.pt',
            help        = 'Model path to be loaded. If a folder, all contained .pt files will be loaded in a committee model.'
            )
        
        self.parser.add_argument(
            '--out_all',
            action      = 'store_true',
            help        = 'Out all evolution as png frames.'
            )
        
        self.parser.add_argument(
            '--AR',
            action      = 'store_true',
            help        = 'Estimates the Aspect Ratio of the evolving configuration. This makes sense only for very simple shapes (e.g. rectangles or ellipses).'
            )
    

class EvaluationParser( NonTrainParser ):
    '''
    This class is an implementation of an argument parser for evaluation of models
    '''
    
    def __init__(self):
        
        super().__init__()
        
        self.parser.add_argument(
            '--init_geo',
            type        = str,
            default     = '',
            help        = 'Path for a .py file containing a definition of an initial geometry. The definition should be into two lists containing shape functions as in PF.geometry.geo_utils called "shapelist_add" and "shapelist_remove".'
            )
        
        self.parser.add_argument(
            '--load_image',
            type        = str,
            default     = '',
            help        = 'Path for an image that will be taken as initial condition.'
            )
        
        self.parser.add_argument(
            '--scatter',
            action      = 'store_true',
            help        = 'Activate scatter mode. Model outputs will be saved independently and not collapsed in an aggregate prediction.'
            )
        
        self.parser.add_argument(
            '--dual',
            action      = 'store_true',
            help        = 'Activate dual dynamics for imported image or generated geometry.'
            )
        
        self.parser.add_argument(
            '--params_list',
            type        = float,
            nargs       = '+',
            help        = 'List containig the parameters to be passed to the predictor; to be consistent with num_params'
            )
        
        self.parser.add_argument(
            '--save_every',
            type        = int,
            default     = 1,
            help        = 'Saving frequency of generated states in evolution'
            )
         
    def parse_args(self):
        
        args = super().parse_args()
        
        # if an init_geo file has been specified, the flag for using externally defined geometry is raised
        if args.init_geo and args.load_image:
            args.gengeo = False
            warn(
                'Both geometry init file and image init file were specified. Falling back on using image init. If you want to use init_geo file abort and remove "--load_image" flag at launchtime.'
                )
        elif args.init_geo:
            args.gengeo = True
        elif args.load_image and (not args.init_geo):
            args.gengeo = False
        elif (not args.init_geo) and (not args.load_image):
            raise RuntimeError('An initial condition was not specified.')
            
        return args
    
    
class TestParser( NonTrainParser ):
    '''
    This class is an implementation of an argument parser for testing models
    '''
    
    def __init__(self):
        
        super().__init__()
        
        self.parser.add_argument(
            '--test_set',
            type        = str,
            default     = 'data/table_comb_test.txt',
            help        = 'File containing paths of images to be used in testing.'
            )
        
        self.parser.add_argument(
            '--translation',
            action      = 'store_true',
            help        = 'Translate images in training procedure (in principle should not be useful because of translational equivaruance in convolution).'
            )
        
        self.parser.add_argument(
            '--dual_prob',
            type        = float,
            default     = 0.5,
            help        = 'Probability of passing dual dynamics in test'
            )
        
        self.parser.add_argument(
            '--batch',
            type        = int,
            default     = 1,
            help        = 'Batch size (only used when extract_param is true)'
            )
        
    def parse_args(self):
        
        args = super().parse_args()
        if not args.extract_param:
            if args.batch > 1:
                print('Resetting batch to 1 due to testing for evolution modes')
                time.sleep(1)
                args.batch = 1 # <- during testing,  batch is 1 for simplicity except in the case of extract_param
        
        return args
