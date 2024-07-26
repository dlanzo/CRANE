## CRANE
# Convolutional Recurrent Approximation of Nanoscale Evolution
This is a project to approximate PDE evolution of microstructures through Convolutional Recurrent Neural Networks based on the pytorch framework


# Code dependencies
Code is written in python. Dependencies needed are (should be exhaustive list):
- `matplotlib` (visualization; version 3.4.1 working; older versions may produce bugs in logging functions)
- `numpy` (mathematical function and algebra module)
- `torch` (NN definition, training and deployment; GPU utilization; "2.0.0+cu117" with compile, "1.9.0+cu102", "1.10.1+cu113" and "1.13.0+cu117" tested, but versions down to 1.7 should run)
- `torchvision` (computer vision part of torch; "0.10.0+cu102" and "0.11.2+cu113" versions tested)
- `numba` (optional; jit compilation of python code: needed to speed up some operations related to initial conditions. Code should work as intended, if slower, without it)

Installation though pip3 (or in a virtualenv) should be enough.


# Code structure
Source code is contained in src folder. In main, there is a _`train.py`_ script for training a model. Both 2D and 3D data and conservative and non conservative evolutions are implemented. After training is successfull, the NN can be imported and used as a pytorch NN. At the moment there is also a (slightly) higher level _`run_3D.py`_ script, which allows to use a trained model to run evolutions in 3D. Other scripts/a more general version will be implemented in future releases.

- _`train.py`_ launches a training program (select train and validation set with relative args)
- _`run_3D.py`_ handles 3D evolution

It is suggested to use a launch file (e.g. a .sh file) containing "python3 train.py --options" for training. In any case  args will be saved after training starts.
Similarly, consider using an input file for running evolutions.

_Remember that trained model are pytorch nn.Modules and, as such, can be saved/loaded as needed!_

src module:
- _`geometry`_ contains functions related to the definition of initial conditions (required for scripts)
- _`classes`_ contains NN module classes (ConvGRU, CommiteeModel, ...) and initial codition related (Phi)
- _`dataloaders`_ contains classes related to the loading of data
- _`parser`_ contains classes implementing argument parsers; this is the UI interface
- _`utils`_ is a big source file containing recurrent functions in all main programs; at the moment they are all imported into main programs

_IMPORTANT NOTE: some of these modules may contain legacy code, experimental features and broken functions. Enter in these files and modify code at your own risk. Sorry for the mess, a major cleanup will be performed in future releases!._
Use `python3 script.py --help` to have information on possible options

# Folder structure
The scope of folders is as follows:
- _`src`_ contains the src module (see previous section).
- _`data`_ is intended to contain dataset for training/validation/testing purpouses.
- _`models`_ contains deployed models to be synced with the git repository.
- _`out`_ contains the ouput of "CRANE simulations", e.g. the outputs of the _`run_3D.py`_ script.
- _`tools`_ is a folder intended to contain utility scripts (at the moement, examples of runfiles)
- _`train_logs`_ contains the ouput of training procedures (i.e. training statistics, model checkpoints, prediction snapshots at different training stages...)
