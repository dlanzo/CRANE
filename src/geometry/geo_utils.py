# <<< import external stuff <<<
import numpy as np
# --- import external stuff ---

# <<< import numba <<<
try:
    from numba import njit
except ImportError:
    print('It seems that "numba" is not installed on this machine or there are problems in importing it. Falling back on non-jitted versions of scripts. Some operations will be slower. Consider installing it (e.g. by "pip install numba").')
    
    def njit(fun): # <- alternative definition of njit
        return fun
# --- import numba ---

def rectangle_fun(center, theta, sides):
    def shape_fn(x,y):
        x -= center[0]
        y -= center[1]
        x, y = x*np.cos(theta) + y*np.sin(theta), x*np.sin(-theta)+y*np.cos(theta)
        x /= sides[0]/2
        y /= sides[1]/2
        return x**20 + y**20 - 1
            
    return njit(shape_fn)

def ellipse_fun(center, theta, axes):
    def shape_fn(x,y):
        x -= center[0]
        y -= center[1]
        x, y = x*np.cos(theta) + y*np.sin(theta), x*np.sin(-theta)+y*np.cos(theta)
        x /= axes[0]
        y /= axes[1]
        return x**2 + y**2 - 1
            
    return njit(shape_fn)

def square_fun(center, theta, side):
    return rectangle_fun(center, theta, sides=(side, side))

def circle_fun(center, radius):
    return ellipse_fun(center, theta=0, axes=(radius, radius) )
