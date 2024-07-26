from geo_utils import *
import random
import os
import matplotlib.pyplot as plt
import PIL

'''
This file allows the generation of a fake TS. This might be useful for benchamrking pourposes or to augment real TS with simple configurations wich do not require a real simulation (e.g. stationary evolutions).

It is intended to be modified by the user directly, so no UI is implemented
'''

if __name__ == '__main__':
    
    # Generate master folder
    os.mkdir('fake_TS')
    
    for ii in range(20):
        
        # Generate subfolder
        subfoldername = 'fake_TS/output_{id}'.format(id=ii)
        os.mkdir( subfoldername )
        
        center_pos = (500,500)
        radius_pos = 100+125*random.random()
        
        center_neg = (center_pos[0]+100*(random.random()-0.5), center_pos[1]+100*(random.random()-0.5))
        radius_neg = 30*(1+random.random())
        
        phi = Phi( (1000,1000) )
        
        positive_circle = circle_fun(
                            center = center_pos,
                            radius = radius_pos
                            )
        
        negative_circle = circle_fun(
                            center = center_neg,
                            radius = radius_neg
                            )
        
        phi.paint_shape( positive_circle, filler_value=1.0 )
        phi.paint_shape( negative_circle, filler_value=0.0 )
        
        image = PIL.Image.fromarray( 255*phi.give_frame() )
        #plt.imshow(image)
        
        for kk in range(150):
            
            filename = subfoldername+'/test_{timestamp}'.format(timestamp=str(round(kk*0.0025,5)).ljust(9,'0'))+'.png'
            # Saving images
            plt.imsave(filename, image, cmap='gray')
