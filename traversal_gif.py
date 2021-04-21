import imageio
from scipy.misc import imresize
import sys

n_points = int(sys.argv[1])
n_steps = 40
images, filenames = [], []
tot_imgs = n_steps * n_points 
#tot_imgs = 100 #n_steps * n_points 
for i in range(tot_imgs):
    #images.append(imageio.imread('./video_output/sample%d.png' %(i)))
    images.append(imageio.imread('./traversals/sample%d.png' %(i)))
    #images.append(imresize(imageio.imread('./traversals/sample%d.png' %(i)), size=[256, 256]))

imageio.mimsave('./gifs/sample.gif', images, duration = 0.1)
