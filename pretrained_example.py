# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
#import config


def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    #url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    #with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    #    _G, _D, Gs = pickle.load(f)
    #fpath = './results/00009-sgan-custom_dataset-1gpu/network-snapshot-003400.pkl'
    fpath = './results/00007-stylegan2-custom_dataset-1gpu-config-a/network-snapshot-.pkl'
    with open(fpath, mode='rb') as f:
        _G, _D, Gs = pickle.load(f)
        #C:\Users\user\stylegan-master\results\00004-sgan-custom_dataset-1gpu
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    # Pick latent vector.
    #rnd = np.random.RandomState(5)
    #latents = rnd.randn(1, Gs.input_shape[1])
    
        # Pick latent vector.
    #rnd = np.random.RandomState(5) #5
    sk=0
    
        # Pick latent vector.
    #rnd = np.random.RandomState(6) #5
    latents_=[]
    #a=52
    #b=11
    """
    for i in range(1,101,1):
        rnd = np.random.RandomState(i)
        latents = rnd.randn(1, Gs.input_shape[1])
        latents_.append(latents)
    
    for j in range(50):
        latents_mean=j/50*latents_[a]+(1-j/50)*latents_[b]
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents_mean, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        # Save image.
        #os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join('./results/', 'example_{}.png'.format(j))
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
    """
    for i in range(1,101,1):
        rnd = np.random.RandomState(i)
        latents = rnd.randn(1, Gs.input_shape[1])

        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

        # Save image.
        #os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join('./results/', 'example{}.png'.format(sk))
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
        sk += 1
    
    s=100
    images = []
    for i in range(50):
        im = PIL.Image.open('./results/example'+str(i)+'.png') 
        im =im.resize(size=(256, 256), resample=PIL.Image.NEAREST)
        images.append(im)
    for i in range(49,0,-1):
        im = PIL.Image.open('./results/example'+str(i)+'.png') 
        im =im.resize(size=(256, 256), resample=PIL.Image.NEAREST)
        images.append(im)    

    images[0].save('./results/example{}_{}_100.gif'.format(256,1000), save_all=True, append_images=images[1:s], duration=100*5, loop=0)  
        

if __name__ == "__main__":
    main()
