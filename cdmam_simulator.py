# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:18:17 2018

@author: kretz01
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:05:50 2018

@author: kretz01
"""

   
from experiment_object import Detector, Source, CDMAMFF, CDMAM_cell
from physics import Physics
from create_dicom import create_dicom_from_h5
from os.path import join as pjoin
import yaml
import argparse
from time import localtime
import numpy as np
from os.path import normpath
import h5py
import sys

def process_i(detector, phantom, prim_img, physic, SNR, i):
    image = prim_img
    rows, cols = image.shape
    
    """add scattering"""
    crow, ccol = int(rows/2), int(cols/2)
    f_size = 2
    lsf = np.zeros(image.shape)
    lsf[crow-f_size:crow+f_size, ccol-f_size:ccol+f_size] = 1
    SPR = phantom.SPR 
    
    img_scat = physic.add_scattering(image, lsf, SPR)
    
    """blur image"""
    bl_img = detector.consider_blur(img_scat, method='gaussian')

    """digitize detector output (linear)"""
    dg = detector.get_dg_output(bl_img)
    
    """add white noise to image"""
    dg = detector.compute_noise(dg, SNR, method='white_noise')
    return dg.flatten()

def do_it(n_img, detector, phantom, prim_img, physic, SNR, path):
#    print('Modelling',n_img,'images on 1 core')
    nx = detector.npxx
    ny = detector.npxy
    imgs = np.zeros((nx*ny,n_img))
#    t1=time.time()
    for i in range(n_img):
        imgs[:,i] = process_i(detector, phantom, prim_img, physic, SNR, i)
#    t2 = time.time()
#    print('Processing with 1 core took',(t2-t1)/60,'min.')
    return imgs


def main(cfg_file):
    if 'win' in sys.platform:
        home_path = 'N:\\MatMoDatPrivate\\kretz01\\linux\\'
    elif 'linux' in sys.platform:
        home_path = '/home/kretz01/san/'
    else:
        raise OSError
    
    print('Home: {}'.format(home_path))
    print()
    print('Using: {}'.format(cfg_file))
    cfg_fp = normpath(pjoin('.','config',cfg_file))
    #cfg_fp = pjoin(home_path,'SpyderProjects','Simulation','mammomat_settings.yml')
    
    with open(cfg_fp) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)
    
    
    fn = home_path+'SpyderProjects/SaveSimRes/'
    lt = localtime()
    y,m,d = lt[0:3]
    date = '%02i%02i%02i' % (d,m,y)
    rid = "{:d}".format(np.random.randint(1000))
    savename = normpath(pjoin('.','Results','sim_data_'+date+'_'+rid+'.hdf5'))
#    savename = normpath(fn+'sim_data_randompos_'+date+'_'+rid+'.hdf5')
    print(savename)
    
    material = cfg["source"]["material"]
    kVp = cfg["source"]["kVp"]
    mAs = cfg["source"]["mAs"]
    eBin = cfg["source"]["eBin"]
    
    npx = cfg["detector"]["npx"]
    npy = cfg["detector"]["npy"]
    pxpitch = cfg["detector"]["pitch"] 
    SPR = cfg["detector"]["SPR"]
    SNR = cfg["detector"]["SNR"]
    bits = cfg["detector"]["bits"]
    calibration = cfg["detector"]["calibration"]
     
    
#    num_workers = cfg["general"]["nworkers"]
    n_img = cfg["general"]["nimg"]
    n_pos = cfg["general"]["npos"]
        
    dimx = npx*pxpitch
    dimy = npy*pxpitch
    
    imgs = np.zeros((npx*npy,0))
    shifts = np.zeros((2,0))
    
    source = Source(cfg['source'], np.array([0,0,700]), 0.54, 0.33, home_path)
    physic = Physics()
    
    if cfg["phantom"]["singleCell"]:
        cdmam = CDMAM_cell(au_diameter=cfg["phantom"]["diam"], au_thickness=cfg["phantom"]["thick"]*1e-3, pos=np.array([0,0,71]), SPR=SPR, home_path=home_path)
    else:
        fname = home_path+cfg["phantom"]["file"]
        cdmam = CDMAMFF(fname=fname, pos=np.array([0,0,71]), SPR=SPR, home_path=home_path)
    
    for j in range(0,n_pos):
        xshift = np.random.choice((-1,1))*np.random.rand()*pxpitch
        yshift = np.random.choice((-1,1))*np.random.rand()*pxpitch
        shift = np.vstack((xshift*np.ones((1,n_img)),yshift*np.ones((1,n_img))))
        detector = Detector(npx, npy, np.array([xshift,yshift,0]), dimx, dimy, bits, calibration, home_path)
        cdmamXRAY = physic.compute_image(detector, source, 0 , cdmam ,0)
        dset = do_it(n_img=n_img, detector=detector, phantom=cdmam, prim_img=cdmamXRAY, physic=physic, SNR=SNR, path=home_path)
        # dset = np.zeros((npx*npy,n_img))
        imgs = np.concatenate((imgs, dset),axis=1)
        shifts = np.concatenate((shifts, shift),axis=1)
        del detector, cdmamXRAY, dset
        print(str(j+1)+'/'+str(n_pos)+' done.')
    
    file = h5py.File(savename, "a")
    dset_img = file.create_dataset("imgdata", data=imgs)
    dset_img.attrs['Shift']=shifts
    dset_img.attrs['Nx']=npx
    dset_img.attrs['Ny']=npy
    dset_img.attrs['Pitch']=pxpitch
    dset_img.attrs['SNR']=SNR
    dset_img.attrs['SPR']=SPR
    dset_img.attrs['material']=material
    dset_img.attrs['kVp']=kVp
    dset_img.attrs['mAs']=mAs
    dset_img.attrs['eBin']=eBin        
    dset_img.attrs['NImg']=imgs.shape[1]
    
    create_dicom_from_h5(file)
    file.close()
    print('Saved '+savename)
    
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="selenia_settings0.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()
    main(args.config)