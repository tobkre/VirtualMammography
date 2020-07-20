# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 08:51:08 2018

@author: kretz01
"""

import pydicom
import numpy as np
import h5py
from os.path import normpath, join
from time import localtime

def create_dicom_from_h5(f):
    count = 0
    
    lt = localtime()
    y,m,d = lt[0:3]
    date = '{:02}{:02}{:02}'.format(d,m,y)
    
    spath = normpath(join('.', 'Results', 'DICOM'))
    
    dset = f["imgdata"]
    data = dset[:]
    shifts = dset.attrs['Shift']
    npx = dset.attrs['Nx']
    npy = dset.attrs['Ny']
    pxpitch = dset.attrs['Pitch']
    SNR = dset.attrs['SNR']
    nimg = dset.attrs['NImg']
    try:
        SPR = dset.attrs['SPR']
    except:
        SPR = 0
    try:
        material = dset.attrs['material']
    except:
        material = 'undef'
    try:
        kVp = dset.attrs['kVp']
    except:
        kVp = 0
    try:
        mAs = dset.attrs['mAs']
    except:
        mAs = 0
    try:
        eBin = dset.attrs['eBin']
    except:
        eBin = 0
    
    ds = pydicom.dcmread(normpath(join('','EE1ADC58')))
    
    for i in range(0,nimg):
        cut_img = np.reshape(data[:,i], (npy, npx))
    #    img[-npy:,-npx:]=cut_img
        img = np.uint16(cut_img)
        fname='simulation_{}_{}_{:03d}.dcm'.format(material,date,count)
        path = normpath(join(spath,fname))
        ds.PixelData = img.tobytes()
        ds.Rows = npy
        ds.Columns = npx
        ds.KVP = kVp
        ds.XRayTubeCurrent = mAs
        ds.AnodeTargetMaterial = material.upper()
        ds.save_as(path)
        count += 1
        
    print('Material: {}, Current: {} mAs, Voltage: {} kVp, SNR {}'.format(material, mAs, kVp, SNR))
