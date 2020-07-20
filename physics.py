# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:48:55 2017

@author: kretz01
"""
import numpy as np
import scipy.interpolate as iplt
from scipy.integrate import quad
import matplotlib.pyplot as plt
from os.path import normpath
#import physics_core
#import progressbar
from siddon import calc_attenuation_parameters
import pdb

from simulate_xray_spectrum import SimulateXRaySpectrum

class Physics:
    """
    Object to handle the physics for a specified experiment.
    """
    def __init__(self):
        self.hallo = 'hallo'
        self.intensity = 0
        self.attenuation = 0
        
    def get_intensity(self, intensity):
        return self.intensity
    
    def get_attenuation(self):
        return self.attenuation

    def calc_intensity_new(self, I0, E_sampled, p_object, length):
        mu = p_object.calc_mu_per_mm(E_sampled)
        return I0 * np.exp(- mu * length)
    
    def calc_intensity(self, spectrum, E_sample_size=60):
        E_min, E_max = min(spectrum[:,0]), max(spectrum[:,0])
        E_sampled = np.linspace(E_min, E_max, E_sample_size)
        intensities = iplt.interp1d(spectrum[:,0], spectrum[:,1], kind='cubic')
        red_intensities = np.empty_like(E_sampled)
        red_spectrum = np.empty((len(E_sampled), 2))
        for (i, E) in enumerate(E_sampled):
            I0 = intensities(E)
            attenuation = self.calc_attenuation(E)
            #print(attenuation)
            red_intensities[i] = I0 * np.exp(- attenuation)
        red_spectrum[:,0] = E_sampled
        red_spectrum[:,1] = red_intensities
        return red_spectrum
        
    def calc_attenuation(self, E):
        attenuation = -0.08418929 * E + 3.23485
        attenuation_length = 2 #lenght in cm
        density = 0.95 #density in g/cm^3
        return attenuation * density * attenuation_length
    
    def calc_total_energy(self, spectrum):
        '''Warning this is a sampled version of total energy that is strongly
        dependent on sampling size. No better solution now.'''
        E_sample_size = 100
        energy = 0
        E_min, E_max = min(spectrum[:,0]), max(spectrum[:,0])
        E_sampled = np.linspace(E_min, E_max, E_sample_size)
        intensities = iplt.interp1d(spectrum[:,0], spectrum[:,1], kind='cubic')
        #print(quad(intensities, E_min, E_max))
        for E in E_sampled:
            energy += intensities(E) * E
        return energy
    
    def calc_alpha_min_max(self, start, end, p_object):
        """
        Calculation of entrance and exit point for a given p_object for a 
        straight line parametrized through 
        start + alpha * (end-start)
        See Siddon for further information.
        
        Args:
            start (vector)    : startpoint corresponding to source point
            end (vector)      : endpoint corresponding to pixel center
            p_object (object) : object 
            
        Returns:
            parameter characterizing the entrance point (a_min) and the exit 
            point (a_max) for the straight line.
            
            a_min (float): min parameter
            a_max (float): max parameter
        """
        nx = p_object.n_vox_x + 1    
        ny = p_object.n_vox_y + 1     
        nz = p_object.n_vox_z + 1
       
        a_x_1 = 0
        a_x_nx = 1
        
        a_y_1 = 0
        a_y_ny = 1
        
        a_z_1 = 0
        a_z_nz = 1
        
        if start[0]-end[0] != 0:
            a_x_1 = (p_object.x_plane(1)-start[0])/(end[0]-start[0])
            a_x_nx = (p_object.x_plane(nx)-start[0])/(end[0]-start[0])
            
        if start[1]-end[1] != 0:
            a_y_1 = (p_object.y_plane(1)-start[1])/(end[1]-start[1])
            a_y_ny = (p_object.y_plane(ny)-start[1])/(end[1]-start[1])
            
        if start[2]-end[2] != 0:
            a_z_1 = (p_object.z_plane(1)-start[2])/(end[2]-start[2])
            a_z_nz = (p_object.z_plane(nz)-start[2])/(end[2]-start[2])
        
        a_min = max(0, min(a_x_1, a_x_nx), min(a_y_1, a_y_ny), min(a_z_1, a_z_nz))
        a_max = min(1, max(a_x_1, a_x_nx), max(a_y_1, a_y_ny), max(a_z_1, a_z_nz))
        #print(np.linalg.norm(((a_max-a_min)*(start-end))))
        return a_min, a_max
        
    def trace_ray(self, start, end, I, E_sampled, compression_plate=0, phantom=0, breast_support=0):
        """
        Calculate attenuation effects for a ray with given spectrum, travelling
        to up to three objects.
        
        Args:
            start (vector) 
            end (vector)
            spectrum (array)
            compression_plate (object, optional)
            phantom (object, optional)
            breast_support (object, optional)
            
        Returns:
            float: Deposited Energy at end point
        """
        
        if compression_plate:
            a_min_cp, a_max_cp = self.calc_alpha_min_max(start, end, compression_plate)
            cp_att_length = np.linalg.norm((a_max_cp-a_min_cp)*(end-start))
            I = self.calc_intensity_new(I, E_sampled, compression_plate, cp_att_length)
            
        if phantom:
#            spectrum = self.siddon_algorithm(start, end, phantom, spectrum)
            length, tissue = calc_attenuation_parameters(start, end, phantom)
            mu = phantom.calc_mu_per_mm_fast(tissue, E_sampled)
            I = I * np.exp(- np.sum(np.transpose(mu) * length, axis=1))
            
        if breast_support:
            a_min_bs, a_max_bs = self.calc_alpha_min_max(start, end, breast_support)
            bs_att_length = np.linalg.norm((a_max_bs-a_min_bs)*(end-start))
            I = self.calc_intensity_new(I, E_sampled, breast_support, bs_att_length)
            
        return I #np.sum(spectrum[:,1])#self.calc_total_energy(spectrum)
    
    def compute_image(self, detector, source, compression_plate, phantom, breast_support, E_sample_size=60):
        """
        Loop over all pixels of the detector to calculate a pixel value for a 
        given source and objects to produce a simulated x-ray image in this
        virtual experiment.
        
        Args:
            detector
            source
            compression_plate
            phantom
            breast_support
            
        Returns:
            array: 2d image array for a given experiment
            
        TODO:
            - consider geometrical blurring by multiple source points
            - time consuming!!!
                - 
        """
        n_rays = detector.npxx * detector.npxy
#        pbar = progressbar.ProgressBar(max_value=n_rays)
        image = np.empty((detector.npxy, detector.npxx))
        n_ray = 0
        
        E_min, E_max = min(source.Spectrum[:,0]), max(source.Spectrum[:,0])
        E_sampled = np.linspace(E_min, E_max, E_sample_size)
        I0 = source.get_intensity(E_sampled)
        
        response = detector.get_response(E_sampled)
        norm_fact = np.sum(I0 * E_sampled * response)
        
        #x_pos0_v, y_pos0_v = source.src_pts just use for more then 1 src point
        x_pos0_v, y_pos0_v = source.src_pts[0]
        z_pos0 = source.geometry.z
        pos_0 = np.array([x_pos0_v, y_pos0_v, z_pos0])
        
        x_pos1_v, y_pos1_v = detector.x_grid, detector.y_grid
        z_pos1 =  detector.geometry.z      
        
        for x_ind in range(0, detector.npxx):
            for y_ind in range(0,detector.npxy):
                pos_1 = np.array([x_pos1_v[y_ind, x_ind], y_pos1_v[y_ind, x_ind], z_pos1]) 
                red_intensities = self.trace_ray(pos_0, pos_1, I0, E_sampled, compression_plate, phantom, breast_support)
                image[y_ind,x_ind] = np.sum(red_intensities*response*(np.roll(E_sampled, -1)-E_sampled)[0])
#                pbar.update(n_ray)
                n_ray += 1
        return image  
    
    def compute_scattering(self, primary_image, scatter_kernel):
        fft_sig = np.fft.fft2(primary_image)
        fshift = np.fft.fftshift(fft_sig)
        
        f_ishift = np.fft.ifftshift(fshift*scatter_kernel)
        scatter = np.abs(np.fft.ifft2(f_ishift))
        return scatter

def process_i(detector, source, phantom, prim_img, SNR, i):
    image = prim_img
    """add scattering"""
    rows, cols = image.shape
    crow, ccol = int(rows/2), int(cols/2)
    f_size = 2
    lsf = np.zeros(image.shape)
    lsf[crow-f_size:crow+f_size, ccol-f_size:ccol+f_size] = 1
    SPR = phantom.SPR 
    
    scatter = physic.compute_scattering(image, lsf)
    scatter = scatter * (SPR*np.mean(image)/np.mean(scatter))
    image = image + scatter
    
    """blur image"""
    bl_img = detector.consider_blur(image, method='gaussian')#TODO: check if gaussian point spread function is a good assumption

    """digitize detector output (linear)"""
    dg = detector.get_dg_output(bl_img, 14, 0.2148*2032443611.0515957)
    
    """add white noise to image"""
    dg = detector.compute_noise(dg, SNR=SNR, method='white_noise')
    return dg.flatten()

def do_it_multi(diam, thickness, n_img, num_cores, detector, source, phantom, prim_img, path):
    print('Modelling',n_img,'images on', num_cores,'cores with diam =',diam,'and thickness =',thickness)
    nx = detector.npxx
    ny = detector.npxy
    imgs = np.zeros((nx*ny,n_img))
    t1=time.time()
    results=Parallel(n_jobs=num_cores, verbose=10)(delayed(process_i)(detector, source, phantom, i) for i in range(n_img))
    imgs = np.array(results).T
    t2 = time.time()
    print('Processing with',num_cores,'cores took',(t2-t1)/60,'min.')
    return imgs
    
def do_it(diam, thickness, n_img, detector, source, phantom, prim_img, SNR, path):
    print('Modelling',n_img,'images on 1 core with diam =',diam,'and thickness =',thickness)
    nx = detector.npxx
    ny = detector.npxy
    imgs = np.zeros((nx*ny,n_img))
    t1=time.time()
    for i in range(n_img):
        imgs[:,i] = process_i(detector, source, phantom, prim_img, SNR, i)
    t2 = time.time()
    print('Processing with 1 core took',(t2-t1)/60,'min.')
    return imgs
    
if __name__=='__main__':
    
    from experiment_object import Detector, Source, Plate, VoxelObject, CDMAM_phantom
    from scipy.ndimage.filters import gaussian_filter
    import h5py
    from joblib import Parallel, delayed
    import time
    import yaml
    import sys
    from os.path import join as pjoin
    
    cfg_file = 'mammomat_settings.yml'
    
    if 'win' in sys.platform:
        home_path = 'N:\\MatMoDatPrivate\\kretz01\\linux\\'
    elif 'linux' in sys.platform:
        home_path = '/home/kretz01/san/'
    else:
        raise OSError

    print('Home:',home_path)
    pmma_path = home_path+'DataFiles/AttenuationFiles/pmma_attenuation.txt'
    
    cfg_fp = pjoin(home_path,'SpyderProjects','Simulation','config',cfg_file)
    #cfg_fp = pjoin(home_path,'SpyderProjects','Simulation','mammomat_settings.yml')
    
    with open(cfg_fp) as fp:
        cfg = yaml.load(fp)
    
    npx = 294 #better use 258 for 18mm since this yield ~0.07mm per px
    #npx = 441
    pxpitch = 0.06 
    dim = npx*pxpitch
    detector = Detector(npx, npx, np.array([0,0,0]), dim, dim, home_path)
    
    source = Source(cfg['source'], np.array([0,0,700]), 0.54, 0.33, home_path)
#    compression_plate = Plate(1.19, [0,0,100], 50, 50, 50, pmma_path)
#    breast_support = Plate(1.19, [0,0,0], 50, 50, 70, pmma_path)
    physic = Physics()
    
    num_workers=10
    n_img = 500
#    SPR = 0.18
    
    SPR = 0.485
    
    SNR = 0.85 * 5.8
    
#    diams = np.array([0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.31, 0.4, 0.5, 0.63, 0.8, 1., 1.25, 1.6, 2.])
#    diams = np.array([0.5, 0.63, 0.8, 1., 1.25, 1.6, 2.])
    diams = np.array([1.])
#    thicks = np.array([0.03e-3, 0.04e-3, 0.05e-3, 0.06e-3, 0.08e-3, 0.10e-3, 0.13e-3, 0.16e-3, 0.20e-3, 0.25e-3, 0.36e-3, 0.50e-3, 0.71e-3, 1e-3, 1.42e-3, 2e-3])
    thicks = np.array([0.36e-3])
    
    for au_diam in diams:
        for au_thick in thicks:
            #au_diam = 1.#0.63 #in mm
            #au_thick = 0.71e-3#.08e-3 #in mm
            phantom = CDMAM_phantom(au_diameter=au_diam, au_thickness=au_thick, pos=np.array([0,0,71]), SPR=SPR, home_path=home_path)

#            xshift = np.random.choice((-1,1))*np.random.rand()*pxpitch
#            yshift = np.random.choice((-1,1))*np.random.rand()*pxpitch
            xshift = 0
            yshift = 0
            shift = np.vstack((xshift*np.ones((1,n_img)),yshift*np.ones((1,n_img))))
            detector = Detector(npx, npx, np.array([xshift,yshift,0]), dim, dim, home_path)
            prim_img = physic.compute_image(detector, source, 0 , phantom ,0)
            imgs = do_it(diam=au_diam, thickness=au_thick, n_img=n_img, detector=detector, source=source, phantom=phantom, prim_img=prim_img, SNR=SNR, path=home_path)
            
            fn = home_path+'SpyderProjects/SaveSimRes/'
            #savename = normpath(fn+'sim_'+str(int(au_diam*1e2))+'_'+str(int(au_thick*1e6))+'_'+str(int(n_img))+'_nn.hdf5')
            savename = normpath(fn+'sim_'+str(int(au_diam*1e2))+'_'+str(int(au_thick*1e6))+'_'+str(int(n_img))+'_data'+'.hdf5')
            file = h5py.File(savename, "a")
            dset_img = file.create_dataset("imgdata", data=imgs)
            dset_img.attrs['shift']=shift
            dset_img.attrs['Nx']=npx
            dset_img.attrs['Ny']=npx
            dset_img.attrs['pitch']=pxpitch
            dset_img.attrs['diameter']=au_diam
            dset_img.attrs['thickness']=au_thick
            dset_img.attrs['n_img']=n_img
            dset_img.attrs['SNR']=SNR
            dset_img.attrs['SPR']=SPR
            file.close()
            print('Saved:',savename)
    
#    img = np.reshape(imgs[:,0], (294, 294))
#    print(np.mean(img[-20:,:]))
##    print(np.mean(img[-20:,:])/np.mean(img[147-6:147+6,147-6:147+6]))
#    print(np.std(img[-20:,:]))
#    img=np.reshape(imgs[:,0], (npx,npx))
#    npx = 145
#    img = img[int(129-npx/2):int(129+npx/2), int(129-npx/2):int(129+npx/2)]
#    pimg = prim_img[int(129-npx/2):int(129+npx/2), int(129-npx/2):int(129+npx/2)]
#    plt.imsave('N://MatMoDatPrivate//kretz01//simu_16_13_spr_high.png', img, cmap='gray', vmin=284, vmax=784)
#    plt.figure()
#    plt.imshow(img, cmap='gray', vmin=284, vmax=784)
#    plt.show()