# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:16:04 2017

@author: kretz01
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
from simulate_xray_spectrum import SimulateXRaySpectrum
from os.path import normpath
from os.path import join as pjoin
import h5py
import matplotlib.pyplot as plt

class Object:
    """
    Any kind of object cosists of a center position which is settled to the
    center of the bottom plane for 3d objects.
    
    Attributes:
        pos (array)          : 3d position array of object center
        width (int)          : width of object in mm
        height (int)         : height of object in mm
        depth (int, optional): depth of oject in mm
    """
    def __init__(self, pos, width, height, depth=0):
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]
        self.width = width
        self.height = height
        self.flat = True
        if depth:
            self.flat = False
            self.depth = depth
            
class Detector:
    """
    Detector is a 2d object with specified geometry and number of pixels in 
    x and y direction.
    
    Attributes:
        npxx (int)  : number of pixels in x direction
        npxy (int)  : number of pixels in y direction
        pos (array) : 3d position array of detector center
        width (int) : width of detector in mm
        height (int): height of detecor in mm
    """
    def __init__(self, npxx:int, npxy:int, pos, width, height, bits, calibration, home_path):
        self.geometry = Object(pos, width, height)
        self.__px_x_size = width/npxx
        self.__px_y_size = width/npxy
        self.npxx = npxx
        self.npxy = npxy
        self.bit_depth = bits
        self.max_signal = calibration
        self.x_pos = np.linspace(self.geometry.x - self.geometry.width/2, 
                                 self.geometry.x + self.geometry.width/2, 
                                 self.npxx) + self.__px_x_size/2
        self.y_pos = np.linspace(self.geometry.y - self.geometry.height/2, 
                                 self.geometry.y + self.geometry.height/2, 
                                 self.npxy) + self.__px_y_size/2
        self.x_grid, self.y_grid = np.meshgrid(self.x_pos, self.y_pos)
        self.z_grid = np.ones_like(self.x_grid) * self.geometry.z
        self.grid = np.array([self.x_grid, self.y_grid, self.z_grid]).T.reshape(self.npxx, self.npxy, 3)
        self.__CsI_data = np.loadtxt(normpath(pjoin(home_path,'attenuation_files','CsI_attenuation.txt')), skiprows=3)
        self.__response = interp1d(self.__CsI_data[:,0]*1000, self.__CsI_data[:,1], kind='cubic')
        
        self.__dens = 4.51 # in g/cm^2 ??
        self.__thickness = 0.25 #in mm
        #self.__length = 
    def get_response(self, E):
        return  1 - np.exp(- self.__response(E) * self.__dens /10 * self.__thickness)
    
    def consider_blur(self, image, method='gaussian'):
        blurred_image = image
        if method=='gaussian':
            blurred_image = gaussian_filter(image, sigma=0.7)
        elif method=='MTF':
            blurred_image = 0
        return blurred_image
        
    def compute_noise(self, image, SNR, method='white_noise'):
        if method=='white_noise':
#            sig_bl = 2.1
            sig_bl = 0.55
            sig = 2*SNR*sig_bl*np.sqrt(np.pi)
            noise = np.random.normal(loc=0,scale=sig,size=image.shape)# normal
            #noise = np.random.normal(loc=0,scale=(1.5/1.)*5.8*1.75,size=image.shape)
            noise = gaussian_filter(noise, sigma=sig_bl)
        return (noise + image)
    
    def get_dg_output(self, image):
        #white_noise = np.random.normal(loc=0, scale=1, size=image.shape)
        #add noise
        
        #digitized with bit depth n
        #2^n
        return image * 2**self.bit_depth / self.max_signal

class Source:
    """
    A X-Ray source is an object characterized through a 2d focal plane. For a 
    anode material and tube voltage a characteristic spectrum is produced. 
    For geometric blurring due to infinite focal area number of source points
    is variable.
    
    Attributes:
        material (str)       : anode material (molybdenum, tungsten or rhodium)
        voltage (float)      : tube voltage in kV
        pos (array)          : 3d position array of source center
        focal_width (int)    : focal width of source in mm
        focal_height (int)   : focal height of source in mm
        home_path (str)      : linux or windows based homepath
        filter_dict (dict)   : Dictionary describing the filter with two fields (material and thickness in mm)
        n_pts (int, optional): number of point sources (blurring effects)
    """
    def __init__(self, cfg, pos, focal_width, focal_height, home_path, n_pts=1):
        self.geometry = Object(pos, focal_width, focal_height)
        
        self.material = cfg["material"]
        self.voltage = cfg["kVp"]
        self.filters = cfg["filters"]
        self.deltaE = cfg["eBin"]
        self.mAs = cfg["mAs"]
        
        self.__spect_sim = SimulateXRaySpectrum(anode_material=self.material, 
                                                voltage=self.voltage, path=home_path)
        self.__spectrum = self.__spect_sim.compute_spectrum(self.deltaE)
        self.fluencePerExposure = self.__spect_sim.compute_fluence_per_exposure(self.__spectrum[:, 0])
        
        self.__filt_spectrum = self.__spectrum.copy()
        for key in self.filters:
            print(key)
            self.__filt_spectrum= self.__spect_sim.filter_spectrum(spectrum=self.__filt_spectrum, material=key, thickness=self.filters[key])
        
        self.EMean = (self.__filt_spectrum[:,0]*self.__filt_spectrum[:,1]*self.deltaE).sum()/(self.__filt_spectrum[:,1]*self.deltaE).sum()
        self.TotalFluencePerExposure = self.calc_fluence_per_exp()
        self.TotalFluencePerAirKerma = self.TotalFluencePerExposure*114.5
        self.Exposure = self.calc_exposure(self.__filt_spectrum)
        self.AirKerma = self.calc_airkerma(self.__filt_spectrum)
        self.Spectrum = np.concatenate(
                (self.__filt_spectrum[:,0].reshape(-1,1), (self.__filt_spectrum[:,1]/self.AirKerma*self.mAs).reshape(-1,1)), axis=1)
        
        self.src_pts = np.random.random(size=(n_pts, 2))
        self.src_pts[:,0] = self.src_pts[:,0] * self.geometry.width - self.geometry.width/2. + self.geometry.x
        self.src_pts[:,1] = self.src_pts[:,1] * self.geometry.height - self.geometry.height/2. + self.geometry.y
        self.__iplt = interp1d(self.__filt_spectrum[:,0], self.__filt_spectrum[:,1], kind='cubic', bounds_error=False, fill_value=0)
        
        print('Computing X-Ray Spectrum for',self.material,'anode at',self.voltage,'kV and',self.mAs,'mAs.')
    
    def calc_exposure(self, spectrum):
        return (spectrum[:,1]/self.fluencePerExposure).sum()
    
    def calc_airkerma(self, spectrum):
        return self.calc_exposure(spectrum)/114.5
    
    def calc_fluence_per_exp(self):
        return (self.fluencePerExposure*self.normalize_spectrum(self.__filt_spectrum)[:,1]).sum()
    
    def get_intensity(self, E):
        return self.__iplt(E)
        
    def plot_points(self):
        plt.figure()
        plt.plot(self.src_pts[:,0], self.src_pts[:,1], 'o')
        plt.show()
    
    def normalize_spectrum(self, spectrum):
        out = spectrum.copy()
        out[:,1] /= out[:,1].sum()+1e-12
        return out
        
class Plate:
    """
    A plate is a 3d object of homogenous material with provided energy
    dependend x-ray attenuation information.
    
    Attributes:
        density (float): material density in g/cm^3
        pos (array)    : 3d position array of plate center
        width (int)    : width of plate in mm
        height (int)   : height of plate in mm
        depth (int)    : depth of plate in mm
        path (str)     : file path for material attenuation information
    """
    def __init__(self, density, pos, width, height, depth, path:str):
        self.geometry = Object(pos=pos, width=width, height=height, depth=depth)
        self.__data = np.loadtxt(path, skiprows=5) #mass attenuation
        self.__mu = interp1d(self.__data[:,0]*1000, self.__data[:,1], kind='cubic')
        self.density = density #density in g/cm^3
        self.n_vox_x = 1
        self.n_vox_y = 1
        self.n_vox_z = 1
        
    def x_plane(self, i):
        return self.geometry.x -self.geometry.width/2. + i*self.geometry.width
    
    def y_plane(self, j):
        return self.geometry.y -self.geometry.height/2. + j*self.geometry.height
    
    def z_plane(self, k):
        return self.geometry.z + k*self.geometry.depth
    
    def get_mu_per_mm(self, E):
        return self.__mu(E) * self.density / 10
    
class VoxelObject:
    """
    A Voxel object is a 3d object specified by its geometry. Each voxel contains 
    an identifier to characterize stored tissue.
    0: vacuum
    1: fatty
    2: glandular
    3: ill mass
    
    Attributes:
        voxel_array (array): 3d array with tissue representation
        vx_size (int)      : (unit) size of cubic voxels
        pos (array)        :  3d position array of object center
    """
    def __init__(self, voxel_array, vx_size, pos, home_path='.'):
        self.voxels = voxel_array
        self.vx_length = vx_size
        self.n_vox_x = voxel_array.shape[0]
        self.n_vox_y = voxel_array.shape[1]
        self.n_vox_z = voxel_array.shape[2]
        self.l_vox_x = vx_size #mm
        self.l_vox_y = vx_size #mm
        self.l_vox_z = vx_size #mm
        self.geometry = Object(pos=pos, width=self.n_vox_x*self.vx_length,
                               height=self.n_vox_y*self.vx_length,
                               depth=self.n_vox_z*self.vx_length)
        
        self.__GLANDULAR_DENS = 1.02 # g/cm^3
        self.__FATTY_DENS = 0.95 # g/cm^3 
        self.__MASS_DENS = 1.044 # g/cm^3
        
        self.__FAT_DATA = np.loadtxt(normpath(pjoin(home_path, 'attenuation_files','adipose_attenuation.txt')), skiprows=5)
        self.__GLA_DATA = np.loadtxt(normpath(pjoin(home_path, 'attenuation_files','glandular_attenuation.txt')), skiprows=5)
        self.__MAS_DATA = np.loadtxt(normpath(pjoin(home_path, 'attenuation_files','carcinoma_attenuation.txt')), skiprows=5)
        self.__MU_FAT = interp1d(self.__FAT_DATA[:,0]*1000, self.__FAT_DATA[:,1], kind='cubic')
        self.__MU_GLA = interp1d(self.__GLA_DATA[:,0]*1000, self.__GLA_DATA[:,1], kind='cubic')
        self.__MU_MAS = interp1d(self.__MAS_DATA[:,0]*1000, self.__MAS_DATA[:,1], kind='cubic')
    
    def x_plane(self, i):
        return self.geometry.x -self.geometry.width/2. + i*self.vx_length
    
    def y_plane(self, j):
        return self.geometry.y -self.geometry.height/2. + j*self.vx_length

    def z_plane(self, k):
        return self.geometry.z + k*self.vx_length
    
    def get_voxel(self, i, j, k):
        z = self.voxels.shape[2] - 1
        return self.voxels[i,j,z-k] 
    
    def calc_mu_per_mm(self, tissue:int, E):
        if tissue==0:
            return 0
        elif tissue==1:
            #self.density = self.__FATTY_DENS
            return self.__MU_FAT(E) * self.__FATTY_DENS / 10
            #self.__data = self.__FAT_DATA
        elif tissue==2:
            #self.density = self.__GLANDULAR_DENS
            return self.__MU_GLA(E) * self.__GLANDULAR_DENS / 10
            #self.__data = self.__GLA_DATA
        elif tissue==3:
            #self.density = self.__MASS_DENS
            return self.__MU_MAS(E) * self.__MASS_DENS / 10
            #self.__data = self.__MAS_DATA
        else:
            raise ValueError('Unknown tissue')
            
    def calc_mu_per_mm_fast(self, tissue, E):
        #try:
        mu = np.zeros((len(tissue), len(E)), dtype=np.float64)    
        mu[np.equal(1, tissue),:] = self.__MU_FAT(E) * self.__FATTY_DENS / 10
        mu[np.equal(2, tissue),:] = self.__MU_GLA(E) * self.__GLANDULAR_DENS / 10
        mu[np.equal(3, tissue),:] = self.__MU_GLA(E) * self.__MASS_DENS / 10
        return mu
        #except:
        #    raise ValueError('Unknown tissue')
        #print(E, tissue)
        #self.__mu = interp1d(self.__data[:,0]*1000, self.__data[:,1], kind='cubic')
        #return self.__mu(E) * self.density / 10
        
class CDMAM_cell:
    """
    A Voxel object is a 3d object specified by its geometry. Each voxel contains 
    an identifier to characterize stored tissue.
    0: vacuum
    1: fatty
    2: glandular
    3: ill mass
    
    Attributes:
        voxel_array (array): 3d array with tissue representation
        vx_size (int)      : (unit) size of cubic voxels
        pos (array)        :  3d position array of object center
    """
    def __init__(self, au_diameter, au_thickness, pos, SPR, home_path):
        self.diam = au_diameter
        self.thick = au_thickness
        self.l_vox_x = 0.05 #mm
        self.l_vox_y = 0.05 #(au_diameter/100) #mm
        self.l_vox_z = 1*10**(-5) #au_thickness #mm
        
        self.n_vox_x = 200
        self.n_vox_y = 200
        self.n_vox_z = 200
        
        self.geometry = Object(pos=pos, width=self.n_vox_x*self.l_vox_x,
                               height=self.n_vox_y*self.l_vox_y,
                               depth=self.n_vox_z*self.l_vox_z)
        
        #print(self.n_vox_x, self.n_vox_y, self.n_vox_z)
        self.SPR = SPR #scatter-to primary ratio
        self.voxels = np.ones((self.n_vox_x,self.n_vox_y,self.n_vox_z))
        self.__fill_voxel()
        
        
        
        self.__PMMA_DENS = 1.19 # g/cm^3
        self.__AU_DENS = 19.302# * 0.5042629524111005# g/cm^3 
        self.__AL_DENS = 2.6989# g/cm^3
        self.__BD_DENS = 2.6989 *65# g/cm^3
        
        self.__PMMA_DATA = np.loadtxt(normpath(pjoin(home_path,'attenuation_files','pmma_attenuation.txt')), skiprows=5)
        self.__AU_DATA = np.loadtxt(normpath(pjoin(home_path,'attenuation_files','au_attenuation.txt')), skiprows=5)
        self.__AL_DATA = np.loadtxt(normpath(pjoin(home_path,'attenuation_files','al_attenuation.txt')), skiprows=5)
        
        self.MU_PMMA = interp1d(self.__PMMA_DATA[:,0]*1000, self.__PMMA_DATA[:,2], kind='cubic')
        self.MU_AU = interp1d(self.__AU_DATA[:,0]*1000, self.__AU_DATA[:,2], kind='cubic')
        self.MU_AL = interp1d(self.__AL_DATA[:,0]*1000, self.__AL_DATA[:,2], kind='cubic')
    
    def __fill_voxel(self):
        #al_ind = int(0.5 / self.l_vox_z)
        #print(al_ind)
        #self.voxels[:,:,:al_ind] = 2
        
        bound = int(0.25 / self.l_vox_x)
        #print(bound)
#        self.voxels[:bound,:,:] = 4
#        self.voxels[-bound:,:,:] = 4
#        self.voxels[:,:bound,:] = 4
#        self.voxels[:,-bound:,:] = 4
        
        y, x = np.ogrid[-np.floor(self.n_vox_x/2):np.floor(self.n_vox_x/2), -np.floor(self.n_vox_y/2):np.floor(self.n_vox_y/2)]
        
        #mask = np.empty((self.voxels.shape[:2]), dtype=np.bool)
        #print(mask.shape)
        #mask[:,:] = False
        #mask[10:-10,10:-10] = True
        #print(mask)
        #print(x*x+y*y)
        #print((self.n_vox_x/2 + 50)*(self.n_vox_y/2 + 50))
        mask_mid = x*x+y*y <= self.diam/self.l_vox_x * self.diam/self.l_vox_y /4
        x += self.n_vox_x/4
        y += self.n_vox_y/4
        mask_corner = x*x+y*y < self.diam/self.l_vox_x * self.diam/self.l_vox_y /4
        au_ind = int(self.thick / self.l_vox_z)
        #print(self.thick, '/', self.l_vox_z)
        #print(au_ind)
        #print(np.any(mask), mask.shape)
        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.imshow(mask)
        self.voxels[mask_mid,:au_ind] = 3
        self.voxels[mask_corner,:au_ind] = 3
    
    def show_phantom(self,cut):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        pl_ind = int(0.5 / self.l_vox_z)
        fig = plt.figure()
        if cut=='x':
            plt.imshow(self.voxels[pl_ind,:,:])
        elif cut=='y':
            plt.imshow(self.voxels[:,pl_ind,:])
        elif cut=='z':
            image = self.voxels
            image[0] *= self.l_vox_x
            image[1] *= self.l_vox_y
            image[2] *= self.l_vox_z
            #pl_ind = 0.5
            #plt.imshow(self.voxels[:,:,pl_ind])
            plt.imshow(image[:,:,24])
        elif cut=='3d':
            ax = fig.gca(projection='3d') 
            ax.voxels(self.voxels[0], self.voxels[1], self.voxels[2])
        plt.show()
        
    def x_plane(self, i):
        return self.geometry.x -self.geometry.width/2. + i*self.l_vox_x
    
    def y_plane(self, j):
        return self.geometry.y -self.geometry.height/2. + j*self.l_vox_y

    def z_plane(self, k):
        return self.geometry.z + k*self.l_vox_z
    
    def get_voxel(self, i, j, k):
        z = self.voxels.shape[2] - 1
        return self.voxels[i,j,z-k] 
             
    def calc_mu_per_mm_fast(self, tissue, E):
        #try:
        mu = np.zeros((len(tissue), len(E)), dtype=np.float64)    
        mu[np.equal(1, tissue),:] = self.MU_PMMA(E) * self.__PMMA_DENS / 10 #per mm
        mu[np.equal(2, tissue),:] = self.MU_AL(E) * self.__AL_DENS / 10
        mu[np.equal(3, tissue),:] = self.MU_AU(E) * self.__AU_DENS / 10
        mu[np.equal(4, tissue),:] = self.MU_AL(E) * self.__BD_DENS / 10
        return mu
    
class CDMAMFF:
    """
    A Voxel object is a 3d object specified by its geometry. Each voxel contains 
    an identifier to characterize stored tissue.
    0: vacuum
    1: fatty
    2: glandular
    3: ill mass
    
    Attributes:
        voxel_array (array): 3d array with tissue representation
        vx_size (int)      : (unit) size of cubic voxels
        pos (array)        :  3d position array of object center
    """
    def __init__(self, fname, pos, SPR, home_path):
        file = h5py.File(fname, 'r')
        dataset = file['data']
        
        self.voxels = np.transpose(dataset, (2,1,0))
        mask = dataset.attrs['Mask']
        #2pos = dataset.attrs['2Pos']
        xyres = dataset.attrs['XYRes']
        zres =  dataset.attrs['ZRes']
        
        self.l_vox_x = float(xyres) #mm
        self.l_vox_y = float(xyres) #(au_diameter/100) #mm
        self.l_vox_z = float(zres/1000) #au_thickness #mm
        
        self.n_vox_x = self.voxels.shape[1]
        self.n_vox_y = self.voxels.shape[0]
        self.n_vox_z = self.voxels.shape[2]
        
        self.geometry = Object(pos=pos, width=self.n_vox_x*self.l_vox_x,
                               height=self.n_vox_y*self.l_vox_y,
                               depth=self.n_vox_z*self.l_vox_z)
        
        #print(self.n_vox_x, self.n_vox_y, self.n_vox_z)
        self.SPR = SPR #scatter-to primary ratio    
        
        self.__PMMA_DENS = 1.19 # g/cm^3
        self.__AU_DENS = 19.302# * 0.5042629524111005# g/cm^3 
        self.__AL_DENS = 2.6989# g/cm^3
        self.__BD_DENS = 19.445# g/cm^3
        
        self.__PMMA_DATA = np.loadtxt(normpath(pjoin(home_path,'attenuation_files/','pmma_attenuation.txt')), skiprows=5)
        self.__AU_DATA = np.loadtxt(normpath(pjoin(home_path,'attenuation_files/','au_attenuation.txt')), skiprows=5)
        self.__AL_DATA = np.loadtxt(normpath(pjoin(home_path,'attenuation_files','al_attenuation.txt')), skiprows=5)
        
        self.MU_PMMA = interp1d(self.__PMMA_DATA[:,0]*1000, self.__PMMA_DATA[:,2], kind='cubic')
        self.MU_AU = interp1d(self.__AU_DATA[:,0]*1000, self.__AU_DATA[:,2], kind='cubic')
        self.MU_AL = interp1d(self.__AL_DATA[:,0]*1000, self.__AL_DATA[:,2], kind='cubic')
    
    def show_phantom(self, zind):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(self.voxels[:,:,zind])
        plt.show()
        
    def x_plane(self, i):
        return self.geometry.x -self.geometry.width/2. + i*self.l_vox_x
    
    def y_plane(self, j):
        return self.geometry.y -self.geometry.height/2. + j*self.l_vox_y

    def z_plane(self, k):
        return self.geometry.z + k*self.l_vox_z
    
    def get_voxel(self, i, j, k):
        z = self.voxels.shape[2] - 1
        return self.voxels[i,j,z-k] 
             
    def calc_mu_per_mm_fast(self, tissue, E):
        #try:
        mu = np.zeros((len(tissue), len(E)), dtype=np.float64)    
        mu[np.equal(0, tissue),:] = self.MU_PMMA(E) * self.__PMMA_DENS / 10 #per mm
        mu[np.equal(3, tissue),:] = self.MU_AL(E) * self.__AL_DENS / 10
        mu[np.equal(2, tissue),:] = self.MU_AU(E) * self.__AU_DENS / 10
        mu[np.equal(1, tissue),:] = self.MU_AU(E) * self.__BD_DENS / 10
        return mu

if __name__=='__main__':
    import pylab
#    au_diam=1.
#    au_thick=0.36*10**(-3)
#    SPR = 0.18
#    home_path = 'N:/MatMoDatPrivate/kretz01/linux/'
#    cdmam1 = CDMAM_phantom(au_diameter=au_diam, au_thickness=au_thick, pos=np.array([0,0,71]), SPR=SPR, home_path=home_path)
##    cdmam2 = CDMAMFF('N:/MatMoDatPrivate/kretz01/linux/MATLAB/VirtualCDMAM/VirtualCDMAM.h5',pos=np.array([0,0,71]), SPR=SPR, home_path=home_path)
#
#    npx = 2323 #better use 258 for 18mm since this yield ~0.07mm per px
#    npy = 3654
#    pxpitch = 0.06 
#    dimx = npx*pxpitch
#    dimy = npy*pxpitch
#    
#    detector = Detector(npx, npy, np.array([0,0,0]), dimx, dimy, home_path)

#    cdmam.show_phantom('x')
#    cdmam.show_phantom('y')
    home_path = '.'
    
#    cfg_w={'source':{'material': "tungsten", 'kVp': 28, 'mAs': 155, 'eBin': 0.5, 'filters':{'Be': 0.3, 'Rh':0.05}}}
#    cfg_mo={'source':{'material': "molybdenum", 'kVp': 28, 'mAs': 155, 'eBin': 0.5, 'filters':{'Be': 0.3, 'Rh':0.025}}}
    cfg_w={'source':{'material': "tungsten", 'kVp': 28, 'mAs': 155, 'eBin': 0.5, 'filters':{'Be': 0.3}}}
    cfg_mo={'source':{'material': "molybdenum", 'kVp': 28, 'mAs': 155, 'eBin': 0.5, 'filters':{'Be': 0.3}}}
    
    
    w_sourc = Source(cfg_w['source'], np.array([0,0,700]), 0.54, 0.33, home_path)
    E_min, E_max = min(w_sourc.Spectrum[:,0]), max(w_sourc.Spectrum[:,0])
    E_sampled = np.arange(E_min, E_max, cfg_w['source']['eBin'])
    w_I0 = w_sourc.get_intensity(E_sampled)
    
    mo_sourc = Source(cfg_mo['source'], np.array([0,0,700]), 0.54, 0.33, home_path)
    mo_I0 = mo_sourc.get_intensity(E_sampled)
       
    pylab.figure()
#    pylab.plot(E_sampled, w_I0/np.sum(w_I0), linewidth=1.5, label='W  + Rh(0.05)')
#    pylab.plot(E_sampled, mo_I0/np.sum(mo_I0), linewidth=1.5, label='Mo + Rh(0.025)')
    pylab.plot(E_sampled, w_I0/np.sum(w_I0), linewidth=1.5, label='W')
    pylab.plot(E_sampled, mo_I0/np.sum(mo_I0), linewidth=1.5, label='Mo')
    pylab.xlabel('photon energy [keV]')
    pylab.ylabel('relative intensity')
    pylab.xlim(0., 40.)
#    pylab.grid()
    pylab.legend()
    pylab.rc('axes', linewidth=0.75)
    pylab.rc('font', size=13)
    plt.savefig('unfiltered_spectrum.jpg', dpi=300)
#    pylab.rc('figure', figsize=[16, 10])
    pylab.show()
    #e_max = 40.
    #source = Source('rhodium', e_max, np.array([0,0,700]), 0.54, 0.33)
    #plt.figure()
    #plt.plot(source.spectrum[:,0], source.spectrum[:,1])
    #print(np.max(source.spectrum[:,1]))