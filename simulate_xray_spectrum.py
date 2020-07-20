# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:55:00 2017

@author: kretz01
"""
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from os.path import normpath
from os.path import join as pjoin
import time

class SimulateXRaySpectrum():
    """
    class
    
    Keyword arguments:
        voltage         -- tube voltage in range 18kV and 42kV 
        anode_material  -- string representing anode material.
                           Choose molybdenum, rhodium or tungsten.
    """    
    def __init__(self, anode_material, voltage, path=''):  
        self.path = path
        
        if anode_material == 'molybdenum':
            self.material = 'molybdenum'
            filename = normpath(pjoin('.','attenuation_files','molybdenum_coefficients.npy'))
        elif anode_material == 'rhodium':
            self.material = 'rhodium'
            filename = normpath(pjoin('.','attenuation_files','rhodium_coefficients.npy'))
        elif anode_material == 'tungsten':
            self.material = 'tungsten'
            filename = normpath(pjoin('.','attenuation_files','tungsten_coefficients.npy'))
        else:
            raise ValueError('Unknown anode material:',anode_material)
        
        try:
            self.coefficients = np.load(filename)
        except:
            print('File',filename,'not found!')
            
        self.min_energy = np.min(self.coefficients[:,0])
        #print(self.min_energy)
        self.step = self.coefficients[1,0] - self.coefficients[0,0]

        if voltage < 18 or voltage > 40:
            raise ValueError('Voltage out of range [18kV, 40kV]')
        else:
            self.voltage = voltage
        
        self.spectrum = 0
    
    def set_voltage(self, V):
        self.voltage = V
        print('Voltage has changed. Please recompute spectrum')
        
    def __get_coefficients(self, E):
        ind = np.argwhere(np.equal(E, self.coefficients[:, 0])).flatten()[0]
        a0 = self.coefficients[ind, 1]
        a1 = self.coefficients[ind, 2]
        a2 = self.coefficients[ind, 3]
        a3 = self.coefficients[ind, 4]
        #print(E, a0, a1, a2 , a3)
        return a0, a1, a2, a3
            
    def compute_spectrum(self, keV_bin=1):
        """Computes the X-Ray spectrum for a given anode material and tube voltage.
        Methods are taken form Boones 1997."""
#        self.energies = np.arange(np.ceil(self.min_energy), np.floor(self.voltage), 1) # energy resolution 1 keV
#        self.energies = np.arange(self.min_energy, self.voltage+keV_bin, keV_bin)
        self.energies = np.arange(1, 40 + keV_bin, keV_bin)
        self.spectrum = np.zeros((len(self.energies),2))
#        self.spectrum = np.zeros((40,2))
#        self.spectrum[:,0] = np.arange(1,40+1,1)
        self.spectrum[:,0] = self.energies
        for (i,E) in enumerate(self.energies):
            if E>=self.min_energy and E<=self.voltage:
                a0, a1, a2, a3 = self.__get_coefficients(E)
#               self.spectrum[int(E), 1] = a0 + a1 * self.voltage + a2 * self.voltage**2 + a3 * self.voltage**3
                self.spectrum[i, 1] = (a0 + a1 * self.voltage + a2 * self.voltage**2 + a3 * self.voltage**3)/keV_bin
        return self.spectrum
    
    def compute_fluence_per_exposure(self, E):
        """
        """
        air_dat = np.loadtxt(normpath(self.path+'DataFiles/AttenuationFiles/air_attenuation.txt'), skiprows=5)
        mu_air = interp1d(air_dat[:,0]*1000, air_dat[:,2], kind='linear', bounds_error=False, fill_value=np.inf)
        return 5.43e5/(mu_air(E)*E)
    
    def normalize_spectrum(self, spectr):
        spectr[:,1] = spectr[:,1]/spectr[:,1].sum()
        return spectr
    
    def __parse_filter(self, material):
        if material == 'AIR':
            density = 0.0012 #g/cm^3
            f_filename = normpath(pjoin('.','attenuation_files','air_attenuation.txt'))        
        elif material == 'Al':
            density = 2.6989 #g/cm^3
            f_filename = normpath(pjoin('.','attenuation_files','al_attenuation.txt'))        
        elif material == 'Be':
            density = 1.848 #g/cm^3
            f_filename = normpath(pjoin('.','attenuation_files','be_attenuation.txt')) 
        elif material == 'Mo':
            density = 10.28 #g/cm^3
            f_filename = normpath(pjoin('.','attenuation_files','mo_attenuation.txt'))
        elif material == 'Rh':
            density = 12.38 #g/cm^3
            f_filename = normpath(pjoin('.','attenuation_files','rh_attenuation.txt'))
        elif material == 'PMMA':
            density = 1.19 #g/cm^3
            f_filename = normpath(pjoin('.','attenuation_files','pmma_attenuation.txt'))
        else:
            raise ValueError('Unknown filter material:',material)
            
        try:
            attenuation_data = np.loadtxt(f_filename, skiprows=5)
            mu = interp1d(attenuation_data[:,0]*1000, attenuation_data[:,2], kind='linear', bounds_error=False, fill_value=0)
        except:
            print('File',f_filename,'not found!')
        return density, mu
        
    def filter_spectrum(self, spectrum, material, thickness):
        density, mu = self.__parse_filter(material)
        filtered_spectrum = spectrum.copy()
        filtered_spectrum[:,1] *= np.exp(- density * mu(spectrum[:,0]) * thickness / 10)
        return filtered_spectrum 
    
    def __rejection_sampling(self,sample_size, kind='cubic'):
        """Perform Rejection sampling of a given spectrum. 
        For further details go to https://en.wikipedia.org/wiki/Rejection_sampling."""
        sampling = np.ones((sample_size, 2))
        n_points = 0
        x_min, x_max = min(self.spectrum[:,0]), max(self.spectrum[:,0])
        y_min, y_max = min(self.spectrum[:,1]), max(self.spectrum[:,1])
        x_dist = x_max - x_min
        y_dist = y_max - y_min
        #print(x_dist, y_dist)
        int_spect = interp1d(self.spectrum[:,0], self.spectrum[:,1], kind=kind)
        while n_points < sample_size:
            sample_point = np.random.random((1,2)).flatten()
            sample_point[0] = sample_point[0] * x_dist + x_min
            sample_point[1] = sample_point[1] * y_dist + y_min
            #print(sample_point)
            if sample_point[1] <= int_spect(sample_point[0]):
                sampling[n_points] *= sample_point
                n_points += 1
        return sampling
    
    def sample_spectrum(self, n_photons, bins=50):
        """Samples the spectrum, so that there are n_photons photons that form 
        the approx. spectrum all together."""
        if np.any(self.spectrum):
            print('Sampling Spectrum with',n_photons,'photons.')
            print('May need some time! (1E+6 photons approx 500s!)')
            sampled = self.__rejection_sampling(n_photons)
            photon_count, edges = np.histogram(sampled[:,0], bins=bins)
            energies = ((edges + np.roll(edges, -1))/2)[:-1]
            photon_array = np.array([])
            for i in range(0,len(energies)):
                photon_array = np.concatenate((photon_array, np.tile(energies[i], photon_count[i])))
                np.random.shuffle(photon_array)
            return photon_count, energies, photon_array
        else:
            raise ValueError('No spectrum defined. Call compute_spectrum() first.')
            
    def sample_spectrum_fast(self, x_sample_size=60, bins=50):
        """Samples the spectrum, so that there are n_photons photons that form 
        the approx. spectrum all together."""
        if np.any(self.spectrum):
            print('Sampling Spectrum.')
            x_min, x_max = min(self.spectrum[:,0]), max(self.spectrum[:,0])
            x_sampled = np.linspace(x_min, x_max, x_sample_size)
            x_dist = np.mean((np.roll(x_sampled, -1) - x_sampled)[:-1])
            int_spec = interp1d(self.spectrum[:,0], self.spectrum[:,1], kind='cubic')
            photon_count = int_spec(x_sampled) * x_dist
            photon_count = np.round(photon_count)
            print(np.sum(photon_count))
            photon_array = np.array([])
            for i in range(0,len(x_sampled)):
                photon_array = np.concatenate((photon_array, np.tile(x_sampled[i], int(photon_count[i]))))
            np.random.shuffle(photon_array)
            return photon_count, x_sampled, photon_array
        else:
            raise ValueError('No spectrum defined. Call compute_spectrum() first.')
        
if __name__=='__main__':
    import seaborn as sns
    sns.set_style("white")
    sns.set_context("talk") 
    home_path = 'N:/MatMoDatPrivate/kretz01/linux/'
    material = 'tungsten'
    voltage = 31.
    simulator = SimulateXRaySpectrum(anode_material=material, 
                                                voltage=voltage, filter_material='rhodium', filter_thickness=0.015, path=home_path)
    spectrum = simulator.compute_spectrum()
    plt.figure()
    plt.plot(spectrum[:,0],spectrum[:,1], label='Tungsten 31 kV spectrum')
    f_spectrum = simulator.filter_spectrum(spectrum=spectrum)
    plt.plot(f_spectrum[:,0],f_spectrum[:,1], label='Filtered spectrum (Rhodium)')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Photon Flux')
    plt.legend(loc='best')
    plt.show()
    #sampled_30, energ, photons = simulator.sample_spectrum_fast()
#    spectrum_20 = simulator.compute_spectrum()
#    simulator.set_voltage(30.)
#    spectrum_30 = simulator.compute_spectrum()
#    
#    
#    
#    simulator.set_voltage(40.) 
#    spectrum_40 = simulator.compute_spectrum()
#    
#    plt.figure()
#    plt.title(material)
#    plt.plot(spectrum_20[:,0], spectrum_20[:,1])
#    plt.plot(spectrum_30[:,0], spectrum_30[:,1])
#    plt.plot(spectrum_40[:,0], spectrum_40[:,1])
#    plt.axis([0, 45, 0, 6E+7])
#    
#    plt.figure()
#    plt.title(material+' sampled')
#    plt.plot(energ, sampled_30)
#    plt.axis([0, 45, 0, 1E+3])
#    plt.show()
    