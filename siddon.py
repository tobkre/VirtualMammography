# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:20:52 2017

@author: kretz01
"""
import numpy as np

def merge(arr_1, arr_2, arr_3=np.array([])):
    m_arr = np.zeros(len(arr_1)+len(arr_2)+len(arr_3))
    ind = 0
    i = 0
    while i < max(len(arr_1),len(arr_2),len(arr_3)):
        if i < len(arr_1):
            m_arr[ind] = arr_1[i]
            ind += 1
        if i < len(arr_2):
            m_arr[ind] = arr_2[i]
            ind+=1
        if i < len(arr_3):
            m_arr[ind] = arr_3[i]
            ind+=1
        i+=1
    return m_arr

def calc_length(alpha, dist):
    return dist * (np.roll(alpha, -1) - alpha)[:-1]

def calc_alpha_min_max(start, end, p_object):
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
    nx = p_object.n_vox_x
    ny = p_object.n_vox_y
    nz = p_object.n_vox_z
   
    a_x_1 = 0
    a_x_nx = 1
    
    a_y_1 = 0
    a_y_ny = 1
    
    a_z_1 = 0
    a_z_nz = 1
    
    if start[0]-end[0] != 0:
        a_x_1 = (p_object.x_plane(0)-start[0])/(end[0]-start[0])
        a_x_nx = (p_object.x_plane(nx)-start[0])/(end[0]-start[0])
        
    if start[1]-end[1] != 0:
        a_y_1 = (p_object.y_plane(0)-start[1])/(end[1]-start[1])
        a_y_ny = (p_object.y_plane(ny)-start[1])/(end[1]-start[1])
        
    if start[2]-end[2] != 0:
        a_z_1 = (p_object.z_plane(0)-start[2])/(end[2]-start[2])
        a_z_nz = (p_object.z_plane(nz)-start[2])/(end[2]-start[2])
    
    a_min = max(0, min(a_x_1, a_x_nx), min(a_y_1, a_y_ny), min(a_z_1, a_z_nz))
    a_max = min(1, max(a_x_1, a_x_nx), max(a_y_1, a_y_ny), max(a_z_1, a_z_nz))
    #print(np.linalg.norm(((a_max-a_min)*(start-end))))
    return a_min, a_max

def calc_attenuation_parameters(start, end, p_object):
    dist = np.linalg.norm(start - end)
    
    nx = p_object.n_vox_x
    ny = p_object.n_vox_y
    nz = p_object.n_vox_z
    
    dx = p_object.l_vox_x
    dy = p_object.l_vox_y
    dz = p_object.l_vox_z
        
    alpha_min ,alpha_max = calc_alpha_min_max(start, end, p_object)
    #print('alpha_min =',alpha_min, ', alpha_max =',alpha_max)
    
    if alpha_max <= alpha_min:
        #ray does not intersect. return length=0 and tissue=0 
        return np.array([0], dtype=np.float64), np.array([0], dtype=np.int32)
        
    if end[0]-start[0] > 0:
        #print('x>0')
        i_min = np.ceil(nx - (p_object.x_plane(nx) - alpha_min*(end[0]-start[0]) - start[0])/dx)
        i_max = np.floor((start[0] + alpha_max*(end[0]-start[0])-p_object.x_plane(0))/dx)
    else:
        #print('x<0')
        i_min = np.ceil(nx - (p_object.x_plane(nx) - alpha_max*(end[0]-start[0]) - start[0])/dx)
        i_max = np.floor((start[0] + alpha_min*(end[0]-start[0])-p_object.x_plane(0))/dx)
    
    if end[1]-start[1] > 0:
        #print('y>0')
        j_min = np.ceil(ny - (p_object.y_plane(ny) - alpha_min*(end[1]-start[1]) - start[1])/dy)
        j_max = np.floor((start[1] + alpha_max*(end[1]-start[1])-p_object.y_plane(0))/dy)
    else:
        #print('y<0')
        j_min = np.ceil(ny - (p_object.y_plane(ny) - alpha_max*(end[1]-start[1]) - start[1])/dy)
        j_max = np.floor((start[1] + alpha_min*(end[1]-start[1])-p_object.y_plane(0))/dy)
        
    if end[2]-start[2] > 0:
        #print('z>0')
        k_min = np.ceil(nz - (p_object.z_plane(nz) - alpha_min*(end[2]-start[2]) - start[2])/dz)
        k_max = np.floor((start[2] + alpha_max*(end[2]-start[2])-p_object.z_plane(0))/dz)
    else:
        #print('z<0')
        k_min = np.ceil(nz - (p_object.z_plane(nz) - alpha_max*(end[2]-start[2]) - start[2])/dz)
        k_max = np.floor((start[2] + alpha_min*(end[2]-start[2])-p_object.z_plane(0))/dz)
    
    alpha_x = np.zeros(int(i_max-i_min+1))
    alpha_y = np.zeros(int(j_max-j_min+1))
    alpha_z = np.zeros(int(k_max-k_min+1))
    for (ind,i) in enumerate(np.arange(i_min, i_max+1)):
        alpha_x[ind] = (p_object.x_plane(i)-start[0])/(end[0]-start[0])
    
    for (ind,i) in enumerate(np.arange(j_min, j_max+1)):
        alpha_y[ind] = (p_object.y_plane(i)-start[1])/(end[1]-start[1])
        
    for (ind,i) in enumerate(np.arange(k_min, k_max+1)):
        alpha_z[ind] = (p_object.z_plane(i)-start[2])/(end[2]-start[2])
    
    alpha_x = np.sort(alpha_x)
    alpha_y = np.sort(alpha_y)
    alpha_z = np.sort(alpha_z)
#    print(alpha_x)
#    print(alpha_y)
#    print(alpha_z)
    
    alpha = np.empty(len(alpha_x)+len(alpha_y)+len(alpha_z)+2)
    ind = 0
    alpha[ind] = alpha_min
    ind+=1
    for i in np.sort(merge(alpha_x, alpha_y, alpha_z)):
        alpha[ind] = i
        ind+=1
        
    alpha[ind] = alpha_max
    #print('alpha', alpha)
    length = calc_length(alpha, dist)
    alpha_mid = ((np.roll(alpha, -1) + alpha)/2)[:-1]
    i = np.int32(np.ceil((start[0]+alpha_mid*(end[0]-start[0])-p_object.x_plane(0))/dx) -1) #for plane_index to voxel_index, -1 for python index starting from 0
    j = np.ceil((start[1]+alpha_mid*(end[1]-start[1])-p_object.y_plane(0))/dy -1) #for plane_index to voxel_index, -1 for python index starting from 0
    k = np.ceil((start[2]+alpha_mid*(end[2]-start[2])-p_object.z_plane(0))/dz -1) #for plane_index to voxel_index, -1 for python index starting from 0
    # round richtiger Operator???

#    vx_i = (i + np.roll(i, 1))/2
#    vx_i[-1] = i[-1]
#    vx_j = (j + np.roll(j, 1))/2
#    vx_j[-1] = j[-1]
#    vx_k = (k + np.roll(k, -1))/2
#    vx_k[-1] = k[-1]
#    vx_i-=1
#    vx_j-=1
#    vx_k-=1
    i[i<0]=0
    j[j<0]=0
    k[k<0]=0
    tissue = p_object.get_voxel(j.astype(int), i.astype(int), k.astype(int))
    #print(len(tissue), np.sum(np.equal(1, tissue)), np.sum(np.equal(2, tissue)))
    return length.astype(np.float64), tissue.astype(np.int32)

def test_it():
    '''Added doc'''
    from experiment_object import Detector, Source, Plate, VoxelObject, CDMAM_cell
    import matplotlib.pyplot as plt
    home_path='.'
    
    detector = Detector(296, 296, np.array([0,0,0]), 18, 18, 14, 2032443611.0515957, '.')
    cfg_s={'source':{'material': "tungsten", 'kVp': 28, 'mAs': 155, 'eBin': 0.5, 'filters':{'Be': 0.3}}}
    source = Source(cfg_s['source'], np.array([0,0,700]), 0.54, 0.33, home_path)
    diam = np.array([1.6])
    #thick = np.array([.13e-3, .16e-3, .20e-3, .25e-3, .36e-3])
    thick = np.array([.13e-3])
    for au_diam in diam:
        for au_thick in thick:
            phantom = CDMAM_cell(au_diameter=au_diam, au_thickness=au_thick, pos=np.array([0,0,10]), SPR=0.18, home_path=home_path)
            
            n_rays = detector.npxx * detector.npxy
            image = np.empty((detector.npxx, detector.npxy))
            x_pos1_v, y_pos1_v = detector.x_grid, detector.y_grid
            z_pos1 =  detector.geometry.z
            
            n_ray = 0
#            pbar = progressbar.ProgressBar(max_value=n_rays)
            
            E_min, E_max = min(source.Spectrum[:,0]), max(source.Spectrum[:,0])
            E_sampled = np.linspace(E_min, E_max, 60)
            
            I0 = source.get_intensity(E_sampled)
            
            response = detector.get_response(E_sampled)
            
            norm_fact = np.sum(I0 * E_sampled * response)
            
            for x_ind in range(0, detector.npxx):
                for y_ind in range(0,detector.npxy):
                    pos_1 = np.array([x_pos1_v[y_ind, x_ind], y_pos1_v[y_ind, x_ind], z_pos1])
                    
                    length, tissue = calc_attenuation_parameters(np.array([-50,0,700]), pos_1, phantom)              
                    red_intensities = np.empty_like(E_sampled)
                    red_spectrum = np.empty((len(red_intensities), 2))
                    mu = phantom.calc_mu_per_mm_fast(tissue, E_sampled)
                    red_intensities = I0 * np.exp(- np.sum(np.transpose(mu) * length, axis=1)) #* ( 1 + np.random.normal(loc=0.0, scale=1*10**(-2)))

#                    #zwischen = np.sum(red_intensities * E_sampled * response)#/norm_fact
#                    #zwischen,_ = quad(lambda x: interp1d(E_sampled, red_intensities*response, kind='cubic')(x), E_min, E_max)
                    zwischen = np.sum(red_intensities*response*(np.roll(E_sampled, -1)-E_sampled)[0])                    
                    image[x_ind, y_ind] =  zwischen
                    n_ray += 1
#                    pbar.update(n_ray)
            
            """add scattering"""
            rows, cols = image.shape
            crow, ccol = int(rows/2), int(cols/2)
            f_size = 2
            lpf = np.zeros(image.shape)
            lpf[crow-f_size:crow+f_size, ccol-f_size:ccol+f_size] = 1
            
            fft_sig = np.fft.fft2(image)
            fshift = np.fft.fftshift(fft_sig)
            
            f_ishift = np.fft.ifftshift(fshift*lpf)
            scatter = np.abs(np.fft.ifft2(f_ishift))
            
            image = image + scatter
            
            """blur image"""
            bl_img = detector.consider_blur(image, method='gaussian')#TODO: check if gaussian point spread function is a good assumption
            
            """add white noise to image"""
            no_img_bl = detector.compute_noise(bl_img, SNR=9.7, method='white_noise')
           
            """digitize detector output linear"""
            #dg = detector.get_dg_output(no_img_bl, 14, 10**19)
            dg = detector.get_dg_output(no_img_bl)
            
            plt.figure()
            plt.imshow(dg, cmap='gray')
            plt.title(str(au_thick)+', '+str(au_diam))
            plt.figure()
            plt.plot(dg[:,150])
    plt.show()

if __name__=='__main__':
    test_it()
    