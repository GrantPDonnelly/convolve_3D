from astropy.convolution import convolve, convolve_fft
from astropy.modeling.models import Gaussian2D
from photutils.psf.matching import create_matching_kernel
import numpy as np
from astropy.io import fits
from tqdm import tqdm

def convolve_JWST_cube(input_cubefile, target_lam, savename, target_inst, exclusion_threshold=0.9):

    '''
    Smooth a JWST cube to the resolution at a given wavelength.

    input_cubefile (str): the file name of the FITS cube that will be convolved
    target_lam (int or float): the wavelength in micrometers corresponding to the desired resolution
    savename (str): file name to save the convolved cube to
    target_inst (options: "NRS" or "MIRI"): the instrument with the desired resolution to convolve with. 
        This is important because the two instruments have an overlap in wavelengths.
    exclusion_threshold (int or float): the fraction of real data that a pixel should contain after convolution to be kept.
    
    '''

    NRS_handoff = 4.1 # micron

    def get_fwhm_MIRI(lam):
        # From Law+2023 https://iopscience.iop.org/article/10.3847/1538-3881/acdddc
        return 0.033*(lam) + 0.106 # arcsec
    
    # These are from my 2D Gaussian fits for a cube of a star
    def get_x_fwhm_NRS(lamda):
        if lamda < NRS_handoff:
            return (0.01*lamda) + 0.14 # arcsec
        elif lamda >= NRS_handoff:
            return (0.02*lamda) + 0.15 # arcsec
    def get_y_fwhm_NRS(lamda):
        if lamda < NRS_handoff:
            return (0.01*lamda) + 0.16 # arcsec
        elif lamda >= NRS_handoff:
            return (0.01*lamda) + 0.15 # arcsec
    
    def get_spectral_axis(cube):
        wav_short = cube[1].header['CRVAL3']
        delta_lam = cube[1].header['CDELT3']
        lam_n = len(cube[1].data)
        spectral_axis = np.asarray([wav_short + (delta_lam*i) for i in range(lam_n)])
        return spectral_axis

    # Prepare input cube info
    input_cube = fits.open(input_cubefile)
    input_spectral_axis = get_spectral_axis(input_cube)
    input_slice = input_cube[1].data[0]
    if input_cube[0].header['INSTRUME'] == 'NIRSPEC':
        input_cube_type = 'NRS'
    elif input_cube[0].header['INSTRUME'] == 'MIRI':
        input_cube_type = 'MIRI'
    
    # Get target PSF parameters
    if target_inst == 'NRS':
        target_x_fwhm = get_x_fwhm_NRS(target_lam)
        target_sig_arcsec_x = target_x_fwhm/np.sqrt(8*np.log(2))
        target_sig_pix_x = target_sig_arcsec_x/np.sqrt(input_cube[1].header['PIXAR_A2'])
    
        target_y_fwhm = get_y_fwhm_NRS(target_lam)
        target_sig_arcsec_y = target_y_fwhm/np.sqrt(8*np.log(2))
        target_sig_pix_y = target_sig_arcsec_y/np.sqrt(input_cube[1].header['PIXAR_A2'])

    elif target_inst == 'MIRI':
        target_fwhm = get_fwhm_MIRI(target_lam)
        target_sig_arcsec = target_fwhm/np.sqrt(8*np.log(2))
        target_sig_pix_x = target_sig_arcsec/np.sqrt(input_cube[1].header['PIXAR_A2'])
        target_sig_pix_y = target_sig_arcsec/np.sqrt(input_cube[1].header['PIXAR_A2'])
    
    # Grid that the PSFs will be spread onto
    y, x = np.mgrid[0:input_slice.shape[0], 0:input_slice.shape[1]]
    midx, midy = int(np.floor(input_slice.shape[1]/2)), \
                 int(np.floor(input_slice.shape[0]/2))
    
    # Kernel for the target resolution
    target_gauss = Gaussian2D(1, midx, midy, target_sig_pix_x, target_sig_pix_y)
    target_kernel = target_gauss(x, y)
    
    # Loop through each plane and convolve
    convolved_planes = []
    convolved_uncertainty_planes = []
    for i in tqdm(range(len(input_spectral_axis))):
        
        this_lam = input_spectral_axis[i]
        this_slice = input_cube[1].data[i]
        this_slice_unc = input_cube[2].data[i]
        
        if this_lam <= target_lam:
    
            # make the gaussian for this wavelength
            if input_cube_type == 'NRS':
                this_x_fwhm_NRS = get_x_fwhm_NRS(this_lam)
                this_sig_arcsec_x = this_x_fwhm_NRS/np.sqrt(8*np.log(2))
                this_sig_pix_x = this_sig_arcsec_x/np.sqrt(input_cube[1].header['PIXAR_A2'])
            
                this_y_fwhm_NRS = get_y_fwhm_NRS(this_lam)
                this_sig_arcsec_y = this_y_fwhm_NRS/np.sqrt(8*np.log(2))
                this_sig_pix_y = this_sig_arcsec_y/np.sqrt(input_cube[1].header['PIXAR_A2'])

            elif input_cube_type == 'MIRI':
                this_x_fwhm_MIRI = get_fwhm_MIRI(this_lam)
                this_sig_arcsec_x = this_x_fwhm_MIRI/np.sqrt(8*np.log(2))
                this_sig_pix_x = this_sig_arcsec_x/np.sqrt(input_cube[1].header['PIXAR_A2'])
                this_sig_pix_y = this_sig_pix_x
            
            input_gauss = Gaussian2D(1, midx, midy, this_sig_pix_x, this_sig_pix_y)

            # Make the kernel for the resolution of this wavelength
            input_kernel = input_gauss(x, y)

            # Make the matching kernel for this wavelength and the desired resolution
            matching_kernel = create_matching_kernel(input_kernel, target_kernel)

            # First, convolve a set of 1s to see where we need to throw out data
            fake_slice = np.where(this_slice!=this_slice, 0, 1)
            fake_convolved_image = convolve_fft(fake_slice, matching_kernel, 
                                       boundary='fill', fill_value=0, preserve_nan=True)
            keep_mask = np.where(fake_convolved_image>=exclusion_threshold, 1, np.nan)

            # Convolve this wavelength plane of the cube to the target resolution
            convolved_image = convolve_fft(this_slice, matching_kernel, 
                                       boundary='fill', fill_value=0, preserve_nan=True)
            convolved_image = convolved_image*keep_mask
            convolved_planes.append(convolved_image)

            # Now convolve the uncertainties
            unc_matching_kernel = matching_kernel**2
            unc_matching_kernel /= unc_matching_kernel.sum()
            # ^^^ see https://iopscience.iop.org/article/10.3847/2515-5172/abe8df/ampdf
            # this is supposed to work for *Gaussians*
            convolved_unc_image = convolve_fft(this_slice_unc, unc_matching_kernel, 
                                       boundary='fill', fill_value=0, preserve_nan=True)
            convolved_unc_image = convolved_unc_image*keep_mask
            convolved_uncertainty_planes.append(convolved_unc_image)
            
        else:
            convolved_planes.append(this_slice)
            convolved_uncertainty_planes.append(this_slice_unc)

    # Save the new cube
    convolved_planes = np.asarray(convolved_planes)
    convolved_uncertainty_planes = np.asarray(convolved_uncertainty_planes)
    input_cube[1].data = convolved_planes
    input_cube[2].data = convolved_uncertainty_planes
    input_cube.writeto(savename, overwrite=True)
    
    return input_cube