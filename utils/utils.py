import numpy as np
from astropy import coordinates
import astropy.units as u

from utils.constants import *

def find_nearest(array, value):
    """
    This function finds the nearest value in an array and returns the 
    closest value and the corresponding index.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def time_convert(seconds): 
    """
    Converts runtimes in seconds to hours:minutes:seconds format.
    """
    seconds = seconds % (24. * 3600.) 
    hour = seconds // 3600.
    seconds %= 3600.
    minutes = seconds // 60.
    seconds %= 60.
      
    return "%d:%02d:%02d" % (hour, minutes, seconds)


# TODO: implement generating numbered output diretories
def get_default_outdir(infile):
    return infile.parent.joinpath(DEFAULT_OUTDIR)


def get_ebv(ra, dec):
    co = coordinates.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='fk5')
    try:
        table = IrsaDust.get_query_table(co, section='ebv')
        ebv = table['ext SandF mean'][0]
    except:
        return 0.04  # average Galactic E(B-V)

    # If E(B-V) is large, it can significantly affect normalization of the
    # spectrum, in addition to changing its shape.  Re-normalizing the spectrum
    # throws off the maximum likelihood fitting, so instead of re-normalizing,
    # we set an upper limit on the allowed ebv value for Galactic de-reddening.
    if (ebv >= 1.0):
        return 0.04  # average Galactic E(B-V)
    return ebv


# Galactic Extinction Correction
def ccm_unred(wave, flux, ebv, r_v=3.1):
    """ccm_unred(wave, flux, ebv, r_v="")
    Deredden a flux vector using the CCM 1989 parameterization 
    Returns an array of the unreddened flux
    
    INPUTS:
    wave - array of wavelengths (in Angstroms)
    dec - calibrated flux array, same number of elements as wave
    ebv - colour excess E(B-V) float. If a negative ebv is supplied
          fluxes will be reddened rather than dereddened     
    
    OPTIONAL INPUT:
    r_v - float specifying the ratio of total selective
          extinction R(V) = A(V)/E(B-V). If not specified,
          then r_v = 3.1
            
    OUTPUTS:
    funred - unreddened calibrated flux array, same number of 
             elements as wave
             
    NOTES:
    1. This function was converted from the IDL Astrolib procedure
       last updated in April 1998. All notes from that function
       (provided below) are relevant to this function 
       
    2. (From IDL:) The CCM curve shows good agreement with the Savage & Mathis (1979)
       ultraviolet curve shortward of 1400 A, but is probably
       preferable between 1200 and 1400 A.
    3. (From IDL:) Many sightlines with peculiar ultraviolet interstellar extinction 
       can be represented with a CCM curve, if the proper value of 
       R(V) is supplied.
    4. (From IDL:) Curve is extrapolated between 912 and 1000 A as suggested by
       Longo et al. (1989, ApJ, 339,474)
    5. (From IDL:) Use the 4 parameter calling sequence if you wish to save the 
       original flux vector.
    6. (From IDL:) Valencic et al. (2004, ApJ, 616, 912) revise the ultraviolet CCM
       curve (3.3 -- 8.0 um-1). But since their revised curve does
       not connect smoothly with longer and shorter wavelengths, it is
       not included here.
    
    7. For the optical/NIR transformation, the coefficients from 
       O'Donnell (1994) are used
    
    >>> ccm_unred([1000, 2000, 3000], [1, 1, 1], 2 ) 
    array([9.7976e+012, 1.12064e+07, 32287.1])
    """
    wave = np.array(wave, float)
    flux = np.array(flux, float)
    
    if wave.size != flux.size: raise TypeError( 'ERROR - wave and flux vectors must be the same size')

    x = 10000.0/wave
    # Correction invalid for x>11:
    if np.any(x>11):
        return flux 

    npts = wave.size
    a = np.zeros(npts, float)
    b = np.zeros(npts, float)
    
    ###############################
    #Infrared
    
    good = np.where( (x > 0.3) & (x < 1.1) )
    a[good] = 0.574 * x[good]**(1.61)
    b[good] = -0.527 * x[good]**(1.61)
    
    ###############################
    # Optical & Near IR

    good = np.where( (x  >= 1.1) & (x < 3.3) )
    y = x[good] - 1.82
    
    c1 = np.array([ 1.0 , 0.104,   -0.609,  0.701,  1.137, \
                  -1.718,   -0.827, 1.647, -0.505 ])
    c2 = np.array([ 0.0,  1.952,    2.908,   -3.989, -7.985, \
                  11.102,   5.491,  -10.805,  3.347 ] )

    a[good] = np.polyval(c1[::-1], y)
    b[good] = np.polyval(c2[::-1], y)

    ###############################
    # Mid-UV
    
    good = np.where( (x >= 3.3) & (x < 8) )   
    y = x[good]
    F_a = np.zeros(np.size(good),float)
    F_b = np.zeros(np.size(good),float)
    good1 = np.where( y > 5.9 ) 
    
    if np.size(good1) > 0:
        y1 = y[good1] - 5.9
        F_a[ good1] = -0.04473 * y1**2 - 0.009779 * y1**3
        F_b[ good1] =   0.2130 * y1**2  +  0.1207 * y1**3

    a[good] =  1.752 - 0.316*y - (0.104 / ( (y-4.67)**2 + 0.341 )) + F_a
    b[good] = -3.090 + 1.825*y + (1.206 / ( (y-4.62)**2 + 0.263 )) + F_b
    
    ###############################
    # Far-UV
    
    good = np.where( (x >= 8) & (x <= 11) )   
    y = x[good] - 8.0
    c1 = [ -1.073, -0.628,  0.137, -0.070 ]
    c2 = [ 13.670,  4.257, -0.420,  0.374 ]
    a[good] = np.polyval(c1[::-1], y)
    b[good] = np.polyval(c2[::-1], y)

    # Applying Extinction Correction
    
    a_v = r_v * ebv
    a_lambda = a_v * (a + b/r_v)
    
    funred = flux * 10.0**(0.4*a_lambda)   

    return funred


def window_filter(spec,size):
    """
    Estimates the median value of the spectrum 
    within a pixel window.
    """
    med_spec = np.empty(len(spec))
    pix = np.arange(0,len(spec),1)
    for i,p in enumerate(pix):
        # Get n-nearest pixels
        # Calculate distance from i to each pixel
        i_sort =np.argsort(np.abs(i-pix))
        idx = pix[i_sort][:size] # indices we estimate from
        med = np.median(spec[idx])
        med_spec[i] = med
    #
    return med_spec


def interpolate_metal(spec,noise):
    """
    Interpolates over metal absorption lines for 
    high-redshift spectra using a moving median
    filter.
    """
    sig_clip = 3.0
    nclip = 10
    bandwidth= 15
    med_spec = window_filter(spec,bandwidth)
    count = 0 
    new_spec = np.copy(spec)
    while (count<=nclip) and ((np.std(new_spec-med_spec)*sig_clip)>np.median(noise)):
        count+=1
        # Get locations of nan or -inf pixels
        nan_spec = np.where((np.abs(new_spec-med_spec)>(np.std(new_spec-med_spec)*sig_clip)) & (new_spec < (med_spec-sig_clip*noise)) )[0]
        if len(nan_spec)>0:
            inan = np.unique(np.concatenate([nan_spec]))
            buffer = 0
            inan_buffer_upp = np.array([(i+buffer) for i in inan if (i+buffer) < len(spec)],dtype=int)
            inan_buffer_low = np.array([(i-buffer) for i in inan if (i-buffer) > 0],dtype=int)
            inan = np.concatenate([inan,inan_buffer_low, inan_buffer_upp])
            # Interpolate over nans and infs if in spec
            new_spec[inan] = np.nan
            new_spec = insert_nan(new_spec,inan)
            nans, x= nan_helper(new_spec)
            new_spec[nans]= np.interp(x(nans), x(~nans), new_spec[~nans])
        else:
            break
    #
    return new_spec
