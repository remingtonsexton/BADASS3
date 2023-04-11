# -*- coding: utf-8 -*-

import numpy as np
import warnings
from scipy.special import erf, erfc, erfcx
from scipy.special import log_ndtr
"""

    Functions for the computation of the new families of pdfs introduced in 
    Sanders & Evans (2020). The models are constructed via a convolution of an 
    arbitrary kernel with a Gaussian. They have simple convolutions with 
    observational uncertainties. The models are everywhere positive-definite
    so can be used as a simple replacement for the Gauss-Hermite series.
    
    Two options are presented:
        1. Uniform kernel -- this model has negative excess kurtosis.
        2. Laplace kernel -- this model has positive excess kurtosis.
        
    The models have the same parameters as the Gauss-Hermite series: 
        mean = mean velocity
        sigma = dispersion parameter (not equal to the standard deviation)
        h3 = 3rd Gauss-Hermite coefficient
        h4 = 4th Gauss-Hermite coefficient
    
    For each model, the following functions are provided:
    
        *_kernel_pdf(x,err,mean,sigma,h3,h4)
            the probability density function for the models convolved with
            a Gaussian uncertainty distribution of width err.
            (mean, sigma, h3, h4) are the parameters of the models.
            
        log_*_kernel_pdf(x, err, mean, sigma, h3, h4)
            natural logarithm of *_kernel_pdf -- computed in a careful way 
            to avoid under/overflow.
            
        *_kernel_fourier_transform(u, err, mean, sigma, h3, h4)
            the Fourier transform of the error-convolved models
            u is the 'frequency' in units of 1/sigma
            
        *_kernel_variance_kurtosis(sigma, h3, h4)
            the variance and excess kurtosis of the model given the parameters
            
    We also provide functions for the Gauss-Hermite series which are used as 
    a comparison for the new models.
    
    At the bottom of this file, we provide an alternative for the function
    losvd_rfft in ppxf.
            
"""

##==============================================================================
## Special functions
##==============================================================================

def log1mexp(x):
    """log(1 - exp(-x)).
    
    Taken from pymc3.math
    
    This function is numerically more stable than the naive approach.
    For details, see
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    with np.errstate(divide='ignore'):
        return np.where(x < 0.683, np.log(-np.expm1(-x)), np.log1p(-np.exp(-x)))

def logdiffexp(a, b):
    """log(exp(a) - exp(b))"""
    return a + log1mexp(a - b)

def lnerfc(x): 
    """ln erfc(x) = ln (1-erf(x)) = ln \sqrt{2/\pi}\int_x^\infty e^{-t^2} dt
    
    For positive arguments we use the identity
    ln erfc(x) = ln erfcx(x)-x^2
    
    """
    return (x>0.)*(np.log(erfcx(np.abs(x)))-x**2)+\
           (x<=0.)*np.log(erfc(-np.abs(x)))

LOWERLIM=1e-300
def lnerfcx(x):
    """ln erfcx(x) = ln (exp(x^2)erfc(x)) 
    
    For negative arguments we use the identity
    ln erfcx(x) = ln erfc(x)+x^2
    
    """
    return (x<0.)*(np.log(erfc(-np.abs(x))+LOWERLIM)+x**2)+\
           (x>=0.)*np.log(erfcx(np.abs(x))) 

##==============================================================================
## Hermite polynomials as per vdM&F 1993 H_n(x)
## --- Q_n(x) are the cumulative functions from Sellwood & Gerhard 2020
## --- alpha(x) is a unit normal
## --- lnalpha(x) is a log of unit normal
## --- Norm(x, s) is a normal with standard deviation s
##==============================================================================

H = {0:lambda y:1, 
     1:lambda y:np.sqrt(2)*y, 
     2:lambda y:(2*y*y-1)/np.sqrt(2.), 
     3:lambda y:(2*y*y-3)*y/np.sqrt(3), 
     4:lambda y: (4*y**4-12*y**2+3)/np.sqrt(24.), 
     5:lambda y: (4*y**5-20*y**3+15*y)/np.sqrt(60.), 
     6:lambda y: (8*y**6-60*y**4+90*y**2-15)/np.sqrt(720.)}

Q = {0:lambda y: .5*(1+erf(y/np.sqrt(2.))), 1:lambda y:-alpha(y), 
     2:lambda y: .5*(1+erf(y/np.sqrt(2.)))-y*alpha(y), 
     3:lambda y: -alpha(y)*(y*y+2), 
     4:lambda y: 1.5*(1+erf(y/np.sqrt(2.)))-y*alpha(y)*(y*y+3)}

alpha = lambda y: np.exp(-y**2/2.)/np.sqrt(2.*np.pi)
lnalpha = lambda y: -y**2/2.-.5*np.log(2.*np.pi)
Norm = lambda y, s: np.exp(-.5*y**2/s**2)/np.sqrt(2.*np.pi*s**2)

def gauss_hermite_series(v,params):
    '''
        Evaluate a Gauss Hermite series at v for params = gamma,V,sigma,h3,h4
        where gamma is the normalization
        
        see Sanders & Evans (2020) equation (1) for more information
        
        Parameters
        ----------
        v : array_like
            Input array of velocities.
        params : array_like
            Parameters of the Gauss-Hermite series, p=(gamma, V, sigma, h3, h4):
                gamma = normalization
                V = mean velocity
                sigma = dispersion parameter
                h3 = 3rd Gauss-Hermite coefficient
                h4 = 4th Gauss-Hermite coefficient

        Returns
        -------
        f : array_like
            Gauss-Hermite series f(v) = (\gamma/\sigma)\alpha(v)(1+h3 H3(v)+h4 H4(v)) 
    '''
    
    gamma,V,sigma,h3,h4 = params
    y = (v-V)/sigma
    f = gamma*alpha(y)/sigma*(1+h3*H[3](y)+h4*H[4](y))#+h5*H[5](y)+h6*H[6](y))
    
    return f

def gauss_hermite_series_zeroed(v,params):
    '''
        As gauss_hermite_series above but =0 if f(x)<0.
    '''
    f = gauss_hermite_series(v,params)
    f[f<0.]=0.
    return f

def gauss_hermite_series_broadened(v, err, params):
    '''
        Evaluate the convolution of a normal distribution with width err with
        a Gauss Hermite series for params = gamma,V,sigma,h3,h4
        where gamma is the normalization
        
        see Sanders & Evans (2020) Appendix C for more information
        
        Parameters
        ----------
        v : array_like
            Input array of velocities.
        err : array_like
            Input array of velocity errors.
        params : array_like
            Parameters of the Gauss-Hermite series, p=(gamma, V, sigma, h3, h4):
                gamma = normalization,
                V = mean velocity,
                sigma = dispersion parameter,
                h3 = 3rd Gauss-Hermite coefficient,
                h4 = 4th Gauss-Hermite coefficient.

        Returns
        -------
        f : array_like
            Convolution of normal distribution with Gauss-Hermite series 
                \int dv' N(v-v'|err) f(v')
    
    '''
    
    gamma,V,sigma,h3,h4 = params
    yerr = err/sigma
    sigmap = np.sqrt(sigma**2+err**2)
    yp = (v-V)/sigmap

    f = gamma/sigmap*alpha(yp)*(1+h3*(H[3](yp)+np.sqrt(1.5)*yerr**2*H[1](yp))/(sigmap/sigma)**3
            +h4*(H[4](yp)+np.sqrt(3.)*yerr**2*H[2](yp)+np.sqrt(3./8.)*yerr**4*H[0](yp))/(sigmap/sigma)**4)
 
    return f

def cumulative_gauss_hermite_series(v, params):
    '''
        Evaluate the cumulative distribution of the Gauss Hermite series 
        for params = gamma,V,sigma,h3,h4 where gamma is the normalization
        
        see Appendix A of Sellwood & Gerhard (2020) (note typo:
            factor of sigma not necessary in denominator)
        
        Parameters
        ----------
        v : array_like
            Input array of velocities.
        params : array_like
            Parameters of the Gauss-Hermite series, p=(gamma, V, sigma, h3, h4):
                gamma = normalization,
                V = mean velocity,
                sigma = dispersion parameter,
                h3 = 3rd Gauss-Hermite coefficient,
                h4 = 4th Gauss-Hermite coefficient.

        Returns
        -------
        f : array_like
            Cumulative distribution for Gauss-Hermite series int_-\infty^v dv f(v)
        
    '''
    
    gamma,V,sigma,h3,h4 = params
    y = (v-V)/sigma
    f = gamma * (Q[0](y)+h3*(2*Q[3](y)-3*Q[1](y))/np.sqrt(3.)
                    +h4*(4*Q[4](y)-12*Q[2](y)+3*Q[0](y))/np.sqrt(24.))
    
    return f


def variance_kurtosis_gauss_hermite(sigma,h3,h4):
    '''
        Evaluate the variance and excess kurtosis of the Gauss-Hermite series
        with dispersion parameter sigma and GH coefficients h3 & h4
        
        Parameters
        ----------
        sigma : array_like
            Dispersion parameter for Gauss-Hermite series.
        h3 : array_like
            3rd Gauss-Hermite coefficient.
        h4 : array_like
            4th Gauss-Hermite coefficient.

        Returns
        -------
         res : tuple of array_like
             (variance, excess kurtosis) of Gauss-Hermite series
        
    '''
    
    lmbda = 1.+np.sqrt(3./8.)*h4
    variance = 1+(h4*(2*np.sqrt(6.)+3*h4)-3*h3**2)/lmbda**2
    kurtosis = .5/lmbda**4*(16*np.sqrt(6.)*h4-9*h4**2*(8+6*np.sqrt(6.)*h4+5*h4**2)+12*h3**2*(15*h4**2+8*np.sqrt(6.)*h4-8)-108*h3**4)
    
    res = sigma**2*variance, kurtosis/variance**2

    return res

##==============================================================================
##
## Negative kurtosis family of models
## ----------------------------------
## These models are formed from the convolution of a Gaussian with a uniform
## kernel. To introduce skewness, the uniform kernel has a different width/height
## on either side of the axis
##
## K(y) = 1/(2a_+) for 0<y<a_+; 1/(2a_-) for -a_-<y<=0
##
## See Section 4.1 of Sanders & Evans (2020) for more details
##
##==============================================================================

def _uniform_kernel_parameters(h3,h4):
    '''
        Converts the Gauss-Hermite coefficients (h3, h4) for uniform kernel model
        into the corresponding (a,Delta,b,w_0) as outlined in Table 1 of 
        Sanders & Evans (2020)
        
        Parameters
        ----------
        h3 : array_like
            3rd GH coefficient
        h4 : array_like
            4th GH coefficient
            
        Returns
        -------
        (a, delta, b, w0) : tuple of array_like
            Parameters of pdf, width a, skewness Delta (note here capital Delta),
            variance scale b, mean scale w_0.
    '''
    
    if np.any(h4>0):
        warnings.warn("h4<0 passed to _uniform_kernel_parameters "
                      "-- implicitly converting to -|h4|")
        
    h40 = -0.187777
    if np.any(h4<h40):
        warnings.warn("h4<-0.187777 passed to _uniform_kernel_parameters "
                      "-- limiting value of h4 is -0.187777, will return nan")
        
    delta_h3 = 0.82
    delta_h4 = 4.3
    kinf = 1.3999852768764105
    k0=np.sqrt(3.)
    scl_a=2.
    scl=3.3
    
    h4_3 = np.abs(h4/(h3+1e-20))/(-h40)
    delta = np.sign(h3)*(-delta_h3*h4_3+np.sqrt((delta_h3*h4_3)**2+4*delta_h4))/(2*delta_h4)
    a = scl_a/np.sqrt(np.sqrt((1-delta_h4*delta**2)*np.abs(h40/(h4+1e-20)))-1)
    kinf = kinf*np.sqrt(1+delta**2+3*delta**4)
    delta*=a
    b = np.sqrt(1.+a**2/(k0-(k0-kinf)*np.tanh(a/scl))**2)
    w0 = (-(delta/2.)+(delta/3.)*np.tanh(a/scl))/b
    
    return a, delta, b, w0

def uniform_kernel_pdf(x,err,mean,sigma,h3,h4):
    '''
    
        Probability density function for the uniform kernel 
        model from Sanders & Evans (2020)
        
        f_{sigma_e}(x) = f_s(w)/sigma
        
        where w = (x-mean)/sigma, s = sigma_e/sigma
        
        f_s(w) = b/(2a_+a_-)(
            a_+ Phi((bw'+a_-)/t) - a_- Phi((bw'-a_+)/t)
            -2 Delta Phi(bw'/t))
            
        see equation (38) of Sanders & Evans (2020)
            
        Phi(x) is the cumulative of the unit normal.
        
        The parameters of the model (a, delta, b, w_0) are 
        chosen such that h_1~h_2~0 and reproduce the required
        h_3, h_4. See Table 1 of Sanders & Evans (2020). The
        transformations are computed by 
        _uniform_kernel_parameters. These models are only valid
        if h4<0. If h4>0 is passed, the code will use -h4 and give
        a warning.
        
        w' = w-w_0
        t = 1 + b^2 s^2
        a_\pm = a \pm delta
        
        Parameters
        ----------
        x : array_like
            input coordinate (velocity)
        err : array_like
            input coordinate uncertainties
        mean : array_like
            mean velocity
        sigma : array_like
            dispersion parameter (not standard deviation)
        h3 : array_like
            3rd Gauss-Hermite coefficient
        h4 : array_like
            4th Gauss-Hermite coefficient
            
        Returns
        -------
        pdf: array_like
            probability density function
            
    '''
    w = (x-mean)/sigma
    werr = err/sigma
    
    a, delta, b, w0 = _uniform_kernel_parameters(h3, h4)
    t = np.sqrt(1.+b*b*werr*werr)

    am, ap = a-delta, a+delta
    it = 1./(np.sqrt(2.)*t)
    bw = b*(w-w0)
    if type(delta) is not np.ndarray:
        if delta==0:
            pdf = 0.25*b/a*(erf((a-bw)*it)+erf((a+bw)*it))/sigma
            return pdf
    pdf = 0.25*b*(am*erf((ap-bw)*it)+
                   ap*erf((am+bw)*it)
                   -2*delta*erf(bw*it))/(ap*am)/sigma
    
    return pdf
    
def ln_uniform_kernel_pdf(x,err,mean,sigma,h3,h4):
    '''
    
        Natural logarithm of the probability density function
        for the uniform kernel model from Sanders & 
        Evans (2020). Full details are given in 
        uniform_kernel_pdf. This function is optimized for 
        numerical stability to avoid under/overflow (see 
        Appendix E of Sanders & Evans, 2020)
        
        Parameters
        ----------
        x : array_like
            input coordinate (velocity)
        err : array_like
            input coordinate uncertainties
        mean : array_like
            mean velocity
        sigma : array_like
            dispersion parameter (not standard deviation)
        h3 : array_like
            3rd Gauss-Hermite coefficient
        h4 : array_like
            4th Gauss-Hermite coefficient
            
        Returns
        -------
        ln_pdf: array_like
            probability density function
            
    '''
    
    w = (x-mean)/sigma
    werr = err/sigma
    
    a, delta, b, w0 = _uniform_kernel_parameters(h3, h4)
    t = np.sqrt(1.+b*b*werr*werr)

    am, ap = a-delta, a+delta
    it = 1./t
    bw = b*(w-w0)
    
    if type(delta) is not np.ndarray:
        if delta==0.:
            ln_pdf  = np.log(.5*b/a)+np.where((b*w+a)*it<0.,
                           logdiffexp(log_ndtr((bw+a)*it),log_ndtr((bw-a)*it)),
                            logdiffexp(log_ndtr(-(bw-a)*it),log_ndtr(-(bw+a)*it)),
                          )
            ln_pdf -= np.log(sigma)
            
            return ln_pdf

    ln_pdf =  np.log(0.5*b/(ap*am))+np.logaddexp(
                np.log(am)+
                np.where((ap-bw)*it<0.,
                         logdiffexp(log_ndtr((ap-bw)*it),log_ndtr(-bw*it)),
                         logdiffexp(log_ndtr(bw*it),log_ndtr(-(ap-bw)*it))
                        ),
                np.log(ap)+
                np.where((bw+am)*it<0.,
                         logdiffexp(log_ndtr((am+bw)*it),log_ndtr(bw*it))
                         ,logdiffexp(log_ndtr(-bw*it),log_ndtr(-(am+bw)*it))
                        )
                )
    ln_pdf -= np.log(sigma)

    return ln_pdf
    
def uniform_kernel_fourier_transform(u,err,mean,sigma,h3,h4):
    '''
    
        Fourier transform of the probability density function
        for the uniform kernel model from Sanders & 
        Evans (2020). See equation (49) of Sanders & Evans
        (2020) for more information.
        
        Parameters
        ----------
        u : array_like
            frequency coordinate (in units of 1/sigma)
        err : array_like
            input coordinate uncertainties
        mean : array_like
            mean velocity
        sigma : array_like
            dispersion parameter (not standard deviation)
        h3 : array_like
            3rd Gauss-Hermite coefficient
        h4 : array_like
            4th Gauss-Hermite coefficient
            
        Returns
        -------
        fft: array_like
            Fourier transform of probability density function
            
    '''
    werr = err/sigma
    a, delta, b, w0 = _uniform_kernel_parameters(h3, h4)
    t = np.sqrt(1.+b*b*werr*werr)
    
    am, ap = a-delta, a+delta
    ub = u/b
    ## Equation (56)
    fft =  .5*((np.exp(1j*ap*ub)-1)/(1j*ap*ub+1e-100)+2*(u==0)
               -(np.exp(-1j*am*ub)-1)/(1j*am*ub+1e-100))*\
            np.exp(-.5*(t*ub)**2+1j*w0*u)
    
    ## Rotate by model mean
    fft *= np.exp(1j*mean/sigma*u)
    
    return fft

def uniform_kernel_variance_kurtosis(sigma,h3,h4):
    '''
        Evaluate the variance and excess kurtosis of the 
        uniform kernel model from Sanders & Evans (2020). 
        See Table D2 of Sanders & Evans (2020) for more 
        information.
        
        Parameters
        ----------
        sigma : array_like
            Dispersion parameter.
        h3 : array_like
            3rd Gauss-Hermite coefficient.
        h4 : array_like
            4th Gauss-Hermite coefficient.

        Returns
        -------
         res : tuple of array_like
             (variance, excess kurtosis) of uniform kernel 
             model.
        
    '''
        
    a, delta, b, w0 = _uniform_kernel_parameters(h3, h4)
    variance = (1.+a*a/3.+delta**2/12.)/b/b*sigma**2
    kurtosis = -1./120.*(16.*a**4-4*a**2*delta**2+delta**4)/(1.+a*a/3.+delta**2/12.)**2
    
    return variance, kurtosis


##==============================================================================
##
## Positive kurtosis family of models
## ----------------------------------
## These models are formed from the convolution of a Gaussian with a Laplace
## kernel. To introduce skewness, the Laplace kernel has a different width
## on either side of the axis.
##
## K(y) = exp(-y/a_+)/(2a_+) for y>=0; exp(y/a_-) for y<0
##
## See Section 4.2 of Sanders & Evans (2020) for more details
##
##==============================================================================


def _laplace_kernel_parameters(h3,h4):
    '''
        Converts the Gauss-Hermite coefficients (h3, h4) into the corresponding
        (a,Delta,b,w_0) for Laplace kernel model as outlined in Table 1 of 
        Sanders & Evans (2020)
        
        Parameters
        ----------
        h3 : array_like
            3rd GH coefficient
        h4 : array_like
            4th GH coefficient
            
        Returns
        -------
        (a, delta, b, w0) : tuple of array_like
            Parameters of pdf, width a, skewness Delta (note here capital Delta),
            variance scale b, mean scale w_0.
    '''
    
    if np.any(h4<0):
        warnings.warn("h4>0 passed to _laplace_kernel_parameters "
                      "-- implicitly converting to -|h4|")
        
    h40 = 0.145461
    if np.any(h4>h40):
        warnings.warn("h4>0.145461 passed to _laplace_kernel_parameters "
                      "-- limiting value of h4 is 0.145461, will return nan")
        
    delta_h4 = 2.
    delta_h3 = 0.37
    scl=2.25
    scl_a=1.6
    scl_a3=1.1
    k0=1./np.sqrt(2.)
    kinf = 1.0806510105505178
    
    acoeff = delta_h4*h40/(np.abs(h4+1e-10))
    bcoeff = -delta_h3/np.abs(h3+1e-10)*(scl_a/scl_a3)**2
    ccoeff = (h40/np.abs(h4+1e-10)-1+(scl_a/scl_a3)**2)
    delta = np.sign(h3)*(-bcoeff-np.sqrt(bcoeff**2-4*acoeff*ccoeff))/(2*acoeff)
    a = scl_a/np.sqrt(h40*(1+delta_h4*delta**2)/np.abs(h4+1e-10)-1)
    
    kinf = kinf*np.sqrt(1+3*delta**2)
    b = np.sqrt(1.+a**2/(k0-(k0-kinf)*np.tanh(a/scl))**2)
    delta*=a
    w0 = (-delta+(8.*delta/7.)*np.tanh(5.*a/scl/4.))/b
    
    return a, delta, b, w0


def laplace_kernel_pdf(x,err,mean,sigma,h3,h4):
    '''
    
        Probability density function for the Laplace kernel 
        model from Sanders & Evans (2020)
        
        f_{sigma_e}(x) = f_s(w)/sigma
        
        where w = (x-mean)/sigma, s = sigma_e/sigma
        
        f_s(w) = b/(4a_+)exp((t^2-2a_+bw')/(2a_+^2))erfc((t^2-a_+bw')/(\sqrt{2}ta_+))
                 +b/(4a_-)exp((t^2+2a_-bw')/(2a_-^2))erfc((t^2+a_-bw')/(\sqrt{2}ta_-))
            
        see equation (41) of Sanders & Evans (2020)

        
        The parameters of the model (a, delta, b, w_0) are 
        chosen such that h_1~h_2~0 and reproduce the required
        h_3, h_4. See Table 1 of Sanders & Evans (2020). The
        transformations are computed by 
        _laplace_kernel_parameters. These models are only valid
        if h4>0. If h4<0 is passed, the code will use |h4| and give
        a warning.
        
        w' = w-w_0
        t = 1 + b^2 s^2
        a_\pm = a \pm delta
        
        Parameters
        ----------
        x : array_like
            input coordinate (velocity)
        err : array_like
            input coordinate uncertainties
        mean : array_like
            mean velocity
        sigma : array_like
            dispersion parameter (not standard deviation)
        h3 : array_like
            3rd Gauss-Hermite coefficient
        h4 : array_like
            4th Gauss-Hermite coefficient
            
        Returns
        -------
        pdf: array_like
            probability density function
            
    '''
    
    w = (x-mean)/sigma
    werr = err/sigma
    
    a, delta, b, mean_w = _laplace_kernel_parameters(h3, h4)
    t = np.sqrt(1.+b*b*werr*werr)
    ap = a+delta
    am = a-delta
    
    argU = (t*t-2*ap*b*(w-mean_w))
    positive_term = np.zeros_like(x)
    prefactor = (b/(4.*ap))
    if type(h4) is np.ndarray:
        prefactor = prefactor[argU<0.]
    positive_term[argU<0.] = prefactor*np.exp((argU/2./ap**2)[argU<0.])*\
                                erfc(((t*t-ap*b*(w-mean_w))/np.sqrt(2)/t/ap)[argU<0.])
    prefactor = (b/ap)
    if type(h4) is np.ndarray:
        prefactor = prefactor[argU>0.]
    positive_term[argU>0.]=np.sqrt(np.pi/8.)*prefactor*alpha((b*(w-mean_w)/t)[argU>0.])*\
                            erfcx(((t*t-ap*b*(w-mean_w))/np.sqrt(2)/t/ap)[argU>0.])
    
    argU = (t*t+2*am*b*(w-mean_w))
    negative_term = np.zeros_like(x)
    prefactor = (b/(4.*am))
    if type(h4) is np.ndarray:
        prefactor = prefactor[argU<0.]
    negative_term[argU<0.] = prefactor*np.exp((argU/2./am**2)[argU<0.])*\
                                erfc(((t*t+am*b*(w-mean_w))/np.sqrt(2)/t/am)[argU<0.])
    prefactor = (b/am)
    if type(h4) is np.ndarray:
        prefactor = prefactor[argU>0.]
    negative_term[argU>0.]=np.sqrt(np.pi/8.)*prefactor*alpha((b*(w-mean_w)/t)[argU>0.])*\
                            erfcx(((t*t+am*b*(w-mean_w))/np.sqrt(2)/t/am)[argU>0.])
    
    pdf = (positive_term + negative_term)/sigma
    
    return pdf

def ln_laplace_kernel_pdf(x,err,mean,sigma,h3,h4):
    '''
    
        Natural logarithm of the probability density function
        for the Laplace kernel model from Sanders & 
        Evans (2020). Full details are given in 
        laplace_kernel_pdf. This function is optimized for 
        numerical stability to avoid under/overflow (see 
        Appendix E of Sanders & Evans, 2020)
        
        Parameters
        ----------
        x : array_like
            input coordinate (velocity)
        err : array_like
            input coordinate uncertainties
        mean : array_like
            mean velocity
        sigma : array_like
            dispersion parameter (not standard deviation)
        h3 : array_like
            3rd Gauss-Hermite coefficient
        h4 : array_like
            4th Gauss-Hermite coefficient
            
        Returns
        -------
        ln_pdf: array_like
            probability density function
            
    '''
    w = (x-mean)/sigma
    werr = err/sigma
    a, delta, b, mean_w = _laplace_kernel_parameters(h3, h4)
    t = np.sqrt(1.+b*b*werr*werr)
    
    ap = a+delta
    am = a-delta
    
    argU = (t*t-2*ap*b*(w-mean_w))
    positive_term = np.zeros_like(x)
    
    prefactor = np.log(b/(4.*ap))
    if type(h4) is np.ndarray:
        prefactor = prefactor[argU<0.]
    positive_term[argU<0.] = prefactor+(argU/2./ap**2)[argU<0.]+\
                                lnerfc(((t*t-ap*b*(w-mean_w))/np.sqrt(2)/t/ap)[argU<0.])
    
    prefactor = np.log(b/ap)
    if type(h4) is np.ndarray:
        prefactor = prefactor[argU>0.]
    positive_term[argU>0.]=.5*np.log(np.pi/8.)+prefactor+lnalpha((b*(w-mean_w)/t)[argU>0.])+\
                            lnerfcx(((t*t-ap*b*(w-mean_w))/np.sqrt(2)/t/ap)[argU>0.])
    
    argU = (t*t+2*am*b*(w-mean_w))
    negative_term = np.zeros_like(x)
    
    prefactor = np.log(b/(4.*am))
    if type(h4) is np.ndarray:
        prefactor = prefactor[argU<0.]
    negative_term[argU<0.] = prefactor+(argU/2./am**2)[argU<0.]+\
                                lnerfc(((t*t+am*b*(w-mean_w))/np.sqrt(2)/t/am)[argU<0.])
    prefactor = np.log(b/am)
    if type(h4) is np.ndarray:
        prefactor = prefactor[argU>0.]
    negative_term[argU>0.]=.5*np.log(np.pi/8.)+prefactor+lnalpha((b*(w-mean_w)/t)[argU>0.])+\
                            lnerfcx(((t*t+am*b*(w-mean_w))/np.sqrt(2)/t/am)[argU>0.])
    
    ln_pdf = np.logaddexp(positive_term,negative_term)-np.log(sigma)

    return ln_pdf

def laplace_kernel_fourier_transform(u,err,mean,sigma,h3, h4):
    '''
    
        Fourier transform of the probability density function
        for the Laplace kernel model from Sanders & 
        Evans (2020). See equation (48) of Sanders & Evans
        (2020) for more information.
        
        Parameters
        ----------
        u : array_like
            frequency coordinate (in units of 1/sigma)
        err : array_like
            input coordinate uncertainties
        mean : array_like
            mean velocity
        sigma : array_like
            dispersion parameter (not standard deviation)
        h3 : array_like
            3rd Gauss-Hermite coefficient
        h4 : array_like
            4th Gauss-Hermite coefficient
            
        Returns
        -------
        fft: array_like
            Fourier transform of probability density function
            
    '''
    werr = err/sigma
    a, delta, b, w0 = _laplace_kernel_parameters(h3, h4)
    t = np.sqrt(1.+b*b*werr*werr)
    
    am, ap = a-delta, a+delta
    ub = u/b
    # Equation (55)
    fft = .5*(1./(1-1j*ap*ub)+1./(1+1j*am*ub))*np.exp(-.5*(t*ub)**2+1j*w0*u)

    # Rotate by model mean
    fft *= np.exp(1j*mean/sigma*u)
    
    return fft 


def laplace_kernel_variance_kurtosis(sigma,h3,h4):
    '''
        Evaluate the variance and excess kurtosis of the 
        Laplace kernel model from Sanders & Evans (2020). 
        See Table D2 of Sanders & Evans (2020) for more 
        information.
        
        Parameters
        ----------
        sigma : array_like
            Dispersion parameter.
        h3 : array_like
            3rd Gauss-Hermite coefficient.
        h4 : array_like
            4th Gauss-Hermite coefficient.

        Returns
        -------
         res : tuple of array_like
             (variance, excess kurtosis) of Laplace kernel 
             model.
        
    '''
        
    a, delta, b, w0 = _laplace_kernel_parameters(h3, h4)
    variance = (1.+a*a*2+delta**2)/b/b*sigma**2
    kurtosis = 6*(2*a**4+12*a**2*delta**2+delta**4)/(1.+a*a*2+delta**2)**2
    res = variance, kurtosis
    
    return res


##==============================================================================
## Alternative for ppxf's function losvd_rfft
## 
## Computes the fft of the losvd
##==============================================================================

def ppxf_losvd_rfft_new_family(pars, nspec, moments, nl, ncomp, vsyst, factor,
                               sigma_diff):
    '''
        Alternative for ppxf's losvd_rfft using the Laplace and uniform kernels
        
        def losvd_rfft(pars, nspec, moments, nl, ncomp, vsyst, factor, sigma_diff):
            """
            Analytic Fourier Transform (of real input) of the Gauss-Hermite LOSVD.
            Equation (38) of `Cappellari (2017)
            <https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..798C>`_

            """
    '''
    
    losvd_rfft = np.empty((nl, ncomp, nspec), dtype=complex)
    p = 0
    for j, mom in enumerate(moments):  # loop over kinematic components
        for k in range(nspec):  # nspec=2 for two-sided fitting, otherwise nspec=1
            s = 1 if k == 0 else -1  # s=+1 for left spectrum, s=-1 for right one
            vel, sig = vsyst + s*pars[0 + p], pars[1 + p]
            a, b = [vel, sigma_diff]/sig
            w = np.linspace(0, np.pi*factor*sig, nl)
            
            losvd_rfft[:, j, k] = np.exp(1j*a*w - 0.5*(1 + b**2)*w**2)

            if mom > 2:
                if pars[3 + p]>0.:
                    losvd_rfft[:, j, k] = laplace_kernel_fourier_transform(
                                            w, sigma_diff, vel, sig, 
                                            pars[2+p], pars[3+p])
                else:
                    losvd_rfft[:, j, k] = uniform_kernel_fourier_transform(
                                            w, sigma_diff, vel, sig, 
                                            pars[2+p], pars[3+p])
        p += mom

    return np.conj(losvd_rfft)

##==============================================================================
