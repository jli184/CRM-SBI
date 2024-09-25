import numpy as np 
from astropy import constants as const
from astroML import time_series
from model_lamppost import make_temp_profile, make_response_func

def DRW_params(logMbh, logMdot, z = 0., wavelength = 4000.):
    
    '''
    
    calculate DRW parameters from logMbh, logMdot, and redshift
    
    '''
    
    #Mdot -> Lbol -> L5100 -> Mi following Richards+2006
    eta = 0.1
    Lbol = eta*(10**logMdot)*2e33/(365*60*60*24)*(const.c.value*1e3)**2 ## in erg/s
    lum5100 = 0.1*Lbol ## erg/s
    lum2500 = (lum5100/5100e-10)*(2500/5100)**0.5*(2500e-10) ## erg/s
    Mi_z2 = np.log10(lum2500/(const.c.value*1e3/2500e-10)/4/np.pi/(3.08e19)**2)/(-0.4)-48.6-2.5*np.log10(1+2)
    Mi = Mi_z2+0.596
    
    ## from MacLeod+2010 Equation 7 / Table 1
    params_SF = [-0.57, -0.479, 0.117, 0.11, 0.07]
    params_tau = [2.4, 0.17, -0.05, 0.12, -0.7]
    
    SF_inf = 10**(params_SF[0]+params_SF[1]*np.log10(wavelength/4000.)+\
                  params_SF[2]*(Mi+23)+params_SF[3]*np.log10(10**logMbh/1e9)+params_SF[4]*np.log10(1+z))
    tau_drw = 10**(params_tau[0]+params_tau[1]*np.log10(wavelength/4000.)+\
                  params_tau[2]*(Mi+23)+params_tau[3]*np.log10(10**logMbh/1e9)+params_tau[4]*np.log10(1+z))
    
    return SF_inf, tau_drw ; 



def make_delayLC(lc_time, lc_flux, tf_time, tf):

    '''
    
    make delayed light curve given response function
    recommend to simulate longer light curves and crop later 
    to avoid incomplete response to the responce function

    input:
    lc_time, lc_flux: original light curve
    tf_time, tf: response function

    output:
    lc_delay: flux of the delayed light curve (time grid is the same as input)
                at t<tau_max (max of the response function), lc_delay=0

    '''
    
    tau_max = np.max(tf_time)
    lc_delay = np.zeros(len(lc_time))
    
    for it in range(len(lc_time)):
        if lc_time[it] < tau_max:
            lc_delay[it] = 0
        else:
            ind = np.where(( (lc_time>=(lc_time[it]-tau_max)) & (lc_time<=lc_time[it]) ))
            t_interp = np.abs(lc_time[ind] - lc_time[it]) 
            tf_interp = np.interp(t_interp, tf_time, tf)
            tf_interp = tf_interp / np.sum(tf_interp)
            lc_delay[it] = np.sum(tf_interp * lc_flux[ind])
    
    return lc_delay

## simulator for multiband light curves
def make_multiband_lc(params=[1.,8.,0.]):

    '''
    simulates multiband light curves (6 LSST filters) with no gap and no flux uncertainties

    input - three parameters: inclination, logMbh, logMdot

    output - light curves in tensor, size = n x 7 : time + light curves in 6 bands 

    '''
    
    ## initialize the random seed
    np.random.seed()
    
    cosinc, logMbh, logMdot = np.asarray(params[:3])
    incl = np.arccos(cosinc) ## in radian
    Mbh = 10**logMbh
    Mdot = 10**logMdot
    
    baseline = 180 ## baseline of 180 day
    wavelengths = [3751, 4740, 6172, 7500, 8678, 9711] ## LSST filters: http://svo2.cab.inta-csic.es/theory/fps/
    lc_time = np.arange(0, 3*baseline, 1) ## simulate a longer lightcurve and crop later

    ## make DRW light curves
    ## use u-band DRW parameters for the "driving light curve"
    sf_inf, tau_drw = DRW_params(logMbh, logMdot, z=0, wavelength=wavelengths[0])
    lc_flux = time_series.generate_damped_RW(lc_time, tau=tau_drw, z=0, SFinf = sf_inf)
    
    ## make power-law light curves
    #lc_flux = time_series.generate_power_law(len(lc_time),1,3)

    tau, tf = make_response_func(incl=incl, Mbh=Mbh, Mdot=Mdot, wavelength=wavelengths[0])

    ## randomly choose a starting point avoid any artifacts on the edges
    #print(np.arange(np.max(tau),len(lc_time)-baseline))
    ind = int(np.random.choice(np.arange(np.max(tau),len(lc_time)-baseline),size=1)[0])

    lc_true_out = np.zeros((baseline,len(wavelengths)+1))
    lc_err_out = np.zeros((baseline,len(wavelengths)+1))
    for i in range(len(wavelengths)):
        tau, tf = make_response_func(incl=incl, Mbh=Mbh, Mdot=Mdot, wavelength=wavelengths[i])
        lc_delay = make_delayLC(lc_time, lc_flux, tau, tf)
        lc_delay_err = np.random.normal(loc=lc_delay,scale=0.01*np.abs(lc_delay)) ## add 1% uncertainty
        lc_true_out[:,i+1] = lc_delay[int(ind):int(ind+baseline)]     
        lc_err_out[:,i+1] = lc_delay_err[int(ind):int(ind+baseline)]  
        
    lc_time = lc_time[int(ind):int(ind+baseline)]
    lc_true_out[:,0] = lc_time-lc_time[0]     
    lc_err_out[:,0] = lc_time-lc_time[0]     
    
    return lc_true_out.flatten(), lc_err_out.flatten();
