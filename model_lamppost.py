import numpy as np 
from astropy import constants as const


def make_temp_profile(r, Mbh=1e8, Mdot=1, albedo=0, r_in = 3, hx = 6):
    
    '''
    make temperature profile for a viscous + irradiated disk

    input - 
    r: radius to evaluate temperature for, in cgs
    Mbh: black hole mass, in solar mass
    Mdot: accretion rate, in solar mass/yr
    a: albedo of accretion disk, unitless
    incl: inclination of the disk
    r_in: radius of ISCO (set at 3 r_s)
    hx: height of x-ray corona (set at 6 r_s)

    output - 
    total temperature profile 
    viscous disk temperature profile 
    irradiated disk temperature profile 
    '''
    
    ## const, unit conversion to cgs
    G = const.G.value*1e3
    sigma_sb = const.sigma_sb.value*1e3
    c = const.c.value*1e2
    Mbh_cgs = Mbh*const.M_sun.value*1e3
    Mdot_cgs = Mdot*(const.M_sun.value*1e3/365/86400)
    r_s = 2*G*Mbh_cgs/c**2 ## Schwarzschild radius
    
    ## fixed parameters
    eta = 0.1
    r_in = r_in*r_s
    hx = hx*r_s
    
    ## calculations
    Lb = eta*Mdot_cgs*c**2
    x = np.sqrt(r**2+hx**2)
    T4 = 3*G*Mbh_cgs*Mdot_cgs/8/np.pi/sigma_sb/r**3*(1-np.sqrt(r_in/r)) + \
            Lb*(1-albedo)*hx/4/np.pi/sigma_sb/x**3
    T4_vis = 3*G*Mbh_cgs*Mdot_cgs/8/np.pi/sigma_sb/r**3*(1-np.sqrt(r_in/r))
    T4_lamp = Lb*(1-albedo)*hx/4/np.pi/sigma_sb/x**3
    
    return T4**(1/4), T4_vis**(1/4), T4_lamp**(1/4);


def make_powerlaw_temp_profile(r, alpha=3/4, Mbh=1e8, Mdot=1, albedo=0, \
                                     incl=0, hx = 6, r_in=3):
                                         
    G = const.G.value*1e3
    sigma_sb = const.sigma_sb.value*1e3
    c = const.c.value*1e2
    Mbh_cgs = Mbh*const.M_sun.value*1e3
    Mdot_cgs = Mdot*(const.M_sun.value*1e3/86400/365)
    
    r_s = 2*G*Mbh_cgs/c**2
    hx = hx*r_s
    eta = 0.1
    Lb = eta*Mdot_cgs*c**2
    
    r1 = 1*86400*c ## scaled at 1 light day
    if r_in*r_s > r1:         
        r0_in = np.ceil(r_in*r_s/r1)*r1
    else:
        r0_in = r1  
        
    Tv_4 = 3*G*Mbh_cgs*Mdot_cgs/8/np.pi/sigma_sb/r1**3*(1-np.sqrt(r_in/r0_in))*(r1/r)**(alpha*4)     
    Ti_4 = hx*(1-albedo)*Lb/4/np.pi/sigma_sb/np.sqrt(hx**2+r1**2)**3*(r1/r)**(alpha*4)     
    
    return (Tv_4+Ti_4)**(1/4);



def make_response_func(tau_max=30, incl=0, Mbh=1e8, Mdot=1, wavelength=4000, albedo=0.,r_in = 3, hx = 6):

    '''
    calculate the response function

    input -
    tau_max: the max tau value to evaluate 
    Mbh: black hole mass, in solar mass
    Mdot: accretion rate, in solar mass/yr
    a: albedo of accretion disk, unitless
    incl: inclination of the disk 
    wavelength: observed wavelength

    output -
    tau: time delay grid, in days
    tf: normalized response function 
    '''
    
    tau = np.arange(0,tau_max+0.1,0.1)
    tau = tau*86400
    
    ## const, unit conversion to cgs
    G = const.G.value*1e3
    sigma_sb = const.sigma_sb.value*1e3
    c = const.c.value*1e2
    Mbh_cgs = Mbh*const.M_sun.value*1e3
    Mdot_cgs = Mdot*(const.M_sun.value*1e3/86400/365)
    r_s = 2*G*Mbh_cgs/c**2 ## Schwarzschild radius
    wavelength_cgs = wavelength*1e-8
    h = const.h.value*1e7
    k = const.k_B.value*1e7
    
    ## some fixed parameters
    r_in = np.max([np.min(tau)*c,3*r_s]) ## start beyond ISCO
    r_out = 30*86400*c ## 30 light days
    hx = hx*r_s

    ## start calculation
    r_grid = np.logspace(np.log10(r_in),np.log10(2*r_out),num=500) ## radius grid in cm
    phi_grid = np.linspace(0,np.pi*2,num=len(r_grid)) ## azimuthal grid
    T_r, T_vis, T_lamp = make_temp_profile(r_grid, Mbh=Mbh, Mdot=Mdot, albedo=albedo, r_in = r_in/r_s, hx = hx/r_s)

    ## make a 2D array of psi(r,phi) and calculate the response function
    rr = (r_grid[1:]+r_grid[:-1])/2.
    dr = np.diff(r_grid)
    dphi = np.diff(phi_grid)
    temp =  np.interp(rr,r_grid,T_r)
    rf_radial = 2*h**2*c**3/k/wavelength_cgs**6/temp**2* \
            np.exp(h*c/wavelength_cgs/k/temp)/(np.exp(h*c/wavelength_cgs/k/temp)-1)**2* \
            ((1-albedo)*hx/4/np.pi/sigma_sb/(hx**2+rr**2)**(3/2))*(temp**(-3))/4*rr*dr*dphi*np.cos(incl)

    rf = np.repeat(rf_radial.reshape(len(rf_radial),1),len(phi_grid)-1,axis=1)
    rf[np.where(np.isnan(rf))] = 0.

    ## time lag as function of r, phi
    ## use a gaussian with certain width (e.g.,0.05d) to evaluate tf    
    rr, phiphi = np.meshgrid((r_grid[1:]+r_grid[:-1])/2,(phi_grid[1:]+phi_grid[:-1])/2)
    tt = rr/c*(1-np.cos(phiphi)*np.sin(incl))
    tt = tt.T.flatten()
    rf = rf.flatten()
    
    sigma = 0.1*86400 ##0.1days
    tf = np.zeros(len(tau))
    for itau in range(len(tau)):
        delta = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*((tt-tau[itau])/sigma)**2)   
        tf[itau] = np.sum(rf*delta)
    
    tf = tf/np.trapz(tf,x=tau)
    
    return tau/86400, tf;
    
