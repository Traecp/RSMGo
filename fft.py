import numpy as np
from scipy.fftpack import fft, fftfreq, fftshift, fft2
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import peak
from scipy import stats
from lmfit import Parameters, minimize
from numba import jit
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter

#mpl.rcParams['font.size'] = 18.0
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.handletextpad'] = 0.5
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['figure.subplot.bottom'] = 0.13
mpl.rcParams['figure.subplot.top'] = 0.93
mpl.rcParams['figure.subplot.left'] = 0.14
mpl.rcParams['figure.subplot.right'] = 0.915
mpl.rcParams['image.cmap'] = "jet"
mpl.rcParams['savefig.dpi'] = 300

pi = np.pi 

@jit
def trapezoid_func(x,y,W,H,beta):
    dim1, dim2 = x.shape
    shape = np.ones(shape=x.shape)
    for i in range(dim1):
        for j in range(dim2):
            tmp1 = x[i,j]/np.tan(np.radians(beta))
            tmp2 = (W-x[i,j])/np.tan(np.radians(beta))
            if (y[i,j]<H) and ((y[i,j]>=tmp1) or (y[i,j]>=tmp2)):
                shape[i,j] = 0
    return shape
    
def Lorentzian(x0,w,x):
	""" normalised area lorentzian (integral = 1)"""
	return (1./2.0/np.pi)*(w/((x-x0)**2+w**2/4.0))
	
def gauss(x0,w,x):
	""" normalised area gaussian"""
	return np.sqrt(4.*np.log(2)/np.pi)*(1./w)*np.exp(-((4.*np.log(2))/(w**2))*(x-x0)**2)
    
def gauss2D(x0,y0,wx, wy, x,y):
    sgx = wx/(2*np.sqrt(2*np.log(2)))
    sgy = wy/(2*np.sqrt(2*np.log(2)))
    return 1/(2*pi*sgx*sgy)* np.exp(-((x-x0)**2/(2*sgx**2) + (y-y0)**2/(2*sgy**2)))
	
def pseudo_Voigt(x0,w,mu,x):
	return mu*gauss(x0,w,x) + (1-mu)*Lorentzian(x0,w,x)

def linear_model(params, x):
    a = params["slope"].value
    b = params["intercept"].value
    return a*x+b 
def obj_linear(pars, x, y):
    err = y - linear_model(pars, x)
    return err
def linear_fit(x,y):
    pars = Parameters()
    pars.add("slope", value=1)
    pars.add("intercept", value=1)
    result = minimize(obj_linear, pars, args=(x,y))
    mod = linear_model(result.params, x)
    return result.params, mod
    
def objective(pars,y,x):
	err =  y - model(pars,x)
	return err
def model(params, x):
    BG = params["BG"].value
    n  = len(params.keys()) -1
    n  = int(n/4)#Number of peaks
    A_variables = []
    X_variables = []
    W_variables = []
    MU_variables = []
    for i in range(n-1):
        A_variables.append("A%d"%i)
        X_variables.append("X%d"%i)
        W_variables.append("W%d"%i)
        MU_variables.append("MU%d"%i)
    A_variables.append("Amp_X")
    X_variables.append("Cen_X")
    W_variables.append("Wid_X")
    MU_variables.append("Mu_X")

    m  = BG
    for i in range(n):
        ampl = params[A_variables[i]].value
        center = params[X_variables[i]].value
        width  = params[W_variables[i]].value
        mu     = params[MU_variables[i]].value
        # m = m + ampl * Lorentzian(center, width, x)
        m = m + ampl * pseudo_Voigt(center, width, mu, x)
    return m
    
def pars_init(x_list, y_list):
    """ Initialize parameters for peaks 
    x_list: list of x peaks 
    y_list: list of y peaks
    returns: params 
    """
    params = Parameters()
    BG     = 1e4
    params.add("BG", value = BG)
    n      = len(x_list)
    A_variables = []
    X_variables = []
    W_variables = []
    MU_variables = []
    for i in range(n):
        A_variables.append("A%d"%i)
        X_variables.append("X%d"%i)
        W_variables.append("W%d"%i)
        MU_variables.append("MU%d"%i)
    W  = np.ones(n)*0.003
    MU = np.ones(n)*0.5
    for i in range(n):
        params.add(X_variables[i], value = x_list[i], min = x_list[i]-0.001, max = x_list[i]+0.001)
        params.add(A_variables[i], value = y_list[i], min = 0)
        params.add(W_variables[i], value = W[i], min=0.0001)
        params.add(MU_variables[i], value = MU[i], min=0, max=1)
        
    # For a central peak at zero:
    params.add("Cen_X", value=0, min=-0.001,max=0.001)
    params.add("Amp_X", value=1e5)
    params.add("Wid_X", value=0.01,min=0.0001)
    params.add("Mu_X", value=0.5,min=0,max=1)
    print ("number of params: %d"%len(params.keys()))
    return params
		
def fit(x,y,threshold):
	ind     = peak.indexes(y, thres=threshold, min_dist=20)
	x_peaks = peak.interpolate(x,y,ind, width=8, func = peak.gaussian_fit)
	new_ind = peak.get_index_from_values(x, x_peaks)
	y_peaks = y[new_ind]
	param_init = pars_init(x_peaks, y_peaks)
	result = minimize(objective, param_init, args=(y,x))
	y = model(result.params, x)
	return result.params, y
    
def get_maxima(x,y,threshold=0.05, min_dist=10, window=10):
    y_smooth = savgol_filter(y, 7, 2)
    ind     = peak.indexes(y_smooth, thres=threshold, min_dist=min_dist)
    x_peaks = peak.interpolate(x,y,ind, width=window, func = peak.lorentzian_fit)#lorentzian_fit
    new_ind = peak.get_index_from_values(x, x_peaks)
    y_peaks = y[new_ind]
    return (x_peaks, y_peaks)
    
def get_maxima_2(x,y,threshold=0.05, min_dist=10, window=10):
    y_smooth = savgol_filter(y, 7, 2)
    ind     = peak.indexes(y_smooth, thres=threshold, min_dist=min_dist)
    x_peaks = x[ind]
    y_peaks = y[ind]
    return (x_peaks, y_peaks)
    
def linear_regression(x,y,threshold=0.04, min_dist=10, window=10, Q = "Qx"):
    ix,iy=get_maxima(x,y,threshold=threshold, min_dist=min_dist, window=window)
    order = np.arange(len(ix))
    slope, intercept, r_value, p_value, std_err = stats.linregress(ix, order)
    fit_line = slope*ix + intercept
    fig,ax = plt.subplots(1,2, figsize=(10,6))
    ax[0].plot(x,y, color="orange", lw=3)
    ax[0].scatter(ix, iy, color="g")
    ax[1].plot(ix, order, "ko")
    ax[1].plot(ix, fit_line, "r-")
    ax[1].set_xlabel("Peak position ($\AA^{-1}$)")
    ax[1].set_ylabel("Peak order")
    ax[1].set_title("Slope = %.2f +/- %.2f $\AA$"%(slope*2*np.pi, std_err))
    ax[0].set_xlabel("%s ($\AA^{-1}$)"%Q )
    ax[0].set_ylabel("Intensity (a.u.)")
    ax[0].set_yscale("log")
    plt.show()
    
def plot_period(x,y,threshold=0.04, min_dist=10, window=10, Q = "Qx"):
    ix,iy=get_maxima(x,y,threshold=threshold, min_dist=min_dist, window=window)
    order = np.arange(len(ix))
    pars, fit_line = linear_fit(ix, order)
    slope = pars["slope"].value
    std_err = pars["slope"].stderr
    
    fig,ax = plt.subplots(1,2, figsize=(10,6))
    ax[0].plot(x,y, color="orange", lw=2)
    ax[0].plot(ix, iy, "go")
    ax[1].plot(ix, order, "ko")
    ax[1].plot(ix, fit_line, "r-")
    ax[1].set_xlabel("Peak position ($\AA^{-1}$)")
    ax[1].set_ylabel("Peak order")
    ax[1].set_title("Slope = %.2f $\pm$ %.2f $\AA$"%(slope*2*np.pi, std_err*2*np.pi))
    ax[0].set_xlabel("%s ($\AA^{-1}$)"%Q )
    ax[0].set_ylabel("Intensity (a.u.)")
    ax[0].set_yscale("log")
    fig.subplots_adjust(left=0.15, top = 0.9, right=0.95, bottom=0.2, wspace=0.3, hspace=0.2)
    plt.show()
    
def transform_Fourier(x, y):
    """ FFT of data with x: momentum transfer in Angstrom^-1 or nm^-1, y is the intensity"""
    npt = x.size
    sr  = x[1] - x[0] #Sampling rate
    sr  = sr/2/np.pi
    xf  = fftfreq(npt, sr)
    xf  = fftshift(xf)
    tf  = fft(y)
    tf  = fftshift(tf)
    tf  = tf/tf.max()
    half= int(npt/2)
    out_x = xf[half:]
    out_y = np.abs(tf)
    out_y = out_y[half:]
    return (out_x, out_y)
    
def plotmap(fn,vmin,vmax):
    Qx,Qz,data = getmap(fn)
    fig = plt.figure(figsize=(8,10))
    ax  = fig.add_subplot(111)
    # cax = fig.add_axes([0.9,0.2,0.1,0.8])
    levels=np.linspace(vmin, vmax, 50)
    img=ax.contourf(Qx, Qz, np.log10(data), levels, vmin=vmin, vmax=vmax)
    # cb = plt.colorbar(img, cax=cax)
    plt.show()
    
def plotit(fn):
    qx, qz, data = getmap(fn)
    fig,ax = plt.subplots(1,1)
    ax.contourf(qx,qz,np.log10(data), 50)
    ax.axvline(0)
    ax.axhline(0)
    plt.show()
    
def getmap(fn):
    f5 = h5py.File(fn, "r")
    data = f5["/data"].value
    Q    = f5["/Q"].value 
    Qx   = Q[:,:,0]
    Qz   = Q[:,:,2]
    f5.close()
    return Qx, Qz, data
if __name__== "__main__":

    fn = "Reconstructed/ALWR5_Qxz_nosym_hkl.h5"
    Qx, Qz, data= getmap(fn)
    xcut = data[1463:1703].sum(axis=0)/(1703-1463)
    q    = np.linspace(Qx.min(), Qx.max(), xcut.size)
    cut_ind = np.logical_and(q>0.0018, q<0.0296)
    cut = xcut[cut_ind]
    x0  = q[cut_ind]
    ncut= np.log10(cut+1e-6)
    plot_period(x0,cut,threshold=0.004, min_dist=15, window=5, Q = "Qx")
    out_xcut = np.vstack([q,xcut])
    header = "Xcut from row 1463:1703\nQx \t Intensity"
    np.savetxt("ALWR5_Qx_projection_nosym.dat", out_xcut.T, header=str(header))
    # ax[0].plot(qz, i, "ko")
    # ax[0].plot(qz, f, "b-", lw=3, label="Fit - Third order")
    # p = argrelextrema(f, np.less)
    # print("Third order minima: ", qz[p]) #Find minima of peaks
    # ax[0].plot(qz[p], f[p], "ro")
    