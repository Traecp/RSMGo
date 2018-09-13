import pyFAI, fabio, threading
from os.path import join, isfile, splitext
from queue import PriorityQueue
from pyFAI.multi_geometry import MultiGeometry
import numpy as np
from matplotlib.pylab import figure, plot, show, gca
from Reconstruction import RunInit
import peak
import os, sys, time
import multiprocessing

_NUM_THREADS = multiprocessing.cpu_count()*3

def read_configuration_file(cfgfn):
	"""
	cfgfn is the filename of the configuration file.
	the function return an object containing information from configuration file (cf inside cfg file).
	"""
	
	try:
		s=open(cfgfn,"r")
	except:
		print( " Error reading configuration file " ,  cfgfn	)		
		exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
		print ("*** print_exception:")
		traceback.print_exception(exceptionType, exceptionValue, exceptionTraceback,
                              limit=None, file=sys.stdout)
		raise Exception
	class Parameters():
		exec(s.read())
	cfg = Parameters()
	s.close()
	return cfg
    
class Worker(threading.Thread):
    def __init__(self, filePath, dataQueue):
        threading.Thread.__init__(self)
        self.dataQueue = dataQueue
        self.filePath  = filePath
    def run(self):
        if isfile(self.filePath):
            self.dataQueue.put(self.filePath)
            self.dataQueue.task_done()
            
def get_img_dataset(img_folder, img_prefix, img_extension, img_digit, beg, end):
    q = PriorityQueue()
    num_threads = _NUM_THREADS
    for j in range(beg, end+1, num_threads):
        threads = []
        for i in range(j, j+num_threads):
            if i<=end:
                img_filename = "{img_prefix}{img_num:0{img_digit}}.{ext}".format(img_prefix=img_prefix, img_num=i, img_digit=img_digit, ext=img_extension)
                fn = join(img_folder, img_filename)
                thr = Worker(fn, q)
                thr.start()
                threads.append(thr)
        for thr in threads:
            thr.join()
    fileList = []
    while not q.empty():
        data = q.get()
        fileList.append(data)
    q.join()
    return sorted(fileList)

   
class Integrator(threading.Thread):
    def __init__(self, mg, p, img_data, mask, bin_pts):
        threading.Thread.__init__(self)
        self.mg = mg
        self.phi_point = p
        self.bin_pts = bin_pts
        self.img_data = img_data
        self.mask = mask
    def run(self):
        self.tth, self.I = self.mg.integrate1d(self.img_data*self.mask, npt=self.bin_pts)
        print("\nIntegrated done for phi point = %d"%self.phi_point)
        
def get_maxima(x,y,threshold=0.3):
    ind     = peak.indexes(y, thres=threshold, min_dist=20)
    x_peaks = peak.interpolate(x,y,ind, width=20, func = peak.lorentzian_fit)#lorentzian_fit
    new_ind = peak.get_index_from_values(x, x_peaks)
    y_peaks = y[new_ind]
    return (x_peaks, y_peaks)
    
def run_integration(configFile, peak_detection_ratio=0.1):
    Run = RunInit(configFile)
    params = Run.params
    mask = fabio.open(params.maskFile).data
    number_of_run = Run.number_of_run
    image_extension = params.image_extension
    img_folder = params.img_folder
    img_prefix = params.img_prefix
    img_digit  = int(params.img_digit)
    poni_files = params.poni_files
    img_begin = np.asarray(params.img_begin)
    img_end   = np.asarray(params.img_end)
    
    Qlength = [np.sqrt((Run.all_Q0[r]**2).sum(axis=2)) for r in range(number_of_run)]
    m=np.array([q.min() for q in Qlength])
    M=np.array([q.max() for q in Qlength])
    Qmin = m.min()
    Qmax = M.max()
    print( "Q min = %f"%Qmin)
    print( "Q max = %f"%Qmax)
    maxipix = pyFAI.detector_factory("Maxipix_5x1")
    ais = [pyFAI.load(pf) for pf in poni_files]
    for ai in ais:
        ai.detector = maxipix
    wl = ai.wavelength*1e10
    tth_min = np.degrees(np.arcsin(Qmin*wl/4/np.pi))*2
    tth_max = np.degrees(np.arcsin(Qmax*wl/4/np.pi))*2
    print( "tth min = %f"%tth_min)
    print( "tth max = %f"%tth_max)
    
    allData   = Run.allData_allRun
    mg = MultiGeometry(poni_files, radial_range=(tth_min, tth_max))
    phi_points = allData.shape[1]
    bin_pts = 1000
    I_tot = np.zeros(bin_pts)
        
    num_threads = _NUM_THREADS
    # """
    for j in range(0, phi_points, num_threads):
        threads = []
        for i in range(j, j+num_threads):
            if i<phi_points:
                img_data = allData[:,i,:,:]
                thr = Integrator(mg, i, img_data, mask, bin_pts)
                thr.start()
                threads.append(thr)
        for thr in threads:
            thr.join()
            I_tot += thr.I
    # """
    
    I_tot = I_tot/phi_points
    q_tth = np.linspace(Qmin, Qmax, bin_pts)
    tth_q = np.linspace(tth_min, tth_max, bin_pts)
    outData = np.vstack([tth_q, q_tth, I_tot])
    outData = outData.T 
    outFile = splitext(configFile)[0]+"_Integrated_intensity_Q.dat"
    np.savetxt(outFile, outData, header=str("2Theta (deg) \t Q (1/A) \t Intensity"), fmt="%6.4f")
    (tth_peak,I_peak) = get_maxima(thr.tth, I_tot, threshold=peak_detection_ratio)
    tth_peak = np.asarray(tth_peak)
    Q_shell = 4*np.pi*np.sin(np.radians(tth_peak/2))/wl
    Qout = np.vstack([tth_peak, Q_shell])
    Qout = Qout.T 
    outFile2 = splitext(configFile)[0]+"_tth_Q_peaks.dat"
    np.savetxt(outFile2, Qout, fmt="%6.4f", header=str("Q max = %.4f\n2Theta (deg) \t Q value (1/A)"%Qmax))
    print("TTH peaks: ")
    print(tth_peak)
    fig=figure()
    
    ax = fig.add_subplot(111)
    ax.plot(thr.tth, I_tot, lw=2)
    ax.plot(tth_peak, I_peak, "ro")
    # for x in tth_peak:
        # ax.axvline(x, color="r", lw=1)
    ax.set_xlabel("2Theta (deg)")
    ax.set_ylabel("Summed intensity (a.u.)")
    title = splitext(configFile)[0]+" integrated intensity over phi scan"
    ax.set_title(title)
    show()
    saveImg = splitext(configFile)[0]+"_integrated_intensity.png"
    fig.savefig(saveImg)
    
# Usage: python Integrated_intensity_3DRSM.py PtSi.conf (0.1) >> Optional (peak_detection_ratio)
if __name__== "__main__":
    print(len(sys.argv))
    if len(sys.argv)<=2:
        peak_detection_ratio = float(0.1)
    else:
        peak_detection_ratio = float(sys.argv[2])
    Config_fname = sys.argv[1]
    Config_fname = os.path.normpath(Config_fname)
    print(Config_fname)
    print(peak_detection_ratio)
    t0 = time.time()
    run_integration(Config_fname, peak_detection_ratio)
    t1=time.time()
    print("Total running time: ", t1-t0, "seconds")
    