#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, glob, threading
from queue import PriorityQueue
import fabio, pyFAI
import numpy as np
import sys, time
import volume_reconstruction as fillvolume
import gc
from imgheader import *
from sys import stdout
import h5py
from os.path import isfile, join
import pyopencl as cl
import multiprocessing

_NUM_THREADS = multiprocessing.cpu_count()*2
def save_cmap(fn, vol):
    h5=h5py.File(fn, "w")
    g=h5.create_group("volume")
    g.create_dataset("data", data=vol, compression="gzip", compression_opts=9)
    h5.close()
    print("Map %s is saved."%fn)
    
class ProgressBar:
    def __init__(self, duration):
        self.duration = duration
        self.prog_bar = '[]'
        self.fill_char = '|'
        self.width = 60
        self.__update_amount(0)

    def update_time(self, elapsed_secs):
        self.__update_amount((elapsed_secs / float(self.duration)) * 100.0)
        self.prog_bar += ' %d/%d images' % (elapsed_secs, self.duration)
        
    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = int((len(self.prog_bar) / 2) - len(str(percent_done)))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])
        
    def __str__(self):
        return str(self.prog_bar)
         

class Worker(threading.Thread):
    def __init__(self, i, filePath, dataQueue):
        threading.Thread.__init__(self)
        self.dataQueue = dataQueue
        self.filePath  = filePath
        self.order_number = i
    def run(self):
        if isfile(self.filePath):
            img = fabio.open(self.filePath)
            data= img.data
            motor=get_motor(img.header)
            self.dataQueue.put((self.order_number, self.filePath, data, motor))
            self.dataQueue.task_done()
            stdout.write("\r Got %s"%self.filePath)
            stdout.flush()
            
def get_img_dataset(img_folder, img_prefix, img_extension, img_digit, beg, end):
    dataQueue = PriorityQueue()
    num_threads = _NUM_THREADS
    fileList = []
    runData  = []
    runMotor = []
    for j in range(beg, end+1, num_threads):
        threads = []
        for i in range(j, j+num_threads):
            if i<=end:
                img_filename = "{img_prefix}{img_num:0{img_digit}}.{ext}".format(img_prefix=img_prefix, img_num=i, img_digit=img_digit, ext=img_extension)
                fn = join(img_folder, img_filename)
                thr = Worker(i, fn, dataQueue)
                thr.start()
                threads.append(thr)
        for thr in threads:
            thr.join()
    # dataQueue.sort()
    while not dataQueue.empty():
        data = dataQueue.get()
        fileList.append(data[1])
        runData.append(data[2])
        runMotor.append(data[3])
    dataQueue.join()
    runData = np.array(runData, dtype=np.float32)
    return (fileList, runData, runMotor)
    
            
def perpendicular_to(v):
    #Find a unit vector perpendicular to a vector v.
    if v[1]==0 and v[2]==0:
        if v[0]!=0:
            x = np.cross(v, np.array([0,1,0])).astype(np.float32)
            return x/np.linalg.norm(x)
    else:
        x=np.cross(v, np.array([1,0,0])).astype(np.float32)
        return x/np.linalg.norm(x)
        
def vectangle(v1,v2):
    v1=np.array(v1)
    v2=np.array(v2)
    cosalpha = np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
    alpha = np.arccos(cosalpha)
    return alpha
    

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
#--------------------------------------------------------------------------------------------------------
# Rotation, Projection and Orientation Matrix
#--------------------------------------------------------------------------------------------------------

def Rotation(angle, axis):
    """
    Right-handed rules, positive rotation = clockwise looking from Origin toward the +infinity
    angle is in radians
    """
    if type(axis) == type("") :
        rotation_axis = {"x":0, "y":1, "z":2}[axis[0]]
    assert((rotation_axis >= 0 and rotation_axis <= 3))
    if axis[1] == "+":
        sign = 1.0
    else:
        sign = -1.0
    # angle = np.radians(angle) * sign
    angle *= sign
    ret_val = np.zeros([3, 3], np.float32)
    i1 = rotation_axis
    i2 = (rotation_axis + 1) % 3
    i3 = (rotation_axis + 2) % 3
    ret_val[i1, i1  ] = 1
    ret_val[i2, i2  ] = np.cos(angle)
    ret_val[i3, i3  ] = np.cos(angle)
    ret_val[i2, i3  ] = -np.sin(angle)
    ret_val[i3, i2  ] = np.sin(angle)
    return ret_val

#--------------------------------------------------------------------------------------------------------

def Mirror(mirror_axis):
    x,y,z = {"x":[-1,1,1], "y":[1,-1,1], "z":[1,1,-1],0:[-1,1,1], 1:[1,-1,1], 2:[1,1,-1]}[mirror_axis]
    mat = np.array([[x,0,0],[0,y,0],[0,0,z]])
    return mat 

#--------------------------------------------------------------------------------------------------------

def ascii_gen2operator(generator):
    I = np.identity(3)
    gendict = {  '1':I,
            '1~':-I,
            '2(x)':Rotation(np.radians(180),"x+"),
            '2(y)':Rotation(np.radians(180),"y+"),
            '2(z)':Rotation(np.radians(180),"z+"),
            '2(110)':np.array([[0,1,0],[1,0,0],[0,0,-1]]),
            '3(z)':Rotation(np.radians(120),"z+"),
            '3(111)':np.array([[0,1,0],[0,0,1],[1,0,0]]),
            '4(x)':Rotation(np.radians(90),"x+"),
            '4(y)':Rotation(np.radians(90),"y+"),
            '4(z)':Rotation(np.radians(90),"z+"),
            '4(z)~':-I*Rotation(np.radians(90),"z+"),
            'm(x)':Mirror(0),
            'm(y)':Mirror(1),
            'm(z)':Mirror(2)
          }
          
    return gendict[generator]

#--------------------------------------------------------------------------------------------------------

def existing_combination(candidate_op,op_list,sigma):
    for op in op_list:
        absdiff = abs(candidate_op - op)
        if absdiff.sum()<sigma:
            return True
    return False
    
#--------------------------------------------------------------------------------------------------------

def genlist2oplist(genlist):
    base_op_list = [ascii_gen2operator(gen) for gen in genlist]
    I = np.identity(3)
    combined_op_list = [I]
    while 1:
        new_add_in_cycle = False
        for op in combined_op_list:
            for base_op in base_op_list:
                newop = np.dot(op,base_op)
                if not existing_combination(newop,combined_op_list,0.01):
                    combined_op_list.append(newop)
                    new_add_in_cycle = True
        if not new_add_in_cycle:
            break
    return np.array(combined_op_list).astype(np.float32)

#--------------------------------------------------------------------------------------------------------

def Sample_Rotation(sample_angles, rot_dir):
    """
    This code consider: from inner-most to outer-most = phi (Z-), omega (Y-)
    sample_angles: list of sample rotation angles (in degrees), from inner-most to outer-most
    rot_dir: direction of rotation of each circle, following the same order as in sample_angles
    """
    tmp = np.eye(3)
    for i in range(len(sample_angles)):
        ang = np.radians(sample_angles[i])
        rot = Rotation(ang, rot_dir[i])
        tmp = np.dot(rot, tmp)
    return tmp.astype(np.float32)
#--------------------------------------------------------------------------------------------------------

def DET(rot1, rot2, rot3):
    """ 
    Rotation matrix for the detector
    rot1: RotZ --> moving horizontally
    rot2: RotY --> moving vertically
    rotations angles rot1, rot2 are in radians, issued from pyFAI poni file. ==> Attention pyFAI convention about rotations axis.
    """
    R1 = Rotation(rot1, "z-")
    R2 = Rotation(rot2, "y-")
    R3 = Rotation(rot3, "x+")
    tmp = np.dot(R2, R3)
    tmp = np.dot(R1, tmp)
    return tmp.astype(np.float32)

#-------------------------------------------------------------------------------------------------------- 

def P0():
    """ direct beam vector from sample to the detector"""
    p0 = np.array([ -1, 0, 0 ])
    return p0

def mag_max(l):
    """
    l is a list of coordinates
    """
    return max(np.sqrt(np.sum(l * l, axis= -1)))
    

class RunInit:
    def __init__(self, Config_fname):
        print ('Setting Parameters ...')
        self.params = read_configuration_file(Config_fname)
        self.ponifiles = self.params.poni_files
        self.azimuthalIntegrators = [pyFAI.load(poni) for poni in self.ponifiles]
        self.ai0 = self.azimuthalIntegrators[0]
        self.pixel_size = np.float(self.ai0.pixel1)
        self.lmbda = np.float(self.ai0.wavelength)
        ip = np.array(self.params.in_plane_dir).astype(np.float32)
        op = np.array(self.params.out_plane_dir).astype(np.float32)
        ip = ip/np.linalg.norm(ip)
        op = op/np.linalg.norm(op)
        ortho = np.cross(op, ip)
        self.Umatrix = np.array([ip, ortho, op])
        UB_rot_dir = self.params.UB_rotation_dir
        UB_rot_ang = float(self.params.UB_rotation_angle)
        UB_rot_ang = np.radians(UB_rot_ang)
        rot_UB_operator = Rotation(UB_rot_ang, UB_rot_dir)
        self.Umatrix = np.dot(rot_UB_operator, self.Umatrix)
        self.params.Umatrix = self.Umatrix
        self.Bmatrix = np.eye(3).astype(np.float32)
        self.params.img_digit = int(self.params.img_digit)
        MD = self.params.MD
        
        self.interp_factor = self.params.interpolation_number
        if self.params.making_slice:
            self.number_of_slice = int(self.params.number_of_slice)
            self.dQ0 = np.array(self.params.slice_thickness, dtype=np.float32)
            self.dQ1 = np.array(self.params.slice_extent_1, dtype=np.float32)
            self.dQ2 = np.array(self.params.slice_extent_2, dtype=np.float32)
            Qoff = np.array(self.params.vector_offset, dtype=np.float32)
            self.G = []
            self.Qoff=[]
            H1 = np.array(self.params.vector_H1, dtype=np.float32)
            H2 = np.array(self.params.vector_H2, dtype=np.float32)
            for s in range(self.number_of_slice):
                vect_a = H1[s]
                vect_bp = H2[s]
                vect_a = vect_a/np.linalg.norm(vect_a)
                vect_c = np.cross(vect_a,vect_bp)
                vect_c = vect_c/np.linalg.norm(vect_c)
                vect_b = np.cross(vect_c,vect_a)
                
                g = np.array([vect_a,vect_b,vect_c])
                self.G.append(g)
                qoff2= np.dot(g,Qoff[s])
                self.Qoff.append(qoff2)
                
            self.G = np.asarray(self.G).astype(np.float32)
            self.Qoff = np.asarray(self.Qoff)
        # """
        self.ascii_gen_list = self.params.symmetry_operators
        
        self.scale_factor = self.params.scale_factor
        self.making_slice = self.params.making_slice
        self.making_volume = self.params.making_volume
        self.making_shell = self.params.making_shell
        self.making_pole_figure = self.params.making_pole_figure
        if self.making_slice: 
            self.sliceName = self.params.slice_outname
            self.slice_dim = int(self.params.slice_dim)
        if self.making_volume:
            self.cube_dim = int(self.params.cube_dim)
            self.number_of_volume = int(self.params.number_of_volume)
            self.volumeName = self.params.volume_outname
            vC = np.array(self.params.volume_center, dtype=np.float32)
            self.volume_center = []
            for v in range(self.number_of_volume):
                vvc = np.dot(self.Bmatrix,vC[v])
                self.volume_center.append(vvc)
            self.volume_center = np.asarray(self.volume_center)
            self.volume_extent = np.array(self.params.volume_extent, dtype=np.float32)
        if self.making_shell:
            self.number_of_shell = int(self.params.number_of_shell)
            self.shell_dim = int(self.params.shell_dim)
            self.shell_extent = float(self.params.shell_extent)
            self.shell_center = np.array(self.params.shell_center, dtype=np.float32)
            self.shell_thickness = float(self.params.shell_thickness)
            self.Q_shell = np.array(self.params.shell_radius, dtype=np.float32)
            self.shellName = self.params.shell_outname
        if self.making_pole_figure:
            self.number_of_figure = int(self.params.number_of_figure)
            self.pole_size = int(self.params.pole_size)
            self.use_custom_projection_plane = self.params.use_custom_projection_plane
            self.rotation_axis = self.params.lab_rotation_axis.upper()
            axis = {"X": np.array([1,0,0],dtype=np.float32),"Y": np.array([0,1,0],dtype=np.float32),"Z": np.array([0,0,1],dtype=np.float32)}
            self.lab_rotation_axis = axis[self.rotation_axis]
            self.crystal_rotation_axis = np.dot(self.Umatrix.T, self.lab_rotation_axis)
            self.pole_normal_vector = self.crystal_rotation_axis/np.linalg.norm(self.crystal_rotation_axis)
            if self.use_custom_projection_plane:
                pnv = np.array(self.params.custom_projection_plane_normal, dtype=np.float32)
                self.pole_normal_vector = pnv/np.linalg.norm(pnv)
            tmp2 = perpendicular_to(self.pole_normal_vector)
            tmp1 = np.cross(tmp2, self.pole_normal_vector)
            self.pole_G = np.array([tmp1, tmp2, self.pole_normal_vector])
            self.pole_thickness = float(self.params.pole_thickness)
            self.Qpole = np.array(self.params.pole_radius, dtype=np.float32)
            self.pole_name = self.params.pole_name
        self.ops_list = genlist2oplist(self.ascii_gen_list)
        self.apply_sym=0
        if len(self.ops_list)>1:
            self.apply_sym=1
        # """
        self.making_volume = self.params.making_volume
        if self.making_volume:
            self.cube_dim = 2*(int(self.params.cube_dim)//2)+1
            self.number_of_volume = int(self.params.number_of_volume)
            self.volumeName = self.params.volume_outname
            vC = np.array(self.params.volume_center, dtype=np.float32)
            self.volume_center = []
            for v in range(self.number_of_volume):
                self.volume_center.append(vC[v])
            self.volume_center = np.asarray(self.volume_center)
            self.volume_extent = np.array(self.params.volume_extent, dtype=np.float32)
        self.number_of_run = int(self.params.number_of_run)
        print("Collecting data ...")
        self.flist = []
        self.allData_allRun = []
        self.allMotor_allRun= []
        self.img_begin = np.array(self.params.img_begin, dtype=np.int16)
        self.img_end   = np.array(self.params.img_end, dtype=np.int16)
        for i in range(self.number_of_run):
            (fl, runData, runMotor) = self.get_img_list(self.img_begin[i], self.img_end[i])
            self.flist.append(fl)
            self.allData_allRun.append(runData)
            self.allMotor_allRun.append(runMotor)
        self.allData_allRun = np.asarray(self.allData_allRun)
        
        print("\nData collection done.")
        img = fabio.open(self.flist[0][0])
        data = img.data
        self.dim1, self.dim2 = data.shape

        print ('Image Dimension :', self.dim1, self.dim2)

        self.normal_to_pol = np.array(self.params.normal_to_pol).astype(np.float32)
        self.pol_degree    = float(self.params.pol_degree)
        
        print ('Computation of Initial Projection Coordinates Q0 for all runs ...')
        
        self.all_Q0   = []
        self.all_POLA = []
        self.all_C3   = []
        for run in range(self.number_of_run):
            ai = self.azimuthalIntegrators[run]
            ai.wavelength = ai.wavelength*1e10 #For convenient purpose, I want the wavelength to be in Angstroem
            dist = ai.dist
            self.det_origin_X = np.float(ai.poni2) #Use poni coordinates in meters directly
            self.det_origin_Y = np.float(ai.poni1)
            self.p0 = P0()
            X_array_tmp = self.pixel_size*(np.arange(self.dim2)) - self.det_origin_X
            Y_array_tmp = self.pixel_size*(np.arange(self.dim1)) - self.det_origin_Y

            XX,YY = np.meshgrid(X_array_tmp, Y_array_tmp)
            XY = np.zeros((self.dim1, self.dim2, 2), dtype=np.float32)
            XY[:,:,0] = XX
            XY[:,:,1] = YY
            XY_tmp = np.tensordot(XY, MD, axes = ([2],[1]))
            
            XX = XY_tmp[:,:,0]
            YY = XY_tmp[:,:,1]

            P_total_tmp = np.zeros((self.dim1, self.dim2 , 3), dtype=np.float32)
            P_total_tmp[:, :, 0] = dist
            P_total_tmp[:, :, 1] = XX
            P_total_tmp[:, :, 2] = YY
            Q0 = np.zeros((self.dim1, self.dim2, 3), dtype=np.float32)
            PP = np.tensordot(P_total_tmp, DET(ai.rot1, ai.rot2, ai.rot3), axes=([2], [1]))
            PP_modulus = np.sqrt(np.sum(PP * PP, axis= -1))
            Q0_tmp = PP.T / PP_modulus.T
            Q0 = 2.0*np.pi*((Q0_tmp.T + self.p0) / ai.wavelength).astype(np.float32)
            self.all_Q0.append(Q0)
            
            print('Computation of Polarisation Correction for run %d...'%(run+1))
            
            P0xn = np.cross(self.p0, self.normal_to_pol, axis=0)
            P0xn_modulus = np.sqrt(np.sum(P0xn * P0xn, axis= -1))
            POL_tmp = self.pol_degree * (1 - ((P0xn * PP).sum(axis= -1) / (P0xn_modulus * PP_modulus)) ** 2)
            POL_tmp += (1 - self.pol_degree) * (1 - ((self.normal_to_pol * PP).sum(axis=-1) / PP_modulus) ** 2)        
            POL_tmp = POL_tmp.astype(np.float32)
            self.all_POLA.append(POL_tmp)
            
            print('Computation of Flux Density and Parallax Correction for run %d...'%(run+1))
            C3 = (dist ** 3 / (dist ** 2 + XX * XX + YY * YY) ** (3/2.)).astype(np.float32)
            self.all_C3.append(C3)
            
        print ('Estimation of Qmax ...')
        corners = []
        for run in range(self.number_of_run):
            corners.append(self.all_Q0[run][0, 0, :])
            corners.append(self.all_Q0[run][0, self.dim2 - 1, :])
            corners.append(self.all_Q0[run][self.dim1 - 1, 0, :])
            corners.append(self.all_Q0[run][self.dim1 - 1, self.dim2 - 1, :])
            
        corners = np.asarray(corners)
        self.Qmax = mag_max(corners) # maximal magnitude for reciprocal vector corresponding to the corners pixels
        print ('-----------------------------')
        print ('Qmax = ', self.Qmax)
        gc.collect()
        
    def get_img_list(self, beg, end):
        (fl,runData, runMotor) = get_img_dataset(self.params.img_folder, self.params.img_prefix, self.params.image_extension, self.params.img_digit, beg, end)#img_folder, img_prefix, img_extension, img_digit, beg, end
        return (fl, runData, runMotor)
        
#--------------------------------------------------------------------------------------------------------

def main3d(Run):
    tbegin=time.time()
    params = Run.params
    params.phi_step = np.array(params.phi_step, dtype=np.float32)
    ascii_gen_list = params.symmetry_operators
    ops_list = genlist2oplist(ascii_gen_list)
    apply_sym=0
    if len(ops_list)>1:
        apply_sym=1
    number_of_run = params.number_of_run
    Bmatrix = Run.Bmatrix
    Bi      = np.linalg.inv(Bmatrix)

    flist = Run.flist
    total = 0
    for run in range(number_of_run):
        total += len(flist[run])
    p = ProgressBar(total)
    Filter = fabio.open(params.maskFile).data.astype(np.float32)
    (dim1, dim2)=Filter.shape
    last_run = 0
    if not Run.making_volume:
        Run.number_of_volume = int(1)
        Run.cube_dim = int(1)
    if not Run.making_shell:
        Run.number_of_shell = int(1)
        Run.shell_dim = int(1)
    if not Run.making_slice:
        Run.number_of_slice = int(1)
        Run.slice_dim = int(1)
    if not Run.making_pole_figure:
        Run.number_of_figure = int(1)
        Run.pole_size = int(1)
    #GPU 
    gpu_enable = int(params.gpu_enable)
    if gpu_enable:
        platform = cl.get_platforms()[int(params.platform_id)]
        device   = platform.get_devices()[int(params.device_id)]
        context  = cl.Context([device])
        queue    = cl.CommandQueue(context)
        mf       = cl.mem_flags
        kernel_code = open("kernelCode.cl", "r").read()
        kernel_pars = {"number_of_volume":Run.number_of_volume, \
                       "nx":Run.cube_dim, \
                       "ny":Run.cube_dim, \
                       "nz":Run.cube_dim, \
                       "dim1":dim1, \
                       "dim2":dim2, \
                       "dimsym":ops_list.shape[0],\
                       "number_of_shell": Run.number_of_shell,\
                       "sx": Run.shell_dim,\
                       "sy": Run.shell_dim,\
                       "sz": Run.shell_dim,\
                       "number_of_figure": Run.number_of_figure,\
                       "px": Run.pole_size,\
                       "py": Run.pole_size,\
                       "slice_size": Run.slice_dim
                       }
        prog     = cl.Program(context, kernel_code%kernel_pars).build()
        data_gpu = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Filter)
        Qfin_gpu = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Run.all_Q0[0])
        Filter_gpu   = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Filter)
        symOps_gpu   = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ops_list)
    if Run.making_volume:
        Volume = np.zeros((Run.number_of_volume, Run.cube_dim, Run.cube_dim, Run.cube_dim), dtype=np.float32)
        Mask   = np.zeros((Run.number_of_volume, Run.cube_dim, Run.cube_dim, Run.cube_dim), dtype=np.uint32)
        if gpu_enable:
            volCenter_gpu   = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Run.volume_center)
            volExtent_gpu   = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Run.volume_extent)
            Volume_gpu   = cl.Buffer(context, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=Volume)
            Mask_gpu   = cl.Buffer(context, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=Mask)
        
    if Run.making_shell:
        ShellVolume = np.zeros((Run.number_of_shell,Run.shell_dim, Run.shell_dim, Run.shell_dim), dtype=np.float32)
        ShellMask   = np.zeros((Run.number_of_shell,Run.shell_dim, Run.shell_dim, Run.shell_dim), dtype=np.uint32)
        if gpu_enable:
            Q_shell_gpu = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Run.Q_shell)
            shell_center_gpu = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Run.shell_center)
            # shell_extent_gpu = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Run.shell_extent)
            shell_extent_gpu = np.float32(Run.shell_extent)
            # shell_thickness_gpu = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Run.shell_thickness)
            shell_thickness_gpu = np.float32(Run.shell_thickness)
            ShellVolume_gpu   = cl.Buffer(context, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=ShellVolume)
            ShellMask_gpu   = cl.Buffer(context, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=ShellMask)
        
        
    if Run.making_slice:
        SliceImage = np.zeros((Run.number_of_slice,Run.slice_dim,Run.slice_dim), dtype=np.float32)
        SliceMask  = np.zeros((Run.number_of_slice,Run.slice_dim,Run.slice_dim), dtype=np.uint32)
        if gpu_enable:
            G_gpu  = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Run.G)
            dQ0_gpu= cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Run.dQ0)
            dQ1_gpu= cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Run.dQ1)
            dQ2_gpu= cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Run.dQ2)
            Qoff_gpu= cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Run.Qoff)
            SliceImage_gpu = cl.Buffer(context, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=SliceImage)
            SliceMask_gpu = cl.Buffer(context, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=SliceMask)
        
    if Run.making_pole_figure:
        PoleData = np.zeros((Run.number_of_figure, Run.pole_size, Run.pole_size), dtype=np.float32)
        PoleMask = np.zeros((Run.number_of_figure, Run.pole_size, Run.pole_size), dtype=np.uint32)
        if gpu_enable:
            PoleData_gpu = cl.Buffer(context, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=PoleData)
            PoleMask_gpu = cl.Buffer(context, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=PoleMask)
            Qpole_gpu = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Run.Qpole)
            # pole_thickness_gpu = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Run.pole_thickness)
            pole_thickness_gpu  = np.float32(Run.pole_thickness)

    #GPU
    # sample_angles = np.zeros(int(Run.params.sample_circles))
    # scanning_motor_index = Run.params.sample_axis.index(Run.params.scanning_motor)
    # sample_rotation_dir = list(Run.params.rot_dir)
    # print("Scanning motor: %s, index: %d"%(Run.params.scanning_motor, scanning_motor_index))
    print("Scanning motor: ", Run.params.scanning_motor)
    for run in range(number_of_run):
        nbfile = 0
        sample_angles = np.zeros(int(Run.params.sample_circles))
        print("Scanning motor: %s"%(Run.params.scanning_motor[run]))
        scanning_motor_index = Run.params.sample_axis.index(Run.params.scanning_motor[run])
        sample_rotation_dir = list(Run.params.rot_dir)
        print("Scanning motor: %s, index: %d"%(Run.params.scanning_motor[run], scanning_motor_index))
        for id in range(len(flist[run])):
            data = Run.allData_allRun[run][id]
            motors=Run.allMotor_allRun[run][id]
            data = (data * Filter)/(Run.all_C3[run] * Run.all_POLA[run])
            for sc in range(int(Run.params.sample_circles)):
                sample_angles[sc] = motors[Run.params.sample_axis[sc]]
            U = Run.Umatrix
            if gpu_enable:
                cl.enqueue_copy(queue, data_gpu, data).wait()
            
            for j in range(Run.interp_factor):
                interphi = sample_angles[scanning_motor_index] + j/Run.interp_factor*params.phi_step[run]
                sample_angles[scanning_motor_index] = interphi
                R = Sample_Rotation(sample_angles, sample_rotation_dir)
                Q = np.tensordot (Run.all_Q0[run] , R.T , axes=([2], [1]))
                Qfin = np.tensordot (Q , U.T , axes=([2], [1]))
                if gpu_enable:
                    cl.enqueue_copy(queue, Qfin_gpu, Qfin).wait()
                if Run.making_volume:
                    if gpu_enable:
                        prog.volReconstruction(queue, data.shape, None, volCenter_gpu, volExtent_gpu, Volume_gpu, Mask_gpu, Qfin_gpu, data_gpu, Filter_gpu,np.int32(apply_sym),symOps_gpu).wait()
                    else:
                        fillvolume.volume(Run.volume_center, Run.volume_extent, Volume, Mask, Qfin, data, Filter,apply_sym,ops_list)
                    
                if Run.making_shell:
                    if gpu_enable:
                        prog.extract_shell(queue, data.shape, None, Q_shell_gpu, shell_center_gpu, shell_extent_gpu, shell_thickness_gpu, ShellVolume_gpu, ShellMask_gpu, Qfin_gpu, data_gpu, Filter_gpu,np.int32(apply_sym),symOps_gpu).wait()
                    else:
                        fillvolume.extract_shell(Run.Q_shell, Run.shell_center, Run.shell_extent, Run.shell_thickness, ShellVolume, ShellMask, Qfin, data, Filter,apply_sym,ops_list)
                    
                if Run.making_slice:
                    if gpu_enable:
                        prog.extract_slice(queue, data.shape, None, np.int32(Run.number_of_slice), dQ0_gpu, dQ1_gpu, dQ2_gpu, Qoff_gpu, SliceImage_gpu, SliceMask_gpu,\
                                           Qfin_gpu, data_gpu, Filter_gpu, np.int32(apply_sym), symOps_gpu, G_gpu).wait()
                    else:
                        fillvolume.extract_slice(Run.number_of_slice,Run.dQ0, Run.dQ1, Run.dQ2, Run.Qoff,SliceImage,SliceMask,Qfin,data,Filter,apply_sym,ops_list,Run.G)
                
                if Run.making_pole_figure:
                    if gpu_enable:
                        prog.stereo_projection(queue, data.shape, None, Qpole_gpu, pole_thickness_gpu, PoleData_gpu, PoleMask_gpu,\
                                               Qfin_gpu, data_gpu, Filter_gpu, np.int32(apply_sym), symOps_gpu).wait()
                    else:
                        fillvolume.stereo_projection(Run.Qpole, Run.pole_thickness, PoleData, PoleMask, Qfin, data, Filter,apply_sym,ops_list)
                
                print('interpolation #%d on %d'%(j+1,Run.interp_factor))
            nbfile += 1
            timeI2 = time.time()
            p.update_time(nbfile + last_run)
            print ('------------------------------------------------------------')
            print (p)
            print ('------------------------------------------------------------')
            print ('\n')
        last_run += nbfile

    print ('3D Intensity Distribution : Done')
    ##################################
    #GPU
    if gpu_enable:
        Qfin_gpu.release()
        data_gpu.release()
        Filter_gpu.release()
        symOps_gpu.release()

    #GPU
    if Run.making_volume:
        if gpu_enable:
            # Getting data from gpu back
            cl.enqueue_copy(queue, Volume, Volume_gpu).wait()
            cl.enqueue_copy(queue, Mask, Mask_gpu).wait()
            Volume_gpu.release()
            Mask_gpu.release()
            volExtent_gpu.release()
            volCenter_gpu.release()
        for v in range(Run.number_of_volume):
            filter_ids = np.where(Mask[v]!=0)
            Volume[v][filter_ids] = Volume[v][filter_ids]/Mask[v][filter_ids]
            save_cmap(Run.volumeName[v], Volume[v])
            
    if Run.making_shell:
        if gpu_enable:
            cl.enqueue_copy(queue, ShellVolume, ShellVolume_gpu).wait()
            cl.enqueue_copy(queue, ShellMask, ShellMask_gpu).wait()
            ShellVolume_gpu.release()
            ShellMask_gpu.release()
            Q_shell_gpu.release()
            shell_center_gpu.release()
            # shell_extent_gpu.release()
            # shell_thickness_gpu.release()
        
        for sh in range(Run.number_of_shell):
            filter_ids = np.where(ShellMask[sh]!=0)
            ShellVolume[sh][filter_ids] = ShellVolume[sh][filter_ids]/ShellMask[sh][filter_ids]
            save_cmap(Run.shellName[sh], ShellVolume[sh])
            
    if Run.making_slice:
        if gpu_enable:
            cl.enqueue_copy(queue, SliceImage, SliceImage_gpu).wait()
            cl.enqueue_copy(queue, SliceMask, SliceMask_gpu).wait()
            SliceImage_gpu.release()
            SliceMask_gpu.release()
        for s in range(Run.number_of_slice):
            mapout = np.zeros_like(SliceImage[s])
            mapout[np.where(SliceMask[s]!=0)] = SliceImage[s][np.where(SliceMask[s]!=0)]/SliceMask[s][np.where(SliceMask[s]!=0)]
            
            tmp2 = mapout * params.scale_factor
            wi = fabio.cbfimage.cbfimage(data=tmp2.astype(np.int32))
            mapOutName = params.slice_outname[s]
            wi.write(mapOutName)
            Qoutname = mapOutName.split(".")[0]+"_hkl.h5"
            print("Slice %s saved."%mapOutName)
            Qoffset = np.dot(Run.Qoff[s], Run.G[s])
            x=np.linspace(-Run.dQ1[s], Run.dQ1[s], Run.slice_dim)
            y=np.linspace(-Run.dQ2[s], Run.dQ2[s], Run.slice_dim)
            x,y=np.meshgrid(x,y)
            z = np.zeros(x.shape)
            q = np.zeros((Run.slice_dim, Run.slice_dim, 3))
            q[:,:,0] = x+Qoffset[0]
            q[:,:,1] = y+Qoffset[1]
            q[:,:,2] = z+Qoffset[2]
            Gi = np.linalg.inv(Run.G[s])
            Qn = np.tensordot(q, Gi,  axes=([2],[1]))
            HKL= np.tensordot(Qn, Bi, axes=([2],[1]))
            h5file = h5py.File(Qoutname, "w")
            h5file.create_dataset("/Q", data=HKL, compression='gzip', compression_opts=9)
            h5file.create_dataset("/data", data=tmp2, compression='gzip', compression_opts=9)
            h5file.close()
            print("HKL coordinates saved.")
            rsmViewer_fn = mapOutName.split(".")[0]+"_rsmviewer.h5"
            # save2RSMviewer(tmp2, HKL, rsmViewer_fn)
            
    if Run.making_pole_figure:
        if gpu_enable:
            cl.enqueue_copy(queue, PoleData, PoleData_gpu).wait()
            cl.enqueue_copy(queue, PoleMask, PoleMask_gpu).wait()
            PoleData_gpu.release()
            PoleMask_gpu.release()
        for p in range(Run.number_of_figure):
            mapout = np.zeros_like(PoleData[p])
            mapout[np.where(PoleMask[p]!=0)] = PoleData[p][np.where(PoleMask[p]!=0)]/PoleMask[p][np.where(PoleMask[p]!=0)]
            
            tmp = mapout * params.scale_factor
            wi = fabio.cbfimage.cbfimage(data=tmp.astype(np.int32))
            mapOutName = Run.pole_name[p]
            wi.write(mapOutName)
            print("Pole %s saved."%mapOutName)
            
    ###################################
    print ('Normal END')
    gc.collect()
    tend=time.time()
    print("Total time for this operation: %.3f s"%(tend-tbegin))
    
def readH5(h5file):
    f=h5py.File(h5file,"r")
    data = f["/data"].value
    Q = f["/Q"].value
    f.close()
    return (Q, data)
    
def save2RSMviewer(data, Q, fn):
    description = os.path.basename(fn)
    description = os.path.splitext(description)[0]
    intensity   = data
    Qx          = Q[:,:,0]
    Qy          = Q[:,:,1]
    Qz          = Q[:,:,2]
    f5 = h5py.File(fn, "w")
    s  = f5.create_group(description)
    s.create_dataset("intensity", data = intensity, compression="gzip", compression_opts=9)
    s.create_dataset("Qx", data = Qx, compression="gzip", compression_opts=9)
    s.create_dataset("Qy", data = Qy, compression="gzip", compression_opts=9)
    s.create_dataset("Qz", data = Qz, compression="gzip", compression_opts=9)
    s.create_dataset("description", data = description)
    f5.close()
    print("%s is saved for RSMviewer"%fn)
    
if __name__=="__main__":
    if len(sys.argv) < 1:
        print('Usage : python Rec.py config.conf')
    else:
        Config_fname = sys.argv[1]
        Run = RunInit(Config_fname)
        if Run.params.run_all:
            main3d(Run)
    sys.exit()
