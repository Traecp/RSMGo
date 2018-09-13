/*Kernel code for volume reconstruction*/
/*#pragma OPENCL EXTENSION cl_khr_fp64 : enable*/
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define NV %(number_of_volume)d //number of volumes to be reconstructed == Volume.shape[0]
#define NX %(nx)d //Volume.shape[1] 
#define NY %(ny)d //Volume.shape[2] 
#define NZ %(nz)d //Volume.shape[3]
#define dim1 %(dim1)d //data.shape[0]
#define dim2 %(dim2)d //data.shape[1]
#define dimsym %(dimsym)d //ops_list.shape[0]
#define Data(i,j) data[j + i*dim2] //matrix 2D concern the data image && correctors [dim1, dim2]
#define filter(i,j) Filter[j + i*dim2] //matrix 2D concern the data image && correctors [dim1, dim2]
#define QFin(i, j, k) Qfin[k + j*3 + i*3*dim2] //Matrix 3D concern images vectors [dim1, dim2, 3]
#define VOL(i, j, k, t) Volume[t + k*NZ + j*NZ*NY +i*NX*NY*NZ] //4D matrix concern the final reconstructred data volumes
#define MAS(i, j, k, t) Mask[t + k*NZ + j*NZ*NY +i*NX*NY*NZ] //4D matrix concern the final reconstructred data volumes
/********For shells: ****/
#define NS %(number_of_shell)d
#define SX %(sx)d 
#define SY %(sy)d 
#define SZ %(sz)d 
#define SVOL(i, j, k, t) ShellVolume[t + k*SZ + j*SZ*SY +i*SX*SY*SZ] //4D matrix concern the final reconstructred data volumes
#define SMASK(i, j, k, t) ShellMask[t + k*SZ + j*SZ*SY +i*SX*SY*SZ] //4D matrix concern the final reconstructred data volumes

/********For pole figures: ****/
#define NP %(number_of_figure)d
#define PX %(px)d 
#define PY %(py)d
#define Pdata(i, j, k) PoleData[k + j*PX + i*PX*PY]
#define Pmask(i, j, k) PoleMask[k + j*PX + i*PX*PY]

/********For slicing: ****/
#define Slice_size %(slice_size)d
#define SlImage(i, j, k) SliceImage[k + j*Slice_size + i*Slice_size*Slice_size]
#define SlMask(i, j, k) SliceMask[k + j*Slice_size + i*Slice_size*Slice_size]

inline void atomicAdd_float(volatile __global float *addr, float val)
{
    union{
        unsigned int u32;
        float f32;
        } next, expected, current;
    current.f32 = *addr;
    do{
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg( (volatile __global unsigned int *)addr,
        expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}

__kernel void volReconstruction(__global const float *Qcenter, //2D matrix
           __global const float *Qextent, //vector 1D
           __global float *Volume,  //matrix 4D
           __global unsigned int *Mask,    //matrix 4D
           __global const float *Qfin,    //matrix 3D
           __global const float *data,    //matrix 2D
           __global const float *Filter,  //matrix 2D
           int apply_sym,              //scalar integer
           __global const float *ops_list) //matrix 3D
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int v;
    unsigned int flag;
    int i,j,k,l,m,n;
    int nz_2, ny_2, nx_2;
    float pv, one, qfx, qfy, qfz,rotqfx, rotqfy, rotqfz, q0x, q0y, q0z, dq;
    nx_2 = (int) (NX-1)/ 2;
    ny_2 = (int) (NY-1)/ 2;
    nz_2 = (int) (NZ-1)/ 2;
    one = 1.0;
    for(v=0;v<NV;v++)
    {
        if (filter(x,y) !=0.0)
        {
            qfx = QFin(x,y,0);
            qfy = QFin(x,y,1);
            qfz = QFin(x,y,2);
            q0x = Qcenter[0 + 3*v];
            q0y = Qcenter[1 + 3*v];
            q0z = Qcenter[2 + 3*v];
            dq  = Qextent[v];
            if (apply_sym)
            {
                for (n=0;n<dimsym;n++)
                {
                    rotqfx = (ops_list[0+3*0+9*n]*qfx +ops_list[1+3*0+9*n]*qfy+ops_list[2+3*0+9*n]*qfz);
                    rotqfy = (ops_list[0+3*1+9*n]*qfx +ops_list[1+3*1+9*n]*qfy+ops_list[2+3*1+9*n]*qfz);
                    rotqfz = (ops_list[0+3*2+9*n]*qfx +ops_list[1+3*2+9*n]*qfy+ops_list[2+3*2+9*n]*qfz);
                    i = (int) (nx_2  * (one + (rotqfx-q0x)/dq ));
                    j = (int) (ny_2  * (one + (rotqfy-q0y)/dq ));
                    k = (int) (nz_2  * (one + (rotqfz-q0z)/dq ));
                    if ((0<=i<NX) && (0<=j<NY) && (0<=k<NZ))
                    {
                        pv = Data(x,y);
                        flag = (int)(pv!=0);
                        atomicAdd_float(&VOL(v,i,k,j), pv);
                        atom_add(&MAS(v,i,k,j), flag);
                    }
                }
            }
            i = (int) (nx_2  * (one + (qfx-q0x)/dq ));
            j = (int) (ny_2  * (one + (qfy-q0y)/dq ));
            k = (int) (nz_2  * (one + (qfz-q0z)/dq ));
            if ((i<NX) && (j<NY) && (k<NZ))
            {
                pv = Data(x,y);
                flag = (int)(pv!=0);
                atomicAdd_float(&VOL(v,i,k,j), pv);
                atom_add(&MAS(v,i,k,j), flag);
            }
        }
    }
}

__kernel void extract_shell(__global const float *Q_shell,
           __global const float *Qcenter,
           const float Qextent,
           const float TOL,
           __global float *ShellVolume, 
           __global unsigned int *ShellMask, 
           __global const float *Qfin,
           __global const float *data,
           __global const float *Filter,
           int apply_sym,
           __global const float *ops_list)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int i,j,k,n,sh;
    unsigned int flag;
    int nz_2, ny_2, nx_2;
    float pv, one, qfx, qfy, qfz,rotqfx, rotqfy, rotqfz;
    float tmp, mds;
    float q0x, q0y, q0z, dq;
    nx_2 = (int) (SX-1)/ 2;
    ny_2 = (int) (SY-1)/ 2;
    nz_2 = (int) (SZ-1)/ 2;
    one = 1.0;
    
    for(sh=0;sh<NS;sh++)
    {
        if (filter(x,y) !=0.0)
        {
            qfx = QFin(x,y,0);
            qfy = QFin(x,y,1);
            qfz = QFin(x,y,2);
            q0x = Qcenter[0];
            q0y = Qcenter[1];
            q0z = Qcenter[2];
            dq  = Qextent;
            tmp = qfx*qfx + qfy*qfy + qfz*qfz;
            mds = sqrt(tmp);
            if ((mds>=Q_shell[sh]-TOL) && (mds<Q_shell[sh]+TOL))
            {
                if (apply_sym)
                {
                    for (n=0;n<dimsym;n++)
                    {
                        rotqfx = (ops_list[0+3*0+9*n]*qfx +ops_list[1+3*0+9*n]*qfy+ops_list[2+3*0+9*n]*qfz);
                        rotqfy = (ops_list[0+3*1+9*n]*qfx +ops_list[1+3*1+9*n]*qfy+ops_list[2+3*1+9*n]*qfz);
                        rotqfz = (ops_list[0+3*2+9*n]*qfx +ops_list[1+3*2+9*n]*qfy+ops_list[2+3*2+9*n]*qfz);
                        i = (int) (nx_2  * (one + (rotqfx-q0x)/dq ));
                        j = (int) (ny_2  * (one + (rotqfy-q0y)/dq ));
                        k = (int) (nz_2  * (one + (rotqfz-q0z)/dq ));
                        if ((0<=i<SX) && (0<=j<SY) && (0<=k<SZ))
                        {
                            pv = Data(x,y);
                            flag = (int)(pv!=0);
                            atomicAdd_float(&SVOL(sh,i,k,j), pv);
                            atom_add(&SMASK(sh,i,k,j), flag);
                        }
                    }
                }
                i = (int) (nx_2  * (one + (qfx-q0x)/dq ));
                j = (int) (ny_2  * (one + (qfy-q0y)/dq ));
                k = (int) (nz_2  * (one + (qfz-q0z)/dq ));
                if ((i<SX) && (j<SY) && (k<SZ))
                {
                    pv = Data(x,y); 
                    flag = (int)(pv!=0);
                    atomicAdd_float(&SVOL(sh,i,k,j), pv);
                    atom_add(&SMASK(sh,i,k,j), flag);
                }
            }
        }
    }
}

__kernel void stereo_projection(__global const float *Q_pole,
           const float TOL,
           __global float *PoleData,
           __global unsigned int *PoleMask,
           __global const float *Qfin, 
           __global const float *data,
           __global const float *Filter,
           int apply_sym,
           __global const float *ops_list)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int i,j,n,sh;
    unsigned int flag;
    int ny_2, nx_2;
    float pv, one, qfx, qfy, qfz,rotqfx, rotqfy, rotqfz;
    float tmp, mds, px, py, chi, phi;
    nx_2 = (int) (PX/ 2);
    ny_2 = (int) (PY/ 2);
    one = 1.0;
    
    for(sh=0;sh<NP;sh++)
    {
        if (filter(x,y) !=0.0)
        {
            qfx = QFin(x,y,0);
            qfy = QFin(x,y,1);
            qfz = QFin(x,y,2);
            tmp = qfx*qfx + qfy*qfy + qfz*qfz;
            mds = sqrt(tmp);
            if ((mds>=Q_pole[sh]-TOL) && (mds<Q_pole[sh]+TOL))
            {
                if (apply_sym)
                {
                    for (n=0;n<dimsym;n++)
                    {
                        rotqfx = (ops_list[0+3*0+9*n]*qfx +ops_list[1+3*0+9*n]*qfy+ops_list[2+3*0+9*n]*qfz);
                        rotqfy = (ops_list[0+3*1+9*n]*qfx +ops_list[1+3*1+9*n]*qfy+ops_list[2+3*1+9*n]*qfz);
                        rotqfz = (ops_list[0+3*2+9*n]*qfx +ops_list[1+3*2+9*n]*qfy+ops_list[2+3*2+9*n]*qfz);
                        chi   = acos(rotqfz/mds);
                        phi   = atan2(rotqfy, rotqfx);
                        px    = tan(chi/2.0)*cos(phi);
                        py    = tan(chi/2.0)*sin(phi);
                        i = (int) (ny_2  * (one + py));
                        j = (int) (nx_2  * (one + px));
                        if ((0<=i<PX) && (0<=j<PY))
                        {
                            pv = Data(x,y);
                            flag = (int)(pv!=0);
                            atomicAdd_float(&Pdata(sh,i,j) , pv);
                            atom_add(&Pmask(sh,i,j), flag);
                        }
                    }
                }
                chi   = acos(qfz/mds);
                phi   = atan2(qfy, qfx);
                px    = tan(chi/2.0)*cos(phi);
                py    = tan(chi/2.0)*sin(phi);
                i = (int) (ny_2  * (one + py));
                j = (int) (nx_2  * (one + px));
                if ((0<=i<PX) && (0<=j<PY))
                {
                    pv = Data(x,y); 
                    flag = (int)(pv!=0);
                    atomicAdd_float(&Pdata(sh,i,j) , pv);
                    atom_add(&Pmask(sh,i,j), flag);
                }
            }
        }
    }
}

__kernel void extract_slice(int number_of_slice,
           __global const float  *dQ0, 
           __global const float  *dQ1,
           __global const float  *dQ2,
           __global const float  *Qoff, 
           __global float  *SliceImage, 
           __global unsigned int  *SliceMask,
           __global const float  *Qfin,
           __global const float  *data,
           __global const float  *Filter,
           int apply_sym,
           __global const float  *ops_list,
           __global const float  *G)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int half_size, s;
    int i,j,l,m, n;
    unsigned int flag;
    float qfx_minus_Qxoff,qfy_minus_Qyoff,qfz_minus_Qzoff, rotqfx_minus_Qxoff, rotqfy_minus_Qyoff, rotqfz_minus_Qzoff, Qxoff,Qyoff,Qzoff,qfx,qfy,qfz,pv,rotqfx,rotqfy,rotqfz,twodQ1,twodQ2;
    half_size = (int) (Slice_size/2.);
    float one = 1.0;
    for(s=0; s<number_of_slice; s++)
    {
        if (filter(x,y) !=0.0)
        {
            Qxoff = Qoff[0+3*s];
            Qyoff = Qoff[1+3*s];
            Qzoff = Qoff[2+3*s];
            qfx = QFin(x,y,0);
            qfy = QFin(x,y,1);
            qfz = QFin(x,y,2);
            if (apply_sym)
            {
                for (n=0;n<dimsym;n++)
                    {
                        rotqfx = (ops_list[0+3*0+9*n]*qfx +ops_list[1+3*0+9*n]*qfy+ops_list[2+3*0+9*n]*qfz);
                        rotqfy = (ops_list[0+3*1+9*n]*qfx +ops_list[1+3*1+9*n]*qfy+ops_list[2+3*1+9*n]*qfz);
                        rotqfz = (ops_list[0+3*2+9*n]*qfx +ops_list[1+3*2+9*n]*qfy+ops_list[2+3*2+9*n]*qfz);
                        rotqfx_minus_Qxoff = (G[0+3*0+9*s]*rotqfx + G[1+3*0+9*s]*rotqfy + G[2+3*0+9*s]*rotqfz) - Qxoff;
                        rotqfy_minus_Qyoff = (G[0+3*1+9*s]*rotqfx + G[1+3*1+9*s]*rotqfy + G[2+3*1+9*s]*rotqfz) - Qyoff;
                        rotqfz_minus_Qzoff = (G[0+3*2+9*s]*rotqfx + G[1+3*2+9*s]*rotqfy + G[2+3*2+9*s]*rotqfz) - Qzoff;
                        if  ((fabs(rotqfx_minus_Qxoff)<=dQ1[s]) && (fabs(rotqfy_minus_Qyoff)<=dQ2[s]) && (fabs(rotqfz_minus_Qzoff)<=dQ0[s]))
                        {
                            j = (int) (half_size * (one +  rotqfx_minus_Qxoff/dQ1[s]));
                            i = (int) (half_size * (one +  rotqfy_minus_Qyoff/dQ2[s]));
                            if ((0<=i<Slice_size) && (0<=j<Slice_size))
                            {
                                pv   = Data(x,y);
                                flag = (int)(pv!=0);
                                atomicAdd_float(&SlImage(s,i,j), pv);
                                atom_add(&SlMask(s,i,j), flag);
                            }
                        }
                    }
            }
            qfx_minus_Qxoff = (G[0+3*0+9*s]*qfx + G[1+3*0+9*s]*qfy + G[2+3*0+9*s]*qfz) - Qxoff;
            qfy_minus_Qyoff = (G[0+3*1+9*s]*qfx + G[1+3*1+9*s]*qfy + G[2+3*1+9*s]*qfz) - Qyoff;
            qfz_minus_Qzoff = (G[0+3*2+9*s]*qfx + G[1+3*2+9*s]*qfy + G[2+3*2+9*s]*qfz) - Qzoff;
            if  ((fabs(qfx_minus_Qxoff)<=dQ1[s]) && (fabs(qfy_minus_Qyoff)<=dQ2[s]) && (fabs(qfz_minus_Qzoff)<=dQ0[s]))
            {
                j = (int) (half_size * (one +  qfx_minus_Qxoff/dQ1[s]));
                i = (int) (half_size * (one +  qfy_minus_Qyoff/dQ2[s]));
                if ((0<=i<Slice_size) && (0<=j<Slice_size))
                {
                    pv = Data(x,y);
                    flag = (int)(pv!=0);
                    atomicAdd_float(&SlImage(s,i,j), pv);
                    atom_add(&SlMask(s,i,j), flag);
                }
            }
        }
    }
}

__kernel void volume_rotation(int num_vol,
                              __global const float *AllVolume,
                              __global const unsigned int *AllMask,
                              __global float *OutVolume,
                              __global unsigned int *OutMask,
                              int num_ops,
                              __global const float *ops_list
                              )
                              
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int NPTX = get_global_size(0);
    int NPTY = get_global_size(1);
    int NPTZ = get_global_size(2);
    unsigned int flag;
    float Xc, Yc, Zc, xx, yy, zz, intensity;
    int v, n, p, q, r;
    Xc = (NPTX-1)/2.0;
    Yc = (NPTY-1)/2.0;
    Zc = (NPTZ-1)/2.0;
    xx = x-Xc;
    yy = y-Yc;
    zz = z-Zc;
    for (v=0; v<num_vol; v++)
    {
        if (AllMask[z + y*NPTZ + x*NPTZ*NPTY + v*NPTZ*NPTY*NPTX])
        {
            for (n=0;n<num_ops;n++)
            {
                p = (int)(ops_list[0+3*0+9*n]*xx + ops_list[1+3*0+9*n]*yy + ops_list[2+3*0+9*n]*zz + Xc);
                q = (int)(ops_list[0+3*1+9*n]*xx + ops_list[1+3*1+9*n]*yy + ops_list[2+3*1+9*n]*zz + Yc);
                r = (int)(ops_list[0+3*2+9*n]*xx + ops_list[1+3*2+9*n]*yy + ops_list[2+3*2+9*n]*zz + Zc);
                if ((0<=p<NPTX) && (0<=q<NPTY) && (0<=r<NPTZ))
                {
                    intensity = AllVolume[z + y*NPTZ + x*NPTZ*NPTY + v*NPTZ*NPTY*NPTX];
                    flag = (int)(intensity != 0);
                    atomicAdd_float(&OutVolume[r + q*NPTZ + p*NPTZ*NPTY + v*NPTZ*NPTY*NPTX], intensity);
                    atom_add(&OutMask[r + q*NPTZ + p*NPTZ*NPTY + v*NPTZ*NPTY*NPTX], flag);
                }
            }
        }
    }
}