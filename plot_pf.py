import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import fabio, os

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


x = np.linspace(-1,1,300)
y = np.linspace(-1,1,300)
x,y = np.meshgrid(x,y)

fn = ["PtSi/PF_PtSi_10_subtracted_bg.cbf", "PtSi/PF_PtSi_13_subtracted_bg.cbf", "PtSi/PF_PtSi_14_subtracted_bg.cbf", "PtSi/PF_PtSi_15_subtracted_bg.cbf", "PtSi/PF_Si111_subtracted_bg.cbf"]

for f in fn:
    fig,ax = plt.subplots(1,1)
    img=fabio.open(f).data
    lv = np.linspace(3,4.5,50)
    p  = ax.contourf(x,y,np.log10(img), lv)
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.colorbar(p, label="log10(Intensity)", format="%.2f")
    plt.savefig(os.path.splitext(f)[0]+"_jet.png")
    plt.close()