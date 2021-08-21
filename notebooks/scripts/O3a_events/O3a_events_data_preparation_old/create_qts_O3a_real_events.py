import pandas as pd
import os
import numpy as np
import pycbc.types
import pycbc
import pylab
gwtc2_events = pd.read_csv('../../data/O3a_events/gwtc2_events.csv')
mergers_gwtc2 = gwtc2_events['name']

odir = '../../data/qts/gwtc-2'
if not os.path.exists(odir):
    os.makedirs(odir)
    os.makedirs(odir+'/H1')
    os.makedirs(odir+'/L1')
    os.makedirs(odir+'/V1')

q_msmall = (4,10)
q_mlarge = (3,7)

for i,merger in enumerate(mergers_gwtc2):
    
    indir='../../data/O3a_events/'+merger
    gps = gwtc2_events['GPS'][i]
    m1z = gwtc2_events['mass_1_source'][i]*(1+gwtc2_events['redshift'])[i]
    if m1z<60:
        q = q_msmall
    else:
        q = q_mlarge

    for det in ['H1','L1','V1'] :
        fname =indir+'/'+det+'.txt'
        if os.path.isfile(fname):
            times,strain = np.loadtxt(fname).T

            hdata_1=pycbc.types.TimeSeries(strain,delta_t=times[1]-times[0],epoch = times[0],dtype='float64',copy=True)
            hdata_whitened = hdata_1.whiten(8,2)* 1E-21

            times_h1, freqs_h1,power_h1 = hdata_whitened.qtransform(1/1024,logfsteps=100,qrange=q,frange = (15,1000))
            ## extra thing added to adjust the clim in presence of noise!
            st,end = np.where(times_h1 <= gps-0.2)[0][-1],np.where(times_h1 >= gps+0.1)[0][0]

            pylab.figure(figsize=[7,4])
            pylab.pcolormesh(times_h1[st:end],freqs_h1,\
                     power_h1[:,st:end]**0.5/np.max(power_h1[:,st:end]**0.5),cmap="viridis")
            pylab.xlim(gps-0.2,gps+0.1)
            pylab.yscale("log",basey=2)
            #ylab.colorbar()
            # pylab.clim(0,20)
            pylab.axis("off")
            pylab.margins(0,0)
            pylab.gca().xaxis.set_major_locator(pylab.NullLocator())
            pylab.gca().yaxis.set_major_locator(pylab.NullLocator())
            pylab.savefig(odir+ '/'+det+'/'+merger+'-whitened.png',bbox_inches="tight",pad_inches = 0)
            #pylab.show()
            pylab.close()

            times_h1, freqs_h1,power_h1 = hdata_1.qtransform(1/1024,logfsteps=100,qrange = q,frange = (15,1000))
            st,end = np.where(times_h1 <= gps-0.2)[0][-1],np.where(times_h1 >= gps+0.1)[0][0]

            pylab.figure(figsize=[7,4])
            pylab.pcolormesh(times_h1[st:end],freqs_h1,power_h1[:,st:end]**0.5/np.max(power_h1[:,st:end]**0.5),cmap="viridis")
            pylab.xlim(gps-0.2,gps+0.1)
            pylab.yscale("log",basey=2)
            #ylab.colorbar()
            # pylab.clim(0,20)
            pylab.axis("off")
            pylab.margins(0,0)
            pylab.gca().xaxis.set_major_locator(pylab.NullLocator())
            pylab.gca().yaxis.set_major_locator(pylab.NullLocator())
            pylab.savefig(odir+ '/'+det+'/'+merger+'.png',bbox_inches="tight",pad_inches = 0)
            pylab.close()
            pylab.figure()
            pylab.plot(hdata_1.sample_times,hdata_1,label='TS')
            pylab.plot(hdata_whitened.sample_times,hdata_whitened,label='TS whitened')
            pylab.xlim(gps-0.2,gps+0.1)
            pylab.axvline(gps,color='k',ls='dashed',alpha=0.5,label='gps')
            pylab.legend()
            pylab.savefig(indir+ '/'+det+'.png',bbox_inches="tight",pad_inches = 0)
        else:
            print(fname, 'does not exist')
