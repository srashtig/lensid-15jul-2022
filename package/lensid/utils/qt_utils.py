import pylab
import pycbc.noise
import pycbc.psd
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt


def inj_psds_HLV(psd_mode="analytical", sample_rate= 2**12 asd_dir=None):
    """
    Generates power spectral densities for the H,L,V detectors.

    Parameters:
        psd_mode='analytical'(default, advanced Ligo, Virgo PSDs ) or 'load'.
        asd_dir = path to directory in which psd files exist. H1.txt, L1.txt,V1.txt
        sample_rate(float) = sample rate of the signal. Default: 4096
    Returns:
        three numpy arrays: psd_H, psd_L, psd_V

    """

    if psd_mode == "analytical":
        flow = 0.0
        delta_f = 1.0 / 16        
        flen = int(sample_rate / (2 * delta_f)) + 1
        psd_H = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
        psd_L = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
        psd_V = pycbc.psd.analytical.AdvVirgo(flen, delta_f, flow)

    elif psd_mode == "load":
        filename_H = asd_dir+"/H1.txt"
        filename_L = asd_dir+"/L1.txt"
        filename_V = asd_dir+"/V1.txt"
        psd_H = pycbc.types.load_frequencyseries(filename_H) ** 2
        psd_L = pycbc.types.load_frequencyseries(filename_L) ** 2
        psd_V = pycbc.types.load_frequencyseries(filename_V) ** 2
    return psd_H, psd_L, psd_V


q_msmall = (4, 10)
q_mlarge = (3, 7)
q_wide = (3, 30)


def inject_noise_signal_custom(signal, psd, duration=128, whitened=False, seed=None):
    """
    Adds gaussian noise to a given signal using a given PSD. Optionally whitens the signal.

    Parameters:

        signal: input pycbc types timeseries.
        
        psd: powers spectral density to generate noise realisation from.
        
        duration(int): duration of the output noise signal, only if
            whitened = True, default =128s.
        whitened(bool): whiten the signal True/False(default).
        
        seed(int): random seed for the noise realisation, default: None.

    Returns:
        pycbc timeseries: noise_signal of 8s.

    """
    delta_t = 1.0 / signal.sample_rate
    tsamples = int(duration / delta_t)

    ts = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=seed)
    ts.start_time += signal.end_time - duration / 2
    noise_signal = ts.add_into(signal)
    if whitened == True:

        noise_signal_whitened = (
            noise_signal.whiten(4,4) * 1e-21
        )
        return noise_signal_whitened.time_slice(
            signal.end_time - 6, signal.end_time + 2
        )
    else:
        return noise_signal.time_slice(
            signal.end_time - 6, signal.end_time + 2
        )


def plot_qt_from_ts(
    noise_signal, t_gps, qrange, outfname="test", save=True, fhigh=1000,normalised = True
):

    """
    Plots and saves the nomalized Qtransform of the given signal.

    Parameters:
        signal (pycbc Timeseries): input time domain signal of the event.
        
        t_gps(int): GPS time of the event in seconds. Qtransform will be 
            plotted around (t_gps-0.2,t_gps+0.1).
            
        qrange(tuple(qmin,qmax)): input to the qtransform function in pycbc.
        
        outfname(str): output filename, saved as png. Default: 'test'.
        
        save(bool): Save the Qtransform plot, else it will show the figure.
        
        fhigh(int): High frequency cut for the Qtransform plot.
        
        normalised: Divide power in each pixel by maximum power. Default : True

    Returns:
        float: maximum power in the Qtransform in the window 
            (t_gps-0.2,t_gps+0.1).

    """
    times, freqs, power = noise_signal.qtransform(
        1 / 1024, logfsteps=100, qrange=qrange, frange=(15, fhigh)
    )
    st, end = (
        np.where(times <= t_gps - 0.2)[0][-1],
        np.where(times >= t_gps + 0.1)[0][0],
    )

    pylab.figure(figsize=[7, 4])
    if normalised == True: 
        pylab.pcolormesh(
            times[st:end],
            freqs,
            power[:, st:end] ** 0.5 / np.max(power[:, st:end] ** 0.5),
            cmap="viridis",
        )  
    else:
          pylab.pcolormesh(
            times[st:end],
            freqs,
            power[:, st:end] ** 0.5,
            cmap="viridis",
        )  
    pylab.xlim(t_gps - 0.2, t_gps + 0.1)
    pylab.yscale("log", basey=2)
    if save == True:
        pylab.axis("off")
        pylab.margins(0, 0)
        pylab.gca().xaxis.set_major_locator(pylab.NullLocator())
        pylab.gca().yaxis.set_major_locator(pylab.NullLocator())
        pylab.savefig(outfname + ".png", bbox_inches="tight", pad_inches=0)
        pylab.close()
    else:
        pylab.show()
    return np.max(power[:, st:end] ** 0.5)

def plot_ts(timseseries_arr,labels_arr, gps, outfname='test',title=''):
    """ Plot and save the strain timeseries centered around time (gps)"""
    
    plt.figure(figsize=(20,5))
    for i,ts in enumerate(timseseries_arr):
        plt.plot(ts.sample_times, ts.data, label = labels_arr[i])

    plt.xlim(gps-0.1,gps+0.05)
    plt.axvline(gps,color='k',ls='dashed',alpha=0.5,label='gps')
    plt.legend()
    plt.title(title)
    plt.savefig(outfname+'.png')
    plt.close()