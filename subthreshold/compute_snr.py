__author__ = "haris.k"

# """
# This is a script to compute snr using pycbc modules
# Last modified on 2020-03-07
# """
import bilby.gw.conversion as conversion
import bilby
import csv
from pycbc.waveform.generator import FDomainCBCGenerator
from pycbc.waveform.generator import FDomainDetFrameGenerator
from pycbc import waveform, filter, psd
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="This is stand alone code for computing snr for injection with O3a PSD"
)
parser.add_argument("-ifile", "--ifile", help="Input dat file name")
parser.add_argument(
    "-network",
    "--network",
    help="Input dat file name: LHV or LH",
    default="LHV")

args = parser.parse_args()
ifile = args.ifile
network = args.network

samples = np.genfromtxt(ifile, names=True)
m1z, m2z, ldist = samples["m1z"], samples["m2z"], samples["ldistance"]
ra_samples, dec_samples = samples["ra"], samples["dec"]
a_1, a_2, tilt_1, tilt_2, phi_jl, phi_12 = (
    samples["a1"],
    samples["a2"],
    samples["tilt_1"],
    samples["tilt_2"],
    samples["phi_jl"],
    samples["phi_12"],
)
theta_jn, psi, phase, fref = (
    samples["theta_jn"],
    samples["psi"],
    samples["phase"],
    samples["f_ref"],
)
t0 = samples["t0"]

iota, s1x, s1y, s1z, s2x, s2y, s2z = conversion.transform_precessing_spins(
    theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, m1z, m2z, fref, phase
)
# s1x =  0*0.261 * np.ones(len(m1z))
# s1y =  0*0.797 * np.ones(len(m1z))
# s1z =  0.011*np.ones(len(m1z))#0.356 * np.ones(len(m1z))
# s2x = 0*-0.375 *np.ones(len(m1z))
# s2y =  0*0.257 *np.ones(len(m1z))
# s2z =  0.211*np.ones(len(m1z))#0.005 *np.ones(len(m1z))
# iota = 0.922*np.ones(len(m1z))#1.369 *np.ones(len(m1z))

delta_f = 1.0 / 64
f_lower = 20.0
f_high = 3000.0
psd_length = int(2000.0 / delta_f)
psd_dir = "/home/haris.k/runs/lensing/O3_injections/data/O3a_representative_psd"
h1_asd_data = np.loadtxt(
    "%s/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt" %
    psd_dir)
l1_asd_data = np.loadtxt(
    "%s/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt" %
    psd_dir)
v1_asd_data = np.loadtxt("%s/O3-Virgo_sensitivity_asd.txt" % psd_dir)

o3a_psd = {}
o3a_psd["H1"] = psd.read.from_numpy_arrays(h1_asd_data[:, 0], np.power(
    h1_asd_data[:, 1], 2), psd_length, delta_f, 18.0)
o3a_psd["L1"] = psd.read.from_numpy_arrays(l1_asd_data[:, 0], np.power(
    l1_asd_data[:, 1], 2), psd_length, delta_f, 18.0)
o3a_psd["V1"] = psd.read.from_numpy_arrays(v1_asd_data[:, 0], np.power(
    v1_asd_data[:, 1], 2), psd_length, delta_f, 18.0)

net = []
for det in network:
    net.append(det + "1")
print(("Network:{}".format(net)))

generator = FDomainDetFrameGenerator(
    FDomainCBCGenerator,
    0.0,
    variable_args=[
        "mass1",
        "mass2",
        "spin1x",
        "spin1y",
        "spin1z",
        "spin2x",
        "spin2y",
        "spin2z",
        "f_ref",
        "tc",
        "ra",
        "dec",
        "inclination",
        "polarization",
        "distance",
    ],
    detectors=net,
    delta_f=delta_f,
    f_lower=f_lower,
    approximant="IMRPhenomPv2",
)

with open("%s_%s_withsnr.dat" % (ifile.split(".")[0], network), "a") as file:
    file.write(
        "ldistance\tm1z\tm2z\tra\tdec\tiota\tpsi\tt0\ts1x\ts1y\ts1z\ts2x\ts2y\ts2z\ttheta_jn\tphi_jl\ttilt_1\ttilt_2\tphi_12\ta_1\ta_2\tphase\tf_ref\tsnr\n"
    )

snr = np.zeros(len(ldist))
for ii in range(len(ldist)):
    data = generator.generate(
        mass1=m1z[ii],
        mass2=m2z[ii],
        spin1x=s1x[ii],
        spin1y=s1y[ii],
        spin1z=s1z[ii],
        spin2x=s2x[ii],
        spin2y=s2y[ii],
        spin2z=s2z[ii],
        tc=t0[ii],
        ra=ra_samples[ii],
        dec=dec_samples[ii],
        inclination=iota[ii],
        polarization=psi[ii],
        distance=ldist[ii],
        f_ref=fref[ii],
    )
    snrsq = 0

    for det in net:
        snrsq = snrsq + filter.matchedfilter.sigmasq(
            data[det],
            psd=o3a_psd[det],
            low_frequency_cutoff=20.0,
            high_frequency_cutoff=1500,
        )
    snr[ii] = np.sqrt(snrsq)
    with open("%s_%s_withsnr.dat" % (ifile.split(".")[0], network), "a") as file:
        file.write(
            "%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" %
            (ldist[ii],
             m1z[ii],
                m2z[ii],
                ra_samples[ii],
                dec_samples[ii],
                iota[ii],
                psi[ii],
                t0[ii],
                s1x[ii],
                s1y[ii],
                s1z[ii],
                s2x[ii],
                s2y[ii],
                s2z[ii],
                theta_jn[ii],
                phi_jl[ii],
                tilt_1[ii],
                tilt_2[ii],
                phi_12[ii],
                a_1[ii],
                a_2[ii],
                phase[ii],
                fref[ii],
                snr[ii],
             ))
