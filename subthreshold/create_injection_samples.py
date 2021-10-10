#!/usr/bin/python
__author__ = "haris.k, apratim.ganguly"
# ----------------------------------------------------------------
import argparse
import sys
import csv
import os
import numpy as np
from cosmology_models import LCDM
from scipy import special
from random import shuffle
from scipy.interpolate import interp1d

# ----------------------------------------------------------------
lcdm = LCDM(0.3)

parser = argparse.ArgumentParser(
    description="Create a dat file with BBH injection samples with a given z and component masses distribution"
)
parser.add_argument(
    "-outfile",
    "--outfile",
    help="name of output file",
    required=True)
parser.add_argument(
    "-z_pdf_model",
    "--z_pdf_model",
    help="z_pdf_models: Dominik, Belczynski, PopIII and Primordial (see Liang Dai et al. PRD,2017 for details), Uniform, Rmin, Rmax",
    required=True,
)
parser.add_argument(
    "-z_max",
    "--z_max",
    help="readshift upper limit",
    type=float,
    required=True)
parser.add_argument(
    "-mass_pdf_model",
    "--mass_pdf_model",
    help="comp. mass_pdf_models: powerlaw1, powerlaw2, (see doi:10.3847/2041-8205/833/1/L1) ,schechter, lognormal, gaussian  (see Liang Dai et al. PRD,2017), powerpluspeak ",
    required=True,
)
parser.add_argument(
    "-f_ref",
    "--f_ref",
    help="f_ref",
    type=float,
    default=20.0)
parser.add_argument(
    "-n",
    "--n",
    type=int,
    help="number of samples",
    required=True)


args = parser.parse_args()
outfile = args.outfile
z_pdf_model = args.z_pdf_model
z_max = args.z_max
mass_pdf_model = args.mass_pdf_model
f_ref = args.f_ref
n = args.n


def Heaviside(x):
    """ Heaviside step function """
    return np.piecewise(x, [x < 0, x >= 0], [0, 1])


def m1m2_to_mchirpeta(mass1, mass2):
    eta = mass1 * mass2 / (mass1 + mass2) / (mass1 + mass2)
    mc = (mass1 + mass2) * np.power(eta, 3.0 / 5.0)
    return (mc, eta)


def mchirpeta_to_m1m2(mchirp, eta):
    mtot = mchirp * np.power(eta, -3.0 / 5)
    fac = np.sqrt(1.0 - 4.0 * eta)
    return (mtot * (1.0 + fac) / 2.0, mtot * (1.0 - fac) / 2.0)


def rejection_sample(pdf_bins, pdf_vals, xmin=None, xmax=None, n=1000):

    interp = interp1d(pdf_bins, pdf_vals, fill_value=0, bounds_error=False)
    evaluated_bins = np.linspace(
        np.amin(pdf_bins) * 0.95, np.amax(pdf_bins) * 1.05, 10000
    )
    evaluated_interp = interp(evaluated_bins)
    pdf = interp1d(
        evaluated_bins,
        evaluated_interp / np.amax(evaluated_interp) * 0.95,
        bounds_error=False,
    )

    if xmin is None:
        xmin = evaluated_bins.min()
    if xmax is None:
        xmax = evaluated_bins.max()
    x = np.linspace(xmin, xmax, 1000)
    y = pdf(x)
    pmin = 0.0
    pmax = y.max()

    # Counters
    naccept = 0
    ntrial = 0

    # Keeps generating numbers until we achieve the desired n
    ran = np.zeros(n)  # output list of random numbers
    while naccept < n:
        # print(ntrial)
        x = np.random.uniform(xmin, xmax, 100000)  # x'
        y = np.random.uniform(pmin, pmax, 100000)  # y'
        indices = np.where(y < pdf(x))
        arr = x[indices]
        # print(arr)
        if len(arr) > n - naccept:
            num = n - naccept
        else:
            num = len(arr)
        ran[naccept: naccept + num] = arr[0:num]
        naccept = naccept + num
        ntrial += 1
    return ran


def schechter_mass_distribution(z, m_lower=5.0, gamma_z=6.0):
    """
        The function return Schechter source frame probability distribution
        of mass ms for given a redshift z.
        Ref: Eq.(21), Liang Dai et al. PRD(2017)
        """
    m = np.linspace(m_lower, 200.0, 5000)
    m_prime = 3 * np.power((1 + z) / 2.5, 0.5)
    pdf = (
        Heaviside(m - m_lower)
        / (m_prime * special.gamma(1 + gamma_z))
        * np.power((m - m_lower) / m_prime, gamma_z)
        * np.exp(-(m - m_lower) / m_prime)
    )
    norm = np.sum(pdf) * (m[1] - m[0])
    return m, pdf / norm


def sample_alpha(n):
    u = np.random.rand(n)
    alpha_min = 0.01
    alpha_max = 100
    k1 = 1.0 - np.power(alpha_min, 2)
    k2 = 1.0 - 1.0 / np.power(alpha_max, 2)
    idx1 = np.where(u < 0.5)[0]
    idx2 = np.where(u >= 0.5)[0]
    alpha_samples = np.zeros(len(u))
    alpha_samples[idx1] = np.sqrt(2 * k1 * u[idx1] + alpha_min ** 2)
    alpha_samples[idx2] = 1.0 / np.sqrt(1 - k2 * (2 * u[idx2] - 1))
    return alpha_samples


z_pdf_data = np.genfromtxt("../../data/z_PDF.dat", names=True)
z = z_pdf_data["z"]
delta_z = z[1] - z[0]
z_pdf = z_pdf_data[z_pdf_model]
idx = idx = (np.abs(z - z_max)).argmin() + 1
z = z[:idx]
z_pdf_nomr = np.sum(z_pdf[:idx]) * delta_z
z_pdf = z_pdf[:idx] / z_pdf_nomr

n_buffer = 2 * n
z_samples = []
if mass_pdf_model == "schechter":
    m1s_samples = []
    m2s_samples = []
    for ii in range(len(z)):
        n_zbin = int(n_buffer * z_pdf[ii] * delta_z)
        z_samples = np.append(
            z_samples, z[ii] - 0.5 * delta_z + delta_z * np.random.rand(n_zbin)
        )
        ms, pdf_ms = schechter_mass_distribution(z[ii])
        delta_ms = ms[1] - ms[0]
        m1s_samples = np.append(
            m1s_samples, np.random.choice(ms, size=n_zbin, p=pdf_ms * delta_ms)
        )
        m2s_samples = np.append(
            m2s_samples, np.random.choice(ms, size=n_zbin, p=pdf_ms * delta_ms)
        )
    index = np.arange(len(z_samples))
    np.random.shuffle(index)
    z_samples = z_samples[index[:n]]
    m1s_samples = m1s_samples[index[:n]]
    m2s_samples = m2s_samples[index[:n]]
else:
    for ii in range(len(z)):
        n_zbin = int(n_buffer * z_pdf[ii] * delta_z)
        z_samples = np.append(
            z_samples, z[ii] - 0.5 * delta_z + delta_z * np.random.rand(n_zbin)
        )
    index = np.arange(len(z_samples))
    np.random.shuffle(index)
    z_samples = z_samples[index[:n]]
    if mass_pdf_model == "lognormal":
        temp1 = np.random.normal(np.log(25.0), 0.18, n)
        temp2 = np.random.normal(np.log(25.0), 0.18, n)
        m1s_samples, m2s_samples = np.exp(temp1), np.exp(temp2)
    elif mass_pdf_model == "gaussian":
        m1s_samples = np.random.normal(8.0, 1.5, 2 * n)
        index = np.where(m1s_samples >= 5)
        m1s_samples = m1s_samples[index][:n]
        m2s_samples = np.random.normal(8.0, 1.5, 2 * n)
        index = np.where(m2s_samples >= 5)
        m2s_samples = m2s_samples[index][:n]
    elif mass_pdf_model == "powerlaw1":
        mmin = 10.0
        mmax = 50.0
        u1 = np.random.rand(4 * n)
        u2 = np.random.rand(4 * n)
        m1s_samples = np.exp(u1 * (np.log(mmax / mmin)) + np.log(mmin))
        m2s_samples = np.exp(u2 * (np.log(mmax / mmin)) + np.log(mmin))
        index = np.where(m1s_samples + m2s_samples <= 100)
        m1s_samples = m1s_samples[index][:n]
        m2s_samples = m2s_samples[index][:n]
    elif mass_pdf_model == "powerlaw2":
        mmin = 10.0
        mmax = 50.0
        u1 = np.random.rand(4 * n)
        qmin = 10.0 / 50.0
        q = np.random.rand(4 * n) * (1 - qmin) + qmin
        c = 1.35 / (np.power(mmin, -1.35) - np.power(mmax, -1.35))
        m1s_samples = 1 / \
            np.power(np.power(mmin, -1.35) - 1.35 * u1 / c, 1 / 1.35)
        m2s_samples = q * m1s_samples
        index = np.where(m1s_samples + m2s_samples <= 100)
        m1s_samples = m1s_samples[index]
        m2s_samples = m2s_samples[index]
        index = np.where(m2s_samples > 10)
        m1s_samples = m1s_samples[index][:n]
        m2s_samples = m2s_samples[index][:n]
    elif mass_pdf_model == "powerpluspeak":
        mmin = 10.0
        mmax = 50.0
        qmin = 10.0 / 50.0
        q = np.random.rand(4 * n) * (1 - qmin) + qmin
        pdf_bins, pdf_vals = np.loadtxt(
            "../../data/power_law_plus_peak_m1_source_gwtc2.txt", unpack=True
        )
        m1s_samples = rejection_sample(pdf_bins, pdf_vals, mmin, mmax, n=4 * n)
        m2s_samples = q * m1s_samples
        index = np.where(m1s_samples + m2s_samples <= 100)
        m1s_samples = m1s_samples[index]
        m2s_samples = m2s_samples[index]
        index = np.where(m2s_samples > 10)
        m1s_samples = m1s_samples[index][:n]
        m2s_samples = m2s_samples[index][:n]

m1z_samples = m1s_samples * (1 + z_samples)
m2z_samples = m2s_samples * (1 + z_samples)

ldistance = np.zeros(len(z_samples))
for ii in range(len(z_samples)):
    ldistance[ii] = lcdm.luminosity_distance_z(z_samples[ii])

a1_samples = np.random.rand(n) * 0.9
a2_samples = np.random.rand(n) * 0.9
tilt_1_samples = np.arccos(np.random.rand(n) * 2 - 1)
tilt_2_samples = np.arccos(np.random.rand(n) * 2 - 1)
theta_jn_samples = np.arccos(np.random.rand(n) * 2 - 1)
phi_jl_samples = np.random.rand(n) * 2 * np.pi
phi_12_samples = np.random.rand(n) * 2 * np.pi
f_ref_samples = np.zeros(n) + f_ref

ra_samples = np.random.rand(n) * 2 * np.pi
dec_samples = np.arcsin(np.random.rand(n) * 2 - 1)
phase_samples = np.random.rand(n) * 2 * np.pi
psi_samples = np.random.rand(n) * 2 * np.pi
t_samples = np.random.rand(n) * 24 * 60.0 * 60 + 1249852257.0  # GW190814


with open(outfile + "_%s_%s.dat" % (mass_pdf_model, z_pdf_model), "w") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(
        ["z\tldistance\tm1z\tm2z\ta1\ta2\ttilt_1\ttilt_2\ttheta_jn\tphi_jl\tphi_12\tphase\tra\tdec\tpsi\tf_ref\tt0"])
    writer.writerows(
        zip(
            z_samples,
            ldistance,
            m1z_samples,
            m2z_samples,
            a1_samples,
            a2_samples,
            tilt_1_samples,
            tilt_2_samples,
            theta_jn_samples,
            phi_jl_samples,
            phi_12_samples,
            phase_samples,
            ra_samples,
            dec_samples,
            psi_samples,
            f_ref_samples,
            t_samples,
        )
    )
