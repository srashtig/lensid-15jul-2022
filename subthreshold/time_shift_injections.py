import argparse

__author__ = "haris.k"

import argparse

# Here we shift the injections in days, thus we can use same injection files for various observation windows
# The input injections are distributed in one day. By shifting them in days won't change time delays, antenna response functions
import numpy as np, csv

parser = argparse.ArgumentParser(
    "Script to shift the injections in days, thus we can use same injection files for various observation windows. The input injections are distributed in one day. By shifting them in days will not change time delays, antenna response functions."
)
parser.add_argument(
    "-i_file",
    "--i_file",
    type=str,
    help="Input injection file with injections distributed in a day",
    required=True,
)
parser.add_argument(
    "-obs_time",
    "--obs_time",
    type=float,
    help="Observation tiome window in which injections are supposed to be redistributed(in sec)",
    default=15811200.0,
)

args = parser.parse_args()
i_file = args.i_file
T = args.obs_time

data = np.genfromtxt(i_file, names=True)
day = 60 * 60 * 24.0
extra_days = int(T / day)
t_ext = np.random.randint(extra_days, size=len(data["t0"])) * day
t_shifted = data["t0"] + t_ext
out_file = i_file.split(".dat")[0] + "_withshiftedtime.dat"
with open(out_file, "w") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(
        [
            "ldistance\tm1z\tm2z\tra\tdec\tiota\tpsi\tt0\tt0_shifted\ts1x\ts1y\ts1z\ts2x\ts2y\ts2z\ttheta_jn\tphi_jl\ttilt_1\ttilt_2\tphi_12\ta_1\ta_2\tphase\tf_ref\tsnr"
        ]
    )
    writer.writerows(
        zip(
            data["ldistance"],
            data["m1z"],
            data["m2z"],
            data["ra"],
            data["dec"],
            data["iota"],
            data["psi"],
            data["t0"],
            t_shifted,
            data["s1x"],
            data["s1y"],
            data["s1z"],
            data["s2x"],
            data["s2y"],
            data["s2z"],
            data["theta_jn"],
            data["phi_jl"],
            data["tilt_1"],
            data["tilt_2"],
            data["phi_12"],
            data["a_1"],
            data["a_2"],
            data["phase"],
            data["f_ref"],
            data["snr"],
        )
    )
