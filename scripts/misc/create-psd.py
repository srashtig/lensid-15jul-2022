import glob
import os, sys
from argparse import ArgumentParser

import numpy as np
from scipy.interpolate import interp1d

import lal
# Bug in lal series (or some fix due to bayestar)
from ligo.skymap import bayestar

parser = ArgumentParser(
    "Create a PSD in LIGOLW XML, reading in a flat file"
)
parser.add_argument("-i", "--input", default="",
    help="Input psd ascii file"
)
parser.add_argument("-d", "--input-dir", default="",
    help="Directory containing psd files. All of which will be added.")
parser.add_argument("--instrument", default=None,
                    help="Instrument name")
parser.add_argument(
    "--is-asd", action='store_true', default=False,
    help="Supply if asd instead of psd."
)
parser.add_argument("-o", "--output", required=True,
    help="Output table in LIGOLW XML format"
)
parser.add_argument("--f0", type=int, default=5,
    help="Start frequency, defaults to 5 Hz."
)
parser.add_argument("--fmax", type=int, default=5000,
    help="Stop frequency, defaults to 5000 Hz."
)
parser.add_argument("--df", type=int, default=1,
    help="Frequency steps, default to 1 Hz"
)

args = parser.parse_args()
def create_sampled_psd(psd_file):
    freq, asd = np.loadtxt(psd_file, unpack=True)
    psd = asd**2 if args.is_asd else asd
    # interpolant to sample PSD at uniform time steps
    interpolant = interp1d(freq, psd, kind='cubic',
                           bounds_error=False, fill_value=1e-40)
    sampled_psd = interpolant(np.arange(args.f0, args.fmax, args.df))
    series = lal.CreateREAL8FrequencySeries(
        instrument, args.f0, args.f0, args.df, lal.SecondUnit, len(sampled_psd)
    )
    series.data.data = sampled_psd
    return series


if (args.input and args.input_dir) or (args.input + args.input_dir == ""):
    print("Need either and input file or a directory storing PSD/ASD")
    sys.exit(1)

psd_dict = dict()
for ff in glob.glob(args.input_dir + "/*") if args.input_dir else (args.input,):
        instrument, *_ = os.path.splitext(os.path.basename(ff))
        psd_dict[args.instrument if args.instrument else instrument] = create_sampled_psd(ff)
print(psd_dict.keys())
# Write in LIGO LW XML
from glue.ligolw import utils as ligolw_utils
xmldoc = lal.series.make_psd_xmldoc(psd_dict)
ligolw_utils.write_filename(xmldoc, args.output, gz=False, verbose=True)
