import matplotlib.pylab as plt
import numpy as np
import healpy as hp
from ligo.skymap.io import fits
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='This is stand alone code for converting .fits skymaps to .npz')

    parser.add_argument(
        '-infile',
        '--infile',
        help='input .fits file ',
        required=True)
    parser.add_argument(
        '-outfile',
        '--outfile',
        help='out .npz file',
        required=True)
    parser.add_argument(
        '-force',
        '--force',
        help='overwrite the output .npz file if exists already..(0/1)',
        default=0)

    args = parser.parse_args()
    filename1 = args.infile
    out_filename1 = args.outfile
    force = args.force
    _main(filename1, out_filename1, force)


def _main(filename1, out_filename1, force=0):
    if (os.path.isfile(filename1) == 1):
        # print(out_filename1)
        if (os.path.isfile(out_filename1) != 1) or (force == 1):
            m = fits.read_sky_map(filename1)
            plt.figure(1)
            out = hp.cartview(
                (m[0]), cbar=False, title=None, fig=1, return_projected_map=True, rot=(
                    0, -180, 180))
            data = np.ma.getdata(out)
            np.savez(out_filename1, data=data)
            plt.close()
    else:
        print('input fits file: %s not found' % filename1)


if __name__ == '__main__':
    main()
