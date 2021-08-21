import pandas as pd
import os

gwtc2_events = pd.read_csv('../../data/O3a_events/gwtc2_events.csv')
mergers_gwtc2 = gwtc2_events['name']
odir = '../../data/bayestar_skymaps/gwtc-2'
if not os.path.exists(odir):
    os.makedirs(odir)
for i,merger in enumerate(mergers_gwtc2):
    indir='../../data/O3a_events/'+merger
    fits1=indir+'/bayestar.fits.gz'
    fits2=indir+'/bayestar.fits'
    fname1=odir+'/'+merger+'.npz'
    print(merger)
    os.system('python ../src/fits_to_cart.py --infile %s --outfile %s'%(fits2,fname1))
    os.system('python ../src/fits_to_cart.py --infile %s --outfile %s'%(fits1,fname1))
