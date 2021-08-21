import pandas as pd
import os

gwtc2_events = pd.read_csv('../../data/O3a_events/gwtc2_events.csv')
mergers_gwtc2 = gwtc2_events['name']
odir = '../../data/PE_skymaps/gwtc-2'
if not os.path.exists(odir):
    os.makedirs(odir)
for i,merger in enumerate(mergers_gwtc2):
    indir='../../data/O3a_events/PE_skymaps/'
    fits1=indir+gwtc2_events['commonName'][i]+'_PublicationSamples.fits'
    fname1=odir+'/'+merger+'.npz'
    print(merger)
    os.system('python ../src/fits_to_cart.py --infile %s --outfile %s'%(fits1,fname1))
