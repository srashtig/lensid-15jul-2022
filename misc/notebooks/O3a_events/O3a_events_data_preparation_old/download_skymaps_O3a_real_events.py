import pandas as pd
import os
from ligo.gracedb.rest import GraceDb

sg = GraceDb()  # put your credentials here


gwtc2_events = pd.read_csv('../../data/O3a_events/gwtc2_events.csv')
mergers_gwtc2 = gwtc2_events['name']
for i, merger in enumerate(mergers_gwtc2):
    odir = '../../data/O3a_events/' + merger
    id = gwtc2_events['superevent'][i]  # graceid
    try:
        r = sg.files(id, "bayestar.fits.gz")
        outfile = open(odir + '/' + "bayestar.fits.gz", "wb")
        outfile.write(r.read())
        outfile.close()
    except BaseException:
        print(merger + ' not found, grace id: ' + id)

  #  os.system('cd %s && { curl -O https://gracedb.ligo.org/api/superevents/%s/files/bayestar.fits ; cd - ; }'%(odir,gwtc2_events['graceid'][i]))
   # os.system('cd %s && { curl -O https://gracedb.ligo.org/api/superevents/%s/files/bayestar.fits,0 ; cd - ; }'%(odir,gwtc2_events['graceid'][i]))
