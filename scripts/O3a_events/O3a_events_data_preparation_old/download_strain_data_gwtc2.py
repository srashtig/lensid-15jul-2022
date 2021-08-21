from ligo.gracedb.rest import GraceDb
from pycbc import catalog
import os
import pandas as pd
import numpy as np
mergers_gwtc2=list(catalog.Catalog(source='gwtc-2').mergers.keys())

sg = GraceDb(cred='/tmp/x509up_p2107792.file2UMIf5.1')
O3a_superevent=pd.read_csv('event-superevent.csv',header=None)

keys = ['commonName','GPS', 'chi_eff',  'chirp_mass',  'chirp_mass_source',  'far',  'far_unit',  'mass_1_source',  'mass_2_source', 'network_matched_filter_snr', 'redshift', 'version']

info_gwtc2=pd.DataFrame(columns=keys)
info_gwtc2['name'] = mergers_gwtc2
info_gwtc2['graceid'] = ''
info_gwtc2['superevent'] = ''

#info_gwtc2=info_gwtc2.set_index('name')
for i,merger in enumerate(mergers_gwtc2):
    odir='../../data/O3a_events/'+merger
    if not os.path.exists(odir):
        os.makedirs(odir)
    m = catalog.Merger(merger,source='gwtc-2')
    for det in ['H1', 'L1','V1']:
        if os.path.isfile(odir+'/'+ det + '.txt') ==False:
            try:
                m.strain(det).save(odir+'/'+ det + '.txt')
            except:
                print(merger , det)

    for key in keys:
        info_gwtc2[key][i] = m.data[key]
    
    idx=np.where(O3a_superevent[0]==m.common_name)[0][0]
    info_gwtc2['superevent'][i] = O3a_superevent[1][idx]
    print(info_gwtc2['superevent'][i])
    trange = str(info_gwtc2['GPS'][i]-0.5) +'..' +str(info_gwtc2['GPS'][i]+0.5)
    for s in sg.superevents(query= trange ,columns=['superevent_id']):
        sup_id = s['superevent_id']
        print(i,merger , sup_id)
        if sup_id is not None:
            info_gwtc2['graceid'][i] = str(sup_id)
info_gwtc2.to_csv('../../data/O3a_events/gwtc2_events.csv')