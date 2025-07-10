#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:35:38 2023

@author: timhermans
"""

import pandas as pd
import numpy as np
solver = 't_tide'
file = 'vigo-vigo-esp-ieo.csv'


surge = pd.read_csv('/Users/timhermans/Documents/PostDoc/Phase3_statistical_surges/Analysis/detide_GESLA3/detided_GESLA3_hourly/t_tide_results/detided_GESLA3_hourly/'+file)
ticon = pd.read_csv('/Users/timhermans/Documents/PostDoc/Phase3_statistical_surges/Analysis/detide_GESLA3/TICON/TICON.txt',sep='\t')

lat = surge.lat.values[0]
lon = surge.lon.values[0]

ticon = pd.concat([ticon .columns.to_frame().T, ticon], ignore_index=True)
ticon.columns = ['lat','lon','const','ampl','phase','ampl_std','phase_std','pctMissing','numObs','lenTempGap','date_start','date_end','record_source']
ticon['const'] = [k.replace(' ','') for k in ticon['const']]
ticon['ampl'] = ticon['ampl'].astype('float')/100 #cm to m

ticon_site = ticon.iloc[np.where((np.abs(ticon.lat.astype('float') - lat) <.1) & (np.abs(ticon.lon.astype('float')-lon) <.1))[0]]


ampl_est = pd.read_csv('/Users/timhermans/Documents/PostDoc/Phase3_statistical_surges/Analysis/detide_GESLA3/detided_GESLA3_hourly/t_tide_results/constituents_hourly/amplitude/'+file)
phase_est = pd.read_csv('/Users/timhermans/Documents/PostDoc/Phase3_statistical_surges/Analysis/detide_GESLA3/detided_GESLA3_hourly/t_tide_results/constituents_hourly/phase/'+file)

ampl_est.columns = [k.replace(' ','') for k in ampl_est.columns]

for k in ticon_site.const:
    if k in ampl_est.columns:
        print('consituent: '+k)
        print('a')
        print('ticon '+str(ticon_site[ticon_site.const==k].ampl.values[0]))
        print(solver+' '+str(np.nanmean(ampl_est[k])))
        print('phase')
        print('ticon '+str(ticon_site[ticon_site.const==k].phase.values[0]))
        print(solver+' '+str(np.nanmean(phase_est[k])))
