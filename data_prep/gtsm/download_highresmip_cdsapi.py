#!/usr/bin/env python
import cdsapi
import numpy as np

model = 'hadgem3_gc31_hm'
years_hist = np.arange(1951,2015)
years_fut = np.arange(2016,2051)
experiments = ['historical','future']

client = cdsapi.Client()
for experiment in experiments:
    if experiment=='historical':
        years = years_hist
    else:
        years = years_fut
   
    for year in years:
            
        dataset = 'sis-water-level-change-timeseries-cmip6'
        request=    {
                'variable': 'storm_surge_residual',
                'experiment': experiment,
                 'model': model,
                'temporal_aggregation':'10_min',
                'year': str(year),
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
               # 'format': 'zip',
            }
        target = 'surge_'+str(year)+'.zip'
        

        
        client.retrieve(dataset, request, target).download()