#!/usr/bin/env python
import cdsapi
import numpy as np

c = cdsapi.Client()

for year in np.arange(2003,2018):
    for month in ['01', '02', '03','04', '05', '06','07', '08', '09','10', '11', '12']:

        c.retrieve(
            'sis-water-level-change-timeseries',
            {
                'variable': 'total_water_level',#'storm_surge_residual',
                'experiment': 'era5_reanalysis',
                'year': str(year),
                'month': month,
                'format': 'zip',
            },
            'codec_era5_'+str(year)+'_'+str(month)+'.zip')