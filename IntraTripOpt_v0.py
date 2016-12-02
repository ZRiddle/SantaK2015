# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 01:11:51 2015

@author: zach.riddle
"""

from Fine_Tuning_v9 import *
import numpy as np
import pandas as pd

# Read in data
Santa = Trips(path='_v11_greed')
total_wrw_start = Santa.wrw

# Loop through all trips
for tripid in Santa.trip_metrics.index:
    # Optimize sorting
    wrw_start = Santa.trip_metrics.ix[tripid].WRW
    Santa.T[tripid] = intra_trip_swap(Santa.T[tripid].copy())
    Santa.update_trip_metrics(tripid)
    gains = wrw_start - Santa.trip_metrics.ix[tripid].WRW
    if gains:
        print 'Trip #'+str(tripid)+' - WRW gain = %.0f'%(gains)

print '%.0f - Starting WRW\n%.0f - Ending WRW'%(total_wrw_start,Santa.wrw)

Santa.save_data(path='_v11_greed')



# Need to take group of presents and decide on longitude cuts vs latitude cuts

