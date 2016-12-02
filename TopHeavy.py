# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 18:27:51 2015

@author: zach.riddle
"""

from Fine_Tuning_v9 import *
from util import *
import numpy as np
import pandas as pd


# Read in gifts
with open('gifts.pkl','r') as f:
    gifts = pickle.load(f)

gifts.drop(['TripId','Cluster'],axis=1,inplace=True)

# Plot Weight distribution
grp = gifts[['Weight','LB']].round(0)
grp.groupby('Weight').count().plot(kind='bar')
gifts['Distance'] = gifts.LB

'''
Alg -
    Take all gifts over 49w
    Initialize 1500 trips
    Loop:    
        Start with heaviest remaining presents
        Add to trips in an 'optimal' way
        


'''
def wrw_trip(g):
    wrw = (g.Distance.cumsum()*g.Weight).sum()
    wrw += (g.Distance.sum()+g.iloc[-1].LB)*10
    return wrw
    
    
# Small test
# Slice at Africa
mask = np.logical_and(gifts.Longitude > 25, gifts.Longitude < 26)
mask = np.logical_and(gifts.Latitude > -60, mask)
print mask.sum()
test = gifts[mask].copy()
print gifts[mask].Weight.sum()

# Weight = 8747
# Set Trip Num = 9
n_trips = 9


###########################################################
###########################################################
# First test old way
# Get total weight
n_trips = 9
w_total = test.Weight.sum()

# Split into 2 trips - by weight
weights = test.sort_values('Longitude').Weight.cumsum()/w_total

step = 1.000000001/n_trips
cut = 0
g = []
while cut < 1:
    mask = np.logical_and(weights > cut,weights <= cut+step)
    # copy subset
    g1 = test[mask].sort_values('Latitude',ascending=False)
    # Rebuild Haversine Distances
    d = haversine_vect(g1.iloc[:-1],g1.iloc[1:])
    g1.iloc[1:,4] = d
    g1.iloc[0,4] = g1.iloc[0,3]
    # Add to list
    g.append(g1)
    cut += step

# Simple Cluster WRW
wrw_cluster = 0
for tr in g:
    wrw_cluster += wrw_trip(tr)

# WRW vs Distance
plt.figure(1)
N = np.arange(g[0].shape[0])
plt.plot(N,g[0].Distance.cumsum() - g[0].LB)
plt.plot(N,(g[0].Distance.cumsum() - g[0].LB)*g[0].Weight)

###########################################################
###########################################################
# 25 Presents > 48w

plt.figure('Test Trips',figsize = (12,9))
plt.scatter(test['Longitude'], test['Latitude'], s=test['Weight'], color='gray')
for k in range(15):
    plt.axvline(25+k/10.)
plt.xlim(24.95,26.05)
plt.ylim(-50,90)

heavy = test.Weight > 47
plt.scatter(test[heavy]['Longitude'], test[heavy]['Latitude'], s=test[heavy]['Weight'], color='red')


# Split a trip in 2 on a present?
#trip = test.sort_values('Latitude',ascending=False)
trip = g[0].copy()
opt = np.zeros((trip.shape[0],2))
opt[:,0] = (trip.Weight.sum()-trip.Weight.cumsum()) * (trip.Distance.cumsum()-trip.LB)
opt[1:,1] = np.array(-2*10*trip.LB)[:-1]

plt.figure('Optimal Cut')
#plt.plot(opt[:,0])
#plt.plot(opt[:,1])

plt.plot(opt.sum(axis=1))
opt_i = opt.sum(axis=1).argmax()
plt.axvline(opt_i)

plt.title('Optimal Cut = '+str(opt_i))

g1 = trip.iloc[:opt_i].copy()
g2 = trip.iloc[opt_i:].copy()
d = haversine_vect(g2.iloc[:-1],g2.iloc[1:])
g2.iloc[1:,4] = d
g2.iloc[0,4] = g2.iloc[0,3]

print int(wrw_trip(trip))
print int(wrw_trip(g1) + wrw_trip(g2))

# Do I split on Longs or Lats?


# Create function to find optimal split
# return of 0 means np split it optimal
def optimal_split(trip,calc_gain=False):
    opt = np.zeros((trip.shape[0],2))
    opt[:,0] = (trip.Weight.sum()-trip.Weight.cumsum()) * (trip.Distance.cumsum()-trip.LB)
    opt[1:,1] = np.array(-2*10*trip.LB)[:-1]
    opt_i = opt.sum(axis=1).argmax()
    
    if calc_gain and opt_i:
        # Split into 2 trips
        g1 = trip.iloc[:opt_i].copy()
        g2 = trip.iloc[opt_i:].copy()
        d = haversine_vect(g2.iloc[:-1],g2.iloc[1:])
        g2.iloc[1:,4] = d
        g2.iloc[0,4] = g2.iloc[0,3]
        gain = wrw_trip(trip) - (wrw_trip(g1) + wrw_trip(g2))
        return gain
    return opt_i


###################################
# Test out for current solution

Santa = Trips(path='_v10_greed')

cuts = []
for tripid in Santa.trip_metrics.index:
    cuts.append(optimal_split(Santa.T[tripid],calc_gain=True))

cuts = np.array(cuts)






###########################################################
###########################################################
# Swap heavy presents to earlier

def intra_trip_swap(trip,swap=True,verbose=False):
    '''
    Loops through presents within a trip and finds an optimal ordering
    '''
    
    # Start at last present
    n = trip.shape[0]
    
    for p in range(n-1,1,-1):
        # Check if swap is benefitial
        weights = np.zeros(3)
        delta = np.zeros(3)
        
        weights[:2] = trip.iloc[p-1:p+1].Weight
        if p < n-1:
            weights[2] = trip.iloc[p+1:].Weight.sum()
        
        # Create Distance Matrix
        nn = 3
        if p < n-1:
            nn = 4
        DMat = np.zeros((nn,nn))
        for i in range(nn):
            DMat[i] = haversine_vect(trip.iloc[p-2+i],trip.iloc[p-2:p+2])
        
        delta[0] = DMat[0,2] + DMat[1,2] - DMat[0,1]
        delta[1] = DMat[0,2] - DMat[0,1] - DMat[1,2]
        if p < n-1:
            delta[2] = DMat[0,2] + DMat[1,3] - DMat[0,1] - DMat[2,3]
        
        gain = (delta*weights).sum()
        if verbose:
            print 'Present',p,'- Gain = %.0f'%(gain)


        if swap and gain < 0:
            # Save temp present
            temp_row = trip.iloc[p].copy()
            # Move earlier present into later
            trip.iloc[p] = trip.iloc[p-1]
            trip.iloc[p-1] = temp_row
            
            # Recalc Distances
            d = haversine_vect(trip.iloc[:-1],trip.iloc[1:])
            trip.iloc[1:,4] = d
            trip.iloc[0,4] = trip.iloc[0,3]
    return trip



_ = intra_trip_swap(Santa.T[0].copy(),swap=True)











