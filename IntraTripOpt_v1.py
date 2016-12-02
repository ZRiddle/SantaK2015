# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 01:11:51 2015

@author: zach.riddle
"""

from Fine_Tuning_v9 import *
import numpy as np
import pandas as pd
import time

# Read in data
Santa = Trips(path='_v10')
total_wrw_start = Santa.wrw

gifts = Santa.gifts
stime = time.time()
# Loop through all trips
for tripid in Santa.trip_metrics.index:
    # Optimize sorting
    wrw_start = Santa.trip_metrics.ix[tripid].WRW
    Santa.T[tripid] = intra_trip_swap(Santa.T[tripid].copy(),verbose=False)
    Santa.update_trip_metrics(tripid)
    gains = wrw_start - Santa.trip_metrics.ix[tripid].WRW
    if gains:
        print 'Trip #'+str(tripid)+'\n%.0f - WRW gain'%(gains)

print '%.0f - Starting WRW\n%.0f - Ending WRW'%(total_wrw_start,Santa.wrw)
print 'Runtime = %.2f'%((time.time()-stime)/60.)
Santa.save_data(path='_v12')

#701

# Need to take group of presents and decide on longitude cuts vs latitude cuts

'''
mask = gifts.Latitude > -91
mask = np.logical_and(mask, gifts.Longitude > 142)
mask = np.logical_and(mask, gifts.Longitude < 142.57)
print '\nPresents =',mask.sum(),'\nWeight = %.0f'%gifts[mask].Weight.sum()


plot_all_presents(gifts)
plt.scatter(gifts[mask]['Longitude'], 
            gifts[mask]['Latitude'], 
            s=gifts[mask]['Weight'], 
            color='red')
plt.xlim(110,160)

temp = gifts[mask].copy()
temp.sort_values('Latitude', ascending=False, inplace=True)
temp.drop(['TripId','Cluster'],axis=1,inplace=True)

temp['Distance'] = np.array(temp.LB)
d = haversine_vect(temp.iloc[:-1],temp.iloc[1:])
temp.iloc[1:,4] = d

# Compare total wrw to vertical to horizontal
wrw_one = wrw_trip(temp)
wrw_v = optimal_split(temp,True)

g = split_trips(temp)
wrw_h = wrw_trip(g[0]) + wrw_trip(g[1])

print '%.0f - Total\n%.0f - Lat Split\n%.0f - Long Split'%(wrw_one,wrw_v,wrw_h)
'''


def create_trip_df(Santa):
    cols = T[0].columns.tolist()
    cols.append('Waste')
    cols.append('TripId')
    sub = pd.DataFrame(columns=cols)
    for tID in Santa.trip_metrics.index:
        # Add trip to DF
        temp = Santa.T[tID].copy()
        temp['Waste'] = (temp.Distance.cumsum() - temp.LB) * temp.Weight
        temp['TripId'] = tID
        sub = pd.concat([sub,temp])

    return sub




def plot_worst_gifts(gifts,worst=50):
    plt.figure("Worst Gifts",figsize=(17,11))
    plt.scatter(gifts['Longitude'], gifts['Latitude'], s=gifts['Weight'], color='gray')
    w_ind = gifts.sort_values('Waste').tail(worst).index
    plt.scatter(gifts.ix[w_ind]['Longitude'], 
                gifts.ix[w_ind]['Latitude'], 
                s=gifts.ix[w_ind]['Weight'], 
                color='red')
    
    plt.title('Worst Gifts',fontsize=22)
    plt.tight_layout()



trip_list = create_trip_df(Santa)
    
plot_worst_gifts(trip_list,300)
plot_worst_trips(Santa,50)


# Find new trips
mask_SPF = Santa.trip_metrics.SPF == 1
Santa.trip_metrics[mask_SPF].sort_values('AvgLong').head()
Santa.trip_metrics[np.logical_not(mask_SPF)].sort_values('AvgLong').head()
# 23, 33, 32, 1835, 1225, 1209
trlist = [1835, 1225, 1209,1840]

plot_all_presents(gifts)
for t in trlist:
    Santa.plot_trip(t,newplot=False)

plt.xlim(-181,-160)
plt.ylim(-91,-76)

plt.ylim(-91,80)

plt.ylim(50,80)


'''

m = np.logical_or(m,trip_list.TripId == 33)
m = np.logical_or(m,trip_list.TripId == 23)
m = np.logical_or(m,trip_list.TripId == 21)
'''
m = trip_list.TripId == 32
tripS = trip_list[m].copy()

m = trip_list.TripId == 1835
tripN = trip_list[m].copy()

m = np.logical_or(m,trip_list.TripId == 1225)
m = np.logical_or(m,trip_list.TripId == 1209)
m = np.logical_or(m,trip_list.TripId == 1840)

trip = trip_list[m].copy()

trip.sort_values('Latitude', ascending=False, inplace=True)
#temp.drop(['TripId','Cluster'],axis=1,inplace=True)
#trip['Distance'] = np.array(temp.LB)

d = haversine_vect(trip.iloc[:-1],trip.iloc[1:])
trip.iloc[1:,4] = d

for ntr in range(2,7):
    ww = np.zeros(4)
    for i in range(3):
        g = split_trips(trip,ntr,i)
        ww1 = 0
        for g1 in g:
            ww1 += wrw_trip(g1)
        ww[i] = ww1
        print '%.0f - %.0f trips - Return Trip %.0f'%(ww[i],ntr,i)
    print ''
    
# Try pushing gift to end
wrw_start = wrw_trip(trip)
g = trip.shape[0] -1
while g > 0:
    g -= 1
    ind = np.array(trip.index)
    ind_g = ind[g]
    ind[g:-1] = ind[g+1:]
    ind[-1] = ind_g
    
    temp = trip.ix[ind].copy()
    d = haversine_vect(temp.iloc[:-1],temp.iloc[1:])
    temp.iloc[1:,4] = d
    temp.iloc[0,4] = temp.iloc[0,3]
    gain = wrw_start - wrw_trip(temp)
    if gain > 0:
        print '%.0f - g\n%.1f - Weight\n   %.0f - Gain'%(g,trip.ix[ind_g].Weight,gain)
        


print '%.0f - Start\n%.0f - End'%(wrw_start,wrw_end)

w_mask = trip.Weight > 3
temp = pd.concat([trip[w_mask].copy(),trip[np.logical_not(w_mask)].sort_values('Latitude')])
d = haversine_vect(temp.iloc[:-1],temp.iloc[1:])
temp.iloc[1:,4] = d
temp.iloc[0,4] = temp.iloc[0,3]

print '%.0f - Start\n%.0f - Small on Return'%(wrw_trip(trip),wrw_trip(temp))

plt.figure('Small On Return',figsize=(17,11))
plot_all_presents(gifts,newplot=False)
plt.plot(temp.Longitude,temp.Latitude)
plt.xlim(-180,-160)
plt.ylim(-91,-75)


plt.plot(temp.Longitude,temp.Latitude)

lat = -80
long1 = 10
long2 = 0
x=[]
for lat in range(0,90,1):
    x.append([lat,haversine((lat,long1),(lat,long2))/AVG_EARTH_RADIUS])
    print lat,haversine((lat,long1),(lat,long2))/AVG_EARTH_RADIUS

plt.figure('Haversine')
x = np.array(x)
plt.plot(x[:,0],x[:,1])




    
# All trips have 2 parts
# Down and Up
# Initialize trips with 1s on the up part
#     

def create_retun_trips(trip,swap=True,verbose=False):
    wrw_start = wrw_trip(trip)
    n = trip.shape[0]
    g = n - 1
    while g > 0:
        g -= 1
        
        delta = np.zeros(3)
        weights = np.zeros(3)
        # Set weights
        weights[0] = trip.iloc[g].Weight
        weights[1] = trip.iloc[g+1:].Weight.sum()
        weights[2] = 10
        
        # Set distnace changes
        Dp = np.zeros(3)
        if g:
            Dp = haversine_vect(trip.iloc[g],trip.iloc[g-1:g+2])
            Dp11 = haversine_vect(trip.iloc[g-1],trip.iloc[g+1])
        else:
            Dp[2] = haversine_vect(trip.iloc[g],trip.iloc[g+1])
            Dp[0] = trip.iloc[g].LB
            Dp11 = trip.iloc[g+1].LB
        Dpn= haversine_vect(trip.iloc[g],trip.iloc[-1])
        CDp= trip.iloc[g+2:].Distance.sum()
        
        delta[0] = CDp + Dpn + Dp11 - Dp[0]
        delta[1] = Dp11 - sum(Dp)
        delta[2] = Dpn + trip.iloc[g].LB - trip.iloc[-1].LB + delta[1]
        
        gain = sum(delta*weights)
        if gain < 0 and swap:
        
            ind = np.array(trip.index)
            ind_g = ind[g]
            ind[g:-1] = ind[g+1:]
            ind[-1] = ind_g
        
            trip = trip.ix[ind].copy()
            # Fix distance at g and at end
            trip.iloc[g,4] = Dp11
            trip.iloc[-1,4] = Dpn
            if verbose:
                print '%.0f - Weight, %.0f - g\n   %.0f - Gain'%(trip.ix[ind_g].Weight,g,-gain)
            
            #g -= 1
    if verbose:
        print '%.0f - Total Gain'%(wrw_start - wrw_trip(trip))
    return trip


tt = create_retun_trips(trip,swap=True,verbose=True)


Santa = Trips(path='_v10_greed')
total_wrw_start = Santa.wrw

def create_retun_trips_all(Santa):
    total_wrw_start = Santa.wrw
    stime = time.time()
    # Loop through all trips
    for tripid in Santa.trip_metrics.index:
        # Optimize sorting
        wrw_start = Santa.trip_metrics.ix[tripid].WRW
        Santa.T[tripid] = create_retun_trips(Santa.T[tripid].copy(),swap=True,verbose=False)
        Santa.update_trip_metrics(tripid)
        gains = wrw_start - Santa.trip_metrics.ix[tripid].WRW
        if gains:
            print 'Trip #'+str(tripid)+'\n%.0f - WRW gain'%(gains)
    
    print '%.0f - Starting WRW\n%.0f - Ending WRW'%(total_wrw_start,Santa.wrw)
    print 'Runtime = %.2f'%((time.time()-stime)/60.)
    return Santa
    
Santa.save_data(path='_v10_greed')



def intra_trip_swap_all(Santa):
    total_wrw_start = Santa.wrw
    stime = time.time()
    # Loop through all trips
    for tripid in Santa.trip_metrics.index:
        # Optimize sorting
        wrw_start = Santa.trip_metrics.ix[tripid].WRW
        Santa.T[tripid] = intra_trip_swap(Santa.T[tripid].copy(),verbose=False)
        Santa.update_trip_metrics(tripid)
        gains = wrw_start - Santa.trip_metrics.ix[tripid].WRW
        if gains:
            print 'Trip #'+str(tripid)+'\n%.0f - WRW gain'%(gains)
    
    print '%.0f - Starting WRW\n%.0f - Ending WRW'%(total_wrw_start,Santa.wrw)
    print 'Runtime = %.2f'%((time.time()-stime)/60.)
    return Santa
    
Santa.save_data(path='_v10_greed')



######################################
# Given North trip and SP trip, Loop through presents in north

def move_south(tripN,tripS):
    '''
    Greedy, Bottom-up approach
    Assumes new present will at the top of the new trip
    Should try to start from bottom of downward trip, not return part
    '''
    n = tripN.shape[0]
    
    for p in range(n-2,1,-1):
        weights = np.zeros(3)
        delta = np.zeros(3)
        present = tripN.iloc[p].copy()
        
        weights[0] = tripS.Weight.sum() + 10
        weights[1] = present.Weight
        weights[2] = tripN.iloc[p+1:].Weight.sum() + 10
        
        DS0 = haversine_vect(present, tripS.iloc[0])
        D11 = haversine_vect(tripN.iloc[p-1],tripN.iloc[p+1])
        
        delta[0] = present.LB + DS0 - tripS.iloc[0].LB
        delta[1] = present.LB - tripN.iloc[:p+1].Distance.sum()
        delta[2] = D11 - haversine_vect(present,tripN.iloc[p-1:p+2]).sum()
        
        gain = sum(weights*delta)
        if gain < 0:
            # New South
            newSouth = pd.concat([tripN.iloc[p-1:p+1],tripS])
            tripS = newSouth.iloc[1:].copy()
            # New North
            tripN = pd.concat([tripN.iloc[:p],tripN.iloc[p+1:]])
            
            # Fix row p
            tripN.iloc[p,4] = D11
            # Fix top of South Trip
            tripS.iloc[0,4] = tripS.iloc[0,3]
            tripS.iloc[1,4] = DS0
    return tripN,tripS

#
#new_wrw = wrw_trip(newSouth) + wrw_trip(newNorth)











