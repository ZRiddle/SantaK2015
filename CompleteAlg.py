# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 22:07:19 2015

@author: zach.riddle
"""


from Fine_Tuning_v9 import *
import numpy as np
import pandas as pd
import time

from util import *

# Read in Data
# 1 and 2 complete
Santa = Trips(path='_v5')
total_wrw_start = Santa.wrw
print 'Initial WRW = %.0f'%total_wrw_start

# 3 - Optimal trip counts
#   - Cap SP at ~600
#   - Overfill top
#   - Aim for 1450-1500 trips?


# First - SP
wrw_start = Santa.wrw
'''
for ppp in range(3):
    print '\n\n'
    print '~'*33
    print 'Outer Loop #'+str(ppp)
    SPmask = Santa.trip_metrics.SPF == 1
    print 'Trip Count = %.0f'%SPmask.sum()
    for q in range(15):
        wrw_beg = Santa.wrw
        # Choose random trip
        SPmask = Santa.trip_metrics.SPF == 1
        SPtrips = Santa.trip_metrics[SPmask].index
        rand_trip = np.random.choice(SPtrips)
        # Choose a random
        nearby_ct = np.random.randint(5,10)
        # Find Optimal
        n_opt = Santa.load_balance(rand_trip,nearby_ct=nearby_ct,verbose=False,w_max = 600)
        if n_opt != nearby_ct+1:
            www = 1
            #print '\nLoop #'+str(q)
            #print 'Nearby Count = %.0f\nOptimal Count = %.0f'%(nearby_ct+1,n_opt)
            #print '%.0f - WRW Start\n%.0f - WRW End'%(wrw_beg,Santa.wrw)
                                    
    Santa.save_data(path='_v2')


for ppp in range(5):
    print '\n\n'
    print '~'*33
    print 'Outer Loop #'+str(ppp)
    SPmask = Santa.trip_metrics.SPF == 2
    print 'Trip Count = %.0f'%SPmask.sum()
    for q in range(15):
        wrw_beg = Santa.wrw
        # Choose random trip
        SPmask = Santa.trip_metrics.SPF == 2
        SPtrips = Santa.trip_metrics[SPmask].index
        rand_trip = np.random.choice(SPtrips)
        # Choose a random
        nearby_ct = np.random.randint(5,10)
        # Find Optimal
        n_opt = Santa.load_balance(rand_trip,nearby_ct=nearby_ct,verbose=False,w_max = 1100)
        if n_opt != nearby_ct+1:
            print '\nLoop #'+str(q)
            print 'Nearby Count = %.0f\nOptimal Count = %.0f'%(nearby_ct+1,n_opt)
            #print '%.0f - WRW Start\n%.0f - WRW End'%(wrw_beg,Santa.wrw)
                                    
    Santa.save_data(path='_v2')

for ppp in range(5):
    print '\n\n'
    print '~'*33
    print 'Outer Loop #'+str(ppp)
    SPmask = Santa.trip_metrics.SPF == 0
    print 'Trip Count = %.0f'%SPmask.sum()
    for q in range(15):
        wrw_beg = Santa.wrw
        # Choose random trip
        SPmask = Santa.trip_metrics.SPF == 0
        SPtrips = Santa.trip_metrics[SPmask].index
        rand_trip = np.random.choice(SPtrips)
        # Choose a random
        nearby_ct = np.random.randint(5,10)
        # Find Optimal
        n_opt = Santa.load_balance(rand_trip,nearby_ct=nearby_ct,verbose=False,w_max = 1200)
        if n_opt != nearby_ct+1:
            print '\nLoop #'+str(q)
            print 'Nearby Count = %.0f\nOptimal Count = %.0f'%(nearby_ct+1,n_opt)
            #print '%.0f - WRW Start\n%.0f - WRW End'%(wrw_beg,Santa.wrw)
                                    
    Santa.save_data(path='_v2')



print '\nTotal trip count = %.0f'%Santa.trip_metrics.shape[0]
'''

# 4 - Create Return Trips
#   - 1 pass, Greedy
'''
Santa = create_retun_trips_all(Santa)
Santa.save_data(path='_v3')
'''

# 5 - Push Heavy presents south
#   - Greedy, bottom-up
def swap_down(p,tripj,tripk,swap=True,top=False,check=False):
    n = tripk.shape[0]
    present = tripj.iloc[p].copy()
    
    # Put heavy presents on down part  
    if present.Latitude < tripk.Latitude.min():
        q = np.array(tripk.Latitude).argmin()+1
    else:
        # Big presents at front of trip
        if present.Weight > 3 or top:
            q = np.array(tripk.Latitude < present.Latitude).nonzero()[0].min()
        else:
            q = np.array(tripk.Latitude < present.Latitude).nonzero()[0].max() + 1
    
    ##TODO - Make this it's own function to do arbitrarily
    # changes for j,k,p
    delta = np.zeros(3)
    weights = np.zeros(3)
    
    # Set weights
    weights[0] = tripj.iloc[p+1:].Weight.sum() + 10
    weights[1] = tripk.iloc[q:].Weight.sum() + 10
    weights[2] = present.Weight
    
    # Last Present
    if p == tripj.shape[0]-1:
        delta[0] = tripj.iloc[p-1].LB - present.Distance - present.LB
    elif p == 0:
        delta[0] = tripj.iloc[p+1].LB - tripj.iloc[p+1].Distance - present.LB
    else:
        delta[0] = haversine_vect(tripj.iloc[p-1],tripj.iloc[p+1]) - tripj.iloc[p:p+2].Distance.sum()
        
    Dpq= haversine_vect(present,tripk.iloc[max(0,q-1):q+1])
    
    if q == n:
        delta[1] = present.LB + Dpq[0] - tripk.iloc[-1].LB
        delta[2] = tripk.Distance.sum() + Dpq[0] - tripj.iloc[:p+1].Distance.sum()
    elif q == 0:
        delta[1] = present.LB + Dpq[0] - tripk.iloc[0].LB
        delta[2] = present.LB - tripj.iloc[:p+1].Distance.sum()
    else:
        delta[1] = Dpq.sum() - tripk.iloc[q].Distance
        delta[2] = tripk.iloc[:q].Distance.sum() + Dpq[0] - tripj.iloc[:p+1].Distance.sum()
    
    gain = sum(delta*weights)
    if gain < 0 and swap:
        if check:
            wrw_start = wrw_trip(tripj) + wrw_trip(tripk)
        # Update trip j
        tripj.drop(present.name,inplace=True)
        if p == 0:
            tripj.iloc[0,4] = tripj.iloc[0,3]
        elif p < tripj.shape[0] - 1:
            tripj.iloc[p,4] = haversine_vect(tripj.iloc[p-1],tripj.iloc[p])
        
        # Update trip k
        p_ind = present.name
        # Add present
        tripk = tripk.append(present)
        # Shift everything after q, +1
        ind = np.array(tripk.index)
        ind[q+1:] = ind[q:-1]
        # Set position q to present
        ind[q] = p_ind
        # reindex
        tripk = tripk.ix[ind].copy()
        # Update distances
        if q == 0:
            tripk.iloc[0,4] = tripk.iloc[0,3]
            tripk.iloc[1,4] = Dpq[0]
        else:
            tripk.iloc[q:q+2,4] = Dpq
            
        if check:
            wrw_end = wrw_trip(tripj) + wrw_trip(tripk)
            print 'Present %.0f to %.0f\n%.0f - Gain\n%.0f - Delta WRW'%(p,q,-gain,wrw_start-wrw_end)
            
        return tripj,tripk,1
    return tripj,tripk,0
       
'''
~~ Alg ~~
    - Index tops of all South Pole Trips
    - For j in Mid-Tier Trips
        - For p in presents in j (heaviest first)
            - Find closest South Pole trip
            - Try to swap
    
    - Repeat process:
        Mid -> Bot
        Top -> Mid
        Top -> Bot
        
        and upwards!

Comments:
    Need a way to destroy/merge trips, there are too many!
'''

def push_downward(Santa,SPF_top,SPF_bot):
    SP_index = pd.DataFrame(columns = ['Longitude'])
    SPmask = Santa.trip_metrics.SPF == SPF_bot
    
    for tr in SPmask[SPmask].index:
        SP_index = SP_index.append(pd.DataFrame(index=[tr],
                                                data=[Santa.T[tr].iloc[0].Longitude],
                                                columns = ['Longitude']))
    #1312
    total_start_wrw = Santa.wrw
    
    for w_limit in [45,40,35]:
        Midmask = Santa.trip_metrics.SPF == SPF_top
        for midtrip in Midmask[Midmask].index:
            # Get trip j
            tripj = Santa.T[midtrip].copy()
            # Loop through presents, weight first
            p_list = np.array(tripj.Weight).argsort()[::-1]
            start_wrw = Santa.wrw
            
            p_ind=0
            p = p_list[p_ind]        
            while tripj.iloc[p].Weight >= w_limit:
                # Find nearest SP trip
                k = ((SP_index - tripj.iloc[p].Longitude)**2).sort_values('Longitude').index[0]
                tripk = Santa.T[k].copy()
                # Record Start WRW
                #start_wrw = wrw_trip(tripj) + wrw_trip(tripk)
                tj,tk,success = swap_down(p,tripj,tripk)
                if success:
                    tripj = tj.copy()
                    # Save trip k
                    Santa.T[k] = tk.copy()
                    Santa.update_trip_metrics(k)
                    
                    # Deindex p_list
                    p_list[p_list>p] -= 1
                p_ind +=1
                p = p_list[p_ind]
            # Save trip j
            Santa.T[midtrip] = tripj
            Santa.update_trip_metrics(midtrip)
            
            end_wrw = Santa.wrw
            if start_wrw - end_wrw:
                print '~~~ Trip %.0f ~~~\n%.0f - Gain'%(midtrip,start_wrw-end_wrw)
        
        print '\n%.0f - Starting WRW\n%.0f - Ending WRW'%(total_start_wrw,Santa.wrw)
    
    light_trips = Santa.trip_metrics.Weight < 150
    
    for tr in light_trips[light_trips].index:
        Santa.destroy_diffuse(tr)
        
    Santa = create_retun_trips_all(Santa)
    return Santa


#push_downward(Santa,SPF_top=2,SPF_bot=1)
#push_downward(Santa,SPF_top=0,SPF_bot=2)
#push_downward(Santa,SPF_top=0,SPF_bot=1)

#Santa.save_data('_v6')


#push_downward(Santa,SPF_top=1,SPF_bot=2)
#push_downward(Santa,SPF_top=2,SPF_bot=0)

# 6 - Optimal swapping
#   - Still assume Lat sorting but with return trips
# Given trip, take a present
# Find nearest trip in top, mid, and bot
# Calc gains
# Swap to optimal

def swap(p,tripj,tripk,swap=True,top=True,check=False):
    n = tripk.shape[0]
    present = tripj.iloc[p].copy()
    
    # Put heavy presents on down part  
    if present.Latitude <= tripk.Latitude.min():
        q = np.array(tripk.Latitude).argmin()+1
    else:
        # Big presents at front of trip
        if present.Weight > 3 or top:
            q = np.array(tripk.Latitude < present.Latitude).nonzero()[0].min()
        else:
            q = np.array(tripk.Latitude < present.Latitude).nonzero()[0].max() + 1
    
    ##TODO - Make this it's own function to do arbitrarily
    # changes for j,k,p
    delta = np.zeros(3)
    weights = np.zeros(3)
    
    # Set weights
    weights[0] = tripj.iloc[p+1:].Weight.sum() + 10
    weights[1] = tripk.iloc[q:].Weight.sum() + 10
    weights[2] = present.Weight
    
    # Last Present
    if p == tripj.shape[0]-1:
        delta[0] = tripj.iloc[p-1].LB - present.Distance - present.LB
    elif p == 0:
        delta[0] = tripj.iloc[p+1].LB - tripj.iloc[p+1].Distance - present.LB
    else:
        delta[0] = haversine_vect(tripj.iloc[p-1],tripj.iloc[p+1]) - tripj.iloc[p:p+2].Distance.sum()
        
    Dpq= haversine_vect(present,tripk.iloc[max(0,q-1):q+1])
    
    if q == n:
        delta[1] = present.LB + Dpq[0] - tripk.iloc[-1].LB
        delta[2] = tripk.Distance.sum() + Dpq[0] - tripj.iloc[:p+1].Distance.sum()
    elif q == 0:
        delta[1] = present.LB + Dpq[0] - tripk.iloc[0].LB
        delta[2] = present.LB - tripj.iloc[:p+1].Distance.sum()
    else:
        delta[1] = Dpq.sum() - tripk.iloc[q].Distance
        delta[2] = tripk.iloc[:q].Distance.sum() + Dpq[0] - tripj.iloc[:p+1].Distance.sum()
    
    gain = sum(delta*weights)
    if gain < 0 and swap:
        if check:
            wrw_start = wrw_trip(tripj) + wrw_trip(tripk)
        # Update trip j
        tripj.drop(present.name,inplace=True)
        if p == 0:
            tripj.iloc[0,4] = tripj.iloc[0,3]
        elif p < tripj.shape[0] - 1:
            tripj.iloc[p,4] = haversine_vect(tripj.iloc[p-1],tripj.iloc[p])
        
        # Update trip k
        p_ind = present.name
        # Add present
        tripk = tripk.append(present)
        # Shift everything after q, +1
        ind = np.array(tripk.index)
        ind[q+1:] = ind[q:-1]
        # Set position q to present
        ind[q] = p_ind
        # reindex
        tripk = tripk.ix[ind].copy()
        # Update distances
        if q == 0:
            tripk.iloc[0,4] = tripk.iloc[0,3]
            tripk.iloc[1,4] = Dpq[0]
        else:
            tripk.iloc[q:q+2,4] = Dpq
            
        if check:
            wrw_end = wrw_trip(tripj) + wrw_trip(tripk)
            print 'Present %.0f to %.0f\n%.0f - Gain\n%.0f - Delta WRW'%(p,q,-gain,wrw_start-wrw_end)
            
    return tripj,tripk,gain


# Choose trip j
for j in Santa.trip_metrics.sort_values('AvgLong').index:
    start_wrw = Santa.wrw
    # Copy Trip
    tripj = Santa.T[j].copy()
    n = tripj.shape[0]
    # Choose present p
    p = n-1
    while p >=0:
        s_wrw = Santa.wrw
        # Find 4 nearest trips
        wmask = Santa.trip_metrics.Weight < 3500
        trips = ((tripj.iloc[p].Longitude - Santa.trip_metrics[wmask].AvgLong)**2).sort_values().head().index
        gains = []
        trips_j = []
        trips_k = []
        trips_n = []
        for tr in trips:
            tripk = Santa.T[tr].copy()
            if tr == j:
                packed = (0,0,0)
            else:
                packed = swap(p,tripj.copy(),tripk.copy())
            
            trips_j.append(packed[0])
            trips_k.append(packed[1])
            gains.append(packed[2])
            trips_n.append(tr)
            
            # Do lower as well
            if tripj.iloc[p].Weight <= 3:
                packed = swap(p,tripj.copy(),tripk.copy(),top=False)
                trips_j.append(packed[0])
                trips_k.append(packed[1])
                gains.append(packed[2])
                trips_n.append(tr)
            
        opt_swap = np.array(gains).argmin()
        
        if gains[opt_swap] < 0:
            # Update
            tripj = trips_j[opt_swap].copy()
            Santa.T[j] = tripj
            Santa.update_trip_metrics(j)
            
            if trips_n[opt_swap] != j:
                Santa.T[trips_n[opt_swap]] = trips_k[opt_swap].copy()
                Santa.update_trip_metrics(trips_n[opt_swap])
            #print '~~ Trip %.0f ~~\n%.0f - gains\n%.0f - Delta WRW'%(p,-gains[opt_swap],s_wrw-Santa.wrw)
        
        p -= 1
        if tripj.shape[0]<4 or tripj.Weight.sum() < 200:
            Santa.destroy_diffuse(j,cap=4000)
            p = -1
            print 'Trip %.0f Destroyed!!!'%(j)
            print 'New Trip Count =',Santa.trip_metrics.shape[0]
    print '%.0f - Gain for Trip %.0f'%(start_wrw-Santa.wrw,j)
    

Santa.save_data('_v5')


# 8 - Intra-trip Opt
