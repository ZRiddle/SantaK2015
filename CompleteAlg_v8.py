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
Santa = Trips(path='_v7_6')
total_wrw_start = Santa.wrw
print 'Initial WRW = %.0f'%total_wrw_start


# 12450524470 - _v7_6
# 12450724470 - _v7_7
# 12455550431 - _v10_greed



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
    
    #light_trips = Santa.trip_metrics.Weight < 150
    
    #for tr in light_trips[light_trips].index:
        #Santa.destroy_diffuse(tr)
        
    Santa = create_retun_trips_all(Santa)
    return Santa

'''
push_downward(Santa,SPF_top=2,SPF_bot=1)
push_downward(Santa,SPF_top=0,SPF_bot=2)
push_downward(Santa,SPF_top=0,SPF_bot=1)

Santa.save_data('_v5')
'''

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




#Santa = intra_trip_swap_all(Santa)
'''
# 7 - Optimal cuts
for tr in Santa.trip_metrics.index:
    g,g1,g2 = optimal_split(Santa.T[tr].copy(),True)
    if g:
        
        Santa.T[tr] = g1.copy()        
        Santa.update_trip_metrics(tr)
        
        # Create New Trip
        new_tripid = len(Santa.T)
        Santa.T.append(g2)
        Santa.update_trip_metrics(new_tripid)
        
        print 'Trip %.0f Split up!'%(tr)
        
Santa.reindex_dataframes()
'''
    
def optimal_swapping(Santa):
    trlist = Santa.trip_metrics.sort_values('AvgLong').index
    # Choose trip j
    for j in trlist:
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
                    packed = swap(p,tripj.copy(),tripk.copy(),top=False)
                
                trips_j.append(packed[0])
                trips_k.append(packed[1])
                gains.append(packed[2])
                trips_n.append(tr)
                
                
            opt_swap = np.array(gains).argmin()
            
            if gains[opt_swap] < 0:
                # Update
                tripj = trips_j[opt_swap].copy()
                
                Santa.T[trips_n[opt_swap]] = trips_k[opt_swap].copy()
                Santa.update_trip_metrics(trips_n[opt_swap])
                
                Santa.T[j] = tripj
                Santa.update_trip_metrics(j)
                #print '~~ Trip %.0f ~~\n%.0f - gains\n%.0f - Delta WRW'%(p,-gains[opt_swap],s_wrw-Santa.wrw)
            
            p -= 1
            if tripj.shape[0]<4 or tripj.Weight.sum() < 150:
                gifts = Santa.destroy_trip(j)
                for gf in gifts:
                    new_trip_ind = ((gf.Longitude - Santa.trip_metrics.AvgLong)**2).sort_values().index[0]
                    new_trip = Santa.T[new_trip_ind].copy()
                    
                    if gf.Latitude <= new_trip.Latitude.min():
                        q = np.array(new_trip.Latitude).argmin()+1
                    else:
                        # Big presents at front of trip
                        if gf.Weight > 2:
                            q = np.array(new_trip.Latitude < gf.Latitude).nonzero()[0].min()
                        else:
                            q = np.array(new_trip.Latitude < gf.Latitude).nonzero()[0].max() + 1
                    
                    p_ind = gf.name
                    # Add present
                    new_trip = new_trip.append(gf)
                    # Shift everything after q, +1
                    ind = np.array(new_trip.index)
                    ind[q+1:] = ind[q:-1]
                    # Set position q to present
                    ind[q] = p_ind
                    # reindex
                    new_trip = new_trip.ix[ind].copy()
                    # Update distances
                    if q == 0:
                        new_trip.iloc[0,4] = new_trip.iloc[0,3]
                        new_trip.iloc[1,4] = haversine_vect(new_trip.iloc[1],gf)
                    elif q == new_trip.shape[0]-1:
                        new_trip.iloc[q,4] = haversine_vect(new_trip.iloc[q-1],new_trip.iloc[q])
                    else:
                        new_trip.iloc[q:q+2,4] = haversine_vect(new_trip.iloc[q-1:q+1],new_trip.iloc[q:q+2])
                    
                    # Update trips
                    Santa.T[new_trip_ind] = new_trip.copy()
                    Santa.update_trip_metrics(new_trip_ind)                
                    
                p = -1
                print 'Trip %.0f Destroyed!!!'%(j)
                print 'New Trip Count =',Santa.trip_metrics.shape[0]
        print '%.0f - Gain for Trip %.0f'%(start_wrw-Santa.wrw,j)
    return Santa

# 8
# Create function to evaluate loss from putting present into trip

def insert_present_gain(present,trip,verbose=False):
    '''
    Evaluates the Loss from adding present to trip
    Inputs : present to add, trip
    Returns : gain in optimal position, new trip
    '''
    # Create Distance Vector
    n = trip.shape[0]
    Dq = np.zeros(n+2)
    CW = np.zeros(n+1)
    Dt = np.zeros(n+2)
    
    Dq[1:-1] = haversine_vect(present,trip)
    Dq[0] = present.LB
    Dq[-1] = present.LB
    
    CW[:-1] = np.array(trip.Weight)[::-1].cumsum()[::-1]
    CW += 10
    
    
    Dt[1:-1] = np.array(trip.Distance)
    Dt[-1] = trip.iloc[-1].LB
    
    gains = (Dq[:-1] + Dq[1:] - Dt[1:]) * CW + (Dt[:-1].cumsum() + Dq[:-1]) * present.Weight
    
    if verbose:
        wrw_start = wrw_trip(trip)
    q_opt = gains.argmin()
    
    new_trip = trip.append(present).copy()
    ind = np.array(new_trip.index)
    ind[q_opt+1:] = ind[q_opt:-1]
    ind[q_opt] = present.name
    
    new_trip = new_trip.ix[ind].copy()
    new_trip.iloc[q_opt,4] = Dq[q_opt]
    
    if q_opt < n:
        new_trip.iloc[q_opt+1,4] = Dq[q_opt+1]
    
    if verbose:
        print '%.0f - Loss\n%.0f - Calculated Loss'%(wrw_trip(new_trip) - wrw_start,gains[q_opt])

    return gains[q_opt],new_trip



def remove_present_gain(trip,verbose=False):
    '''
    Calculates gains for removing each present
    Input : trip
    Returns : gains
    '''
    n = trip.shape[0]
    gains = np.zeros(n)
    
    Dp = np.zeros(n+1)
    Dp1 = np.zeros(n)
    CW = np.zeros(n)
    
    CW[:-1] = (np.array(trip.Weight)[::-1].cumsum()[::-1])[1:]
    CW += 10
    
    Dp[:-1] = np.array(trip.Distance)
    Dp[-1] = trip.iloc[-1].LB
    
    Dp1[1:-1] = haversine_vect(trip.iloc[:-2],trip.iloc[2:])
    Dp1[0] = trip.iloc[1].LB
    Dp1[-1] = trip.iloc[-2].LB
    
    gains = (Dp1 - Dp[:-1] - Dp[1:]) * CW - Dp[:-1].cumsum()*np.array(trip.Weight)
    

    if verbose:
        start_wrw = wrw_trip(trip)
        p = 3
        
        new_trip = trip.drop(trip.index[p]).copy()
        new_trip.iloc[p,4] = Dp1[p]
        print '%.0f - Calculated\n%.0f - Actual'%(gains[p],wrw_trip(new_trip)-start_wrw)
    
    return gains




# Diffuse 

def diffuse(Santa, tripid, w_limit = 1000, east = True, 
            w_long = 50, nearby = 4, verbose = False):
    '''
    Diffuse Presents east or west until weight cap is satisfied
    '''
    
    if east:
        dtrips = np.arange(tripid+1,min(tripid+nearby+1,Santa.trip_metrics.shape[0]))
    else:
        dtrips = np.arange(max(0,tripid-nearby),tripid)
        
    
    trip = Santa.T[tripid].copy()
    cont = True
    stshape = trip.shape[0]
    
    while (trip.Weight.sum() > w_limit or cont) and len(dtrips):
        # First calculate pgains for each present
        pgains = remove_present_gain(trip)
        
        newtripids = []
        gains = []
        newtrips = []
        
        if trip.Weight.sum() > w_limit + w_long:
            if east:
                p = np.array(trip.Longitude).argmax()
            else:
                p = np.array(trip.Longitude).argmin()
                
            present = trip.iloc[p].copy()
        
            newtripids_p = []
            gains_p = []
            newtrips_p = []
            
            # Loop through closest trips
            for dtrip in dtrips:
                tripj = Santa.T[dtrip].copy()
                
                gg,tt = insert_present_gain(present.copy(),tripj.copy())
                
                newtripids_p.append(dtrip)
                gains_p.append(gg)
                newtrips_p.append(tt.copy())
                
            # Find optimal index
            opt_ind = np.array(gains_p).argmin()
            
            newtripid = newtripids_p[opt_ind]
            gains = gains_p[opt_ind]
            newtrip = newtrips_p[opt_ind]
            
            Santa.T[newtripid] = newtrip.copy()
            Santa.update_trip_metrics(newtripid)
            
            trip.drop(trip.index[p],inplace=True)
            
            if p == 0:
                trip.iloc[0,4] = trip.iloc[0,3]
            elif p < trip.shape[0]:
                trip.iloc[p,4] = haversine_vect(trip.iloc[p-1],trip.iloc[p])
        
            Santa.T[tripid] = trip.copy()
            
        else:
            # Loop through presents
            for p in range(trip.shape[0]):
                present = trip.iloc[p].copy()
                
                newtripids_p = []
                gains_p = []
                newtrips_p = []
                
                # Loop through closest trips
                for dtrip in dtrips:
                    tripj = Santa.T[dtrip].copy()
                    
                    gg,tt = insert_present_gain(present.copy(),tripj.copy())
                    
                    newtripids_p.append(dtrip)
                    gains_p.append(gg)
                    newtrips_p.append(tt.copy())
                    
                # Find optimal index
                opt_ind = np.array(gains_p).argmin()
                
                newtripids.append(newtripids_p[opt_ind])
                gains.append(gains_p[opt_ind])
                newtrips.append(newtrips_p[opt_ind])
                
            # Find optimal present to move
            all_gains = (np.array(gains) + pgains)
            opt_p = all_gains.argmin()
    
            newtripid = newtripids[opt_p]
            newtrip = newtrips[opt_p]
            
            if all_gains.min() > 0 or newtrip.Weight.sum() > w_limit or trip.shape[0] < 3:
                cont = False
                    
            if trip.Weight.sum() > w_limit or cont:
                # Update trips
                #wrw_pre = wrw_trip(trip) + wrw_trip(Santa.T[newtripid])
                
                Santa.T[newtripid] = newtrip.copy()
                Santa.update_trip_metrics(newtripid)
                
                trip.drop(trip.index[opt_p],inplace=True)
                
                if opt_p == 0:
                    trip.iloc[0,4] = trip.iloc[0,3]
                elif opt_p < trip.shape[0]:
                    trip.iloc[opt_p,4] = haversine_vect(trip.iloc[opt_p-1],trip.iloc[opt_p])
            
                Santa.T[tripid] = trip.copy()
    if verbose:
        print 'Total presents moved =',stshape-trip.shape[0]
    Santa.update_trip_metrics(tripid)

    return Santa



def destroy_trip(Santa,tripid):
    p_list = Santa.destroy_trip(tripid)
    
    dtrips = np.arange(max(tripid-3,0),min(tripid+4,Santa.trip_metrics.shape[0]))
    
    for present in p_list:
        newtripids_p = []
        gains_p = []
        newtrips_p = []
        
        # Loop through closest trips
        for dtrip in dtrips:
            if dtrip != tripid:
                tripj = Santa.T[dtrip].copy()
                
                gg,tt = insert_present_gain(present.copy(),tripj.copy())
                
                newtripids_p.append(dtrip)
                gains_p.append(gg)
                newtrips_p.append(tt.copy())
            
        # Find optimal index
        opt_ind = np.array(gains_p).argmin()
        
        newtripid = newtripids_p[opt_ind]
        #gain = gains_p[opt_ind]
        newtrip = newtrips_p[opt_ind]
        
        # Update trips
        #wrw_pre = wrw_trip(trip) + wrw_trip(Santa.T[newtripid])
        
        Santa.T[newtripid] = newtrip.copy()
        Santa.update_trip_metrics(newtripid)
        
        Santa.reindex_dataframes()

    return Santa
    


def optimize_trip(Santa,tripid,verbose=True):
    trip = Santa.T[tripid].copy()
    present_ids = np.array(trip.index)
    
    stwrw = wrw_trip(trip)
    
    for p in present_ids:
        # Copy Present
        present = trip.ix[p].copy()
        # Drop from trip
        trip.drop(p,inplace=True)
        # Update distances
        trip.iloc[1:,4] = haversine_vect(trip.iloc[:-1],trip.iloc[1:])
        trip.iloc[0,4] = trip.iloc[0,3]
        # Find optimal position
        _,newtrip = insert_present_gain(present.copy(),trip.copy())
        
        trip = newtrip.copy()
    
    if verbose:
        print '%.0f - Intra-Trip Gain, Trip %.0f'%(stwrw - wrw_trip(trip), tripid)
        
    # Update trips
    Santa.T[tripid] = trip.copy()
    Santa.update_trip_metrics(tripid)
    
    return Santa,stwrw-wrw_trip(trip)


'''
tripid = 2
east = True
w_limit = 2000
ss
'''

# Initialize trips with 1500
def initialize_trips_better(Santa,ntrips=1500,verbose=False):
    T = []
    #total_w = Santa.trip_metrics.Weight.sum()
    gifts = Santa.gifts.drop(['TripId','Cluster'],axis=1).copy().sort_values('Longitude')
    
    # Shift Alaska
    gifts.loc[gifts.Longitude < -169,'Longitude'] += 360
    gifts.Longitude -= 11
    gifts = gifts.sort_values('Longitude')
    
    cut = -180
    step = 360./ntrips
    while cut < 180:
        mask = np.logical_and(gifts.Longitude > cut, gifts.Longitude <= cut+step)
        g = gifts[mask].sort_values('Latitude',ascending = False).copy()  
        if g.shape[0]:
            g['Distance'] = np.array(g.LB)
            g.iloc[1:,4] = haversine_vect(g.iloc[1:],g.iloc[:-1])
            T.append(g.copy())
        cut += step
        #print g.shape[0],g.Weight.sum()
    
    Santa.T = T
    for i in range(len(T)):
        Santa.update_trip_metrics(i)
        
    print Santa.wrw
    return Santa


'''
#mask = Santa.trip_metrics.Weight < 200
for tripid in [2,45]:
    Santa = destroy_trip(Santa,tripid)

Santa.reindex_dataframes()
'''


'''
for tripid in Santa.trip_metrics.index:
    if Santa.T[tripid].shape[0]>2:
        Santa = optimize_trip(Santa,tripid)

Santa.save_data('_v7_5')
'''


totalgain = total_wrw_start - Santa.wrw

w_limit = 1000
east = True

#while w_limit <= 1000:
for i in range(3):
    stwrw = Santa.wrw
    sttime = time.time()
    print '~~~ Weight Limit = %.0f ~~~'%w_limit
    trindex = np.array(Santa.trip_metrics.index)
    #totalgain = 0
    if not east:
        trindex = trindex[::-1]
    for tripid in trindex:
        if Santa.T[tripid].shape[0] > 1:
            print '~~~ TripId %.0f~~~'%tripid
            # Do intra-trip opt
            #if east:
            
            Santa, gain = optimize_trip(Santa,tripid,verbose=False)
            totalgain += gain
            
            # Diffuse
            swrw = Santa.wrw
            Santa = diffuse(Santa, tripid = tripid, w_limit = w_limit, 
                            east = east, w_long = 50, nearby = 5)
            gain = swrw - Santa.wrw
            totalgain += gain
    
            print '%.0f - Diffusion Gain'%(gain)
            if Santa.T[tripid].shape[0] > 1:
                Santa,gain = optimize_trip(Santa,tripid,verbose=False)
                totalgain += gain
                
            print '%.0f - Running Total Gain\n'%totalgain
            
            if not tripid % 100:
                print '~'*22,'\n%.0f - New WRW\n'%(Santa.wrw),'~'*22
        
    print '%.0f - New WRW\nRuntime = %.0f seconds\n'%(Santa.wrw,time.time()-sttime)
    if east:
        east = False
    else:
        east = True
        
    #w_limit += .5
    
    # save    
    Santa.save_data('_v7_7')



print Santa.wrw

tt = create_trip_df(Santa)

print '# Presents =',tt.shape
print 'Weight Max =',Santa.trip_metrics.Weight.max()

plot_worst_trips(Santa,50)

Santa.save_data('_v7_7')


# Starting v6 - 12450584470






