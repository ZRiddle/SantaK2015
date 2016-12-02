# -*- coding: utf-8 -*-
"""
Created on Tue Dec 01 11:04:16 2015

@author: zach.riddle
"""

import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import seaborn
from sklearn.cluster import KMeans
import pickle

from util import *

plot_this = 0
CONST = 600.0


# Weights
# gifts.Weight.sum()/1000
# 1409, minimum of 15 trips
lat_long = ['Latitude','Longitude']


###################################################
###################################################
# Read Data
gifts,samplesub = get_data()
# Add TripId
gifts['TripId'] = np.arange(gifts.shape[0]).astype(int)


######################################################################################
######################################################################################
######################################################################################
# KMeans
# First, Make Antartica It's own goddam cluster

gifts['Cluster'] = 0

ant_mask = gifts['Latitude'] < 4
gifts.loc[ant_mask,'Cluster'] = 2

ant_mask = gifts['Latitude'] < -60
gifts.loc[ant_mask,'Cluster'] = 1

'''
start_time = time.time()
clusters = KMeans(n_clusters=2)
X = clusters.fit_predict(gifts[np.logical_not(ant_mask)][lat_long])
print 'KMeans Runtime = %.3f seconds'%(time.time()-start_time)
'''

# Save Clusters to gifts
#gifts.loc[np.logical_not(ant_mask),'Cluster'] = X+1

# Look at clusters
grp = gifts.groupby('Cluster')['Weight'].aggregate(['sum','count'])
#grp['sum'].plot(kind='bar')

#plt.figure('Clusters',figsize=(12,8))
#m = Basemap(resolution='c',projection='ortho',lat_0=60.,lon_0=-60.)


# Cast as string
gifts.Cluster = gifts.Cluster.astype(str)


# Do recursive Clustering until Clusters have less than 1000kg
# May see improvement using the real distance metric instead of default KMeans one
def recursive_clustering(data,verbose=True):
    # Split into 3 groups instead of 2 if weight between 2000 and 2800
    if data.Weight.sum() < 2950 and data.Weight.sum() > 1970:
        nc = 3
    else:
        nc = 2
    # Split into Clusters
    X = KMeans(n_clusters = nc).fit_predict(data[lat_long])
    # Record New Cluster
    data.Cluster = np.core.defchararray.add(np.array(data.Cluster).astype(str),X.astype(str))
    # Get total weights of clusters
    w = data.groupby('Cluster')['Weight'].sum()
    
    # If Weight is under 2000 then we'll split up those presents later
    if w.sum() > 980*nc:
        # Loop through clusters
        for new_cluster in w.index:
            # Ignore Clusters with Weight under 1000 - They are done
            if w[new_cluster]>1000:
                # Data for this Cluster is recursively Clustered
                mask = data.Cluster == new_cluster
                data.loc[mask] = recursive_clustering(data[mask])
    else:
        if verbose:
            print w
    return data
    

# Save Parent Cluster
#gifts['ParentCluster'] = gifts.Cluster

# Do Recursive Clustering
for c in np.unique(gifts.Cluster):
    cmask = gifts.Cluster==c
    clust = gifts[cmask].copy()
    # Increase Longitude
    clust.Longitude = clust.Longitude*CONST
    clust = recursive_clustering(clust,verbose = False)
    gifts.loc[cmask] = clust
    
gifts.Longitude = gifts.Longitude/CONST



##########################################
# Fix Trips with > 1000
# Inspect new groups
grp = gifts.groupby('Cluster')['Weight'].aggregate(['sum','count'])

ct = 0
while grp['sum'].max() > 1000 and ct < 7:
    print '~'*33,'\nWeight fixing iterations #'+str(ct+1)
    heavy_clusters = grp[grp['sum'] >1000].index
    for heavy in heavy_clusters:
        print 'Fixing',heavy
        # Find it's sister clusters    
        base = heavy[:-1]
        # Figure out if it's 3 or 2 clusters
        if base+str(2) in grp.index:
            sis = []
            for s in [base+str(0),base+str(1),base+str(2)]:
                if s != heavy:
                    sis.append(s)
        else:
            sis = []
            for s in [base+str(0),base+str(1)]:
                if s != heavy:
                    sis.append(s)
        # Get Centers of Clusters
        centers = []
        for s in sis:
            centers.append(gifts[gifts.Cluster==s][lat_long].mean().tolist())
        
        
              
        
        # For the heavy cluster, calculate the distance for every point to the OTHER center
        distances = gifts[gifts.Cluster==heavy][lat_long]
        # Swap sisters if 2 is closer        
        mLong = np.mean(distances.Longitude)
        if len(centers) > 1 and (mLong-centers[1][1])**2 < (mLong-centers[0][1])**2:
            # Swap Centers
            centers = [centers[1],centers[0]]
            # Swap Sisters
            sis = [sis[1],sis[0]]
        
        
        # HaverSine Distances
        distances['Dist'] = (distances.Longitude - centers[0][1])**2        
        # Sort by distances
        distances.sort_values('Dist',inplace=True)
        # Grab indexes in correct order
        d_ind = np.array(distances.index)
        
        w_light = grp.ix[sis[0]]['sum']
        if ct<2:
            w_light -= 100
        w_heavy = grp.ix[heavy]['sum']
        while d_ind.shape[0] > 0 and max(w_light,w_heavy) > 1000:
            # Move closest present to sister cluster
            w_closest = gifts.ix[d_ind[0]].Weight
            if w_closest + w_light < 1000:
                # Swith Cluster
                gifts.ix[d_ind[0],'Cluster'] = sis[0]
                # Shift Weights
                w_light += w_closest
                w_heavy -= w_closest
            
            # Reduce the index
            d_ind = d_ind[1:]
        
        # If w_heavy is still over 1000, move to the other sister
        if w_heavy > 1000 and len(sis) > 1 and ct > 2:
            # For the heavy cluster, calculate the distance for every point to the OTHER center
            distances = gifts[gifts.Cluster==heavy][lat_long]
            # HaverSine Distances
            #distances['Dist'] = [haversine(tuple(distances.iloc[k][lat_long]),centers[1]) for k in range(distances.shape[0])]
            distances['Dist'] = (distances.Longitude - centers[1][1])**2             
            # Sort by distances
            distances.sort_values('Dist',inplace=True)
            # Grab indexes in correct order
            d_ind = np.array(distances.index)
            
            w_light = grp.ix[sis[1]]['sum']
            while d_ind.shape[0] > 0 and max(w_light,w_heavy) > 1000:
                # Move closest present to sister cluster
                w_closest = gifts.ix[d_ind[0]].Weight
                if w_closest + w_light < 1000:
                    # Swith Cluster
                    gifts.ix[d_ind[0],'Cluster'] = sis[1]
                    # Shift Weights
                    w_light += w_closest
                    w_heavy -= w_closest
                
                # Reduce the index
                d_ind = d_ind[1:]
    
    # Regroup Clusters to See max weight
    grp = gifts.groupby('Cluster')['Weight'].aggregate(['sum','count'])
    
    ct += 1


#################################
# Create TripId from Cluster
def Cluster_to_TripId(gifts):
    cl = np.unique(gifts.Cluster)
    for k in range(cl.shape[0]):
        cl_mask = gifts.Cluster == cl[k]
        gifts.loc[cl_mask,'TripId'] = k
    return gifts


# See improvement
gifts = Cluster_to_TripId(gifts)
new_wrw = WRW(gifts,show_warning=False)
print 'WRW after Recursive Clustering =',new_wrw




############################################################
############################################################
# Now Optimize Trips
# Loop Through Trips and sort by Latitude


gifts_new = pd.DataFrame(columns=gifts.columns)
for tID in np.unique(gifts.TripId):
    # Grab a test Trip
    test_trip = gifts[gifts.TripId == tID].copy()
    
    # Sort largest to smallest
    gifts_new = pd.concat([gifts_new,test_trip.sort_values('Latitude',ascending=False)])
    


# Check improvement again

new_wrw = WRW(gifts_new,show_warning=False)
print 'WRW after Latitude Sorting =',new_wrw
print 'Long Mult =',CONST


# Write a submission
write_sub(gifts_new,'Cluster_LatSort_v9.csv')









############################################################
############################################################
# NCLust = 7

# 15383453727 - v1 1.0
# 13842070489 - v2 4.0
# 13569066776 - v3 6.0
# 13214598341 - v4 10.0
# 12925131261 - v5 20.0
# 12749261512 - v6 40.0
# 12647010113 - v7 80.0
# 12607486576 - v8 200.0
# 12603192571 - v9 1000

# 12580666762

# 12539513411 - 300 / nclust = 3

# 12633212464 - CONTS = 300/.1, nclust = 3
# 12541668262 - CONTS = 300/.5, nclust = 3
# 12541239886 - CONTS = 300/1.5, nclust = 2
# 12541239886 - CONTS = 300/1.1, nclust = 2
#  - CONTS = 500/1, nclust = 3, No Antartica


###LBLBLBLBLB
# 12268695239 - Bound



############################################################
############################################################
# Optomize Trips within Clusters
# Need something special for Antartica

############################################################
############################################################
# Now Optimize Trips
# This part isn't done yet

# Copy gifts new over
gifts = gifts_new.copy()
# Calculate the LB for each present
gifts['LB'] = haversine_NP(gifts)

'''
# Grab a test Trip
test_trip = gifts[gifts.TripId == 1200].copy()

# Sort largest to smallest
test_trip.sort_values(by='Latitude',ascending=False,inplace=True)



start_time = time.time()
G=200
test_trip_,wrw,swaps,X = anneal_trip(test_trip,G=G,alpha=.5,beta=10)
print 'Annealing Iterations = %.0f\nRuntime = %.3f seconds'%(G,time.time()-start_time)
print 'Acceptance %% = %.2f'%(np.mean(swaps)*100)

if 0:
    # Plot WRW
    plt.figure('Simulated Annealing',figsize=(12,8))
    WRW = np.array(wrw)
    plt.plot(wrw)
    plt.ylabel('Weighted Reindeer Weariness',fontsize=17)
    plt.xlabel('Iteration',fontsize=17)
    plt.title('Simulated Annealing',fontsize=20)
    plt.tight_layout()
    
    
    plt.figure('Path')
    plt.plot(test_trip['Longitude'],test_trip['Latitude'],color='red',label='Prior')
    plt.plot(test_trip_['Longitude'],test_trip_['Latitude'],color='green',label='Best')
    
    
    plt.figure('Iterations',figsize=(14,9))
    for i in range(9):#range(X.shape[0]):
        
        plt.subplot(int(str(33)+str(i+1)))
        plt.plot(X[i*G/100/9,:,0],X[i*G/100/9,:,1])
        plt.title('Iteration #'+str(i*G/100*9))
        
    plt.tight_layout()

'''

###########################################
###########################################

def intialize_trip_metrics(gifts,tripid):
    # Returns TripId, LB, WRW, and AvgLong for a trip
    trip = gifts[gifts.Cluster==tripid].copy()
    
    # Get the distances
    d = haversine_vect(trip.iloc[:-1],trip.iloc[1:])
    trip['Distance'] = 0
    trip.iloc[1:,6] = d
    trip.iloc[0,6] = trip.iloc[0,5]
    
    # Lower Bound for trip
    # Sum the Lower Bounds * Weights + the largest Lower Bound * 10 for sleigh
    LB = (trip.LB*trip.Weight).sum() + 2*trip.LB.max()*10
    
    # WRW for trip
    # Sum actual cumulative distances * Weights + sled weight including trip home
    wrw = np.sum((trip.Distance).cumsum()*trip.Weight) + (trip.iloc[-1]['LB']+trip.Distance.sum())*10
    
    # Compute Average Longitude
    avg_long = np.mean(trip['Longitude'])
        
    # Drop Cluster and id
    trip.drop(['Cluster','TripId'],axis=1,inplace=True)
    
    return trip,[LB,wrw,avg_long]


    
# Need to generate table with trip Metrics
trip_metrics = grp.copy()
trip_metrics.columns = ['Weight','Ct']

# Need LowerBound of Trip, Trip WRW, WRW Room, and Trip avg Longitude
trip_metrics['LB'] = 0
trip_metrics['WRW'] = 0
trip_metrics['AvgLong'] = 0

# Create Master Trip List
T = []

for t in trip_metrics.index:
    # Get Metrics
    trip_temp,metrics = intialize_trip_metrics(gifts,t)
    # Add Trip to Trip list
    T.append(trip_temp)
    # Add Metrics to Trip Metrics
    trip_metrics.ix[t,['LB','WRW','AvgLong']] = metrics
    
trip_metrics.index = np.arange(trip_metrics.shape[0]).astype(int)

# Save files as pkl
with open('trip_metrics_v1.pkl','w') as f:
    pickle.dump(trip_metrics,f)
with open('trip_list_v1.pkl','w') as f:
    pickle.dump(T,f)





