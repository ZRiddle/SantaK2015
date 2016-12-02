# -*- coding: utf-8 -*-
"""
@author: zach.riddle
"""

import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.pylab import cm
import seaborn
from sklearn.cluster import KMeans
import pickle

from util import *

lat_long = ['Latitude','Longitude']

# Read in Data
# Save files as pkl
with open('trip_metrics_v1.pkl','r') as f:
    trip_metrics = pickle.load(f)
with open('trip_list_v1.pkl','r') as f:
    T = pickle.load(f)
with open('gifts_v1.pkl','r') as f:
    gifts = pickle.load(f)


def add_NorthPole(df):
    h = ['Latitude','Longitude','Weight','LB']
    # Insert start and end
    start = pd.DataFrame(data=[[90,0,0,0]],index=[0],columns = h)
    end = pd.DataFrame(data=[[90,0,10,0]],index=[100001],columns = h)
    return start.append(df[h].copy()).append(end)

def get_distances(trip):
    # Returns an array of the distances
    dist = [0]
    for i in range(1,trip.shape[0]):
        dist.append(haversine(tuple(trip.iloc[i-1][lat_long]),tuple(trip.iloc[i][lat_long])))
    return dist

def WRW_trip(trip,d=[]):
    if len(d) == trip.shape[0]:
        d = np.array(d)
    else:
        d = np.array(trip.Distance)
    return (d.cumsum()*trip.Weight).sum()


def anneal_update(trip,ind,d):
    trip = trip.ix[ind]
    trip['Distance'] = d
    return trip


def anneal_trip(trip,G=500,alpha=0.03,beta=10,verbose=True):
    # Add North Pole to start and end
    trip = add_NorthPole(trip)
    
    # Create column with previous distance
    trip['Distance'] = get_distances(trip)
    best_trip = trip.copy()
    ###############################
    # Simulated Annealing
    
    # Constants
    WRW_pre = WRW_trip(trip) # Prior WRW
    WRW_post = WRW_pre # Post-Swap WRW
    WRW_low = WRW_pre
    
    ind = np.array(trip.index) # Copy index
    # Get the distances
    d = get_distances(trip.ix[ind])
    
    # Records
    WRW = [WRW_pre]
    swaps = []
    X = np.zeros((G/100,trip.shape[0]-2,2))
    
    for g in range(G):
        # Choose a random position to swap
        rand = np.random.randint(trip.shape[0]-3)+1 # Cannot swap first or last
        #rand2 = np.random.randint(trip.shape[0]-2)+1 # Cannot swap first or last
        
        # Swap the index with the one after it
        ind = np.array(trip.index) # Copy index
        a = ind[rand] # copy i
        ind[rand] = ind[rand+1] # copy j to i
        ind[rand+1] = a # copy i to j
        
        # Calculate the new distances for only 3 paths, involving 4 points
        d_temp = get_distances(trip.ix[ind[rand-1:rand+3]])
        # Get the distances
        d[rand:rand+3] = d_temp[1:]
        
        '''
        # Calculate the new distances for only 2 paths, involving 3 points
        d_temp = get_distances(trip.ix[ind[rand2-1:rand2+2]])
        # Get the distances
        d[rand2:rand2+2] = d_temp[1:]
        '''
        
        # Calculate the new WRW
        WRW_post = WRW_trip(trip.ix[ind],d)
        
        if WRW_post < WRW_pre:
            # If it's better update
            trip = anneal_update(trip,ind,d)
            WRW_pre = WRW_post
        else:
            # If it's worse, upate with some probability
            p = max((min(WRW_pre/WRW_post,1)-alpha-(.2*g/G)),0)**beta
            # Generate a random number and swap
            if p > np.random.rand():
                trip = anneal_update(trip,ind,d)
                WRW_pre = WRW_post
                swaps.append(1)
            else:
                # Do nothing
                swaps.append(0)
                
        WRW.append(WRW_pre)
        if WRW_pre < WRW_low:
            WRW_low = WRW_pre
            best_trip = trip.copy()
        # Print outputs
        if verbose and (g+1)%100==0:
            print '%.0f Iteration Complete...'%(g+1)
            print ' -Best WRW = %.0f'%(WRW_low)
            X[g/100,:] = np.array(trip.ix[ind][lat_long])[1:-1]
    return best_trip.iloc[1:trip.shape[0]-1],WRW,swaps,X




# Plot Trips
def plot_trip(T,tripid,newplot=True,color='green'):
    temp = T[tripid]
    if newplot:
        plt.figure('Trip',figsize=(11,8))
    plt.scatter(temp['Longitude'], temp['Latitude'], s=temp['Weight'], color='red')
    plt.plot(temp['Longitude'],temp['Latitude'], color=color)



def plot_all_presents(T):
    # define the colormap
    cmap = plt.cm.RdYlGn
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(len(T))]
    np.random.shuffle(cmaplist)
    # Plot Figure    
    plt.figure("SANTA!!!!!!",figsize=(12,9))
    
    for tripid in range(len(T)):
        temp = T[tripid].iloc[1:-1]
        plt.scatter(temp['Longitude'], temp['Latitude'], s=temp['Weight'], color=cmaplist[tripid])
        plt.plot(temp['Longitude'],temp['Latitude'], color='red')



'''
#####################################################################
#####################################################################
#####################################################################
## Plots
############################################
# Find Worst Trips
headroom = trip_metrics.WRW - trip_metrics.LB
headroom.sort_values(inplace=True,ascending=False)

#plot_trip(T,headroom.index[0])

#plot_all_presents(T)


n = 55
# create the new map
cmap = cm.get_cmap('winter') 
colors = [cmap(1.*i/n) for i in range(n)]
# create the new map
cmap = cm.get_cmap('winter', 33) 

plt.figure('Worst Trips',figsize=(17,11))
plt.scatter(gifts.Longitude,gifts.Latitude,color = 'gray')
c=0
coords = [360,0,180,0]
for tripid in headroom.index[:50]:
    #range(510,530):#
    plot_trip(T,tripid,newplot=False,color=colors[c+1])
    c+=1
    # Get boundries
    if T[tripid].iloc[1:-1].Longitude.min() + 180 < coords[0]:
        coords[0] = T[tripid].iloc[1:-1].Longitude.min() + 180
    if T[tripid].iloc[1:-1].Longitude.max() + 180 > coords[1]:
        coords[1] = T[tripid].iloc[1:-1].Longitude.max() + 180
    if T[tripid].iloc[1:-1].Latitude.min() + 90 < coords[2]:
        coords[2] = T[tripid].iloc[1:-1].Latitude.min() + 90
    if T[tripid].iloc[1:-1].Latitude.max() + 90 > coords[3]:
        coords[3] = T[tripid].iloc[1:-1].Latitude.max() + 90

band = .1
plt.xlim(coords[0]*(1-band)-180,coords[1]*(1+band)-180)
plt.ylim(coords[2]*(1-band)-90,coords[3]*(1+band)-90)
plt.title('Worst Trips - Most Wasted Movement',fontsize=20)
plt.tight_layout()

print 'Calculated Lower Bound for these trips =',trip_metrics.LB.sum()
'''



'''
####################################################
# Find Lightest Trips
headroom = trip_metrics.Weight.copy()
headroom.sort_values(inplace=True)#,ascending=False)

#plot_trip(T,headroom.index[0])
#plot_all_presents(T)


def plot_trips(trips,title=''):
    n = len(trips)+2
    # create the new map
    cmap = cm.get_cmap('winter') 
    colors = [cmap(1.*i/n) for i in range(n)]
    
    plt.figure('Smallest Trips',figsize=(17,11))
    plt.scatter(gifts.Longitude,gifts.Latitude,color = 'gray')
    c=0
    coords = [360,0,180,0]
    for tripid in trips:
        #range(510,530):#
        plot_trip(T,tripid,newplot=False,color=colors[c+1])
        c += 1
        # Get boundries
        if T[tripid].iloc[1:-1].Longitude.min() + 180 < coords[0]:
            coords[0] = T[tripid].iloc[1:-1].Longitude.min() + 180
        if T[tripid].iloc[1:-1].Longitude.max() + 180 > coords[1]:
            coords[1] = T[tripid].iloc[1:-1].Longitude.max() + 180
        if T[tripid].iloc[1:-1].Latitude.min() + 90 < coords[2]:
            coords[2] = T[tripid].iloc[1:-1].Latitude.min() + 90
        if T[tripid].iloc[1:-1].Latitude.max() + 90 > coords[3]:
            coords[3] = T[tripid].iloc[1:-1].Latitude.max() + 90
    
    band = .1
    plt.xlim(coords[0]*(1-band)-180,coords[1]*(1+band)-180)
    plt.ylim(coords[2]*(1-band)-90,coords[3]*(1+band)-90)
    plt.title(title,fontsize=20)
    plt.tight_layout()


plot_trips(headroom.index[:24],title='Lightest Trips - Total Weight < 400')
print 'Lightest Trips:\n',trip_metrics.sort_values('Weight').head()
'''



#### Notes on updating ###
# Adding Gift:
# If gift i is inserted into trip T, then T_i and T_{i+1} Distances need to be updated
# Then for T, all the trip metrics will need to be recomputed

# Removing Gift
# If gift i is removed from trip T, then T_{i+1} Distance needs to be updated
# Then for T, all the trip metrics will need to be recomputed








class Trips:
    '''
    Trips Class for optimizing Santa's routes
    
    Properties:
    T : list of pandas dfs
        Each df is a single trip
    trip_metrics : pandas df
        1 row per trip
    gifts : pandas df
        gifts dataset
    wrw : float
        Weighted Reindeer Weariness - Loss Function
        https://www.kaggle.com/c/santas-stolen-sleigh/details/evaluation
    '''
    
    lat_long = ['Latitude','Longitude']
    
    def __init__(self,T=None,trip_metrics=None,gifts=None,path=''):
        # Set initial tables
        if T == None:
            with open('trip_list'+path+'.pkl','r') as f:
                self.T = pickle.load(f)
        else:
            self.T = T
        if trip_metrics == None:
            with open('trip_metrics'+path+'.pkl','r') as f:
                self.trip_metrics = pickle.load(f)
        else:
            self.trip_metrics = trip_metrics
        if gifts == None:
            with open('gifts'+path+'.pkl','r') as f:
                self.gifts = pickle.load(f)
        else:
            self.gifts = gifts
            
        if 'SPF' not in self.trip_metrics.columns:
            # Add SouthPoleFlag
            self.trip_metrics['SPF'] = 0
            for tr in range(len(self.T)):
                if self.T[tr].Latitude.mean() < -60:
                    self.trip_metrics.iloc[tr,5] = 1
        # Set wrw
        self.wrw = self.trip_metrics.WRW.sum()
        
    
    def haversine(self,v1,v2):
        # calculate haversine
        lat = np.array(np.radians(v1['Latitude'])) - np.array(np.radians(v2['Latitude']))
        lng = np.array(np.radians(v1['Longitude'])) - np.array(np.radians(v2['Longitude']))
        d = np.sin(lat / 2) ** 2 + np.array(np.cos(np.radians(v1['Latitude']))) *\
            np.array(np.cos(np.radians(v2['Latitude']))) * np.sin(lng / 2) ** 2
        h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
        return h  # in kilometers
        
    def haversine_NP(self,v1):
        # calculate haversine
        lat = np.radians(v1['Latitude']) - np.radians(north_pole[0])
        lng = np.radians(v1['Longitude']) - np.radians(north_pole[1])
        d = np.sin(lat / 2) ** 2 + np.cos(np.radians(v1['Latitude'])) * np.cos(np.radians(north_pole[0])) * np.sin(lng / 2) ** 2
        h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
        return h  # in kilometers
        
    def remove_gift(self,tripid,giftid,update=True):
        '''0
        Removes a gift from a trip
        Updates the trip
        Returns the gift to add into another trip
        '''
        # Copy gift
        loose_gift = self.T[tripid].ix[giftid].copy()
        # Record index
        ind = self.T[tripid].index.get_loc(giftid)
        # Delete gift
        self.T[tripid].drop(giftid,inplace=True)
        
        if update:
            # Recalculate the Distance for the gift that is now at the index
            if ind == 0:
                self.T[tripid].iloc[ind,4] = self.haversine_NP(self.T[tripid].iloc[ind][lat_long])
            elif ind < self.T[tripid].shape[0]:
                self.T[tripid].iloc[ind,4] = self.haversine(self.T[tripid].iloc[ind-1][lat_long],self.T[tripid].iloc[ind][lat_long])
        
        
            # Update metrics for trip id
            self.update_trip_metrics(tripid)
                
        # Return gift
        return loose_gift
    
    def update_trip_metrics(self,tripid):
        '''
        Updates the trip_metrics dataframe
        '''
        # Don't change South Pole Flag
        SPF = self.trip_metrics.ix[tripid].SPF
        
        # Updates Weight, Count, LB, WRW, and AvgLong for a trip
        # Weight - sum weights
        weight = self.T[tripid].Weight.sum()
        
        # Count - number of presents
        ct = self.T[tripid].shape[0]
        
        # Lower Bound for trip
        # Sum the Lower Bounds * Weights + the largest LB Distances*2*10 for roundtrip sleigh
        LB = (self.T[tripid].LB*self.T[tripid].Weight).sum() + 2*self.T[tripid].LB.max()*10
        
        # WRW for trip
        # Sum actual cumulative distances * Weights + sled weight including trip home
        wrw = np.sum((self.T[tripid].Distance).cumsum()*self.T[tripid].Weight) 
        wrw += (self.T[tripid].iloc[-1]['LB']+self.T[tripid].Distance.sum())*10
        
        # Compute Average Longitude
        avg_long = np.mean(self.T[tripid]['Longitude'])
        
        # Update row
        self.trip_metrics.ix[tripid] = [weight,ct,LB,wrw,avg_long,SPF]
        
        # Update overall wrw
        self.wrw = self.trip_metrics.WRW.sum()
        
        
    def add_gift(self,tripid,loose_gift):
        '''
        Removes a gift from a trip
        Updates the trip
        Returns the gift to add into another trip
        '''
        # Add to trip
        self.T[tripid] = self.T[tripid].append(loose_gift)
        
        # Put into the correct spot based on Latitude
        g_id = self.T[tripid].index[-1]
        self.T[tripid].sort_values(by='Latitude',inplace=True,ascending=False)
        ind = self.T[tripid].index.get_loc(g_id)
        
        # Recalculate the Distance for the gift that is now at the index and the one after it
        if ind == 0:
            self.T[tripid].iloc[ind,4] = self.T[tripid].iloc[ind]['LB']
        else:
            self.T[tripid].iloc[ind,4] = self.haversine(self.T[tripid].iloc[ind-1][lat_long],self.T[tripid].iloc[ind][lat_long])
        if ind+1 < self.T[tripid].shape[0]:
            self.T[tripid].iloc[ind+1,4] = self.haversine(self.T[tripid].iloc[ind][lat_long],self.T[tripid].iloc[ind+1][lat_long])
        
        # Update metrics for trip id
        self.update_trip_metrics(tripid)
                
    ########################################
                #Plotting
    # Plot Trips
    def plot_trip(self,tripid,newplot=True,color='green'):
        temp = self.T[tripid].copy()
        if newplot:
            plt.figure('Trip',figsize=(8,5))
        plt.scatter(temp['Longitude'], temp['Latitude'], s=temp['Weight'], color='red')
        plt.plot(temp['Longitude'],temp['Latitude'], color=color)
    

    def plot_trips(self,trips,title='Trips',newplot=True):
        n = len(trips)+2
        # create the new map
        cmap = cm.get_cmap('winter') 
        colors = [cmap(1.*i/n) for i in range(n)]
        if newplot:
            plt.figure(title,figsize=(12,8))
            plt.subplot(211)
        else:
            plt.subplot(212)
        plt.scatter(self.gifts.Longitude,gifts.Latitude,color = 'gray')
        c=0
        coords = [360,0,180,0]
        for tripid in trips:
            #range(510,530):#
            self.plot_trip(tripid,newplot=False,color=colors[c+1])
            c += 1
            # Get boundries
            if self.T[tripid].Longitude.min() + 180 < coords[0]:
                coords[0] = self.T[tripid].Longitude.min() + 180
            if self.T[tripid].Longitude.max() + 180 > coords[1]:
                coords[1] = self.T[tripid].Longitude.max() + 180
            if self.T[tripid].Latitude.min() + 90 < coords[2]:
                coords[2] = self.T[tripid].Latitude.min() + 90
            if self.T[tripid].Latitude.max() + 90 > coords[3]:
                coords[3] = self.T[tripid].Latitude.max() + 90
        
        band = .1
        plt.xlim(coords[0]*(1-band)-180,coords[1]*(1+band)-180)
        plt.ylim(coords[2]*(1-band)-90,coords[3]*(1+band)-90)
        plt.title(title,fontsize=20)
        plt.tight_layout()

    def write_sub(self,filename):
        sub = pd.DataFrame(columns=T[0].columns)
        for tID in self.trip_metrics.index:
            # Add trip to DF
            temp = self.T[tID].copy()
            temp['LB'] = tID
            sub = pd.concat([sub,temp])
    
        sub.index.names = ['GiftId']
        sub['TripId'] = sub['LB'].astype(int)
        sub['TripId'].to_csv(filename,header = ['TripId'])


    def save_data(self,path=''):
        # Save files as pkl
        with open('trip_metrics'+path+'.pkl','w') as f:
            pickle.dump(self.trip_metrics,f)
        with open('trip_list'+path+'.pkl','w') as f:
            pickle.dump(self.T,f)
        with open('gifts'+path+'.pkl','w') as f:
            pickle.dump(self.gifts,f)


    
    def destroy_trip(self,tripid):
        '''
        Destroy a trip - returning a list of loose presents
        '''
        # Get list of free gifts
        loose_gifts = []
        for g in self.T[tripid].index:
            l = self.remove_gift(tripid,g,update=False)
            loose_gifts.append(l)
        # Update trip metrics to 0s
        self.trip_metrics.drop(tripid,inplace=True)
        return loose_gifts
        
    def diffuse_gifts(self,loose_gifts):
        # Keep a south pole flag for this algorithm
        SP = 0
        if loose_gifts[-1].Latitude < -60:
            #print 'SP = 1'
            SP = 1
        
        # Step 1 - Add all gifts to the closest trip
        for g in loose_gifts:
            # Find Nearest Trip           
            new_trip = ((self.trip_metrics[self.trip_metrics.SPF == SP].AvgLong - g.Longitude)**2).sort_values().index[0]
            # Add present to trip
            self.add_gift(new_trip,g)
        
        if np.random.rand() > .5:
            self.diffuse_east(SP)
            self.diffuse_west(SP)
        else:
            self.diffuse_west(SP)
            self.diffuse_east(SP)
        
        
        
    def diffuse_east(self,SP,cap=1000):
        # Step 2 - Start with the heaviest trip, Diffuse east
        curr_trip = self.trip_metrics[self.trip_metrics.SPF == SP].sort_values('AvgLong',ascending=True).index[0]
        
        mask = np.logical_and(self.trip_metrics.SPF == SP,self.trip_metrics.Weight>cap)
        eastmost_trip = self.trip_metrics[mask].sort_values('AvgLong',ascending=False).index[0]
        # Keep going east until weight satisfied
        while curr_trip != eastmost_trip and mask.sum():
            # Keep removing presents until weight satisfied
            print '   -Eastward - trip #'+str(curr_trip)
            while self.trip_metrics.ix[curr_trip].Weight > cap:
                # Take east-most gift
                east_gift = self.T[curr_trip].sort_values('Longitude',ascending=False).index[0]
                loose = self.remove_gift(curr_trip,east_gift)
                # Add it to it's nearest trip (Not Itself!)
                new_trip = ((self.trip_metrics[self.trip_metrics.SPF == SP].AvgLong - loose.Longitude)**2).sort_values().index[:2]
                if new_trip[0]==curr_trip:
                    new_trip = new_trip[1]
                else:
                    new_trip = new_trip[0]
                self.add_gift(new_trip,loose)
                
            # Increment current trip to next eastward trip over 1000
            mask = np.logical_and(self.trip_metrics.AvgLong > self.trip_metrics.ix[curr_trip].AvgLong,self.trip_metrics.Weight > cap)
            mask = np.logical_and(mask,self.trip_metrics.SPF == SP)
            if mask.sum():
                curr_trip = self.trip_metrics[mask].sort_values('AvgLong').index[0]
        
    def diffuse_west(self,SP,cap=1000):
        # Step 3 - Diffuse west
        curr_trip = self.trip_metrics[self.trip_metrics.SPF == SP].sort_values('AvgLong',ascending=False).index[0]
        
        # Create Eastward Mask
        mask = np.logical_and(self.trip_metrics.SPF == SP,self.trip_metrics.Weight>cap)
        eastmost_trip = self.trip_metrics[mask].sort_values('AvgLong',ascending=True).index[0]
        # Keep going east until weight satisfied
        while curr_trip != eastmost_trip and mask.sum():
            # Keep removing presents until weight satisfied from current trip
            print '   -Westward - trip #'+str(curr_trip)
            while self.trip_metrics.ix[curr_trip].Weight > cap:
                # Take east-most gift
                east_gift = self.T[curr_trip].sort_values('Longitude',ascending=True).index[0]
                loose = self.remove_gift(curr_trip,east_gift)
                # Add it to it's nearest trip (Not Itself!)
                new_trip = ((self.trip_metrics[self.trip_metrics.SPF == SP].AvgLong - loose.Longitude)**2).sort_values().index[:2]
                if new_trip[0]==curr_trip:
                    new_trip = new_trip[1]
                else:
                    new_trip = new_trip[0]
                self.add_gift(new_trip,loose)
                
            # Increment current trip to next wastward trip over 1000
            mask = np.logical_and(self.trip_metrics.AvgLong < self.trip_metrics.ix[curr_trip].AvgLong,self.trip_metrics.Weight > cap)
            mask = np.logical_and(mask,self.trip_metrics.SPF == SP)    
            if mask.sum():
                curr_trip = self.trip_metrics[mask].sort_values('AvgLong',ascending = False).index[0]
        
    def destroy_diffuse(self,tripid):
        loose_gifts = self.destroy_trip(tripid)
        self.diffuse_gifts(loose_gifts)
    
'''
# Plot the Average Longitude
plt.figure('Trip Longitudes',figsize=(11,9))
temp_metrics = Santa.trip_metrics.copy()
temp_metrics.AvgLong = np.round(temp_metrics.AvgLong,-1).astype(int)
temp_metrics.groupby('AvgLong')['Weight'].sum().plot(kind='bar')
'''


##############################
def swap_worst_trip(Santa,ntry=10,plot_it = True,greedOfTheNoob=True,SPF = True,verbose=1):
    start_wrw = Santa.wrw
    stime = time.time()
    for k in range(ntry):
        pp= '~'*33+'\nLoop #'+str(k)
        
        headroom = Santa.trip_metrics.WRW - Santa.trip_metrics.LB
        headroom.sort_values(inplace=True,ascending=False)
        
        # Plot single worst trip
        #plot_trip(T,headroom.index[0],newplot=False,color=colors[c+1])
        
        # Look at worst trip
        # Choose Worst Trip Stochastically
        p = headroom*0+10
        p /= p.sum()
        w = np.random.choice(np.arange(p.shape[0]),p=p)
        worst_trip = headroom.index[w]
        
        #Santa.trip_metrics.ix[worst_trip]
        #Santa.T[worst_trip].head()
        
        # Find the outlier present in the worst trip
        # Find the outlier points
        dis = (Santa.T[worst_trip].Longitude - Santa.trip_metrics.ix[worst_trip].AvgLong)**2
        dis.sort_values(inplace=True,ascending=False)
        
        # Choose worst Present stochastically
        p = dis.head(20)
        p /= p.sum()
        w = np.random.choice(np.arange(p.shape[0]),p=p)
        outlier = Santa.T[worst_trip].ix[dis.index[w]]
        worst_present = dis.index[w]
        
        # Find trip with closest Avg Long
        # Segregate South Pole
        if SPF:
            SP_mask = Santa.trip_metrics.SPF == Santa.trip_metrics.ix[worst_trip].SPF
            closest_trip = (outlier.Longitude - Santa.trip_metrics[SP_mask].AvgLong)**2
        else:
            closest_trip = (outlier.Longitude - Santa.trip_metrics.AvgLong)**2        
        closest_trip.sort_values(inplace=True)
        
        # Choose new trip stochastically
        new_trip = closest_trip.head(2).index
        if new_trip[0]==worst_trip:
            new_trip = new_trip[1]
        else:
            new_trip = new_trip[0]
        
        if Santa.trip_metrics.ix[new_trip].Weight + Santa.T[worst_trip].ix[worst_present].Weight > 1000:
            if verbose:
                print pp
                print 'Too Heavy'
        else:
            Santa.trip_metrics.ix[closest_trip.head().index]
            if verbose:
                print ' -Worst Trip',worst_trip
                print ' -Worst Present',worst_present
                print ' -New Trip',new_trip
            
            if plot_it:
                Santa.plot_trips([worst_trip,new_trip],title='Before '+str(k))
            
            # Try out the remove and add functions
            # First check the combined WRW for these 2 trips
            wrw0 = Santa.trip_metrics.ix[[worst_trip,new_trip]].WRW.sum()
            
            # Remove it
            pres = Santa.remove_gift(worst_trip,worst_present)
            
            # Add it
            Santa.add_gift(new_trip,pres)
            wrw2 = Santa.trip_metrics.ix[[worst_trip,new_trip]].WRW.sum()
            if verbose:
                print ' -WRW reduction after swap =',1-wrw2/wrw0
                print ' -Absolute WRW Reduction =',wrw0-wrw2
            elif wrw2 < wrw0 and (k+1)%1000==0:
                print pp
                print ' -Sucess! Reduced by',wrw0-wrw2
                
            if plot_it:
                Santa.plot_trips([worst_trip,new_trip],title='After '+str(k),newplot=False)
                
            if wrw0 < wrw2 and greedOfTheNoob:
                # UNDO!!!!
                pres = Santa.remove_gift(new_trip,worst_present)
                Santa.add_gift(worst_trip,pres)                    
                    
    perc_gained_lb = (start_wrw-Santa.wrw) / (start_wrw - Santa.trip_metrics.LB.sum())
    print '\nStarting WRW = %0f\nNew WRW = %.0f\nWRW %% Gained = %.2f%%'%(start_wrw,Santa.wrw,perc_gained_lb*100)
    print 'Runtime (minutes) = %.2f'%((time.time()-stime)/60.0)
    return Santa


'''
# Good Trips
good_trips = []
bad_trips = []

gain = 1
total_g = 0
total_l = 0
beg_wrw = Santa.wrw
best_wrw = beg_wrw
wrw_all = []
#while gain > 0 and len(good_trips) < 10:
for k in range(200):
    # Find the lightest trip
    light_trip = Santa.trip_metrics[Santa.trip_metrics.Weight > 0].sort_values('Weight').index[0]
    
    # Record pre WRW
    wrw_start = Santa.wrw
    w = Santa.trip_metrics.ix[light_trip].Weight    
    
    # Destroy it
    print 'Loop %.0f\n -Destroying trip %.0f\n -Weight = %.0f'%(k,light_trip,w)
    Santa.destroy_diffuse(light_trip)
    
    # Check new WRW
    wrw_end = Santa.wrw
    gain = wrw_start - wrw_end
    print ' -Gain = %.0f'%(gain)
    wrw_all.append(wrw_end/1000000)
    if gain > 0:
        good_trips.append(light_trip)
        total_g += gain
    else:
        #print 'Trip',light_trip,'Was a bad move! STOPPPPPPP!!!!!'
        bad_trips.append(light_trip)
        total_l += gain
    if wrw_end < best_wrw:
        best_wrw = wrw_end

print 'Total trips killed =',len(good_trips)+len(bad_trips)
print 'New WRW =',wrw_end
print 'Total WRW Gain = %0f'%(beg_wrw - wrw_end)
print 'Best WRW = %.0f'%best_wrw

Santa.save_data(path='_v3')



# Check The Capacity For 2 groups
w_top = Santa.trip_metrics[Santa.trip_metrics.SPF==0].Weight.sum()
w_bottom = Santa.trip_metrics[Santa.trip_metrics.SPF==1].Weight.sum()
ct_top = Santa.trip_metrics[Santa.trip_metrics.SPF==0].shape[0]
ct_bottom = Santa.trip_metrics[Santa.trip_metrics.SPF==1].shape[0]

print 'Top Avg Capacity = %.1f'%(w_top/ct_top)
print 'Bottom Avg Capacity = %.1f'%(w_bottom/ct_bottom)

# Look for groups of 4 trips, break them into 5 optimally


wrw_all = np.array(wrw_all)
plt.plot(wrw_all)




#####################################
# Shake Down

wrw_before = Santa.wrw
# Go to 150, Diffuse with Cap of 950 ish, Do swapping
cap = 990
while cap >= 950:
    Santa.diffuse_east(1,cap+5)
    Santa.diffuse_west(1,cap)
    Santa.diffuse_east(0,cap+5)
    Santa.diffuse_west(0,cap)
    cap -= 10
    # Calc new WRW
    gain = wrw_before - Santa.wrw
    wrw_before = Santa.wrw
    print 'WRW Gain = %.0f'%gain


Santa.save_data(path='_v3')
'''

#####################################
# Greedy optimize



# Initialize Class
Santa = Trips(path='_v4')


for omg in range(1):
    Santa = swap_worst_trip(Santa,1000,
                            plot_it=False,
                            greedOfTheNoob=True,
                            SPF=True,
                            verbose=0)
                                
    Santa.save_data(path='_v4')





mask = Santa.trip_metrics.SPF == 1
lb = Santa.trip_metrics[mask].WRW - Santa.trip_metrics[mask].LB
room_SP = lb.sum()

mask = Santa.trip_metrics.SPF == 0
lb = Santa.trip_metrics[mask].WRW - Santa.trip_metrics[mask].LB
room_NP = lb.sum()


longmask = Santa.trip_metrics.AvgLong > 0
longmask = np.logical_and(longmask,Santa.trip_metrics.SPF == 1)
Santa.trip_metrics[longmask].sort_values('AvgLong').head(7)

# [741,736,734,737,738,742,744]





# Check if splitting trips is good?

#################################
# Testing WRW

def split_trip(g_temp):
    # Split into 2 trips
    median_long = g_temp.sort_values('Longitude').iloc[g_temp.shape[0]/2].Longitude
    
    mask = g_temp.Longitude < median_long
    g1 = g_temp[mask].copy()
    g2 = g_temp[np.logical_not(mask)].copy()
    
    
    g1.sort_values('Latitude',ascending=False,inplace=True)
    d = haversine_vect(g1.iloc[:-1],g1.iloc[1:])
    g1.iloc[1:,4] = d
    g1.iloc[0,4] = g1.iloc[0,3]
    
    g2.sort_values('Latitude',ascending=False,inplace=True)
    d = haversine_vect(g2.iloc[:-1],g2.iloc[1:])
    g2.iloc[1:,4] = d
    g2.iloc[0,4] = g2.iloc[0,3]

    return g1,g2
    

def wrw_trip(g1):
    WRW1 = (g1.Distance.cumsum()*g1.Weight).sum()
    WRW1 += (g1.Distance.sum()+g1.iloc[-1].LB)*10
    return WRW1





#g_temp = gifts[mask].copy()
g_temp = Santa.T[1218].copy()
'''
g_temp = g_temp.append(Santa.T[736].copy())
g_temp = g_temp.append(Santa.T[734].copy())
g_temp = g_temp.append(Santa.T[737].copy())
g_temp = g_temp.append(Santa.T[738].copy())
g_temp = g_temp.append(Santa.T[742].copy())
g_temp = g_temp.append(Santa.T[744].copy())
'''


print 'Shape =',g_temp.shape
print 'Weight =',g_temp.Weight.sum()

# sort by latitude
g_temp.sort_values('Latitude',ascending=False,inplace=True)
# calculate marginal distances
# Get the distances
d = haversine_vect(g_temp.iloc[:-1],g_temp.iloc[1:])
g_temp['Distance'] = 0
g_temp.iloc[1:,4] = d
g_temp.iloc[0,4] = g_temp.iloc[0,3]

WRW_start = wrw_trip(g_temp)
print 'WRW single trip =',WRW_start

# Calc room on LB
Dmax = g_temp.LB.max()*2*10
g_LB = (g_temp.LB * g_temp.Weight).sum() + Dmax
room = g_LB-WRW_start



g1,g2 = split_trip(g_temp)

WRW1 = wrw_trip(g1)
WRW2 = wrw_trip(g2)

print 'WRW 2 seperate trips =',WRW2+WRW1

##################################
# Plot

n = 3
# create the new map
cmap = cm.get_cmap('winter') 
colors = [cmap(1.*i/n) for i in range(n)]
# create the new map
cmap = cm.get_cmap('winter', 33) 

plt.figure('2 Trips',figsize=(17,11))
plt.scatter(gifts.Longitude,gifts.Latitude,color = 'gray')

plt.scatter(g1['Longitude'], g1['Latitude'], s=g1['Weight'], color='red')
plt.plot(g1['Longitude'],g1['Latitude'], color=colors[1])

plt.scatter(g2['Longitude'], g2['Latitude'], s=g2['Weight'], color='red')
plt.plot(g2['Longitude'],g2['Latitude'], color=colors[2])

coords = [g_temp.Longitude.min(),g_temp.Longitude.max(),
          g_temp.Latitude.min(),g_temp.Latitude.max()]
band = .1
plt.xlim(coords[0]*(1-band),coords[1]*(1+band))
plt.ylim(coords[2]*(1-band),coords[3]*(1+band))

plt.tight_layout()

gain = WRW1+WRW2-WRW_start

room/2+Dmax
print 'Optimal trip count = ',(-room/Dmax)**.5

# Center of Mass Calculation
def CenterMass(g_temp):
    return (g_temp.Weight*(np.arange(g_temp.shape[0])-g_temp.shape[0]/2.0+.5)).mean()/g_temp.Weight.sum()+.5

print 'Center Mass Combined',CenterMass(g_temp)
print 'Center Mass trip1',CenterMass(g1)
print 'Center Mass trip2',CenterMass(g2)


g1_LB = (g1.LB * g1.Weight).sum()
print 'Optimal Trip Count for trip1',((WRW1-g1_LB+Dmax)/Dmax)**.5

g2_LB = (g2.LB * g2.Weight).sum()
print 'Optimal Trip Count for trip1',((WRW2-g2_LB+Dmax)/Dmax)**.5

#CenterMass(pd.DataFrame({'Weight':[0,10,0]}))

# Use solver to get solution for cubic function
def optimal_trip_ct(n_p,w,Dmax):
    d = -1 * n_p * w /(6 * Dmax/1.4)
    return np.roots([1,n_p*d,0,d])[0].real

w_avg = (g_temp.Distance.sum()-g_temp.LB.max())/g_temp.shape[0]
print 'Avg Weights:\n',w_avg,'\n',\
        (g2.Distance.sum()-g2.LB.max())/g2.shape[0],'\n',\
        (g1.Distance.sum()-g1.LB.max())/g1.shape[0]
print '\nShape =',g_temp.shape

optimal_trip_ct(g_temp.shape[0],w_avg,Dmax/20)
optimal_trip_ct(g1.shape[0],(g1.Distance.sum()-g1.LB.max())/g1.shape[0],Dmax/20)




# 52263451 - 2 trips
# 45937092 - 4 trips
# 45630462 - 7 trips
# 45759094 - 8 trips

'''
g11,g12 = split_trip(g1)
g21,g22 = split_trip(g2)

g111,g112 = split_trip(g11)
g121,g122 = split_trip(g12)
g211,g212 = split_trip(g21)
g221,g222 = split_trip(g22)

wrw_trip(g111) + wrw_trip(g112) + wrw_trip(g121) + wrw_trip(g122) +\
wrw_trip(g211) + wrw_trip(g212) + wrw_trip(g221) + wrw_trip(g222)
'''




Santa.write_sub('sub1.csv')



##########################
# Optimize the South Pole with ACTUAL DISTANCES!!!!!!!!!!!!!






####################################################
# Make smaller trips!!! Especially <-125 Long and > 160 Long and around -25 Long

def split_trips(tr_list,n_trips=2):
    '''
    Takes in the trip list
    Splits into specified number of trips
    '''
    # Build list of presents
    g_temp = T[tr_list[0]].copy()
    for g in tr_list[1:]:
        g_temp = g_temp.append(T[g].copy())
        
    # Get total weight
    w_total = g_temp.Weight.sum()
    
    # Split into 2 trips - by weight
    weights = g_temp.sort_values('Longitude').Weight.cumsum()/w_total
    
    step = 1.000000001/n_trips
    cut = 0
    g = []
    while cut < 1:
        mask = np.logical_and(weights > cut,weights <= cut+step)
        # copy subset
        g1 = g_temp[mask].sort_values('Latitude',ascending=False)
        # Rebuild Haversine Distances
        d = haversine_vect(g1.iloc[:-1],g1.iloc[1:])
        g1.iloc[1:,4] = d
        g1.iloc[0,4] = g1.iloc[0,3]
        # Add to list
        g.append(g1)
        cut += step

    return g
    
# [741,736,734,737,738,742,744]
# [1218,1217,1220,1238,1219,1237]
tr_list = [741,736,734,737,738,742,744]

wrw_start = trip_metrics.ix[tr_list].WRW.sum()
# 6 - 18212923
if 1:
    for n_trips in range(4,20):
        g1 = split_trips(tr_list,n_trips)
        wrw_new = 0
        for g in g1:
            wrw_new += wrw_trip(g)
        print '%.0f - %.0f trips'%(wrw_new,n_trips)






