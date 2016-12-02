# -*- coding: utf-8 -*-
"""
@author: zach.riddle
"""

import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.pylab import cm
import random
import pickle

from util import *

lat_long = ['Latitude','Longitude']

# Read in Data
# Save files as pkl
with open('trip_metrics_v9.pkl','r') as f:
    trip_metrics = pickle.load(f)
with open('trip_list_v9.pkl','r') as f:
    T = pickle.load(f)
with open('gifts.pkl','r') as f:
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



def plot_worst_trips(n_trips=10,lightest = False):
    ## Plots
    ############################################
    # Find Worst Trips
    if lightest:
        headroom = Santa.trip_metrics.Weight.copy()
        headroom.sort_values(inplace=True)
    else:   
        headroom = (Santa.trip_metrics.WRW-Santa.trip_metrics.LB)#-1
        headroom.sort_values(inplace=True,ascending=False)
    
    # create the new map
    cmap = cm.get_cmap('winter') 
    colors = [cmap(1.*i/(n_trips+2)) for i in range(n_trips+2)]
    # create the new map
    cmap = cm.get_cmap('winter', n_trips+2) 
    
    plt.figure('Worst Trips',figsize=(17,11))
    plt.scatter(Santa.gifts.Longitude,gifts.Latitude,color = 'gray')
    c=0
    coords = [360,0,180,0]
    for tripid in headroom.index[:n_trips]:
        #range(510,530):#
        plot_trip(Santa.T,tripid,newplot=False,color=colors[c+1])
        c+=1
        # Get boundries
        if Santa.T[tripid].Longitude.min() + 182 < coords[0]:
            coords[0] = Santa.T[tripid].Longitude.min() + 182
        if Santa.T[tripid].Longitude.max() + 182 > coords[1]:
            coords[1] = Santa.T[tripid].Longitude.max() + 182
        if Santa.T[tripid].Latitude.min() + 92 < coords[2]:
            coords[2] = Santa.T[tripid].Latitude.min() + 92
        if Santa.T[tripid].Latitude.max() + 92 > coords[3]:
            coords[3] = Santa.T[tripid].Latitude.max() + 92
    
    band = .1
    plt.xlim(coords[0]*(1-band)-182,min(coords[1]*(1+band)-182,180))
    plt.ylim(coords[2]*(1-band)-92,min(coords[3]*(1+band)-92,90))
    plt.title('Worst Trips - Most Wasted Movement',fontsize=20)
    plt.tight_layout()
    
    #print 'Calculated Lower Bound for these trips =',Santa.trip_metrics.LB.sum()




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
            with open('gifts.pkl','r') as f:
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
        SPF = 0
        if self.T[tripid].Latitude.min() < -60:
            SPF = 1
        
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
        with open('gifts.pkl','w') as f:
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
        
        
        
    def diffuse_east(self,SP,cap=1000,verbose=False):
        if self.trip_metrics.Weight.max()<cap:
            return 0
        # Step 2 - Start with the heaviest trip, Diffuse east
        if SP > -1:
            curr_trip = self.trip_metrics[self.trip_metrics.SPF == SP].sort_values('AvgLong',ascending=True).index[0]
        else:
            curr_trip = self.trip_metrics.sort_values('AvgLong',ascending=True).index[0]
        mask = self.trip_metrics.Weight>cap
        if SP > -1:
            mask = np.logical_and(self.trip_metrics.SPF == SP,mask)
        if mask.sum():
            eastmost_trip = self.trip_metrics.sort_values('AvgLong',ascending=False).index[0]
        # Keep going east until weight satisfied
        while mask.sum() and curr_trip != eastmost_trip:
            # Keep removing presents until weight satisfied
            if verbose:
                print '   -Eastward - trip #'+str(curr_trip)
            while self.trip_metrics.ix[curr_trip].Weight > cap:
                # Take east-most gift
                east_gift = self.T[curr_trip].sort_values('Longitude',ascending=False).index[0]
                loose = self.remove_gift(curr_trip,east_gift)
                # Add it to it's nearest trip (Not Itself!)
                if SP > -1:
                    new_trip = ((self.trip_metrics[self.trip_metrics.SPF == SP].AvgLong - loose.Longitude)**2).sort_values().index[:2]
                else:
                    new_trip = ((self.trip_metrics.AvgLong - loose.Longitude)**2).sort_values().index[:2]
                
                if new_trip[0]==curr_trip:
                    new_trip = new_trip[1]
                else:
                    new_trip = new_trip[0]
                self.add_gift(new_trip,loose)
                
            # Increment current trip to next eastward trip over 1000
            mask = np.logical_and(self.trip_metrics.AvgLong > self.trip_metrics.ix[curr_trip].AvgLong,self.trip_metrics.Weight > cap)
            if SP > -1:
                mask = np.logical_and(mask,self.trip_metrics.SPF == SP)
            if mask.sum():
                curr_trip = self.trip_metrics[mask].sort_values('AvgLong').index[0]
        
    def diffuse_west(self,SP,cap=1000,verbose=False):
        if self.trip_metrics.Weight.max()<cap:
            return 0
        # Step 3 - Diffuse west
        curr_trip = self.trip_metrics[self.trip_metrics.SPF == SP].sort_values('AvgLong',ascending=False).index[0]
        
        # Create Eastward Mask
        mask = np.logical_and(self.trip_metrics.SPF == SP,self.trip_metrics.Weight>cap)
        if mask.sum():
            eastmost_trip = self.trip_metrics[mask].sort_values('AvgLong',ascending=True).index[0]
        # Keep going east until weight satisfied
        while mask.sum() and curr_trip != eastmost_trip:
            # Keep removing presents until weight satisfied from current trip
            if verbose:
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
    
    
    def optimal_trip_count(self,tr_list,r1=2,r2=10,r3=1,verbose=False):
        '''
        Inputs
        trip list
        Search Parameters
        
        Returns : int
            Optmial trip count
        '''
        wrw_best = 0
        n_best = 0
        g_best = []
        for n_trips in range(r1,r2,r3):
            g1 = self.split_trips(tr_list,n_trips)
            wrw_new = 0
            for g in g1:
                wrw_new += self.wrw_trip(g)
            if wrw_new < wrw_best or wrw_best == 0:
                wrw_best = wrw_new
                n_best = n_trips
                g_best = g1
            if verbose:
                print '%.0f - %.0f trips'%(wrw_new,n_trips)
        if verbose:
            print '\nOptimal Trip Count =',n_best
            print 'Avg Weight Per Trip = %.0f'%(self.trip_metrics.ix[tr_list].Weight.sum()/n_best),'\n'
        return n_best,g_best
    
    
    def split_trips(self,tr_list,n_trips=2):
        '''
        Takes in the trip list
        Splits into specified number of trips
        '''
        # Build list of presents
        g_temp = self.T[tr_list[0]].copy()
        for g in tr_list[1:]:
            g_temp = g_temp.append(self.T[g].copy())
            
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
    
    def wrw_trip(self,g):
        wrw = (g.Distance.cumsum()*g.Weight).sum()
        wrw += (g.Distance.sum()+g.iloc[-1].LB)*10
        return wrw
        
    def load_balance(self,tripid,nearby_ct=3,verbose=False,w_max=1000):
        '''
        Finds optimal trip count for a group a trips and 
        either splits them up or combines them
        
        Inputs
        tripid : Trip ID to balance
        nearby_ct : number of nearby trips to include
        '''
        # Create South Pole Mask
        SP = int(self.trip_metrics.ix[tripid].SPF)
        SPmask = self.trip_metrics.SPF == SP
        # Find nearest trips
        tr_list = np.array(((self.trip_metrics.ix[tripid].AvgLong - \
            self.trip_metrics[SPmask].AvgLong)**2).sort_values().head(nearby_ct+1).index)
        
        # Get weight
        w_total = self.trip_metrics.ix[tr_list].Weight.sum()
        
        # Set search criterea to search around current solution
        min_n = int(np.floor(w_total/w_max))+1
        
        r1 = max(1,max(min_n,nearby_ct-2))
        r2 = max(r1+5,int(r1*1.2))
        r3 = min(1,int((r2-r1)/5.))
        
        # Search for optimal trip count
        n_opt,g = self.optimal_trip_count(tr_list,r1=r1,r2=r2,r3=r3,verbose=verbose)
    
        if n_opt == nearby_ct+1:
            # Already at optimal solution - don't do anything
            return n_opt
        else:
            # split trips with optimal n
            #g = self.split_trips(tr_list,n_opt)
            
            # Rebuild T and trip_metrics
            for i in range(max(n_opt,nearby_ct+1)):
                if i < len(tr_list):
                    if i < n_opt:
                        # Simply Replace this tripid and update metrics
                        self.T[tr_list[i]] = g[i]
                        self.update_trip_metrics(tr_list[i])
                    else:
                        # No more trips to add
                        # Null out the remaining trips in T, Drop from trip metrics
                        self.T[tr_list[i]] = []
                        self.trip_metrics.drop(tr_list[i],inplace=True)
                else:
                    # Need to create a new trip
                    new_tripid = len(self.T)
                    if verbose:
                        print 'New TripId Created =',new_tripid
                    # Add new trip to T
                    self.T.append(g[i])
                    self.update_trip_metrics(new_tripid)
                
        # Check weight constraints
        if self.trip_metrics.Weight.max() > 1000:
            # Smooth out weights if there's too much
            self.diffuse_east(SP,verbose=verbose)
            self.diffuse_west(SP,verbose=verbose)
            
        return n_opt
    
    def swap_worst_trip(self,ntry=10,plot_it = True,greedOfTheNoob=True,SPF = True,
                        verbose=1,w_max=1000,heavy_first = True):
        start_wrw = self.wrw
        stime = time.time()
        for k in range(ntry):
            pp= '~'*33+'\nLoop #'+str(k)
            
            headroom = self.trip_metrics.WRW - self.trip_metrics.LB
            headroom.sort_values(inplace=True,ascending=False)
            
            # Plot single worst trip
            #plot_trip(T,headroom.index[0],newplot=False,color=colors[c+1])
            
            # Look at worst trip
            # Choose Worst Trip Stochastically
            p = headroom+10
            p /= p.sum()
            w = np.random.choice(np.arange(p.shape[0]),p=p)
            worst_trip = headroom.index[w]
            
            #self.trip_metrics.ix[worst_trip]
            #self.T[worst_trip].head()
            
            # Find the outlier present in the worst trip
            # Find the outlier points
            dis = (self.T[worst_trip].Longitude - self.trip_metrics.ix[worst_trip].AvgLong)**2
            
            # The 'bad-ness' of a present is:
            # Distance from mean x Weight x Order
            if heavy_first:
                dis *= self.T[worst_trip].Weight * (np.arange(self.T[worst_trip].shape[0])+10)
                
            dis.sort_values(inplace=True,ascending=False)
            
            # Choose worst Present stochastically
            p = dis.head(25)
            p /= p.sum()
            w = np.random.choice(np.arange(p.shape[0]),p=p)
            outlier = self.T[worst_trip].ix[dis.index[w]]
            worst_present = dis.index[w]
            
            # Find trip with closest Avg Long
            # Segregate South Pole
            if SPF:
                SP_mask = self.trip_metrics.SPF == self.trip_metrics.ix[worst_trip].SPF
                closest_trip = (outlier.Longitude - self.trip_metrics[SP_mask].AvgLong)**2
            else:
                closest_trip = (outlier.Longitude - self.trip_metrics.AvgLong)**2        
            closest_trip.sort_values(inplace=True)
            
            '''
            # Choose new trip stochastically
            new_trip = closest_trip.head(2).index
            if new_trip[0]==worst_trip:
                new_trip = new_trip[1]
            else:
                new_trip = new_trip[0]
            '''
            r = np.floor(random.random()*4)
            new_trip = closest_trip.head(4).index[r]
            
            
            if self.trip_metrics.ix[new_trip].Weight + self.T[worst_trip].ix[worst_present].Weight > w_max:
                if verbose:
                    print pp
                    print 'Too Heavy'
            else:
                #self.trip_metrics.ix[closest_trip.head().index]
                if verbose:
                    print ' -Worst Trip',worst_trip
                    print ' -Worst Present',worst_present
                    print ' -New Trip',new_trip
                
                if plot_it:
                    self.plot_trips([worst_trip,new_trip],title='Before '+str(k))
                
                # Try out the remove and add functions
                # First check the combined WRW for these 2 trips
                wrw0 = self.trip_metrics.ix[[worst_trip,new_trip]].WRW.sum()
                
                # Remove it
                pres = self.remove_gift(worst_trip,worst_present)
                
                # Add it
                self.add_gift(new_trip,pres)
                wrw2 = self.trip_metrics.ix[[worst_trip,new_trip]].WRW.sum()
                if verbose:
                    print ' -WRW reduction after swap =',1-wrw2/wrw0
                    print ' -Absolute WRW Reduction =',wrw0-wrw2
                    
                if plot_it:
                    self.plot_trips([worst_trip,new_trip],title='After '+str(k),newplot=False)
                    
                if wrw0 < wrw2 and greedOfTheNoob:
                    # UNDO!!!!
                    pres = self.remove_gift(new_trip,worst_present)
                    self.add_gift(worst_trip,pres)  
            if w_max > 1000:
                if np.random.rand() > .5:
                    self.diffuse_east(1)
                    self.diffuse_west(1)
                    self.diffuse_east(0)
                    self.diffuse_west(0)
                else:
                    self.diffuse_west(1)
                    self.diffuse_east(1)
                    self.diffuse_west(0)
                    self.diffuse_east(0)
                        
        perc_gained_lb = (start_wrw-self.wrw) / (start_wrw - self.trip_metrics.LB.sum())
        print '\n%.0f - Starting WRW\n%.0f - New WRW\nWRW %% Gained = %.2f%%'%(start_wrw,self.wrw,perc_gained_lb*100)
        print 'Runtime (minutes) = %.2f'%((time.time()-stime)/60.0)
        


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
Santa = Trips(path='_v10_greed')
print 'Starting WRW =',Santa.wrw

'''
#####################
# Alg
# Grab a random trip
# Find nearest 3-5 trips (with SP constraint)
# Determine Optimal slitting - with weight constraint
# Break down
# Repeat

wrw_start = Santa.wrw

for ppp in range(0):
    print '\n\n'
    print '~'*33
    print 'Outer Loop #'+str(ppp)
    print 'Current Trip Count =',Santa.trip_metrics.shape[0]
    for q in range(75):
        wrw_beg = Santa.wrw
        # Choose random trip
        rand_trip = np.random.choice(Santa.trip_metrics.index)
        # Choose a random
        nearby_ct = np.random.randint(3,7)
        # Find Optimal
        n_opt = Santa.load_balance(rand_trip,nearby_ct=nearby_ct,verbose=False,w_max = 1002)
        if n_opt != nearby_ct+1:
            print '\nLoop #'+str(q)
            print 'TripID = %.0f\nNearby Count = %.0f\nOptimal Count = %.0f'%(rand_trip,nearby_ct+1,n_opt)
            print '%.0f - WRW Start\n%.0f - WRW End'%(wrw_beg,Santa.wrw)
    
    for omg in range(1):
        Santa.swap_worst_trip(400,
                              plot_it=False,
                              greedOfTheNoob=True,
                              SPF=True,
                              verbose=0,
                              w_max = 1002)
                                    
    Santa.save_data(path='_v10')







# Need to make this better. 
# Grab 2 trips and optimize instead of random presents
# Need something that tries to combine SP and NSP trips
# No More trip remapping
for omg in range(1):
    print '\nLoop #'+str(omg)
    Santa.swap_worst_trip(300,
                          plot_it=False,
                          greedOfTheNoob=True,
                          SPF=True,
                          verbose=0)#,
                          #w_max = 1025-omg/2.0)
                                    
    Santa.save_data(path='_v10')
'''



'''
##########################
# Optimize the South Pole with ACTUAL DISTANCES!!!!!!!!!!!!!
# Approximate Haversine near the South Pole??

mask = Santa.gifts.Latitude < -60
mask = np.logical_and(mask,Santa.gifts.Latitude > -86)

n = mask.sum()
DistMat = np.zeros((n,n))
for i in range(n):
    DistMat[i] = haversine_vect(Santa.gifts[mask].iloc[i],Santa.gifts[mask])

# -80 to -81
# n = 581
# Avg Distance = 1343

# -85 to -86
# n = 490
# Avg Distance = 640

# -89 to -90
# n = 433
# Avg Distance = 79

stime = time.time()

SPmask = Santa.gifts.Latitude < -60
n = SPmask.sum()
DMat = np.zeros((n,n))

for i in range(n):
    DMat[i] = haversine_vect(Santa.gifts[SPmask].iloc[i],Santa.gifts[SPmask])

DMatIndex = SPmask[SPmask].index

print 'Distance Matrix Runtime = %.2f'%(time.time()-stime)




# Iterate through the SP trips and try to merge them to nearby ones
SP_trips = np.array(Santa.trip_metrics[Santa.trip_metrics.SPF == 1].sort_values('AvgLong').index)
np.random.shuffle(SP_trips)

# Put cap on trips
for k in range(0):
    SP_trips = np.array(Santa.trip_metrics[Santa.trip_metrics.SPF == 1].index)
    np.random.shuffle(SP_trips)
    print '\n','~'*33
    print 'Loop %.0f'%k
    print 'Current Trip Count =',SP_trips.shape[0],'\n'
    
    for j in range(20):
        try:
            tripid = SP_trips[j]
            #print tripid
            ct = np.random.randint(3,8)
            _ = Santa.load_balance(tripid,nearby_ct=ct,verbose=False,w_max=650)
            if _ != ct+1:
                print '%.0f = Current Trip Count\n%.0f = Optimal Trip Count'%(ct+1,_)
                print '\n'
        except:
            print "Key Doesn't Exist Anymore... Skipped"

    Santa.swap_worst_trip(800,
                      plot_it=False,
                      greedOfTheNoob=True,
                      SPF=True,
                      verbose=0,
                      w_max = 750)

    Santa.save_data(path='_v10')




# Optimize Within the south pole
for omg in range(5):
    print '\nLoop #'+str(omg)
    Santa.swap_worst_trip(300,
                          plot_it=False,
                          greedOfTheNoob=True,
                          SPF=True,
                          verbose=0)#,
                          #w_max = 1025-omg/2.0)
                                    
    Santa.save_data(path='_v10')
    
# Optimize Within the south pole - No weight constraints
for omg in range(5):
    print '\nLoop #'+str(omg)
    Santa.swap_worst_trip(300,
                          plot_it=False,
                          greedOfTheNoob=True,
                          SPF=True,
                          verbose=0,
                          heavy_first=False)#,
                          #w_max = 1025-omg/2.0)
                                    
    Santa.save_data(path='_v10')
'''


# Add presents to the south pole trips
for omg in range(2):
    print '\nGreed Loop #'+str(omg)
    ww = np.random.randint(2)
    Santa.swap_worst_trip(300,
                          plot_it=False,
                          greedOfTheNoob=True,
                          SPF=False,
                          verbose=0,
                          heavy_first=ww)#,
                          #w_max = 1025-omg/2.0)
                                    
    Santa.save_data(path='_v10_greed')




# Shave off the number of trips in the north

for ppp in range(0):
    print '\n\n'
    print '~'*33
    print 'Outer Loop #'+str(ppp)
    print 'Current Trip Count =',Santa.trip_metrics.shape[0]
    for q in range(50):
        wrw_beg = Santa.wrw
        # Choose random trip
        rand_trip = np.random.choice(Santa.trip_metrics[Santa.trip_metrics.SPF==0].index)
        # Choose a random
        nearby_ct = np.random.randint(5,13)
        # Find Optimal
        n_opt = Santa.load_balance(rand_trip,nearby_ct=nearby_ct,verbose=False,w_max = 1000)
        if n_opt != nearby_ct+1:
            print '\nLoop #'+str(q)
            print 'TripID = %.0f\nNearby Count = %.0f\nOptimal Count = %.0f'%(rand_trip,nearby_ct+1,n_opt)
            print '%.0f - WRW Start\n%.0f - WRW End'%(wrw_beg,Santa.wrw)
    
    for omg in range(3):
        ww = np.random.randint(2)
        Santa.swap_worst_trip(2000,
                              plot_it=False,
                              greedOfTheNoob=True,
                              SPF=False,
                              verbose=0,
                              heavy_first=ww)#,
                              #w_max = 1025-omg/2.0)
                                        
                                    
    Santa.save_data(path='_v10_greed')



# Add presents to the south pole trips
for omg in range(15):
    print '\nGreed Loop #'+str(omg)
    ww = np.random.randint(2)
    Santa.swap_worst_trip(20000,
                          plot_it=False,
                          greedOfTheNoob=True,
                          SPF=False,
                          verbose=0,
                          heavy_first=ww)#,
                          #w_max = 1025-omg/2.0)
                                    
    Santa.save_data(path='_v10_greed')









total_w = Santa.trip_metrics.Weight.sum()

Santa.trip_metrics[Santa.trip_metrics.SPF == 1].Weight.sum()











Santa.write_sub('sub9.csv')





