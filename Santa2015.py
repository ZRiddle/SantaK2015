# -*- coding: utf-8 -*-
"""
@author: zach.riddle


"""

import pandas as pd
import numpy as np
import time


# Leaderboard
FIRST_PLACE = 12667039084.89030
TENTH_PLACE = 12667039084.89030

# Constants
AVG_EARTH_RADIUS = 6371  # in km
north_pole = (90,0)
weight_limit = 1000
sleigh_weight = 10


#####################
# Functions

# Read in data
def get_data():
    gifts = pd.read_csv('gifts.csv',index_col=0)
    samplesub = pd.read_csv('sample_submission.csv',index_col=0)
    return gifts,samplesub


# Evalutaion metric - using numpy not math
def haversine(point1, point2, miles=False):
    """ Calculate the great-circle distance bewteen two points on the Earth surface.
    :input: two 2-tuples, containing the latitude and longitude of each point
    in decimal degrees.
    Example: haversine((45.7597, 4.8422), (48.8567, 2.3508))
    :output: Returns the distance bewteen the two points.
    The default unit is kilometers. Miles can be returned
    if the ``miles`` parameter is set to True.
    """
    # unpack latitude/longitude
    lat1, lng1 = point1
    lat2, lng2 = point2

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng / 2) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    if miles:
        return h * 0.621371  # in miles
    else:
        return h  # in kilometers

# weighted trip length
def weighted_trip_length(stops, weights): 
    tuples = [tuple(x) for x in stops.values]
    # adding the last trip back to north pole, with just the sleigh weight
    tuples.append(north_pole)
    weights.append(sleigh_weight)
    
    dist = 0.0
    prev_stop = north_pole
    prev_weight = sum(weights)
    for i in range(len(tuples)):        
        dist = dist + haversine(tuples[i], prev_stop) * prev_weight
        prev_stop = tuples[i]   
        prev_weight = prev_weight - weights[i]
    return dist

# WRW
def weighted_reindeer_weariness(sub,gifts):
    # Inputs are the submissions + gifts    
    
    # Join
    all_trips = sub.join(gifts)
    uniq_trips = all_trips.TripId.unique()
    
    if any(all_trips.groupby('TripId').Weight.sum() > weight_limit):
        raise Exception("One of the sleighs over weight limit!")
 
    dist = 0
    for t in uniq_trips:
        this_trip = all_trips[all_trips.TripId==t]
        dist = dist + weighted_trip_length(this_trip[['Latitude','Longitude']], this_trip.Weight.tolist())
    
    return dist    

def write_sub(sub,filename):
    sub['TripId'].to_csv(filename,header = ['TripId'])
    
def print_score(sub,gifts):
    WRR_sub = weighted_reindeer_weariness(sub['TripId'],gifts)
    print '1st Place Comarison = %.3f'%(WRR_sub/FIRST_PLACE)
    print '10th Place Comparison = %.3f'%(WRR_sub/TENTH_PLACE)
    
    
    
'''
####################################################################
####################################################################
####################################################################
gifts,samplesub = get_data()

start_time = time.time()
WRR_sample = weighted_reindeer_weariness(samplesub,gifts)

print 'Sample Submission WRR =',WRR_sample
print 'Score Time = %.2f seconds'%(time.time() - start_time)

# Inspect Weights Distribution
temp = gifts.copy()
temp.Weight = np.round(temp.Weight,0)
grp = temp.groupby('Weight')['Latitude'].count()
grp.plot(kind='bar')

# Lat, -89 to 89
# Long, -179 to 179


# Heaviest First
n_trips = 20
sub1 = gifts.sort(['Weight','Latitude'],ascending=False)
sub1['TripId'] = 0
pres_per_trip = sub1.shape[0]/n_trips
for k in range(pres_per_trip):
    arr = np.arange(pres_per_trip)
    start = k*pres_per_trip
    end = min(sub1.shape[0],(k+1)*pres_per_trip)
    sub1.iloc[start:end,3] = arr


WRR_sub1 = weighted_reindeer_weariness(sub1['TripId'],gifts)
print 'Sample Submission WRR =',WRR_sub1
print 'Comparison to Sample = %.3f'%(WRR_sub1/WRR_sample)
print '1st Place Comarison = %.3f'%(WRR_sub1/FIRST_PLACE)
print '10th Place Comparison = %.3f'%(WRR_sub1/TENTH_PLACE)


sub1['TripId'].to_csv('HeavyFirstSub.csv',header = samplesub.columns)

'''



