from gmplot import gmplot
import os
from operator import itemgetter
import pandas as pd
from ast import literal_eval
from dtw import dtw
import numpy as np
import time
from math import radians, cos, sin, asin, sqrt, atan2

def haversine_distance(x, y):
    lon1 = x[0]
    lat1 = x[1]
    lon2 = y[0]
    lat2 = y[1]

    #convert decimals to radians
    lon1 , lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    #formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = 6371 * c

    return d

def lcs(X, Y):
	#find the length of the strings
    m = len(X)
    n = len(Y)
 
    #declaring the array for storing the dp values
    L = [[None]*(n+1) for i in xrange(m+1)]
    matching_points = []
 
    """Following steps build L[m+1][n+1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0 :
                L[i][j] = 0
			#they match if haversine distance is <=200m
            elif haversine_distance(X[i-1], Y[j-1]) <= 0.2:
                L[i][j] = L[i-1][j-1]+1
                matching_points.append(X[i-1])
            else:
                L[i][j] = max(L[i-1][j] , L[i][j-1])
 
    #L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    #return L[m][n]
    return matching_points

###################
#read trainSet data
trainSet = pd.read_csv('train_set.csv', converters={"Trajectory": literal_eval}, index_col = 'tripId')
#trainSet = trainSet[:500]

#read testSet data
testSet = pd.read_csv('test_set_a2.csv', sep='\t', converters={"Trajectory": literal_eval})

#copy trajectory's lat and lon to our new list of lists 'train_data'
train_data = []
for line in trainSet['Trajectory']:
	temp = []
	for arr in line:
		temp.append(arr[1:])

	train_data.append(temp)

#same procedure for test_data
test_data = []
for line in testSet['Trajectory']:
	temp = []
	for arr in line:
		temp.append(arr[1:])

	test_data.append(temp)

#start clock
print('--- starting calculation ---')
start_time = time.time()

#compute neighbours
lcs_match = [ [], [], [], [], [] ]
j=0
for line1 in test_data:
	i=0
	for line2 in train_data:
		points_matched = lcs(line1, line2)
		lcs_match[j].append( (len(points_matched), points_matched, trainSet['journeyPatternId'].iloc[i]) )
		i+=1
	j+=1

#find 5 max neighbours
neighbours = [ [], [], [], [], [] ]
j=0
for arr in lcs_match:
	for i in range(0,5):
		max_neighbour = max(arr, key=itemgetter(0))
		neighbours[j].append(max_neighbour)
		arr.remove(max_neighbour)

	j+=1

#end clock
end_time = time.time()
print('--- %s seconds elapsed ---' % ( end_time - start_time ) )

#start printing maps
directory = 'A2_maps'
if not os.path.exists(directory):
	os.makedirs(directory)

i=0
for line in test_data:
	#print map from test_data
	lons = []
	lats = []
	for lon, lat in line:
		lons.append(float(lon))
		lats.append(float(lat))

	#place map
	gmap = gmplot.GoogleMapPlotter(lats[0], lons[0], 13)
	#plot points
	gmap.plot(lats, lons, 'green', edge_width=5)
	#draw map
	gmap.draw('./' + directory + '/test_set' + str(i) + '.html')

	#print 5 neighbours
	for j in range(0, 5):
		jp_id = neighbours[i][j][2]

		#find the neighbour in train set so we can draw
		index = 0
		for traj in trainSet['journeyPatternId']:
			if traj == jp_id:
				break

			index+=1

		lats = []
		lons = []
		for lon, lat in train_data[index]:
			lons.append(float(lon))
			lats.append(float(lat))

		mlats = []
		mlons = []
		for mlon, mlat in neighbours[i][j][1]:
			mlons.append(float(mlon))
			mlats.append(float(mlat))

		#place map for neighbours
		gmap = gmplot.GoogleMapPlotter(lats[0], lons[0], 13)
		#plot points for neighbours
		gmap.plot(lats, lons, 'blue', edge_width=5)
		#plot points for matching points
		gmap.plot(mlats, mlons, 'red', edge_width=5)
		
		#draw map for neighbours
		gmap.draw('./' + directory + '/test_set' + str(i) + '_neighbour' + str(j) + 'jpid' + str(jp_id) + '.html')

	i+=1
