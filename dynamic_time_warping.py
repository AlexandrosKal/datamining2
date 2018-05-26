from gmplot import gmplot
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

#read trainSet data
trainSet = pd.read_csv('train_set.csv', converters={"Trajectory": literal_eval}, index_col = 'tripId')
trainSet = trainSet[:100]

#read testSet data
testSet = pd.read_csv('test_set_a1.csv', sep='\t', converters={"Trajectory": literal_eval})

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
	#print line
	temp = []
	for arr in line:
		#print arr
		temp.append(arr[1:])

	test_data.append(temp)

#print test_data
#start clock
print('--- starting calculation ---')
start_time = time.time()

#compute neighbours
dtw_dist = [ [], [], [], [], [] ]
j=0
for line1 in test_data:
	line1 = np.array(line1)
	i=0
	for line2 in train_data:
		line2 = np.array(line2)
		dist, cost, acc, path = dtw(line1, line2, dist=haversine_distance)
		dtw_dist[j].append( (dist, trainSet['journeyPatternId'].iloc[i]) )
		i+=1

	j+=1
	
#sort arrays
#i=0
#for arr in dtw_dist:
#	dtw_dist[i] = sorted(arr, key=itemgetter(0))
#	i+=1

#find 5 nearest neighbours
neighbours = [ [], [], [], [], [] ]
j=0
for arr in dtw_dist:
	for i in range(0,5):
		min_neighbour = min(arr, key=itemgetter(0))
		neighbours[j].append(min_neighbour)
		arr.remove(min_neighbour)

	j+=1

#end clock
end_time = time.time()
print('--- %s seconds elapsed ---'  % ( end_time - start_time) )

print neighbours[0]
print neighbours[1]
print neighbours[2]
print neighbours[3]
print neighbours[4]
print('--- printing maps ---')
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
	gmap.draw('test_set' + str(i) + '.html')

	#print 5 neighbours
	for j in range(0, 5):
#		jp_id = dtw_dist[i][j][1]
		jp_id = neighbours[i][j][1]

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

		#place map for neighbours
		gmap = gmplot.GoogleMapPlotter(lats[0], lons[0], 13)
		#plot points for neighbours
		gmap.plot(lats, lons, 'blue', edge_width=5)
		#draw map for neighbours
		gmap.draw('test_set' + str(i) + '_neighbour' + str(j) + 'jpid' + str(jp_id) + '.html')

	i+=1
