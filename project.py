from gmplot import gmplot
import os
import sys
import pandas as pd
from ast import literal_eval
from operator import itemgetter
from dtw import dtw
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import time
from collections import Counter
from math import radians, cos, sin, asin, sqrt, atan2
import csv

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

def classify(neighbours):
    classified = []
    for n in neighbours:
        jp_ids = []
        for k in range(0,5):
            jp_ids.append(n[k][1])
        temp = Counter(jp_ids).most_common(1)
        classified.append(temp[0][0])
    return classified

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

def find_dtw(trainSet, testSet):
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
    dtw_dist = []
    j=0
    for line1 in test_data:
        line1 = np.array(line1)
        i=0
        dtw_dist.append([])
        for line2 in train_data:
            line2 = np.array(line2)
            dist, cost, acc, path = dtw(line1, line2, dist=haversine_distance)
            poutsa =  trainSet['journeyPatternId'].iloc[i]
            dtw_dist[j].append( (dist, trainSet['journeyPatternId'].iloc[i]) )
            i+=1

        j+=1

    #find 5 nearest neighbours
    neighbours = []
    j=0
    for arr in dtw_dist:
        neighbours.append([])
        for i in range(0,5):
            min_neighbour = min(arr, key=itemgetter(0))
            neighbours[j].append(min_neighbour)
            arr.remove(min_neighbour)

        j+=1

    #end clock
    end_time = time.time()
    print('--- %s seconds elapsed ---'  % ( end_time - start_time) )
    return neighbours

def visualise(trainSet):
    print "-----Visualise-----"
    #read data
    trainSet = trainSet
    #take first 5 with different journeypatternid
    trainSet = trainSet.drop_duplicates('journeyPatternId')
    trainSet = trainSet[:5]



    i = 1
    for trajectory in trainSet['Trajectory']:
        lats = []
        lons = []
        for time, lon, lat in trajectory:
            #get coordinates
            lons.append(float(lon))
            lats.append(float(lat))
        #place map
        gmap = gmplot.GoogleMapPlotter(lats[0], lons[0], 13)
        #plot points
        gmap.plot(lats, lons, 'green', edge_width=5)
        #draw map
        gmap.draw("my_map" + str(i) + ".html")
        i+=1
    print "Maps succesfully drawn."

def a1(trainSet, testSet):
    print "-----A1-----"
    trainSet =  trainSet
    testSet = testSet

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

    neighbours = find_dtw(trainSet, testSet)

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

def a2(trainSet, testSet):
    print "-----A2-----"
    trainSet =  trainSet
    testSet = testSet

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

def classification(trainSet, testSet):
    print "-----Classification-----"
    trainSet = trainSet
    testSet = testSet

    start_time = time.time()
    neighbours = find_dtw(trainSet, testSet)

    classified = classify(neighbours)
    end_time = time.time()
    print('--- Total seconds elapsed(DTW+Voting): %s ---'  % ( end_time - start_time) )

    #write csv
    id = 0
    with open('testSet_JourneyPatternIDs.csv', 'w') as svmf:
        fieldnames = ['Test_Trip_ID', 'Predicted_JourneyPatternID']
        writer = csv.DictWriter(svmf, fieldnames = fieldnames)
        writer.writeheader()
        #x is a number that describes a category
        # we use categories_inv to translate the number to the appropriate category
        for x in classified:
            writer.writerow({'Test_Trip_ID': id, 'Predicted_JourneyPatternID': x})
            id +=1

def cross_validation(trainSet):
    print "-----Cross Validation-----"
    train_data = trainSet

    indexes = train_data.index.values

    kf = KFold(n_splits=10)
    acc = 0
    f = 0
    start_time = time.time()
    for train_index, test_index in kf.split(train_data):
        f += 1
        print "Fold: " + str(f)
        nearest = find_dtw(train_data.loc[indexes[train_index], :],
        train_data.loc[indexes[test_index], :])
        classified = classify(nearest)
        acc += accuracy_score(train_data['journeyPatternId'][indexes[test_index]], classified)

    end_time = time.time()
    acc /= 10
    print "Accuracy = " + str(acc)
    print('--- Total seconds elapsed(CV): %s ---'  % ( end_time - start_time) )

print("Loading data...")
#read trainSet data
trainSet = pd.read_csv('train_set.csv', converters={"Trajectory": literal_eval}, index_col = 'tripId')

#read testSet1 data
testSet1 = pd.read_csv('test_set_a1.csv', sep='\t', converters={"Trajectory": literal_eval})

#read testSet2 data
testSet2 = pd.read_csv('test_set_a2.csv', sep='\t', converters={"Trajectory": literal_eval})

for arg in sys.argv[1:]:
    if arg == "visualise":
        visualise(trainSet)
    elif arg == "a1":
        a1(trainSet[:50], testSet1)
    elif arg == "a2":
        a2(trainSet[:50], testSet2)
    elif arg == "classification":
        classification(trainSet[:50], testSet2)
    elif arg == "cross_validation":
        cross_validation(trainSet[:50])
    else:
        print "Unkonwn arguement: " + str(arg)
