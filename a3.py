from gmplot import gmplot
from operator import itemgetter
import pandas as pd
from ast import literal_eval
from dtw import dtw
import numpy as np
import time
from math import radians, cos, sin, asin, sqrt, atan2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from collections import Counter
import numpy as np
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

def classify(neighbours):
    classified = []
    for n in neighbours:
        jp_ids = []
        for k in range(0,5):
            jp_ids.append(n[k][1])
        temp = Counter(jp_ids).most_common(1)
        classified.append(temp[0][0])
    return classified

train_data = pd.read_csv('train_set.csv',
                       converters={"Trajectory": literal_eval},
                       index_col = 'tripId')

train_data = train_data[:50]
indexes = train_data.index.values
#print indexes

kf = KFold(n_splits=10)
acc = 0
for train_index, test_index in kf.split(train_data):
    nearest = find_dtw(train_data.loc[indexes[train_index], :],
    train_data.loc[indexes[test_index], :])
    classified = classify(nearest)
    acc += accuracy_score(train_data['journeyPatternId'][indexes[test_index]], classified)

acc /= 10
print acc

