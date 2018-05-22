from gmplot import gmplot
import pandas as pd
from ast import literal_eval

#read data
trainSet = pd.read_csv('train_set.csv',
                       converters={"Trajectory": literal_eval},
                       index_col = 'tripId')

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
