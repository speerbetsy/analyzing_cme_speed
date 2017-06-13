import datetime
import sys
#sys.path.insert(0, '../functions')
import cme_graphing

#Our parameters for the data
#Must be between 1994 and 2004
min_year = datetime.datetime(1996,1,1,0,0,0)
max_year = datetime.datetime(2004,12,31,23, 59,59)
print (max_year)
#Must have at least 11 height-time points
min_ht = 10 #Generates 3465 cmes

#Create the lists for those two days where we will store their values
cmes=cme_graphing.find_file(min_year, max_year, min_ht)

x=5 #will run for first 1000 due to time constraints
for index, cme in cmes.iloc[0:x,:].iterrows(): 
    print (type(cme))
    ht = cme.loc['HT_DATA']
    times = ht.DATE_TIME
    heights = ht.HEIGHT
    cme_graphing.height_velocity_graphs(times, heights, str(times[0]))


