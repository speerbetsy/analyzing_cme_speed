import datetime
import cme_graphing

#Our parameters for the data
#Must be between 1994 and 2004
min_year = datetime.datetime(1996,1,1,0,0,0)
max_year = datetime.datetime(2004,12,31,23, 59,59)

#Must have at least 11 height-time points
min_ht = 10 #Generates 3465 cmes

#Create the lists for those two days where we will store their values
cmes=cme_graphing.find_file(min_year, max_year, min_ht)

x= 15#will run for first 1000 due to time constraints
for index, cme in cmes.iloc[0:x,:].iterrows(): 
    ht = cme.loc['HT_DATA']
    times = ht.DATE_TIME.values
    heights = ht.HEIGHT
    #times is in datetime, heights is in Rsun
    cme_graphing.height_velocity_graphs(times, heights, str(times[0]))


