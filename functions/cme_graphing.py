import pickle
from matplotlib import pyplot as plt
import numpy as np
import datetime

def find_file(date_time):
    #First, create the lists for those two days 
    #where we will store their values
    day=[]
    #Second, scan the files for the right dates and time
    with open('../all_cmes.pkl', 'rb') as f:
        data = pickle.load(f)

    ht_data=data['HT_DATA']
    x=0
    while (x<len(ht_data)):
        if (ht_data[x]['DATE_TIME'][0]==date_time):
            day=ht_data[x]
            break
        x+=1
    return (day)

def height_velocity_graphs(day, desc):
    #Extracting data
    height_data=day['HEIGHT']
    time_data=day['DATE_TIME']
    
    #Making calculations
    t=time_data-time_data[0]
    t=(np.array(t, dtype=datetime.datetime))*(10**-9)/60
    
    #Height vs time (Rsun)
    plt.title("Height (Rsun) vs Time "+desc)
    plt.xlabel('Time')
    plt.ylabel('Height (Rsun)')
    plt.plot(t, height_data,'+')
    plt.gcf().autofmt_xdate()
    plt.show()
    
    #Height vs Time (km)
    height_data_km=np.array(height_data)
    height_data_km=height_data_km*695000
    plt.title("Height (km) vs Time "+desc)
    plt.xlabel('Time')
    plt.ylabel('Height (km)')
    plt.plot(t, height_data_km,'+')
    plt.gcf().autofmt_xdate()
    plt.show()
    
    #calcuations for velocity
    diff_height_data=np.diff(height_data_km)
    diff_time_data=np.diff(time_data)
    velocity=diff_height_data/((diff_time_data.astype(float))*10**-9)
    
    #Velocity vs Time
    plt.title("Velocity vs Time " +desc)
    plt.xlabel('Time (min)')
    plt.ylabel('Velocity km/sec')
    plt.plot(t[1:].astype(float), velocity,'+') #one less data point
    plt.show()

   