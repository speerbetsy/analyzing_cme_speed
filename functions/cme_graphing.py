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

def fit_to_function_height(day):
    #Extracting data
    height_data=day['HEIGHT']
    time_data=day['DATE_TIME']
    
    #Making calculations
    t=time_data-time_data[0]
    t=(np.array(t, dtype=datetime.datetime))*(10**-9)/60
    #get centroid point
    diff_t=np.diff(t)
    center_t=t[:-1]+(0.5*diff_t)
    return center_t, height_data

def format_time(time):
    #Making calculations
    t=time-time[0]
    t=(np.array(t, dtype=datetime.datetime))*(10**-9)/60
    #get centroid point
    diff_t=np.diff(t)
    center_t=t[:-1]+(0.5*diff_t)
    return (center_t)

def get_derivative(y,x):
    diff_y=np.diff(y)
    diff_x=np.diff(x)
    derivative=diff_y/((diff_x.astype(float))*10**-9)
    return (derivative)

def fit_to_function_velocity(day):
   return 0
    
def height_graphs(x, y, desc):
    #Height vs time (Rsun)
    plt.title("Height (Rsun) vs Time "+desc)
    plt.xlabel('Time')
    plt.ylabel('Height (Rsun)')
    plt.plot(x, y,'+')
    #plt.gcf().autofmt_xdate()
    plt.show()
    
    #Height vs Time (km)
    y_km=(np.array(y))*695000
    plt.title("Height (km) vs Time "+desc)
    plt.xlabel('Time')
    plt.ylabel('Height (km)')
    plt.plot(x,y_km,'+')
    #plt.gcf().autofmt_xdate()
    plt.show()
 
def velocity(time, v_0, a_0):
    #Velocity 1: Model equation
    return (v_0 + (time * a_0))

def velocity_graphs(x,y, desc):
    #Velocity vs Time
    plt.title("Velocity vs Time " +desc)
    plt.xlabel('Time (min)')
    plt.ylabel('Velocity km/sec')
    plt.plot(x, y,'+') #one less data point
    plt.show()

   