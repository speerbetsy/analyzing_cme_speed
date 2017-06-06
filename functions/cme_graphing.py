import pickle
from matplotlib import pyplot as plt
import numpy as np
import datetime

def find_file_orig(date_time):
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

def find_file(date_min, date_max, min_ht):
    #date_min and date_max are the date ranges
    #Create the list that we will store the values
    days=[]
    #Second, scan the files for the right dates and time
    with open('../all_cmes.pkl', 'rb') as f:
        data = pickle.load(f)

    ht_data=data['HT_DATA']
    x=0
    while (x<len(ht_data)):
        if (ht_data[x]['DATE_TIME'][0]>date_min) and (ht_data[x]['DATE_TIME'][0]<date_max) and (len(ht_data[x]['HEIGHT'])>=min_ht) :
            days.append(ht_data[x])
        x+=1
    return (days)

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

def format_nstime(time): #for time that is in nanoseconds
    #Making calculations
    t=time-time[0]
    t=(np.array(t, dtype=datetime.datetime))*(10**-9)/60
    #get centroid point
    diff_t=np.diff(t)
    center_t=t[:-1]+(0.5*diff_t)
    return (center_t)

def format_time(time):
    #Making calculations
    t=time-time[0]
    t=(np.array(t, dtype=datetime.datetime))
    #get centroid point
    diff_t=np.diff(t)
    center_t=t[:-1]+(0.5*diff_t)
    return (center_t)

def get_derivative(y,x):
    diff_y=np.diff(y)
    diff_x=np.diff(x)
    if (diff_x.dtype=='timedelta64[ns]'):
        derivative=diff_y/(diff_x.astype(float)*10**-9)
    else:
        derivative=diff_y/diff_x
    return (derivative)

def fit_to_function_velocity(day):
   return 0
    
def height_graphs(x, y, desc):
    file = open("../heights/"+desc+"_rsun"+".png", "w")
    #Height vs time (Rsun)
    plt.title("Height (Rsun) vs Time "+desc)
    plt.xlabel('Time')
    plt.ylabel('Height (Rsun)')
    plt.plot(x, y,'+')
    #plt.gcf().autofmt_xdate()
    plt.savefig('../heights/'+desc+'_rsun'+'.png')
    #plt.show()
    file.close()
    
    #Height vs Time (km)
    file = open("../heights/"+desc+"_km"+".png", "w")
    y_km=(np.array(y))*695000
    plt.title("Height (km) vs Time "+desc)
    plt.xlabel('Time')
    plt.ylabel('Height (km)')
    plt.plot(x,y_km,'+')
    #plt.gcf().autofmt_xdate()
    plt.savefig('../heights/'+desc+'_km'+'.png')
    plt.show()
    file.close()
 
def velocity(time, v_0, a_0):
    #Velocity 1: Model equation
    return (v_0 + (time * a_0))

def height(time, h_0, v_0, a_0):
    #Height equation
    return (h_0 + (v_0 * time) + (0.5 * (time**2) * a_0))

def sin_height(time,a_0,a_1,a_2,a_3,a_4,a_5):
    #sinusoidal height equation
    return (((-1.0 * a_0 * np.cos((1/a_1)*time*2*np.pi + a_2))/(2*np.pi/a_1)) + a_3*time + 0.5*a_4*(time**2) + a_5)

def sin_velocity(time,a_0,a_1,a_2,a_3,a_4):
    #the derivative of the above equation
    return (a_0*np.sin(((1/a_1)*time*2*np.pi)+a_2) + a_3 + a_4*time)

def velocity_graphs(x,y, desc):
    file = open("../velocities/"+desc+"_km"+".png", "w")
    #Velocity vs Time
    plt.title("Velocity vs Time " +desc)
    plt.xlabel('Time (min)')
    plt.ylabel('Velocity km/sec')
    plt.plot(x, y,'+') #one less data point
    plt.savefig('../velocities/'+desc+'_km'+'.png')
    plt.show()
    file.close()

   