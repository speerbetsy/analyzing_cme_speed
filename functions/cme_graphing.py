import pickle
from matplotlib import pyplot as plt
import numpy as np
import datetime
import pandas as pd
from scipy.optimize import curve_fit

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
    #Second, scan the files for the right dates and time
    with open('all_cmes.pkl', 'rb') as f:
        cmes = pickle.load(f)

    cme_datetimes = pd.to_datetime(cmes['DATE-OBS'] + ' ' + cmes['TIME-OBS'])
    cmes = cmes.assign(datetime=cme_datetimes)

    filtered_cmes=cmes[(cmes['NUM_DATA_POINTS'] >=min_ht) & (cmes['datetime'] >= date_min) & (cmes['datetime'] <= date_max)]
    return (filtered_cmes)


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
    
def height_velocity_graphs(x, y, desc):
    #Setting up the plot
    fig = plt.figure(1, figsize=(12,12))
    t_iso=desc[0:10]+'T'+desc[11:13]+'-'+desc[14:16]+'-'+desc[17:19]
    file = open("figures/test/"+t_iso+'.png', "w")
    
    #Making calculations and formatting time properly
    t=x-x[0]
    t=(np.array(t, dtype=datetime.datetime))*(10**-9)/60
    
    #Height vs time (km)
    y_km=(np.array(y))*695000
    ax1 = fig.add_subplot(121)
#PLOT FITS
    #Fit 1 H-T: Linear Algebra Least-squares method
    A = np.vstack([t, np.ones(len(t))]).T
    lin = np.linalg.lstsq(A, y_km)[0] #grabs values for velocity
    fit_h_lin = lin[1] + (t * lin[0])
    print('Least-squares linear fit (h-t): v=%f m/s, h=%f R_Sun' % (lin[0]/1000, lin[1]))
    #Fit 2 H-T: Quadratic fit Least-squares method
    A = np.vstack([t**2, t, np.ones(len(t))]).T
    quad = np.linalg.lstsq(A, y_km)[0] #grabs values for acceleration and velocity
    fit_h_quad = quad[2] + (t * quad[1]) + ((t**2)*0.5*quad[0])#quad1=velocity, quad0=acceleration
    print('Least-squares quadratic fit (h-t): a=%f m/s^2, v=%f km/s, h=%f R_Sun' % ((quad[0]*2), quad[1]/1000, quad[2]))
    #plotting
    #ax1.set_title("Height "+x[0].isoformat())
    ax1.set_title("Height "+desc)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Height (km)')
    ax1.plot(t,y_km, '+', label='raw data')
    ax1.plot(t, fit_h_lin, label='lin fit')
    ax1.plot(t, fit_h_quad, label='quad fit')
    ax1.legend(loc=2)

    #Velocity vs Time
    tv=format_nstime(x)
    vy=get_derivative(y_km,x)
    ax2 = fig.add_subplot(122)
#PLOT FITS
    #Fit 1: scipy poly fit
    #param_bounds=([-np.inf,-np.inf, 0,-np.inf,-np.inf],[np.inf, np.inf,np.inf,np.inf, np.inf])
    #popt, pcov = curve_fit(sin_velocity, tv, vx,bounds=param_bounds)
    #print ("Curve fit: a0=%f km/s, a1=%f 1/s, a2=%f (phase), a3=%f km/s, a4=%f m/s^2" %(popt[0]/1000, popt[1], popt[2], popt[3]/1000, popt[4]))

    #popt, pcov = curve_fit(sin_velocity, tv, y)
    #print ("Velocity-Time fit for plt curve function: a0=%f a1=%f \
    #       a2=%f a3=%f a4=%f a5=%f" %(popt[0], popt[1], popt[2], \
     #      popt[3], popt[4], popt[5]))
    #plotting
    #ax2.set_title("Velocity "+ x[0].isoformat())
    ax2.set_title("Velocity "+ desc)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Velocity (km/s)')
    ax2.plot(tv,vy, '+', label='raw data')
    #ax2.plot(tv, sin_velocity(tv, *popt)/1000, 'r-.', label='Scipy Curve Fit')
    ax2.legend(loc=2)

    plt.tight_layout()
    plt.savefig("figures/test/"+t_iso+'.png')
    plt.show()
    file.close()
    
def height_graphs(x, y, desc):
    file = open("figures/"+desc+".png", "w")
    #Height vs time (Rsun)
    plt.title("Height (Rsun) vs Time "+desc)
    plt.xlabel('Time')
    plt.ylabel('Height (Rsun)')
    plt.plot(x, y,'+')
    #plt.gcf().autofmt_xdate()
    plt.savefig('figures/'+desc+'.png')
    #plt.show()
    file.close()
    
    #Height vs Time (km)
    file = open("figures/"+desc+".png", "w")
    y_km=(np.array(y))*695000
    plt.title("Height (km) vs Time "+desc)
    plt.xlabel('Time')
    plt.ylabel('Height (km)')
    plt.plot(x,y_km,'+')
    #plt.gcf().autofmt_xdate()
    plt.savefig("figures/"+desc+'.png')
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
    file = open("figures/"+desc+".png", "w")
    #Velocity vs Time
    plt.title("Velocity vs Time " +desc)
    plt.xlabel('Time (min)')
    plt.ylabel('Velocity km/sec')
    plt.plot(x, y,'+') #one less data point
    plt.savefig("figures/"+desc+'.png')
    plt.show()
    file.close()

   