import pickle
from matplotlib import pyplot as plt
import numpy as np
import datetime
import pandas as pd
from scipy.optimize import curve_fit
from scipy import optimize

rsun = 6.957e8

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
    t=(np.array(t, dtype=datetime.datetime))*(10**-9)#/60
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
    x_center = x[:-1] + diff_x / 2.0
    return (x_center, derivative)

    
def height_velocity_graphs(x, y, desc):  
    #Setting up the plot
    fig = plt.figure(1, figsize=(12,12))
    t_iso=desc[0:10]+'T'+desc[11:13]+'-'+desc[14:16]+'-'+desc[17:19]
    
    #Making calculations and formatting time properly
    t = x.astype(float) * 1e-9  # from nanoseconds to seconds
    t0 = t[0]
    t -= t0
    
    #Height vs time (Rsun)
    y=(np.array(y))*rsun #y is in meters
    ax1 = fig.add_subplot(121)
#PLOT FITS
    #Fit 1 H-T: Linear Algebra Least-squares method
    A = np.vstack([t, np.ones(len(t))]).T
    lin = np.linalg.lstsq(A, y)[0] #grabs values for velocity
    fit_h_lin = lin[1] + (t * lin[0])
    print('Least-squares linear fit (h-t): v=%f km/s, h=%f R_Sun' % (lin[0]/1000, lin[1]/rsun))
    #Fit 2 H-T: Quadratic fit Least-squares method
    A = np.vstack([t**2, t, np.ones(len(t))]).T
    quad = np.linalg.lstsq(A, y)[0] #grabs values for acceleration and velocity
    fit_h_quad = quad[2] + (t * quad[1]) + ((t**2)*0.5*quad[0])#quad1=velocity, quad0=acceleration
    print('Least-squares quadratic fit (h-t): a=%f m/s^2, v=%f km/s, h=%f R_Sun' % ((quad[0]*2), quad[1]/1000, quad[2]/rsun))
    #plotting
    ax1.set_title("Height "+desc)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Height (Rsun)')
    ax1.set_ylim([0,32])
    ax1.plot(t/60,y/rsun, '+', label='raw data')
    ax1.plot(t/60, fit_h_lin/rsun, label='lin fit')
    ax1.plot(t/60, fit_h_quad/rsun, label='quad fit')
    ax1.legend(loc=2)

    #Velocity vs Time
    #tv=format_nstime(x)
    #y is in Rsun
    y_km=y #*695000
    
    tv, vy=get_derivative(y_km,t)
    ax2 = fig.add_subplot(122)
#PLOT FITS
    limits=(np.zeros(5), np.full([5], np.inf))
    #in meters
    limits=([30*1e3, 1, 0, 0, -30], [200*1e3, 1400*60, 2*np.pi, 3500*1e3, 30])
    #Fit 1: scipy poly fit
    popt, pcov = optimize.curve_fit(sin_velocity, tv, vy, p0=[50*1e3, 500*60, 0, 1e5, 10], bounds=limits)
    print ("Curve fit: a0=%f km/s, a1=%f 1/s, a2=%f (phase), a3=%f km/s, a4=%f m/s^2" %(popt[0]/1000, popt[1], popt[2], popt[3]/1000, popt[4]))
    
    #plotting
    ax2.set_title("Velocity "+ desc)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Velocity (Rsun/s)')
    ax2.plot(tv/60,vy/rsun, '+', label='raw data')
    ax2.plot(tv/60, (sin_velocity(tv, *popt))/rsun, '--', label='Scipy Curve Fit')
    ax2.legend(loc=2)

    plt.tight_layout()
    plt.savefig("figures/test/"+t_iso+'.png')
    plt.show()
    
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
    plt.ylabel('Height')
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

   