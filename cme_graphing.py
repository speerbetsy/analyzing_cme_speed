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

def height(time, h_0, v_0, a_0):
    #Height equation
    return (h_0 + (v_0 * time) + (0.5 * (time**2) * a_0))

    
def height_graphs(ax1, x, yarray, desc):
    ax1.set_title("Height "+desc)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Height (Rsun)')
    ax1.set_ylim([0,32])
    for y in yarray:
        ax1.plot(x/60,y[0]/rsun, label=y[1])
    ax1.legend(loc=2)
    return (ax1)    
    
def height_velocity_graphs(x, y, desc): 
    #x is in datetime, y is in Rsun
    #Setting up the plot
    fig = plt.figure(1, figsize=(12,9))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    #Formatting x and y
    #formatting Time properly so it is seconds from start
    t = x.astype(float) * 1e-9  # from nanoseconds to seconds
    t0 = t[0]
    t -= t0
    #Converting y from Rsun to meters
    y=(np.array(y))*rsun #y is now in meters
    
#HEIGHT AND VELOCITY CALCULATIONS
    height_yarrays=[]
    velocity_yarrays=[]
    
    #raw data for height and velocity
    ax1.plot(t/60,y/rsun, '+', label='raw data')
    tv, vy=get_derivative(y,t)
    ax2.plot(tv/60,vy/1000, '+', label='raw data')
    
    #Fit 1: Linear Algebra Least-squares method
    hy1, hlabel1, vy1, vlabel1 = lin_fit_lstsq(t,tv,y)
    height_yarrays.append([hy1, hlabel1])
    velocity_yarrays.append([vy1, vlabel1])
    
    #Fit 2: Quadratic fit Least-squares method
    hy2, hlabel2, vy2, vlabel2=quad_fit_lstsq(t,tv,y)
    height_yarrays.append([hy2, hlabel2])
    velocity_yarrays.append([vy2, vlabel2])
    
    #Fit 3: Linear Curve Fit
    y3,label3=lin_curve_fit_h(t,y)
    height_yarrays.append([y3,label3])
    #Fit 4: Quadratic Curve Fit
    y4, label4=quad_curve_fit_h(t,y)
    height_yarrays.append([y4,label4])

#HEIGHT AND VELOCITY GRAPH PLOTTING
    height_graphs(ax1,t,height_yarrays,desc)
    velocity_graphs(ax2,tv,velocity_yarrays,desc)
    plt.tight_layout()
    t_marks=desc[0:10]+'T'+desc[11:13]+'-'+desc[14:16]+'-'+desc[17:19]
    plt.savefig("figures/test/"+t_marks+'.png')
    plt.show() 
#oscillating curve fit attempts, commented out for now bc they dont work
   # limits=(np.zeros(5), np.full([5], np.inf))
    #in meters
  #  limits=([30*1e3, 1, 0, 0, -30], [200*1e3, 1400*60, 2*np.pi, 3500*1e3, 30])
    #Fit 1: scipy poly fit
  #  popt, pcov = optimize.curve_fit(sin_velocity, tv, vy, p0=[50*1e3, 500*60, 0, 1e5, 10], bounds=limits)
    #print ("Curve fit: a0=%f km/s, a1=%f 1/s, a2=%f (phase), a3=%f km/s, a4=%f m/s^2" %(popt[0]/1000, popt[1], popt[2], popt[3]/1000, popt[4]))

    
def lin_curve_fit_h(t,y):
    lin_opt, lin_cov = curve_fit(lin_height_model, t, y, p0=[2 * rsun, 400 * 1e3])
    curve_fit_lin=lin_height_model(t, *lin_opt)
    label='Linear Curve Fit, v=%.1f km/s h=%.1f rsun'%(lin_opt[1]/1000, lin_opt[0]/rsun)
    return (curve_fit_lin,label)

def lin_fit_lstsq(t,tv,y):
    #Fit 1 H-T: Linear Algebra Least-squares method
    A = np.vstack([t, np.ones(len(t))]).T
    lin = np.linalg.lstsq(A, y)[0] #grabs values for velocity
    fit_h_lin = lin[1] + (t * lin[0])
    hlabel= 'LstSq Linear Fit: v=%.1f km/s, h=%.1f R_Sun' % (lin[0]/1000, lin[1]/rsun)
    fit_v_lin=np.full(len(tv),lin[0])
    vlabel='LstSq Linear Velocity'
    return (fit_h_lin, hlabel, fit_v_lin, vlabel)

def lin_height_model(t, *a):
    return a[0] + a[1] * t

def lin_vel_model(t, *a):
    return np.full(t.shape, a[0])

def quad_curve_fit_h(t,y):
    quad_opt, quad_cov = curve_fit(quad_height_model, t, y, p0=[2 * rsun, 400 * 1e3, 0.0])
    curve_fit_quad=quad_height_model(t, *quad_opt)
    label='Quad Curve Fit, a=%.1f m/s^2, v=%.1f km/s h=%.1f rsun'%(quad_opt[2], quad_opt[1]/1000, quad_opt[0]/rsun)
    return (curve_fit_quad, label)

def quad_fit_lstsq(t,tv,y):
    A = np.vstack([t**2, t, np.ones(len(t))]).T
    quad = np.linalg.lstsq(A, y)[0] #grabs values for acceleration and velocity
    fit_h_quad = quad[2] + (t * quad[1]) + ((t**2)*quad[0])#quad1=velocity, quad0=acceleration  also multiplied a by 0.5
    hlabel='LstSq Quad Fit: a=%.1f m/s^2, v=%.1f km/s, h=%.1f R_Sun' % ((quad[0]*2), quad[1]/1000, quad[2]/rsun)
    fit_v_quad=quad[1]+(quad[0]*tv)
    vlabel='LstSq Quad Velocity'
    return (fit_h_quad, hlabel, fit_v_quad, vlabel)



def quad_height_model(t, *a):
    return a[0] + a[1] * t +  0.5 * a[2] * t ** 2
    
def sin_height(time,a_0,a_1,a_2,a_3,a_4,a_5):
    #sinusoidal height equation
    return (((-1.0 * a_0 * np.cos((1/a_1)*time*2*np.pi + a_2))/(2*np.pi/a_1)) + a_3*time + 0.5*a_4*(time**2) + a_5)

def sin_velocity(time,a_0,a_1,a_2,a_3,a_4):
    #the derivative of the above equation
    return (a_0*np.sin(((1/a_1)*time*2*np.pi)+a_2) + a_3 + a_4*time)

def velocity_graphs(ax2,x,yarray, desc):
    ax2.set_title("Velocity "+desc)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Velocity (km/sec)')
    for y in yarray:
        ax2.plot(x/60,y[0]/1000, label=y[1])
    ax2.legend(loc=2)
    return (ax2)    
    
def velocity(time, v_0, a_0):
    #Velocity 1: Model equation
    return (v_0 + (time * a_0))



   