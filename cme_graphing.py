import pickle
import statistics
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import optimize

RSUN = 6.957e8
AU = 1.49598e11
C2 = 11.4
C3 = 56.0
E2 = np.arctan(np.radians(C2/3600)) * AU
E3 = np.arctan(np.radians(C3/3600)) * AU


def find_file(date_min, date_max, min_ht):
    # date_min and date_max are the date ranges
    # Create the list that we will store the values
    # Second, scan the files for the right dates and time
    with open('all_cmes.pkl', 'rb') as f:
        cmes = pickle.load(f)

    cme_datetimes = pd.to_datetime(cmes['DATE-OBS'] + ' ' + cmes['TIME-OBS'])
    cmes = cmes.assign(datetime=cme_datetimes)

    filtered_cmes = cmes[(cmes['NUM_DATA_POINTS'] >= min_ht) &
                         (cmes['datetime'] >= date_min) &
                         (cmes['datetime'] <= date_max)]
    return (filtered_cmes)


def get_derivative(y, x):
    diff_y = np.diff(y)
    diff_x = np.diff(x)
    if (diff_x.dtype == 'timedelta64[ns]'):
        derivative = diff_y/(diff_x.astype(float)*10**-9)
    else:
        derivative = diff_y/diff_x
    x_center = x[:-1] + diff_x / 2.0
    return (x_center, derivative)


def height_graphs(subplt_height, x, yarray, desc):
    subplt_height.set_title("Height "+desc)
    subplt_height.set_xlabel('Time (min)')
    subplt_height.set_ylabel('Height ($R_{sun}$)')
    subplt_height.set_ylim([0, 32])
    for y in yarray:
        subplt_height.plot(x/60, y[0]/RSUN, label=y[1])
    subplt_height.legend(loc=2)
    return (subplt_height)


def height_velocity_graphs(x, y, desc, n, tscope):
    # x is in datetime, y is in RSUN
    # Setting up the plot
    fig = plt.figure(1, figsize=(12, 9))
    plt.clf()
    subplt_height = fig.add_subplot(121)
    subplt_velocity = fig.add_subplot(122)

    # Formatting x and y
    # formatting Time properly so it is seconds from start
    t = x.astype(float) * 1e-9  # from nanoseconds to seconds
    t0 = t[0]
    t -= t0
    # Converting y from RSUN to meters
    y_height = (np.array(y)) * RSUN  # y is now in meters

# HEIGHT AND VELOCITY CALCULATIONS
    height_yarrays = []
    velocity_yarrays = []

    # Raw data for height and velocity
    subplt_height.plot(t/60, y_height/RSUN, '+', label='Raw Data')
    tv, vy = get_derivative(y_height, t)
    subplt_velocity.plot(tv/60, vy/1000, '+', label='Raw Data')

    # Fit 1: Linear Curve Fit
    height_y_lin, hlabel_lin, lin_opt = lin_curve_fit_h(t, y_height)
    rchisq_lin = (reduced_chi_sq(
            y_height, height_y_lin, tscope)) / (len(y_height) - 2)
    lin_opt = np.append(lin_opt, rchisq_lin)
    hlabel_lin += " $\chi^{2}_{Red}=$%.1f" % rchisq_lin
    height_yarrays.append([height_y_lin, hlabel_lin])

    tv, y_velocity_lin = get_derivative(height_y_lin, t)
    vlabel_lin = "Linear Curve Fit"
    velocity_yarrays.append([y_velocity_lin, vlabel_lin])
    

    # Fit 2: Quadratic Curve Fit
    y_height_quad, hlabel_quad, quad_opt = quad_curve_fit_h(t, y_height)
    rchisq_quad = round((reduced_chi_sq(
            y_height, y_height_quad, tscope)) / (len(y_height) - 3), 2)
    quad_opt = np.append(quad_opt, rchisq_quad)
    hlabel_quad += "\n$\chi^{2}_{Red}=$%.1f" % rchisq_quad
    height_yarrays.append([y_height_quad, hlabel_quad])

    tv, y_velocity_quad = get_derivative(y_height_quad, t)
    vlabel_quad = "Quad Curve Fit"
    velocity_yarrays.append([y_velocity_quad, vlabel_quad])


    try:
        # Fit 3: Oscillating curve fit
        # in meters and seconds
        # a0=amplitude (m/s), a1=freq (s^-1), a2=phase,
        # a3=velocity (m/s), a4= acceleration (m/s^2)
        # limits=([a0min, a1min, a2min, a3min, a4min],
        #          [a0max, a1max, a2max, a3max, a4max])
        short_p = 2 * (np.amin(np.diff(tv)))  # period can't be shorter than twice
        #                                       the shortest step
        limits = ([0, short_p, 0, 0, -20, 0], [1000*1e3, t[-1], 2*np.pi,
                                                       3000*1e3, 30, np.inf])
        #limits = ([-(np.inf), -(np.inf), -(np.inf), -(np.inf), -(np.inf), -
            #       (np.inf)], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        # Fit 3: Oscillating fit
        # Coming up with initial guesses to put into the fit 
        # Amplitude, phase, and period are randomized and changed
        # Velocity, Acceleration, and Height are constant and taken from othe fits 
        amp = np.linspace(0,3000*1e3,100)
        phase = np.linspace(short_p, t[-1], 100)
        period = np.linspace(0, 2*np.pi, 100)
        
        initial_amp = (np.random.choice(amp,1))[0]
        initial_phase = (np.random.choice(phase,1))[0]
        initial_period = (np.random.choice(period,1))[0]
        initial_acc = quad_opt[2]
        initial_vel = lin_opt[1]
        initial_height = quad_opt[2]
        
        oscil_opt, oscil_cov = optimize.curve_fit(sin_height, t, y_height,
                                                  p0=[initial_amp, initial_phase,
                                                      initial_period, initial_vel,
                                                      initial_acc, initial_height],
                                                  bounds=limits)
        hlabel_oscil = "Oscillating Fit: A=%.1f km min$^{-1}$,\nP=%.1f min, $\Phi$=%.1f, a=%.1f m s$^{-2}$,\
        \nv=%.1f km s$^{-1}$, h= %.1f R$_{sun}$,\n" \
        % (oscil_opt[0]/1000, (oscil_opt[1]/60), oscil_opt[2], oscil_opt[4],
           oscil_opt[3]/1000, oscil_opt[5]/RSUN)
        t_500 = np.arange(0, t[-1], (t[-1]/500))
    
        rchisq_oscil = (reduced_chi_sq(
                  y_height, sin_height(
                          t, *oscil_opt), tscope)) / (len(y_height) - 6)
        
        hlabel_oscil += " $\chi^{2}_{Red}=$%.1f" % rchisq_oscil
        vlabel3 = "Oscillating Fit"
        # tv_100 = np.arange(0, tv[-1], (tv[-1])/100)
        tv_500 = np.arange(tv[0], tv[-1], (tv[-1])/500)
        tv_500 = np.append(tv_500, tv[-1])

    # HEIGHT AND VELOCITY GRAPH PLOTTING
        # Plotting the height vs time graph w/out oscillating fit
        height_graphs(subplt_height, t, height_yarrays, desc)
        # Plotting height oscilatting fit
        subplt_height.plot(t_500/60, sin_height(t_500, *oscil_opt)/RSUN,
                           label=hlabel_oscil)
        subplt_height.legend(loc=2)
        # Plotting the velocity vs time graph w/out oscillating fit
        velocity_graphs(subplt_velocity, tv, velocity_yarrays, desc)
        # Plotting velocity oscillating fit
        subplt_velocity.plot(tv_500/60, sin_velocity(tv_500, *oscil_opt[:-1])/1000,
                             label=vlabel3)
        subplt_velocity.legend(loc=2)
        # Formatting the layout and presentation of the height and velocity graphs
        plt.tight_layout()
        plt.savefig("cme_pkls/images/"+desc+"/run"+str(n)+'.png')
        #plt.show()
        oscil_opt = np.append(oscil_opt, rchisq_oscil)
        
    except Exception as ex:
        print('Error: Cant make oscillatory fit because: ', ex)
        oscil_opt = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    
    dict_fits = {'Linear Fit': lin_opt, 'Quad Fit': quad_opt,
                     'Oscillating Fit': oscil_opt, 'Date': desc}
    return (dict_fits)


def histogram(data, desc, bin_min, bin_max):
    bins = np.linspace(bin_min, bin_max, 100)
    plt.hist(data, bins, alpha=0.5, facecolor='blue', label='Oscillating Fit')

    # Formatting the histogram
    plt.xlabel(desc)
    plt.ylabel('Fraction')
    plt.title('1996-2004, ' + desc + "=%.1f"%statistics.mean(data) + ", median=%.1f"%statistics.median(data) + ', # of CMEs = 3465')
    plt.legend(loc=2)
    plt.show()


def lin_curve_fit_h(t, y):
    lin_opt, lin_cov = curve_fit(lin_height_model, t, y,
                                 p0=[2 * RSUN, 400 * 1e3])
    curve_fit_lin = lin_height_model(t, *lin_opt)
    label = 'Linear Curve Fit: v=%.1f km s$^{-1}$,\nh=%.1f R$_{sun}$,' % \
        (lin_opt[1]/1000, lin_opt[0]/RSUN)
    return (curve_fit_lin, label, lin_opt)


def lin_height_model(t, *a):
    return a[0] + a[1] * t


def quad_curve_fit_h(t, y):
    quad_opt, quad_cov = curve_fit(quad_height_model, t, y, p0=[2 * RSUN,
                                                                400 * 1e3,
                                                                0.0])
    curve_fit_quad = quad_height_model(t, *quad_opt)
    label = 'Quad Curve Fit: a = %.1f m s$^{-2}$,\nv = %.1f km s$^{-1}$, h=%.1f R$_{sun}$,' % (
            quad_opt[2], quad_opt[1]/1000, quad_opt[0]/RSUN)
    return (curve_fit_quad, label, quad_opt)


def quad_height_model(t, *a):
    return a[0] + a[1] * t + 0.5 * a[2] * t ** 2


def reduced_chi_sq(y, yu, tscope):
    err = np.array(tscope)
    err[err == 'C2'] = E2
    err[err == 'C3'] = E3
    chisq = ((y - yu) / err) ** 2
    return chisq.sum()


def sin_height(time, a_0, a_1, a_2, a_3, a_4, a_5):
    # sinusoidal height equation
    return (((-1.0 * a_0 * np.cos((1/a_1)*time*2*np.pi + a_2))/(2*np.pi/a_1)) +
            a_3 * time + 0.5 * a_4 * (time**2) + a_5)


def sin_velocity(time, a_0, a_1, a_2, a_3, a_4):
    # the derivative of the above equation
    return (a_0 * np.sin(((1 / a_1) * time * 2 * np.pi) + a_2) +
            a_3 + a_4 * time)


def to_pkl_file(ran, lin_fit_array, quad_fit_array, oscil_fit_array, desc):
    # Save fit and rchi values to .pkl file
    COLUMNS = ('LIN-FIT', 'QUAD-FIT', 'OSCIL-FIT-AMP','OSCIL-FIT-PER','OSCIL-FIT-PHASE','OSCIL-FIT-VEL','OSCIL-FIT-ACC')
    df = pd.DataFrame(columns=COLUMNS, index=ran)
    for n in ran:
        print("this is n: ", n)
        df.loc[n] = pd.Series({'LIN-FIT': lin_fit_array[n],
                            'QUAD-FIT': quad_fit_array[n],
                            'OSCIL-FIT-AMP': oscil_fit_array[n][0],
                            'OSCIL-FIT-PER': oscil_fit_array[n][1],
                            'OSCIL-FIT-PHASE': oscil_fit_array[n][2],
                            'OSCIL-FIT-VEL': oscil_fit_array[n][3],
                            'OSCIL-FIT-ACC':oscil_fit_array[n][4]} )
    print(df)
    df.to_pickle('cme_pkls/pkl_files/'+desc+'.pkl')
    
    
def to_pkl_file2(num, dates, lin_fit_array, quad_fit_array, oscil_fit_array):
    # Save fit and rchi values to .pkl file
    COLUMNS = ('RUN', 'LIN-FIT', 'QUAD-FIT', 'OSCIL-FIT',)
    INDEX = np.arange(0, len(lin_fit_array))
    df = pd.DataFrame(columns=COLUMNS, index=INDEX)
    for n in INDEX:
        df.loc[n] = pd.Series({'RUN': runs[n],
                               'LIN-FIT': lin_fit_array[n],
                               'QUAD-FIT': quad_fit_array[n],
                               'OSCIL-FIT': oscil_fit_array[n]})
    print(df)
    df.to_pickle('cme_pkls/pkl_files/'+desc+'.pkl')



def velocity_graphs(subplt_velocity, x, yarray, desc):
    subplt_velocity.set_title("Velocity "+desc)
    subplt_velocity.set_xlabel('Time (min)')
    subplt_velocity.set_ylabel('Velocity (km s$^{-1}$)')
    for y in yarray:
        subplt_velocity.plot(x/60, y[0]/1000, label=y[1])
    return (subplt_velocity)
