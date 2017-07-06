import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import optimize

rsun = 6.957e8
au = 1.49598e11
C2 = 11.4
C3 = 56.0
E2 = np.arctan(np.radians(C2/3600)) * au
E3 = np.arctan(np.radians(C3/3600)) * au


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


def height_graphs(ax1, x, yarray, desc):
    ax1.set_title("Height "+desc)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Height (Rsun)')
    ax1.set_ylim([0, 32])
    for y in yarray:
        ax1.plot(x/60, y[0]/rsun, label=y[1])
    ax1.legend(loc=2)
    return (ax1)


def height_velocity_graphs(x, y, desc, tscope):
    # x is in datetime, y is in Rsun
    # Setting up the plot
    fig = plt.figure(1, figsize=(12, 9))
    plt.clf()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Formatting x and y
    # formatting Time properly so it is seconds from start
    t = x.astype(float) * 1e-9  # from nanoseconds to seconds
    t0 = t[0]
    t -= t0
    # Converting y from Rsun to meters
    y = (np.array(y)) * rsun  # y is now in meters

# HEIGHT AND VELOCITY CALCULATIONS
    height_yarrays = []
    velocity_yarrays = []

    # Raw data for height and velocity
    ax1.plot(t/60, y/rsun, '+', label='Raw Data')
    tv, vy = get_derivative(y, t)
    ax2.plot(tv/60, vy/1000, '+', label='Raw Data')

    # Fit 1: Linear Curve Fit
    # v_i is the initial velocity for p0
    y1, hlabel1, v_i = lin_curve_fit_h(t, y)
    rchisq = (reduced_chi_sq(y, y1, tscope)) / (len(y) - 2)
    hlabel1 += " R_ChiSq=%.1f" % rchisq
    height_yarrays.append([y1, hlabel1])

    tv, vy1 = get_derivative(y1, t)
    vlabel1 = "Linear Curve Fit"
    velocity_yarrays.append([vy1, vlabel1])

    # Fit 2: Quadratic Curve Fit
    # acc_i is the initial acceleration for p0
    y2, hlabel2, acc_i, h_i = quad_curve_fit_h(t, y)
    rchisq = (reduced_chi_sq(y, y2, tscope)) / (len(y) - 3)
    hlabel2 += "\nR_ChiSq=%.1f" % rchisq
    height_yarrays.append([y2, hlabel2])

    tv, vy2 = get_derivative(y2, t)
    vlabel2 = "Quad Curve Fit"
    velocity_yarrays.append([vy2, vlabel2])

    # Fit 3: Oscillating curve fit
    # in meters and seconds
    # a0=amplitude (m/s), a1=phase (s^-1), a2=phase,
    # a3=velocity (m/s), a4= acceleration (m/s^2)
    # limits=([a0min, a1min, a2min, a3min, a4min],
    #          [a0max, a1max, a2max, a3max, a4max])
    short_p = 2 * (np.amin(np.diff(tv)))  # period can't be shorter than twice
    #                                       the shortest step
    limits = ([10 * 1e3, short_p, 0, 10, -20, 0], [10 * 1e4, t[-1], 2*np.pi,
                                                   300000*1e3, 30, np.inf])
    # Fit 1: scipy poly fit
    popt, pcov = optimize.curve_fit(sin_height, t, y, p0=[87*1e3, t[-1]*0.75,
                                                          0, v_i, acc_i, h_i],
                                    bounds=limits)
    hlabel3 = "Oscillating Fit: a0=%.1f km/s\na1=%.1f 1/s a2=%.1f (phase)\
    \na3=%.1f km/s a4=%.1f m/s^2\na5= %.1f RSUN" %\
        (popt[0]/1000, popt[1]/60, popt[2], popt[3]/1000, popt[4],
         popt[5]/rsun)

    rchisq = (reduced_chi_sq(y, sin_height(t, *popt), tscope)) / (len(y) - 6)
    hlabel3 += " R_ChiSq=%.1f" % rchisq
    height_yarrays.append([sin_height(t, *popt), hlabel3])

    vlabel3 = "Oscillating Fit"
    velocity_yarrays.append([sin_velocity(tv, *popt[:-1]), vlabel3])

# HEIGHT AND VELOCITY GRAPH PLOTTING
    height_graphs(ax1, t, height_yarrays, desc)
    velocity_graphs(ax2, tv, velocity_yarrays, desc)
    plt.tight_layout()
    t_marks = desc[0:10]+'T'+desc[11:13]+'-'+desc[14:16]+'-'+desc[17:19]
    plt.savefig("figures/test/"+t_marks+'.png')
    plt.show()


def lin_curve_fit_h(t, y):
    lin_opt, lin_cov = curve_fit(lin_height_model, t, y,
                                 p0=[2 * rsun, 400 * 1e3])
    curve_fit_lin = lin_height_model(t, *lin_opt)
    label = 'Linear Curve Fit, v=%.1f km/s\nh=%.1f rsun' % (lin_opt[1]/1000,
                                                            lin_opt[0]/rsun)
    return (curve_fit_lin, label, lin_opt[1])


def lin_height_model(t, *a):
    return a[0] + a[1] * t


def quad_curve_fit_h(t, y):
    quad_opt, quad_cov = curve_fit(quad_height_model, t, y, p0=[2 * rsun,
                                                                400 * 1e3,
                                                                0.0])
    curve_fit_quad = quad_height_model(t, *quad_opt)
    label = 'Quad Curve Fit, a=%.1f m/s^2\nv=%.1f km/s h=%.1f rsun' \
            % (quad_opt[2], quad_opt[1]/1000, quad_opt[0]/rsun)
    return (curve_fit_quad, label, quad_opt[2], quad_opt[0])


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


def velocity_graphs(ax2, x, yarray, desc):
    ax2.set_title("Velocity "+desc)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Velocity (km/sec)')
    for y in yarray:
        ax2.plot(x/60, y[0]/1000, label=y[1])
    ax2.legend(loc=2)
    return (ax2)
