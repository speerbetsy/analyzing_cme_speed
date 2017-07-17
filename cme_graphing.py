import pickle
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


def height_velocity_graphs(x, y, desc, tscope):
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
    tv, vy = get_derivative(y, t)
    subplt_velocity.plot(tv/60, vy/1000, '+', label='Raw Data')

    # Fit 1: Linear Curve Fit
    height_y_lin, hlabel_lin, lin_opt = lin_curve_fit_h(t, y_height)
    rchisq_lin = (reduced_chi_sq(y_height, height_y_lin, tscope)) / (len(y_height) - 2)
    hlabel_lin += " $\chi^{2}_{Red}=$%.1f" % rchisq_lin
    height_yarrays.append([height_y_lin, hlabel_lin])

    tv, y_velocity_lin = get_derivative(height_y_lin, t)
    vlabel_lin = "Linear Curve Fit"
    velocity_yarrays.append([y_velocity_lin, vlabel_lin])

    # Fit 2: Quadratic Curve Fit
    y_height_quad, hlabel_quad, quad_opt = quad_curve_fit_h(t, y_height)
    rchisq_quad = (reduced_chi_sq(y_height, y_height_quad, tscope)) / (len(y_height) - 3)
    hlabel_quad += "\n$\chi^{2}_{Red}=$%.1f" % rchisq_quad
    height_yarrays.append([y_height_quad, hlabel_quad])

    tv, y_velocity_quad = get_derivative(y_height_quad, t)
    vlabel_quad = "Quad Curve Fit"
    velocity_yarrays.append([y_velocity_quad, vlabel_quad])

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
    # Fit 3: Oscillating fit
    initial_acc = quad_opt[2]
    initial_vel = lin_opt[1]
    initial_height = quad_opt[2]
    oscil_opt, oscil_cov = optimize.curve_fit(sin_height, t, y_height,
                                              p0=[87*1e3, t[-1]*0.75, 0,
                                                  initial_vel, initial_acc,
                                                  initial_height],
                                              bounds=limits)
    hlabel3 = "Oscillating Fit: A=%.1f km $s^{-1}$,\nP=%.1f 1 $s^{-1}$,\
    $\Phi$=%.1f,\n a=%.1f m $s^{2}$, v=%.1f km $s^{-1}$,\nh= %.1f $R_{sun}$," \
    % (oscil_opt[0]/1000, oscil_opt[1]/60, oscil_opt[2], oscil_opt[3]/1000,
       oscil_opt[4], oscil_opt[5]/RSUN)
    t_100 = np.arange(0, t[-1], (t[-1]/100))

    rchisq = (reduced_chi_sq(y_height, sin_height(t,
                                           *oscil_opt), tscope)) / (len(y_height) - 6)
    hlabel3 += " $\chi^{2}_{Red}=$%.1f" % rchisq

    vlabel3 = "Oscillating Fit"
    tv_100 = np.arange(0, tv[-1], (tv[-1])/100)

# HEIGHT AND VELOCITY GRAPH PLOTTING
    # Plotting the height vs time graph w/out oscillating fit
    height_graphs(subplt_height, t, height_yarrays, desc)
    # Plotting height oscilatting fit
    subplt_height.plot(t_100/60, sin_height(t_100, *oscil_opt)/RSUN, label=hlabel3)
    subplt_height.legend(loc=2)
    # Plotting the velocity vs time graph w/out oscillating fit
    velocity_graphs(subplt_velocity, tv, velocity_yarrays, desc)
    # Plotting velocity oscillating fit
    subplt_velocity.plot(tv_100/60, sin_velocity(tv_100, *oscil_opt[:-1])/1000,
             label=vlabel3)
    subplt_velocity.legend(loc=2)
    # Formatting the layout and presentation of the height and velocity graphs
    plt.tight_layout()
    t_marks = desc[0:10]+'T'+desc[11:13]+'-'+desc[14:16]+'-'+desc[17:19]
    plt.savefig("figures/test/"+t_marks+'.png')
    plt.show()

# RETURN FIT AND RCHI VALUES
    dict_fits = {'Linear Fit': lin_opt, 'Quad Fit': quad_opt,
                 'Oscillating Fit': oscil_opt, 'Date': desc}
    return (dict_fits)


def lin_curve_fit_h(t, y):
    lin_opt, lin_cov = curve_fit(lin_height_model, t, y,
                                 p0=[2 * RSUN, 400 * 1e3])
    curve_fit_lin = lin_height_model(t, *lin_opt)
    label = 'Linear Curve Fit: v=%.1f km $s^{-1}$,\nh=%.1f $R_{sun}$,' % \
        (lin_opt[1]/1000, lin_opt[0]/RSUN)
    return (curve_fit_lin, label, lin_opt)


def lin_height_model(t, *a):
    return a[0] + a[1] * t


def quad_curve_fit_h(t, y):
    quad_opt, quad_cov = curve_fit(quad_height_model, t, y, p0=[2 * RSUN,
                                                                400 * 1e3,
                                                                0.0])
    curve_fit_quad = quad_height_model(t, *quad_opt)
    label = 'Quad Curve Fit: a=%.1f m $s^{2}$,\nv=%.1f km $s^{-1}$,\
    h=%.1f $R_{sun}$,' % (quad_opt[2], quad_opt[1]/1000, quad_opt[0]/RSUN)
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


def to_pkl_file(date, fit_values, rchi_values):
    # Save fit and rchi values to .pkl file
    COLUMNS = ('CME-DATE', 'FIT-VALUES', 'RCHI-VALUES')
    INDEX = np.arange(0, len(fit_values))
    df = pd.DataFrame(columns=COLUMNS, index=INDEX)
    for n in INDEX:
        df.loc[n] = pd.Series({'CME-DATE': date[n],
                               'FIT-VALUES': fit_values[n],
                               'RCHI-VALUES': rchi_values[n]})
    print(df)
    df.to_pickle('cme_fit_rchi.pkl')


def velocity_graphs(subplt_velocity, x, yarray, desc):
    subplt_velocity.set_title("Velocity "+desc)
    subplt_velocity.set_xlabel('Time (min)')
    subplt_velocity.set_ylabel('Velocity (km $s^{-1}$)')
    for y in yarray:
        subplt_velocity.plot(x/60, y[0]/1000, label=y[1])
    return (subplt_velocity)
