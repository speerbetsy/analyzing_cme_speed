import pickle
import pandas as pd
import numpy as np
import matplotlib
from datetime import datetime
from scipy.optimize import curve_fit

matplotlib.use('TkAgg')
from matplotlib import pylab as plt

RSUN = 6.957e8  # Meters


def filter_cmes(date_min, date_max, min_ht):
    with open('all_cmes.pkl', 'rb') as f:
        all_cmes = pickle.load(f)

    cme_datetimes = pd.to_datetime(all_cmes['DATE-OBS'] + ' ' + all_cmes['TIME-OBS'])
    all_cmes = all_cmes.assign(datetime=cme_datetimes)

    filtered_cmes = all_cmes[
        (all_cmes['NUM_DATA_POINTS'] >= min_ht) & (all_cmes['datetime'] >= date_min) & (
        all_cmes['datetime'] <= date_max)]
    return filtered_cmes


def derivative(x, y):
    xd = np.diff(x, 1)
    yd = np.diff(y, 1)

    deriv = yd / xd
    xx = x[:-1] + xd / 2.0

    return xx, deriv


def lin_height_model(t, *a):
    return a[0] + a[1] * t


def quad_height_model(t, *a):
    return a[0] + a[1] * t + 0.5 * a[2] * t ** 2


def lin_vel_model(t, *a):
    return np.full(t.shape, a[0])


def quad_vel_model(t, *a):
    return a[0] + a[1] * t


if __name__ == '__main__':
    min_dt = datetime(1996, 1, 1, 0, 0, 0)
    max_dt = datetime(2004, 12, 31, 23, 59, 59)
    all_cmes = filter_cmes(min_dt, max_dt, 10)
    for i, cme in all_cmes.iloc[0:2].iterrows():
        ht_data = cme.loc['HT_DATA']
        times = ht_data.DATE_TIME.values
        heights = ht_data.HEIGHT.values

        # Prepare data for fitting
        t = times.astype(float) * 1e-9  # Seconds
        t0 = t[0]
        t -= t0
        h = heights * RSUN  # Meters

        lin_opt, lin_cov = curve_fit(lin_height_model, t, h, p0=[2 * RSUN, 400 * 1e3])
        quad_opt, quad_cov = curve_fit(quad_height_model, t, h, p0=[2 * RSUN, 400 * 1e3, 0.0])

        t_diff, v = derivative(t, h)

        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Height plot
        ax1.plot(t / 60, h / RSUN, '+', label='Data')
        ax1.set_title('Height')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('$R_{Sun}$')
        ax1.plot(t / 60, lin_height_model(t, *lin_opt) / RSUN, label='Linear Fit')
        ax1.plot(t / 60, quad_height_model(t, *quad_opt) / RSUN, label='Quadratic Fit')
        ax1.legend(loc=2)

        # Velocity plot
        ax2.plot(t_diff / 60, v / 1e3, '+')
        ax2.set_title('Velocity')
        ax2.set_xlabel('Time (min)')
        ax2.set_ylabel('$km\ s^{-1}$')
        ax2.plot(t_diff / 60, lin_vel_model(t_diff, *lin_opt[1:]) / 1e3, label='Linear Fit')
        ax2.plot(t_diff / 60, quad_vel_model(t_diff, *quad_opt[1:]) / 1e3, label='Quadratic Fit')

        plt.tight_layout()
        plt.show()