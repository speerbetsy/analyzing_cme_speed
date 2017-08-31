import pickle
import statistics
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import optimize
import cme_graphing
import datetime


with open('cme_pkls/final_oscillation/final_oscil_vals_new.pkl', 'rb') as f:
        cmes = pickle.load(f)

print("len of cmes: ", len(cmes))
print(cmes)

# Figuring out dates
dates = cmes['DATE']
print("date 1: ", dates[0][0:4])
d1996 = []
d1997 = []
d1998 = []
d1999 = []
d2000 = []
d2001 = []
d2002 = []
d2003 = []
d2004 = []
for d in dates:
    if (d[0:4]=='1996'):
        d1996.append(d)
    elif (d[0:4]=='1997'):
        d1997.append(d)
    elif (d[0:4]=='1998'):
        d1998.append(d)
    elif (d[0:4]=='1999'):
        d1999.append(d)
    elif (d[0:4]=='2000'):
        d2000.append(d)
    elif (d[0:4]=='2001'):
        d2001.append(d)
    elif (d[0:4]=='2002'):
        d2002.append(d)
    elif (d[0:4]=='2003'):
        d2003.append(d)
    elif (d[0:4]=='2004'):
        d2004.append(d)
        
print("this is 1996: ",len(d1996))
print("this is 1997: ",len(d1997))
print("this is 1998: ",len(d1998))
print("this is 1999: ",len(d1999))
print("this is 2000: ",len(d2000))
print("this is 2001: ",len(d2001))
print("this is 2002: ",len(d2002))
print("this is 2003: ",len(d2003))
print("this is 2004: ",len(d2004))


# Making histograms of acceleration, amplitude, period, and velocity
#ACCELERATION
acc = cmes['ACC']
acc_label = 'm s$^{-2}$'

bins_min = min(acc)
bins_max = max(acc)
bins = np.linspace(bins_min,bins_max, 100) # np.linspace(bin_min, bin_max, space)

# Plot the histogram of the quad and oscilating acceleration
n, bins, patches = plt.hist(acc, bins, alpha=0.5, normed=1, facecolor='blue', label='Acceleration')
max_bin = int(np.argmax(n))
acc_val = bins[max_bin]

# Formatting the histogram
plt.xlabel('Acceleration [m s$^{-2}$]')
plt.ylabel('Fraction')
plt.title('1996-2004, ' + acc_label + '/mean: ' + "=%.1f"%statistics.mean(acc) + "median=%.1f"%statistics.median(acc))
plt.legend(loc=2)
plt.savefig("cme_pkls/final_oscillation/total_average/acceleration.png")
plt.clf()
plt.show()
print("this is the most common acceleration: ",acc_val)



#AMPLITUDE
amp = cmes['AMP']
amp_label = 'km min$^{-1}$'
bins_min = min(amp)
bins_max = max(amp)
bins = np.linspace(bins_min,bins_max, 100) # np.linspace(bin_min, bin_max, space)

# Plot the histogram of the quad and oscilating acceleration
n, bins, patches = plt.hist(amp, bins, alpha=0.5, normed=1, facecolor='blue', label='Amplitude')
max_bin = int(np.argmax(n))
amp_val = bins[max_bin]

# Formatting the histogram
plt.xlabel('Amplitude [km min$^{-1}$]')
plt.ylabel('Fraction')
plt.title('1996-2004, ' + amp_label + '/mean: ' + "=%.1f"%statistics.mean(amp) + "median=%.1f"%statistics.median(amp))
plt.legend(loc=2)
plt.savefig("cme_pkls/final_oscillation/total_average/amplitude.png")
plt.clf()
plt.show()
print("this is the most common amplitude: ",amp_val)



#PERIOD
per = cmes['PER']
per_label = '[minutes]'
bins_min = min(per)
bins_max = max(per)
bins = np.linspace(bins_min,bins_max, 100) # np.linspace(bin_min, bin_max, space)

# Plot the histogram of the quad and oscilating acceleration
n, bins, patches = plt.hist(per, bins, alpha=0.5, normed=1, facecolor='blue', label='Period')
max_bin = int(np.argmax(n))
per_val = bins[max_bin]

# Formatting the histogram
plt.xlabel('Period [minutes]')
plt.ylabel('Fraction')
plt.title('1996-2004, ' + per_label + '/mean: ' + "=%.1f"%statistics.mean(per) + "median=%.1f"%statistics.median(per))
plt.legend(loc=2)
plt.savefig("cme_pkls/final_oscillation/total_average/period.png")
plt.clf()
plt.show()
print("this is the most common period: ",per_val)



#VELOCITY
vel = cmes['VEL']
vel_label = '[km s$^{-1}$]'

bins_min = min(vel)
bins_max = max(vel)
bins = np.linspace(bins_min,bins_max, 100) # np.linspace(bin_min, bin_max, space)

# Plot the histogram of the quad and oscilating acceleration
n, bins, patches = plt.hist(vel, bins, alpha=0.5, normed=1, facecolor='blue', label='Velocity')
max_bin = int(np.argmax(n))
vel_val = bins[max_bin]

# Formatting the histogram
plt.xlabel('Velocity [km s$^{-2}$]')
plt.ylabel('Fraction')
plt.title('1996-2004, ' + vel_label + '/mean: ' + "=%.1f"%statistics.mean(vel) + "median=%.1f"%statistics.median(vel))
plt.legend(loc=2)
plt.savefig("cme_pkls/final_oscillation/total_average/velocity.png")
plt.clf()
plt.show()
print("this is the most common velocity: ",vel_val)



# Comparing quad chi values to oscil chi values
quad_chi = cmes['QUAD-CHI']
oscil_chi = cmes['OSCIL-CHI']
oscil_chi_better_fit = 0
for i in range(len(quad_chi)):
    if (oscil_chi[i] < quad_chi[i]):
        oscil_chi_better_fit += 1
        
print ("these are the total number of oscil fits: ",len(oscil_chi))
print ("and these are the number with higher chi sq values than the quad fits: ",oscil_chi_better_fit)


