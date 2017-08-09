import datetime
import cme_graphing
import matplotlib.pyplot as plt
import numpy as np
import pickle


histogram=plt.figure()

# First, open the pickle file with the cme fit values
with open('cme_fit.pkl', 'rb') as f:
    cmes = pickle.load(f)

# Second, grab the acceleration values for the quadratic and oscillating fit
quad_acc = [el[2] for el in cmes['QUAD-FIT']]
oscil_acc = [el[4] for el in cmes['OSCIL-FIT'] if el[4] !=np.inf]
acc_label = 'Acceleration [m s$^{-2}$]'

# Third, set up the bins and their limits
bins_min = min(quad_acc + oscil_acc)
bins_max = max(quad_acc + oscil_acc)
bins = np.linspace(-30, 30, 100) # np.linspace(bin_min, bin_max, space)

# Fourth, plot the histogram of the quad and oscilating acceleration
plt.hist(oscil_acc, bins, alpha=0.5, normed=1, facecolor='blue', label='Oscillating Fit')
plt.hist(quad_acc, bins, alpha=0.5, normed=1, facecolor='green', label='Quadratic Fit')

# Formatting the histogram
plt.xlabel('Acceleration [m s$^{-2}$]')
plt.ylabel('Fraction')
plt.title('1996-2004, ' + acc_label)
plt.legend(loc=2)
plt.show()

# Second, grab the acceleration values for the quadratic and oscillating fit
quad_vel = [el[1]/1e3 for el in cmes['QUAD-FIT']]
oscil_vel = [el[3]/1e3 for el in cmes['OSCIL-FIT'] if el[3] !=np.inf]
vel_label = 'Velocity [km s$^{-1}$]'

# Third, set up the bins and their limits
bins_min = min(quad_vel + oscil_vel)
bins_max = max(quad_vel + oscil_vel)
bins = np.linspace(50, 1250, 100) # np.linspace(bin_min, bin_max, space)

# Fourth, plot the histogram of the quad and oscilating acceleration
plt.hist(oscil_vel, bins, alpha=0.5, normed=1, facecolor='blue', label='Oscillating Fit')
plt.hist(quad_vel, bins, alpha=0.5, normed=1, facecolor='green', label='Quadratic Fit')

# Formatting the histogram
plt.xlabel('Velocity [m s$^{-2}$]')
plt.ylabel('Fraction')
plt.title('1996-2004, ' + vel_label)
plt.legend(loc=2)
plt.show()

# Plotting histograms of oscillating fit other values
oscil_amp = [np.abs(el[0])/1000 for el in cmes['OSCIL-FIT'] if el[0] !=np.inf]
amp_label = 'Amplitude [km min$^{-1}$]'
amp_min = -375
amp_max = 375
cme_graphing.histogram(oscil_amp, amp_label, amp_min, amp_max)

oscil_per = [el[1] for el in cmes['OSCIL-FIT'] if el[1] !=np.inf]
per_label = 'Period [min]'
per_min = 0
per_max = 2000
cme_graphing.histogram(oscil_per, per_label, per_min, per_max)

import ipdb; ipdb.set_trace()

oscil_phi = [el[2] for el in cmes['OSCIL-FIT'] if el[2] !=np.inf]
phi_label = '$\Phi$'
phi_min = -50
phi_max = 50
cme_graphing.histogram(oscil_phi, phi_label, phi_min, phi_max)

