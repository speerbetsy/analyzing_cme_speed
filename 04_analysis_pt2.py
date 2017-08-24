import datetime
import cme_graphing
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cme_graphing


histogram=plt.figure()
# First, grab the dates bc the files we want are named after the dates

# Our parameters for the data
# Must be between 1994 and 2004
min_year = datetime.datetime(1996, 1, 1, 0, 0, 0)
max_year = datetime.datetime(2004, 12, 31, 23, 59, 59)

# Replicating the data from figure 1 in the paper
date1 = datetime.datetime(1996, 12, 19, 18, 30, 5)
date2 = datetime.datetime(2000, 3, 7, 16, 30, 5)

# Must have at least 11 height-time points
min_ht = 10  # Generates 3465 cmes

# Create the lists for those two days where we will store their values
cmes = cme_graphing.find_file(min_year, max_year, min_ht)

# Second, open all 3424 pkl files
for index, cme in cmes.iloc[234:334, :].iterrows():
    # Grab the height time data as well as the type
    # of telescope
    ht = cme.loc['HT_DATA']
    times = ht.DATE_TIME.values
    
    # Store the date-time info as a string
    # to be used for CME identification in files
    # and graph titles, then create directories
    # for each of the 3,465 CMEs
    desc = str(times[0])
    desc = desc[0:10]+'T'+desc[11:13]+'-'+desc[14:16]+'-'+desc[17:19]
    newpath = 'cme_pkls/pkl_files/' + desc + '.pkl'
    print(newpath)


    with open(newpath, 'rb') as f:
        new_cmes = pickle.load(f)


    # Third, grab the acceleration values for the quadratic and oscillating fit
    quad_acc = [el[2] for el in new_cmes['QUAD-FIT']]
    oscil_acc = new_cmes['OSCIL-FIT-ACC']
    acc_label = 'Acceleration [m s$^{-2}$]'
    
    if (quad_acc == []):
        continue

    # Set up the bins and their limits
    bins_min = min(quad_acc + oscil_acc)
    bins_max = max(quad_acc + oscil_acc)
    bins = np.linspace(-30, 30, 100) # np.linspace(bin_min, bin_max, space)

    # Plot the histogram of the quad and oscilating acceleration
    plt.hist(oscil_acc, bins, alpha=0.5, normed=1, facecolor='blue', label='Oscillating Fit')
    plt.hist(quad_acc, bins, alpha=0.5, normed=1, facecolor='green', label='Quadratic Fit')

    # Formatting the histogram
    plt.xlabel('Acceleration [m s$^{-2}$]')
    plt.ylabel('Fraction')
    plt.title('1996-2004, ' + acc_label)
    plt.legend(loc=2)
    plt.show()

    # Fouth, grab the acceleration values for the quadratic and oscillating fit
    quad_vel = [el[1]/1e3 for el in new_cmes['QUAD-FIT']]
    oscil_vel = new_cmes['OSCIL-FIT-VEL']
    vel_label = 'Velocity [km s$^{-1}$]'

    # Set up the bins and their limits
    bins_min = min(quad_vel + oscil_vel)
    bins_max = max(quad_vel + oscil_vel)
    bins = np.linspace(50, 1250, 100) # np.linspace(bin_min, bin_max, space)

    # Plot the histogram of the quad and oscilating acceleration
    plt.hist(oscil_vel, bins, alpha=0.5, normed=1, facecolor='blue', label='Oscillating Fit')
    plt.hist(quad_vel, bins, alpha=0.5, normed=1, facecolor='green', label='Quadratic Fit')

    # Formatting the histogram
    plt.xlabel('Velocity [m s$^{-2}$]')
    plt.ylabel('Fraction')
    plt.title('1996-2004, ' + vel_label)
    plt.legend(loc=2)
    plt.show()

    # Plotting histograms of oscillating fit other values
    oscil_amp = new_cmes['OSCIL-FIT-AMP']/1e3
    amp_label = 'Amplitude [km min$^{-1}$]'
    amp_min = min(oscil_amp)
    amp_max = max(oscil_amp)
    cme_graphing.histogram(oscil_amp, amp_label, amp_min, amp_max)

    oscil_per = new_cmes['OSCIL-FIT-PER']/60
    per_label = 'Period [minutes]'
    per_min = min(oscil_per)
    per_max = max(oscil_per)
    cme_graphing.histogram(oscil_per, per_label, per_min, per_max)

    oscil_phi = new_cmes['OSCIL-FIT-PHASE']
    phi_label = '$\Phi$'
    phi_min = min(oscil_phi)
    phi_max = max(oscil_phi)
    cme_graphing.histogram(oscil_phi, phi_label, phi_min, phi_max)

