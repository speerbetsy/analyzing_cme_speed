import datetime
import cme_graphing
import matplotlib.pyplot as plt
import numpy as np
import pickle
import statistics
import cme_graphing
import os


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

# Create arrays to store the final values for all 3,465 cmes
date_array = []
amp_array = []
per_array = []
phase_array = []
vel_array = []
acc_array = []
quad_chi_array = []
chi_array = []

# Second, open all 3424 pkl files
for index, cme in cmes.iloc[2686:, :].iterrows():
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
    newpath = 'cme_pkls/pkl_files/' + desc + 'new' +'.pkl'
    
    if not os.path.exists(newpath):
        print("file ",newpath, " does not exist")
        continue
    with open(newpath, 'rb') as f:
        new_cmes = pickle.load(f)

    # Third, grab the amplitude, period, and phase values for the oscillating fit
# AMPLITUDE    
    oscil_amp = new_cmes['OSCIL-FIT-AMP']/1e3
    # if there are no fits, then skip this date and move on to the next in the loop
    if (len(oscil_amp) == 0):
            continue
    amp_label = 'Amplitude [km min$^{-1}$]'
    # Set up the bins and their limits
    amp_min = min(oscil_amp)
    amp_max = max(oscil_amp)
    amp_val = cme_graphing.histogram(oscil_amp, amp_label, amp_min, amp_max, "amplitudenew/"+str(desc)+"index"+str(index))

# PERIOD
    oscil_per = new_cmes['OSCIL-FIT-PER']/60
    per_label = 'Period [minutes]'
    per_min = min(oscil_per)
    per_max = max(oscil_per)
    per_val = cme_graphing.histogram(oscil_per, per_label, per_min, per_max, "periodnew/"+str(desc)+"index"+str(index))
    # must meet minimum reqs (30<amp<400, per>60 min)
    if ((amp_val < 30) or (amp_val >400) or (per_val < 60)):
        print("Does not meet requirements for oscillatory fit")
        continue

# PHASE
    oscil_phi = new_cmes['OSCIL-FIT-PHASE']
    phi_label = '$\Phi$'
    phi_min = min(oscil_phi)
    phi_max = max(oscil_phi)
    phi_val = cme_graphing.histogram(oscil_phi, phi_label, phi_min, phi_max, "phasenew/"+str(desc)+"index"+str(index))

    # Fouth, grab the acceleration values for the quadratic and oscillating fit
# ACCELERATION    
    quad_acc = [el[2] for el in new_cmes['QUAD-FIT']]
    oscil_acc = new_cmes['OSCIL-FIT-ACC']
    acc_label = 'Acceleration [m s$^{-2}$]'

    bins_min = min(quad_acc + oscil_acc)
    bins_max = max(quad_acc + oscil_acc)
    bins = np.linspace(bins_min,bins_max, 100) # np.linspace(bin_min, bin_max, space)

    # Plot the histogram of the quad and oscilating acceleration
    plt.hist(quad_acc, bins, alpha=0.5, normed=1, facecolor='green', label='Quadratic Fit')
    n, bins, patches = plt.hist(oscil_acc, bins, alpha=0.5, normed=1, facecolor='blue', label='Oscillating Fit')
    max_bin = int(np.argmax(n))
    acc_val = bins[max_bin]

    # Formatting the histogram
    plt.xlabel('Acceleration [m s$^{-2}$]')
    plt.ylabel('Fraction')
    plt.title('1996-2004, ' + acc_label + "=quad: %.1f"%statistics.mean(quad_acc) + '/oscil: ' + "=%.1f"%statistics.mean(oscil_acc) + ",quad median=%.1f"%statistics.median(quad_acc) + ",oscil median=%.1f"%statistics.median(oscil_acc) + ', # of CMEs = 3465')
    plt.legend(loc=2)
    plt.savefig("cme_pkls/final_oscillation/accelerationnew/" + str(desc)+str(index) +".png")
    plt.clf()
#    plt.show()

# VELOCITY
    quad_vel = [el[1]/1000 for el in new_cmes['QUAD-FIT']]
    oscil_vel = new_cmes['OSCIL-FIT-VEL']/1000
    vel_label = 'Velocity [km s$^{-1}$]'

    # Set up the bins and their limits
    bins_min = min(quad_vel + oscil_vel)
    bins_max = max(quad_vel + oscil_vel)
    bins = np.linspace(-100, 1000, 100) # np.linspace(bin_min, bin_max, space)

    # Plot the histogram of the quad and oscilating acceleration
    plt.hist(quad_vel, bins, alpha=0.5, normed=1, facecolor='green', label='Quadratic Fit')
    # Getting oscil peak val
    n, bins, patches = plt.hist(oscil_vel, bins, alpha=0.5, normed=1, facecolor='blue',           label='Oscillating Fit')
    max_bin = int(np.argmax(n))
    vel_val = bins[max_bin]

    # Formatting the histogram
    plt.xlabel('Velocity [km s$^{-2}$]')
    plt.ylabel('Fraction')
    plt.title('1996-2004, ' + vel_label + "=quad: %.1f"%statistics.mean(quad_vel) + '/oscil: ' + "=%.1f"%statistics.mean(oscil_vel) + ",quad median=%.1f"%statistics.median(quad_vel) + ",oscil median=%.1f"%statistics.median(oscil_vel) + ', # of CMEs = 3465')
    plt.legend(loc=2)
    plt.savefig("cme_pkls/final_oscillation/velocitynew/" + str(desc)+str(index) +".png")
    plt.clf()
   # plt.show()

#CHISQ Values
    quad = np.array(new_cmes['QUAD-FIT'])
    quad_chi = []
    for i in range(len(quad)):
        quad_chi.append(quad[i][3])
    quad_chi_val = quad_chi[0] #they are all the same
    
    oscil_chi = new_cmes['OSCIL-FIT-CHISQ']
    chi_label = 'Reduced Chi Squared'

    chi_min = min(oscil_chi)
    chi_max = max(oscil_chi)
    chi_val = cme_graphing.histogram(oscil_chi, chi_label, chi_min, chi_max, "chisqnew/"+str(desc)+"index"+str(index))

    #Fifth, appending all important values from oscil fit
    date_array.append(desc)
    amp_array.append(amp_val)
    per_array.append(per_val)
    phase_array.append(phi_val)
    vel_array.append(vel_val)
    acc_array.append(acc_val)
    chi_array.append(chi_val)
    quad_chi_array.append(quad_chi_val)

print("this is len of acceptable cmes: ",len(amp_array), " ", len(phase_array))
cme_graphing.to_pkl_file2(len(amp_array), date_array, amp_array, per_array, phase_array, vel_array, acc_array, chi_array, quad_chi_array)
