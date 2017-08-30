import datetime
import cme_graphing
import numpy as np
import os

# Our parameters for the data
# Must be between 1994 and 2004
min_year = datetime.datetime(1996, 1, 1, 0, 0, 0)
max_year = datetime.datetime(2004, 12, 31, 23, 59, 59)

# Replicating the data from figure 1 in the paper
date1 = datetime.datetime(1996, 12, 19, 18, 30, 5)
date2 = datetime.datetime(2000, 3, 7, 16, 30, 5)

# Must have at least 11 height-time points
min_ht = 11  # Generates 3465 cmes

# Create the lists for those two days where we will store their values
cmes = cme_graphing.find_file(min_year, max_year, min_ht)


# Creating the linear, quadratic, and oscillating fit arrays that will be
# put in the dictionary later to be returned
runs = np.arange(1000)

# First FOR loop: grab a CME
for index, cme in cmes.iloc[0:, :].iterrows():
    # Grab the height time data as well as the type
    # of telescope
    ht = cme.loc['HT_DATA']
    times = ht.DATE_TIME.values
    heights = ht.HEIGHT
    tscope = ht.TEL
    
    # Store the date-time info as a string
    # to be used for CME identification in files
    # and graph titles, then create directories
    # for each of the 3,465 CMEs
    desc = str(times[0])
    desc = desc[0:10]+'T'+desc[11:13]+'-'+desc[14:16]+'-'+desc[17:19]
    newpath = "cme_pkls/images/" + desc +'new'+ "/"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    # Create arrays of all 3 types of fits
    # and the dates and run number to be
    # stored in a pkl file later
    lin_fit_array = []
    quad_fit_array = []
    oscil_fit_array = []
    dates = []
    ran = []
    
    # Second FOR loop: try 1000 fits for that CME
    for n in runs:
        # times is in datetime, heights is in Rsun
        df = cme_graphing.height_velocity_graphs(times, heights, desc,
                                                n, tscope)
        lin_fit_array.append(df['Linear Fit'])
        quad_fit_array.append(df['Quad Fit'])
        oscil_fit_array.append(df['Oscillating Fit'])
        dates.append(df['Date'])
        if (df['Oscillating Fit'][0] != np.inf):
            ran.append(n)
    cme_graphing.to_pkl_file(ran, lin_fit_array, quad_fit_array, oscil_fit_array, desc)
