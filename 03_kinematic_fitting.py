import datetime
import cme_graphing

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


# Creating the linear, quadratic, and oscillating fit arrays that will be
# put in the dictionary later to be returned
dates = []
lin_fit_array = []
quad_fit_array = []
oscil_fit_array = []

x = 200  # will run for first 1000 due to time constraints
for index, cme in cmes.iloc[300:350, :].iterrows():
    ht = cme.loc['HT_DATA']
    times = ht.DATE_TIME.values
    heights = ht.HEIGHT
    tscope = ht.TEL
    # times is in datetime, heights is in Rsun
    df = cme_graphing.height_velocity_graphs(times, heights, str(times[0]),
                                             tscope)
    lin_fit_array.append(df['Linear Fit'])
    quad_fit_array.append(df['Quad Fit'])
    oscil_fit_array.append(df['Oscillating Fit'])
    dates.append(df['Date'])

cme_graphing.to_pkl_file(dates, lin_fit_array, quad_fit_array, oscil_fit_array)
