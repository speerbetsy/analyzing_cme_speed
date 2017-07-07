import datetime
import cme_graphing

# Our parameters for the data
# Must be between 1994 and 2004
min_year = datetime.datetime(1996, 1, 1, 0, 0, 0)
max_year = datetime.datetime(2004, 12, 31, 23, 59, 59)

# Must have at least 11 height-time points
min_ht = 10  # Generates 3465 cmes

# Create the lists for those two days where we will store their values
cmes = cme_graphing.find_file(min_year, max_year, min_ht)

dates = []
fit_values = []
rchi_values = []
x = 5  # will run for first 1000 due to time constraints
for index, cme in cmes.iloc[2:x, :].iterrows():
    ht = cme.loc['HT_DATA']
    times = ht.DATE_TIME.values
    heights = ht.HEIGHT
    tscope = ht.TEL
    # times is in datetime, heights is in Rsun
    f, r = cme_graphing.height_velocity_graphs(times, heights, str(times[0]),
                                               tscope)
    dates.append(str(times[0]))
    fit_values.append(f)
    rchi_values.append(r)

cme_graphing.to_pkl_file(dates, fit_values, rchi_values)

# 1,