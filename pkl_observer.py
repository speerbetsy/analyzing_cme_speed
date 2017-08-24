import pickle
import statistics
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import optimize

with open('cme_pkls/pkl_files/1996-01-13T22-19-18.pkl', 'rb') as f:
    cmes = pickle.load(f)

amp = cmes['OSCIL-FIT-AMP']
amp23 = cmes.loc[997]['OSCIL-FIT-AMP']
print(amp)
print("this is amp 23: ", amp23)


