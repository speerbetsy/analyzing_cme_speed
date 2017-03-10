import pandas as pd
from os import listdir

base_data_path = 'data/raw/'

files = listdir(base_data_path)

for file in files:
     data = pd.read_csv('data/raw/'+files[-1],
            sep='\s+',  comment='#', 
            names=['height', 'date', 'time', 'angle', 'tel', 'fc', 'col', 'row'],
            parse_dates=[[1,2]])
