from bs4 import BeautifulSoup
from urllib import request
import re
import pandas as pd
import os

base_data_path = 'data'
base_url = 'http://cdaw.gsfc.nasa.gov/CME_list/'

r = request.urlopen(base_url).read()
soup = BeautifulSoup(r, 'html.parser')

table = soup.find('table')

year_month_links = table.find_all('a', href=True)


def parse_yht(data):
    return pd.read_csv(data)


for link in year_month_links:
    path_url = '/'.join(link.get('href').split('/')[:-1]) + '/'
    r = request.urlopen(base_url+link.get('href')).read()
    soup = BeautifulSoup(r, 'html.parser')
    events = soup.find_all(href=re.compile("yht"))
    for event in events:
        file_name = event.get('href').split('/')[-1]
        file_path = base_data_path+'/raw/' + file_name

        raw_data = request.urlopen(base_url+path_url+event.get('href')).read()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            file.write(raw_data)
        # data = parse_yht(raw_data)
