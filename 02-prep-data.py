import pandas as pd
from os import listdir
from io import StringIO

BASE_DATA_PATH = 'data/raw/'
COLUMNS = ('DATE-OBS', 'TIME-OBS', 'DETECTOR', 'FILTER', 'OBSERVER', 'FEAT_CODE', 'IMAGE_TYPE', 'YHT_ID',
           'ORIG_HTFILE', 'ORIG_WDFILE', 'UNIVERSAL', 'WDATA', 'WDATA', 'HALO', 'ONSET1', 'ONSET2', 'ONSET2',
           'CEN_PA', 'WIDTH', 'SPEED', 'ACCEL', 'FEAT_PA', 'FEAT_QUAL', 'QUALITY_INDEX', 'REMARK', 'COMMENT',
           'NUM_DATA_POINTS', 'HT_DATA')

file_names = listdir(BASE_DATA_PATH)

df = pd.DataFrame(columns=COLUMNS)

for file_name in file_names:
    with open(BASE_DATA_PATH + file_name) as file:
        lines = file.readlines()
        meta_data = lines[1:27]
        meta_data = [element.split(': ')[1][:-1] for element in meta_data]
        height_time_data = lines[27:]
        data = pd.read_csv(StringIO(''.join(height_time_data)),
                           sep='\s+', comment='#',
                           names=['HEIGHT', 'DATE', 'TIME', 'ANGLE', 'TEL', 'FC', 'COL', 'ROW'],
                           parse_dates=[[1, 2]])

        meta_data.append(len(data))
        meta_data.append(data)
        cme_count = len(df)
        df.loc[cme_count] = meta_data
        if cme_count % 1000 == 0:
            print(cme_count)

df.to_pickle('all_cmes.pkl')
