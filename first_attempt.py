import pandas as pd
import numpy as np
from os import listdir
from io import StringIO
import pickle

def main():
    with open('all_cmes.pkl', 'rb') as f:
        data = pickle.load(f)
    #print (data)
    data.to_csv('out.csv', sep=' ')
    #data.to_csv(r'C:\Users\Betsy\Documents\TCD\cmevo', header=None, index=None, sep=' ', mode='a')
    #file = open("testfile.txt","w")
    #file.write(data)
    #file.close() 
    print ("done")

    
if __name__== "__main__":
    main()