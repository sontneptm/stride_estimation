from glob import glob
import pandas as pd
import numpy as np

def read_files():
    files = glob('./stride_data/data/*')
    datas = []
    for file in files:
        data = pd.read_csv(file, usecols=(1,2,3,11,12)) # 1=x, 2=y, 3=z, 11=stride_len, 12=stride_num
        print(type(data))
        
    return datas

def seperate(data:pd.DataFrame):
    

if __name__ == '__main__':
    data = read_files()

    print(type(data))
    print(data)