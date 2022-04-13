from glob import glob
import pandas as pd
import numpy as np

def read_files():
    files = glob('./stride_data/raw_data/*')
    for file in files:
        data = pd.read_csv(file, usecols=(1,2,3,11,12)) # 1=x, 2=y, 3=z, 11=stride_len, 12=stride_num
        seperate(data)    

def seperate(data:np.ndarray):
    stride_num = data.iloc[:,4].to_numpy()

    x_list = data.iloc[:,0].to_numpy()
    y_list = data.iloc[:,1].to_numpy()
    z_list = data.iloc[:,2].to_numpy()
    stride_len_list = data.iloc[:,3].to_numpy()

    last_stride_num = -1
    sep_index_list = []

    for i in range(len(stride_num)):
        if stride_num[i]<2: continue

        if last_stride_num != stride_num[i]:
            sep_index_list.append(i)
            last_stride_num = stride_num[i]

    print(sep_index_list)
    print(len(sep_index_list))
    
if __name__ == '__main__':
    data = read_files()
