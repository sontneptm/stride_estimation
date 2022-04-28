from glob import glob
from subject import Subject
import pandas as pd
import numpy as np

def read_files():
    subject_string_list = glob('./stride_lab_data/raw_data/*')
    subject_list = []

    for sub_str in subject_string_list:
        name = sub_str[sub_str.index("\\")+1:]
        subject = Subject(name)
        
        data_folder_str_list = glob(sub_str+'/*')

        acc_path = data_folder_str_list[0]
        
        for file in glob(acc_path+'/*'):
            if "1" in file: acc_type = 'l_wrist'
            elif "2" in file: acc_type = 'r_wrist'
            elif "4" in file: acc_type = 'l_ankle'
            elif "5" in file: acc_type = 'r_ankle'
            acc_data = pd.read_csv(file, header=None)
            
            subject.set_acc(acc_data, acc_type=acc_type)

        stride_path = data_folder_str_list[1]
        try:
            stride_data = pd.read_csv(stride_path+"/분할인덱스(왼발).csv")
            subject.set_stride_info(stride_data.to_numpy())
        except:
            print("no split index found")

        pp_path = data_folder_str_list[2]

        for file in glob(pp_path+'/*'):
            if "C" in file:
                if "L" in file: pp_type = 'l_pp'
                elif "R" in file: pp_type = 'r_pp'

                pp_data = pd.read_csv(file, header=None, encoding='ascii', usecols=(2,5))
                subject.set_plantar_pressure(pp_data, pp_type)

        subject_list.append(subject)

    return subject_list

if __name__ == '__main__':
    subject_list = read_files()
    for subject in subject_list:
        if subject.name in ['민세동','원혜연','이경준','이서영','장승완'] :
            print(subject.name)
            subject.find_stride_split_index()
            print()