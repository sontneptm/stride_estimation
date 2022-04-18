from os import sep
import numpy as np

class Subject():
    def __init__(self, name) -> None:
        self.name = name
        self.l_wrist_data = None
        self.r_wrist_data = None
        self.l_ankle_data = None
        self.r_ankle_data = None
        self.stride_info = None
        self.l_plantar_pressure = None
        self.r_plantar_pressure = None
        
    def set_acc(self, data, acc_type):
        if acc_type == "l_wrist":
            self.l_wrist_data = data
        elif acc_type == "r_wrist":
            self.r_wrist_data = data
        elif acc_type == "l_ankle":
            self.l_ankle_data = data
        elif acc_type == "r_ankle":
            self.r_ankle_data = data

    def set_plantar_pressure(self, data, pp_type):
        if pp_type== "l_pp":
            self.l_plantar_pressure = data.to_numpy()
        elif pp_type== "r_pp":
            self.r_plantar_pressure = data.to_numpy()

    def set_stride_info(self, data):
        self.stride_info = data

    def find_stride_split_index(self):
        self.stride_split_index=[]

        step_flag = False

        for i in range(len(self.l_plantar_pressure)):
            time = self.l_plantar_pressure[i][0]
            pp_value = self.l_plantar_pressure[i][1]

            if pp_value < 100 and not step_flag:
                step_flag=True
            elif pp_value > 100 and step_flag:
                step_flag=False
                self.stride_split_index.append(i)

        print(self.stride_split_index)

    def save_as_one_stride(self):
        file = open("stride_lab_data/processed_data/"+self.name+"/pp_data.csv", 'a')

        for info in self.stride_info:
            total_data = []
            stride_length = info[2]
            data = self.l_plantar_pressure[info[0]:info[1]][:,1]

            total_data.append(stride_length)
            total_data = np.concatenate((total_data, data), axis=0)
            total_data = str(total_data)[1:-1].split()
            total_data = str(list(map(int, total_data)))[1:-1]

            file.write(total_data+'\n')





            