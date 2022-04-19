from datetime import datetime, time
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
            self.l_wrist_data = data.to_numpy()
        elif acc_type == "r_wrist":
            self.r_wrist_data = data.to_numpy()
        elif acc_type == "l_ankle":
            self.l_ankle_data = data.to_numpy()
        elif acc_type == "r_ankle":
            self.r_ankle_data = data.to_numpy()

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
            start_time = self.l_plantar_pressure[info[0]][0].replace(" ","")
            end_time = self.l_plantar_pressure[info[1]][0].replace(" ","")

            start_time = datetime.strptime('1'+start_time,'%H:%M:%S.%f')
            start_time = start_time.replace(hour= (start_time.hour + 2))
            end_time = datetime.strptime('1'+end_time,'%H:%M:%S.%f')
            end_time = end_time.replace(hour= (end_time.hour + 2))
            
            l_ankle_start_index, l_ankle_end_index = self.find_index_with_time(type='l_ankle', s_time=start_time, e_time=end_time)

            pp_data = self.l_plantar_pressure[info[0]:info[1]][:,1]
            l_ankle_x = self.l_ankle_data[l_ankle_start_index:l_ankle_end_index][:,1]
            l_ankle_y = self.l_ankle_data[l_ankle_start_index:l_ankle_end_index][:,2]
            l_ankle_z = self.l_ankle_data[l_ankle_start_index:l_ankle_end_index][:,3]

            if len(l_ankle_x) < 20 or len(l_ankle_x) > 60:
                continue
            else:
                while (len(l_ankle_x)<60):
                    l_ankle_x = np.append(l_ankle_x, 0)
                    l_ankle_y = np.append(l_ankle_y, 0)
                    l_ankle_z = np.append(l_ankle_z, 0)
    
            while (len(pp_data)<120):
                pp_data = np.append(pp_data, 0)

            total_data.append(stride_length)
            total_data = np.concatenate((total_data, pp_data), axis=0)
            total_data = np.concatenate((total_data, l_ankle_x), axis=0)
            total_data = np.concatenate((total_data, l_ankle_y), axis=0)
            total_data = np.concatenate((total_data, l_ankle_z), axis=0)

            total_data = str(total_data)[1:-1].split()
            total_data = str(list(map(np.float32, total_data)))[1:-1]
            total_data = total_data.replace(" ", "")

            file.write(total_data+'\n')

    def find_index_with_time(self, type, s_time, e_time):
        if type == 'l_ankle': target=self.l_ankle_data
        elif type == 'r_ankle': target=self.r_ankle_data

        s_min = None
        s_min_index = None
        e_min = None
        e_min_index = None

        for i in range(len(target)):
            time = target[i][0]
            time = datetime.strptime(time, '%H:%M:%S:%f')

            s_sub = np.absolute(s_time-time)
            e_sub = np.absolute(e_time-time)

            if s_min is None:
                s_min = s_sub
                s_min_index = i
                e_min = e_sub
                e_min_index = i
            elif s_sub<s_min:
                s_min = s_sub
                s_min_index = i
            elif e_sub<e_min:
                e_min = e_sub
                e_min_index = i

        return s_min_index, e_min_index






            