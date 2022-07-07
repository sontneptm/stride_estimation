from datetime import datetime
from re import S
from scipy.signal import find_peaks, butter, sosfilt
from scipy.fft import fft
from matplotlib import pyplot as plt
import numpy as np
import os

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

    def convert_timedelata_to_milisec(self, delta):
        return (delta.microseconds/1000) + (delta.seconds*1000)

    def find_stride_split_index(self, type="left"):
        self.stride_split_index=[]

        step_flag = False

        target = self.l_plantar_pressure
        
        if type == 'right':
            target = self.r_plantar_pressure

        for i in range(len(target)):
            pp_value = target[i][1]

            if pp_value < 200 and not step_flag:
                step_flag=True
            elif pp_value > 200 and step_flag:
                step_flag=False
                self.stride_split_index.append(i)

        print(self.stride_split_index)

    def find_peaks_index(self, pp_data) -> list:
        peaks = find_peaks(pp_data, height=2000, distance=20)
        zero_point = -1
        for i in range(len(pp_data)):
            data = pp_data[i]
            if data <= 90:
                zero_point = i
                break

        return [peaks[0][0], peaks[0][1], zero_point]

    def change_str_to_time(self, time_str:str):
        rtn_time = time_str.replace(" ", "")
        try:
            rtn_time = datetime.strptime("1"+rtn_time, '%H:%M:%S.%f')
        except:
            rtn_time = datetime.strptime("1"+rtn_time, '%H:%M:%S')
        rtn_time = rtn_time.replace(hour=(rtn_time.hour+2))

        return rtn_time

    def find_r_swing_index(self, pp_data):
        swing_start = -1
        swing_end = -1
        for i in range(len(pp_data)):
            data = pp_data[i]
            
            if swing_start == -1 and data < 200:
                swing_start = i+1
            if swing_start != -1 and data > 200:
                swing_end = i-1
                break
        
        return swing_start, swing_end

    def extract_gait_features(self, l_pp_data, r_pp_data):
        gait_features = []

        l_pp_len = len(l_pp_data)
        l_time = l_pp_data[:,:1].squeeze()
        l_pp_data = l_pp_data[:, 1:].squeeze()

        r_time = r_pp_data[:,:1].squeeze()
        r_pp_data = r_pp_data[:, 1:].squeeze()

        r_swing_start_index, r_swing_end_index = self.find_r_swing_index(r_pp_data)

        peaks_index = self.find_peaks_index(l_pp_data)
        hs_index = peaks_index[0]
        to_index = peaks_index[1]
        swing_index = peaks_index[2]

        r_swing_start_t = self.change_str_to_time(r_time[r_swing_start_index])
        r_swing_end_t = self.change_str_to_time(r_time[r_swing_end_index])

        initial_contact_t = self.change_str_to_time(l_time[0])
        heel_strike_t = self.change_str_to_time(l_time[hs_index])
        toe_off_t = self.change_str_to_time(l_time[to_index])
        swing_t = self.change_str_to_time(l_time[swing_index])
        last_t = self.change_str_to_time(l_time[l_pp_len-1])

        self.swing_t = swing_t
        self.r_swing_start_t = r_swing_start_t
        self.r_swing_end_t = r_swing_end_t

        stride_time = self.convert_timedelata_to_milisec(np.absolute(initial_contact_t - last_t))
        r_step_time = self.convert_timedelata_to_milisec(np.absolute(initial_contact_t - r_swing_end_t))
        l_step_time = self.convert_timedelata_to_milisec(np.absolute(r_swing_end_t - last_t))
        stance_time = self.convert_timedelata_to_milisec(np.absolute(initial_contact_t - swing_t))
        swing_time = self.convert_timedelata_to_milisec(np.absolute(swing_t - last_t))
        stance_swing_ratio = stance_time / swing_time

        first_double_support_time = self.convert_timedelata_to_milisec(np.absolute(initial_contact_t - r_swing_start_t))
        single_support_time = self.convert_timedelata_to_milisec(np.absolute(r_swing_start_t - r_swing_end_t))
        second_double_support_time = self.convert_timedelata_to_milisec(np.absolute(r_swing_end_t - swing_t))
        hs_time = self.convert_timedelata_to_milisec(np.absolute(initial_contact_t - heel_strike_t))
        full_contact_time = self.convert_timedelata_to_milisec(np.absolute(heel_strike_t - toe_off_t))
        to_time = self.convert_timedelata_to_milisec(np.absolute(toe_off_t - swing_t))

        gait_features.append(stride_time)
        gait_features.append(r_step_time)
        gait_features.append(l_step_time)
        gait_features.append(stance_time)
        gait_features.append(swing_time)
        gait_features.append(stance_swing_ratio)

        gait_features.append(first_double_support_time)
        gait_features.append(single_support_time)
        gait_features.append(second_double_support_time)
        gait_features.append(hs_time)
        gait_features.append(full_contact_time)
        gait_features.append(to_time)

        return gait_features

    def moving_average(self, data:np.ndarray, window_size) -> np.ndarray:
        return np.convolve(data, np.ones((window_size,)) / window_size, mode='same') 

    def check_freq_window(self, data):
        data_freq = fft(data)

        plt.plot(data * 10)
        plt.plot(data_freq)
        plt.show()

    def get_sliced_acc(self, start_idx, end_idx, swing_idx=None, type="l_ankle"):
        target_data = None  
        if type == "l_ankle":   
            target_data = self.l_ankle_data
        elif type == "r_ankle":   
            target_data = self.r_ankle_data
        elif type == "l_wrist":   
            target_data = self.l_wrist_data
        elif type == "r_wrist":   
            target_data = self.r_wrist_data

        acc_x = target_data[start_idx:end_idx][:,1]
        acc_y = target_data[start_idx:end_idx][:,2]
        acc_z = target_data[start_idx:end_idx][:,3]

        acc_x = self.moving_average(acc_x, 5)
        acc_y = self.moving_average(acc_y, 5)
        acc_z = self.moving_average(acc_z, 5)

        # apply low pass filter
        sos = butter(10, 10, 'lowpass', fs=40, output='sos')
        acc_x = sosfilt(sos, acc_x)
        acc_y = sosfilt(sos, acc_y)
        acc_z = sosfilt(sos, acc_z)

        # get rid of acc in stance phase
        if swing_idx is not None:
            acc_x = acc_x[swing_idx:]
            acc_y = acc_y[swing_idx:]
            acc_z = acc_z[swing_idx:]

        return acc_x, acc_y, acc_z

    def get_svm(self, acc_x, acc_y, acc_z):
        rtn_svm = []

        def get_root_square(x,y,z):
            return np.sqrt(np.power(x,2)+np.power(y,2)+np.power(z,2))
        
        rtn_svm = list(map(get_root_square, acc_x, acc_y, acc_z))

        return rtn_svm

    def save_as_one_stride(self):
        os.makedirs("stride_lab_data/processed_data/"+self.name, exist_ok=True)

        file = open("stride_lab_data/processed_data/"+self.name+"/processed_data.csv", 'a')

        for info in self.stride_info:
            total_data = []
            walking_speed = None
            stride_length = info[2]
            r_step_length = info[3]
            l_step_length = info[4]

            start_time = self.change_str_to_time(self.l_plantar_pressure[info[0]][0])
            end_time = self.change_str_to_time(self.l_plantar_pressure[info[1]][0])
            
            r_pp_start_index, r_pp_end_index = self.find_index_by_time(type='r_pp', s_time=start_time, e_time=end_time)

            l_pp_data = self.l_plantar_pressure[info[0]:info[1]][:,1]
            r_pp_data = self.r_plantar_pressure[r_pp_start_index:r_pp_end_index][:,1]

            gait_features = self.extract_gait_features(self.l_plantar_pressure[info[0]:info[1]], self.r_plantar_pressure[r_pp_start_index:r_pp_end_index])

            walking_speed = stride_length/(gait_features[0]/1000.0)

            # 전체 가속도 데이터에서 한 stride에 해당하는 부분 slice
            l_ankle_start_index, l_ankle_end_index = self.find_index_by_time(type='l_ankle', s_time=start_time, e_time=end_time)
            r_ankle_start_index, r_ankle_end_index = self.find_index_by_time(type='r_ankle', s_time=start_time, e_time=end_time)
            l_wrist_start_index, l_wrist_end_index = self.find_index_by_time(type='l_wrist', s_time=start_time, e_time=end_time)
            r_wrist_start_index, r_wrist_end_index = self.find_index_by_time(type='r_wrist', s_time=start_time, e_time=end_time)

            l_ankle_end_index = l_ankle_start_index + (int(info[1]/10 * 4) - int(info[0]/10 * 4))
            r_ankle_end_index = r_ankle_start_index + (int(info[1]/10 * 4) - int(info[0]/10 * 4))
            l_wrist_end_index = l_wrist_start_index + (int(info[1]/10 * 4) - int(info[0]/10 * 4))
            r_wrist_end_index = r_wrist_start_index + (int(info[1]/10 * 4) - int(info[0]/10 * 4))

            l_ankle_swing_index, _ = self.find_index_by_time(type='l_ankle', s_time=self.swing_t, e_time=end_time)
            l_ankle_swing_index = l_ankle_swing_index - l_ankle_start_index     

            r_ankle_swing_start_index, r_ankle_swing_end_index = self.find_index_by_time(type='r_ankle', s_time=self.r_swing_start_t, e_time=self.r_swing_end_t)
            r_ankle_swing_start_index -= r_ankle_start_index
            r_ankle_swing_end_index -= r_ankle_end_index

            l_ankle_x, l_ankle_y, l_ankle_z = self.get_sliced_acc(l_ankle_start_index, l_ankle_end_index, swing_idx=l_ankle_swing_index, type="l_ankle")
            r_ankle_x, r_ankle_y, r_ankle_z = self.get_sliced_acc(r_ankle_start_index, r_ankle_end_index, type="r_ankle")
            #TODO r_ankle acc도 right foot swing phase 구간으로 잘라야함
            l_wrist_x, l_wrist_y, l_wrist_z = self.get_sliced_acc(l_wrist_start_index, l_wrist_end_index, type="l_wrist")
            r_wrist_x, r_wrist_y, r_wrist_z = self.get_sliced_acc(r_wrist_start_index, r_wrist_end_index, type="r_wrist")
            
            # self.check_freq_window(l_ankle_x)
            l_db_y = np.cumsum(np.cumsum(l_ankle_y))
        
            l_ankle_svm = self.get_svm(l_ankle_x,l_ankle_y,l_ankle_z)
            r_ankle_svm = self.get_svm(r_ankle_x,r_ankle_y,r_ankle_z)
            l_wrist_svm = self.get_svm(l_wrist_x,l_wrist_y,l_wrist_z)
            r_wrist_svm = self.get_svm(r_wrist_x,r_wrist_y,r_wrist_z)

            if len(l_ankle_x) < 5 or len(l_ankle_x) > 30:
                continue
            if len(r_ankle_x) < 5 or len(r_ankle_x) > 50:
                continue
            while (len(l_ankle_x)<30):
                l_ankle_x = np.append(l_ankle_x, 0)
                l_ankle_y = np.append(l_ankle_y, 0)
                l_ankle_z = np.append(l_ankle_z, 0)
                l_db_y = np.append(l_db_y, l_db_y[len(l_db_y)-1])
                l_ankle_svm = np.append(l_ankle_svm,0)

            while (len(r_ankle_x)<50):
                r_ankle_x = np.append(r_ankle_x, 0)
                r_ankle_y = np.append(r_ankle_y, 0)
                r_ankle_z = np.append(r_ankle_z, 0)
                r_ankle_svm = np.append(r_ankle_svm, 0)

            while (len(l_wrist_x)<50):
                l_wrist_x = np.append(l_wrist_x, 0)
                l_wrist_y = np.append(l_wrist_y, 0)
                l_wrist_z = np.append(l_wrist_z, 0)
                l_wrist_svm = np.append(l_wrist_svm, 0)

            while (len(r_wrist_x)<50):
                r_wrist_x = np.append(r_wrist_x, 0)
                r_wrist_y = np.append(r_wrist_y, 0)
                r_wrist_z = np.append(r_wrist_z, 0)
                r_wrist_svm = np.append(r_wrist_svm, 0)
    
            while (len(l_pp_data)<125):
                l_pp_data = np.append(l_pp_data, 0)

            while (len(r_pp_data)<125):
                r_pp_data = np.append(r_pp_data, 0)

            total_data.append(walking_speed)
            total_data.append(stride_length)
            total_data.append(r_step_length)
            total_data.append(l_step_length)

            total_data = np.concatenate((total_data, gait_features), axis=0)

            total_data = np.concatenate((total_data, l_pp_data), axis=0)
            total_data = np.concatenate((total_data, r_pp_data), axis=0)
            
            total_data = np.concatenate((total_data, l_db_y), axis=0)

            total_data = np.concatenate((total_data, l_ankle_z), axis=0)
            total_data = np.concatenate((total_data, l_ankle_y), axis=0)
            total_data = np.concatenate((total_data, l_ankle_z), axis=0)

            total_data = np.concatenate((total_data, r_ankle_x), axis=0)
            total_data = np.concatenate((total_data, r_ankle_y), axis=0)
            total_data = np.concatenate((total_data, r_ankle_z), axis=0)

            total_data = np.concatenate((total_data, l_wrist_x), axis=0)
            total_data = np.concatenate((total_data, l_wrist_y), axis=0)
            total_data = np.concatenate((total_data, l_wrist_z), axis=0)
            
            total_data = np.concatenate((total_data, r_wrist_x), axis=0)
            total_data = np.concatenate((total_data, r_wrist_y), axis=0)
            total_data = np.concatenate((total_data, r_wrist_z), axis=0)

            total_data = str(total_data)[1:-1].split()
            total_data = str(list(map(np.float32, total_data)))[1:-1]
            total_data = total_data.replace(" ", "")

            file.write(total_data+'\n')

    def find_index_by_time(self, type, s_time, e_time):
        if type == 'l_ankle': target=self.l_ankle_data
        elif type == 'r_ankle': target=self.r_ankle_data
        elif type == 'l_wrist': target=self.l_wrist_data
        elif type == 'r_wrist': target=self.r_wrist_data
        elif type == 'r_pp': target=self.r_plantar_pressure

        s_min = None
        s_min_index = None
        e_min = None
        e_min_index = None

        for i in range(len(target)):
            time = target[i][0]
            if type=='r_pp':
                time = time.replace(" ", "")
                time = "1"+time
                try:
                    time = datetime.strptime(time, '%H:%M:%S.%f')
                except:
                    time = datetime.strptime(time, '%H:%M:%S')
                time = time.replace(hour=(time.hour+2))
            else:
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
