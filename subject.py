class Subject():
    def __init__(self, name) -> None:
        self.name = name
        self.l_wrist_data = None
        self.r_wrist_data = None
        self.l_ankle_data = None
        self.r_ankle_data = None
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
