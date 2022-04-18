class Subject():
    def __init__(self, name) -> None:
        self.name = name
        self.l_wrist_data = None
        self.r_wrist_data = None
        self.l_ankle_data = None
        self.r_ankle_data = None
        
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
            self.l_plantar_pressure = data
        elif pp_type== "r_pp":
            self.r_plantar_pressure = data


