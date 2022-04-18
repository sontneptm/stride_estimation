class Subject():
    def __init__(self, name) -> None:
        self.name = name
        self.l_wrist_data = None
        self.r_wrist_data = None
        self.l_ankle_data = None
        self.r_ankle_data = None
        
    def set_acc(self, data, acc_type):
        if acc_type == "l_wrist":
            print("check1")
            self.l_wrist_data = data
        elif acc_type == "r_wrist":
            print("check2")
            self.r_wrist_data = data
        elif acc_type == "l_ankle":
            print("check3")
            self.l_ankle_data = data
        elif acc_type == "r_ankle":
            print("check4")
            self.r_ankle_data = data


