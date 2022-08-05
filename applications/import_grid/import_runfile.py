from datetime import datetime
import string

start_time_dependent_data = "PACKAGES FOR EACH LAYER AND STRESS-PERIOD"

class stress_period:
    def __init__(self, string):
        components = string.split(",")
        self.stress_period_number = int(components[0])
        self.arg2 = float(components[1])
        self.timestamp = datetime.strptime(components[2],"%Y%m%d")
        self.arg4 = int(components[3])
        self.arg5 = int(components[4])
        self.packageData = {}
    @classmethod
    def is_stress_period_header(cls, string):
        if string.count(",") != 4:
            return False
        components = string.split(",")
        try:
            datetime.strptime(components[2], '%Y%m%d')
        except ValueError:
            return False
        return True

    @classmethod
    def read_stress_period(cls, linenr, data):
        result = stress_period(data[linenr])
        linenr = result.readstress_period_data(linenr, data)
        return linenr, result

    def readstress_period_data(self, linenr, data):
        linenr+=1
        while package_data_one_stress_period.is_package_header(data[linenr]):
            package, linenr = self.read_one_package( linenr, data)
            self.packageData[package.name] = package
            linenr+=1
        return linenr

    def read_one_package(self, linenr, data):
        components = data[linenr].split(",")
        number= int(components[0].strip())
        name= components[1].strip().strip("()")
        package = package_data_one_stress_period(number, name)
        linenr = package.read(data, linenr)
        return package, linenr

class package_data_one_stress_period:
    def __init__(self, number, name):
        self.number = number
        self.name = name
        self.files = []

    @classmethod
    def is_package_header(cls, string):
        return string.count(",") == 1 and string.count("(") == 1 and string.count(")") == 1

    def read(self, data, linenr):
        done = False
        while not done:
            for _ in range(0, self.number):
                linenr+=1
                f = bcfile.read_bc_file_indirection(data[linenr])
                self.files.append(f)
            done = self.is_package_header(data[linenr+1]) or stress_period.is_stress_period_header(data[linenr+1]) or data[linenr+1].strip()==""
        return linenr



class bcfile:
    def __init__(self, layer, arg1, arg2, filepath):
        self.layer = layer
        self.arg1 = arg1
        self.arg2 = arg2
        self.filepath = filepath

    @classmethod
    def read_bc_file_indirection(cls, string):
        components = string.split(",")
        layer = int(components[0])
        arg1 = float(components[1])
        arg2 = float(components[2])
        filepath = components[3].strip().strip()
        return bcfile(layer, arg1, arg2, filepath)


def read_time_dependent_data(runfilepath):
    stressperiods = []

    with open(runfilepath, "r") as f:
        data = f.readlines()

        linenr = 0
        while data[linenr].strip() != start_time_dependent_data:
            linenr+=1

        linenr+=1
        stressperiod_number = int(data[1].split()[2])
        for _ in range (0,stressperiod_number):
            linenr, sp  = stress_period.read_stress_period(linenr,data)
            stressperiods.append(sp)

    return stressperiods






