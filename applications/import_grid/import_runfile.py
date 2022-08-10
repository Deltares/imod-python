from datetime import datetime
import string

start_time_dependent_data = "PACKAGES FOR EACH LAYER AND STRESS-PERIOD"

class stress_period:
    def __init__(self, string):
        '''
        the string argument should be the header of the stress period in the run-file
        an example of such a header is 14,1.000,20080414,1,1
        The first number is the layer. The meaning of the second, fourth and fifth numbers is not clear;
        hence these are just stored as "arg2", "arg4" and "arg5"
        The third number its the date of the start of the stress period.
        '''
        components = string.split(",")
        self.stress_period_number = int(components[0])
        self.arg2 = float(components[1])
        self.timestamp = datetime.strptime(components[2],"%Y%m%d")
        self.arg4 = int(components[3])
        self.arg5 = int(components[4])
        self.packageData = {}
    @classmethod
    def is_stress_period_header(cls, string):
        '''
        returns true if the input string appears to be the header of a stress period.
        an example of such a header is 14,1.000,20080414,1,1
        '''
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
        '''
        reads the data related to 1 stress period from the run-file.
        "data" is an array where each element is one line from the runfile.
        linenr is the line number of the start of the stress period.
        The function returns the linenumber of the end of this stress period.
        '''

        result = stress_period(data[linenr])
        linenr = result.readstress_period_data(linenr, data)
        return linenr, result

    def readstress_period_data(self, linenr, data):
        '''
        reads the data related to 1 stress period from the run-file.
        "data" is an array where each element is one line from the runfile.
        linenr is the line number of the start of the stress period.
        The function returns the linenumber of the end of this stress period.
        '''
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
    '''
    This class reads and stores the package data for one package in one stress period.
    The package data consists of the paths to idf files used for the package in the current stress period.
    '''
    def __init__(self, number, name):
        self.number = number
        self.name = name
        self.files = []

    @classmethod
    def is_package_header(cls, string):
        '''
        returns true if the input string is the header of package data for 1 package in the runfile,
        in the stress-period block.
        an example of such a header is  "     4,(RIV)"
        The "Riv" means that the packagedata is for a river boundary.
        The 4 means that the number of idf files that describe the rivers is 4 or a multiple of 4
        '''
        return string.count(",") == 1 and string.count("(") == 1 and string.count(")") == 1

    def read(self, data, linenr):
        '''
        reads athe package data for one package in one stress period.
        the line number of the end of the package data is returned.
        '''
        done = False
        while not done:
            for _ in range(0, self.number):
                linenr+=1
                f = bcfile.read_bc_file_indirection(data[linenr])
                self.files.append(f)
            done = self.is_package_header(data[linenr+1]) or stress_period.is_stress_period_header(data[linenr+1]) or data[linenr+1].strip()==""
        return linenr



class bcfile:
    '''
    This class reads and stores the data associated to a file-path specified in a run-file.
    An example of what it reads is the following line:
    1,1.0,0.0,D:\submodel\SUBMODEL_working\CHD\VERSION_1\HEAD_20080416_L1.IDF
    Here, the first integer is the layer number the idf file refers to
    The second and third numbers ar of unknown meaning and are stored as "arg1" and "arg2"
    The string at the end is the path of the idf file that should be used
    '''
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
    '''
    This function reads the time-dependent data block of the runfile.
    It returns the result as a list of stress-periods.
    Each stress period has a start-data and a list of packages.
    Each package has a list of files.

    '''

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






