

class bcolors:   #We use it for the interface
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header():
    print('                                                                                                                                    ')
    print('                                                                                                                                    ')
    print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
    print(bcolors.HEADER+"######################     Initialising ANNADEA Train Cluster Generation module   #####################"+bcolors.ENDC)
    print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
    print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
    print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)

def print_message(message):
    print(UF.TimeStamp(), message)


