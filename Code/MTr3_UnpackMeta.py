#import libraries
import argparse
import csv
import os
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#Set the parsing module
parser = argparse.ArgumentParser(description='Enter training job parameters')
parser.add_argument('--ModelName',help="Which model would you like to use as a base for training (please enter N if you want to train a new model from scratch)", default='Default')

args = parser.parse_args()

#setting main learning parameters
ModelName=args.ModelName
#Loading Directory locations
csv_reader=open('../config',"r")
config = list(csv.reader(csv_reader))
for c in config:
    if c[0]=='AFS_DIR':
        AFS_DIR=c[1]
    if c[0]=='EOS_DIR':
        EOS_DIR=c[1]
csv_reader.close()

#Loading Data configurations
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import UtilityFunctions as UF
print(bcolors.HEADER+"####################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################  Initialising ANNDEA model training module      #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################            Written by Filips Fedotovs            #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################               PhD Student at UCL                 #########################"+bcolors.ENDC)
print(bcolors.HEADER+"###################### For troubleshooting please contact filips.fedotovs@cern.ch ##################"+bcolors.ENDC)
print(bcolors.HEADER+"####################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
#This code fragment covers the Algorithm logic on the first run
Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+Model_Meta_Path+bcolors.ENDC)
if os.path.isfile(Model_Meta_Path):
       Model_Meta_Raw=UF.PickleOperations(Model_Meta_Path, 'r', 'N/A')
       print(Model_Meta_Raw[1])
       Model_Meta=Model_Meta_Raw[0]
       Header=Model_Meta.TrainSessionsData[0][0]+['Train Sample ID','LR','Batch Size','Normalised Epochs']
       New_Data=[Header]
       counter=0
       for TSD in range(len(Model_Meta.TrainSessionsData)):
           for Record in Model_Meta.TrainSessionsData[TSD][1:]:
               counter+=1
               New_Data.append(Record+[Model_Meta.TrainSessionsDataID[TSD],Model_Meta.TrainSessionsParameters[TSD][0],Model_Meta.TrainSessionsParameters[TSD][1],counter])

else:
       print(UF.TimeStamp(),bcolors.WARNING+'Fail! No existing meta files have been found, exiting now'+bcolors.ENDC)
Model_Meta_csv=EOSsubModelDIR+'/'+args.ModelName+'_Out.csv'
UF.LogOperations(Model_Meta_csv,'w', New_Data)
print(UF.TimeStamp(),bcolors.OKGREEN+'CSV output has been saved as '+bcolors.ENDC+bcolors.OKBLUE+Model_Meta_csv+bcolors.ENDC)
print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
exit()


