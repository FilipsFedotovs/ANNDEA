#This simple script is design for studying and selecting an optimal size and resolution of the images

########################################    Import libraries    #############################################
import csv
import argparse
import ast
import random

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
#Loading Directory locations
csv_reader=open('../config',"r")
config = list(csv.reader(csv_reader))
for c in config:
    if c[0]=='AFS_DIR':
        AFS_DIR=c[1]
    if c[0]=='EOS_DIR':
        EOS_DIR=c[1]
csv_reader.close()
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import UtilityFunctions as UF

parser = argparse.ArgumentParser(description='This script helps to visualise the seeds by projecting their hit coordinates to the 2-d screen.')
parser.add_argument('--TrainSampleID',help="What training sample to visualise?", default='SHIP_UR_v1')
parser.add_argument('--NewTrainSampleID',help="What training sample to visualise?", default='SHIP_UR_v1')
parser.add_argument('--f',help="Where are the sets located?", default='/eos/user/')
parser.add_argument('--Sets',help="Name of the training sets?", default='[]')
parser.add_argument('--Type',help="Please enter the sample type: VAL or TRAIN", default='VAL')
parser.add_argument('--ObjectType',help="Please enter the sample type: VAL or TRAIN", default='VERTEX')
########################################     Main body functions    #########################################
args = parser.parse_args()
TrainSampleID=args.TrainSampleID
NewSampleID=args.NewTrainSampleID
Sets=ast.literal_eval(args.Sets)
input_location=args.f
Type=args.Type



print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################   Initialising ANNDEA Training Set Merger            #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
CombinedObject=[]
for s in Sets:
    input_file_location=input_location+'/'+s+'.pkl'
    ObjectSet=UF.PickleOperations(input_file_location,'r', 'N/A')[0]

    CombinedObject+=ObjectSet
    print('Combined set size is ',len(CombinedObject))
random.shuffle(CombinedObject)

if Type=='VAL':
    MetaFile=UF.PickleOperations(input_location+'/'+TrainSampleID+'_info.pkl','r', 'N/A')[0]
    output_file_location=input_location+'/'+NewSampleID+'_info.pkl'
    print(UF.PickleOperations(output_file_location,'w', MetaFile)[1])
    output_file_location=input_location+'/'+NewSampleID+'_'+Type+'_'+args.ObjectType+'_SEEDS_OUTPUT.pkl'

else:
    output_file_location=input_location+'/'+NewSampleID+'_'+Type+'_'+args.ObjectType+'_SEEDS_OUTPUT_1.pkl'


print(UF.PickleOperations(output_file_location,'w', CombinedObject)[1])










