#This simple script prepares 2-segment track seeds for the initial CNN/GNN union
# Part of ANNDEA package
#Made by Filips Fedotovs
#Current version 1.0

########################################    Import libraries    #############################################
import argparse
import sys




######################################## Set variables  #############################################################
#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--i',help="Set number", default='1')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')
parser.add_argument('--BatchID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--MaxPoolSeeds',help="How many seeds?", default='20000')
parser.add_argument('--PY',help="Python libraries directory location", default='.')

######################################## Set variables  #############################################################
args = parser.parse_args()
i=int(args.i)    #This is just used to name the output file
p=args.p
o=args.o
sfx=args.sfx
pfx=args.pfx
BatchID=args.BatchID
########################################     Preset framework parameters    #########################################
MaxPoolSeeds=int(args.MaxPoolSeeds)
#Loading Directory locations
EOS_DIR=args.EOS
AFS_DIR=args.AFS
PY_DIR=args.PY
if PY_DIR!='': #Temp solution
    sys.path=['',PY_DIR]
    sys.path.append('/usr/lib64/python36.zip')
    sys.path.append('/usr/lib64/python3.6')
    sys.path.append('/usr/lib64/python3.6/lib-dynload')
    sys.path.append('/usr/lib64/python3.6/site-packages')
    sys.path.append('/usr/lib/python3.6/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')
import UtilityFunctions as UF #This is where we keep routine utility functions
import pandas as pd #We use Panda for a routine data processing
import math #We use it for data manipulation
import gc  #Helps to clear memory
import numpy as np
#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RVx1c_'+BatchID+'_Link_Fit_Seeds.pkl'
output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+BatchID+'_'+str(0)+'/'+pfx+'_'+BatchID+'_'+o+'_'+str(i)+sfx

base_data=UF.PickleOperations(input_file_location,'r','N/A')[0]

print(UF.TimeStamp(), "Loading is successful, there are total of "+str(len(base_data))+" vertexed seeds...")
base_data=base_data[(i*MaxPoolSeeds):min(((i+1)*MaxPoolSeeds),len(base_data))]
print(UF.TimeStamp(), "Out of these only "+str(len(base_data))+" vertexed seeds will be considered here...")
print(UF.TimeStamp(), "Initiating the seed merging...")
InitialDataLength=len(base_data)
SeedCounter=0
SeedCounterContinue=True
while SeedCounterContinue:
    if SeedCounter>=len(base_data):
       SeedCounterContinue=False
       break
    progress=round(float(SeedCounter)/float(len(base_data))*100,0)
    print(UF.TimeStamp(),'progress is ',progress,' %', end="\r", flush=True) #Progress display
    SubjectSeed=base_data[SeedCounter]
    for ObjectSeed in base_data[SeedCounter+1:]:
        if SubjectSeed.InjectSeed(ObjectSeed):
           base_data.pop(base_data.index(ObjectSeed))
    SeedCounter+=1
print(str(InitialDataLength), "2-track vertices were merged into", str(len(base_data)), 'vertices with higher multiplicity...')
print(UF.PickleOperations(output_file_location,'w', base_data)[1])
print(UF.TimeStamp(), "Saving the results into the file",output_file_location)
