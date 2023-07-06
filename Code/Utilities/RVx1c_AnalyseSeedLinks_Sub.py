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
parser.add_argument('--MaxSegments',help="A maximum number of track combinations that will be used in a particular HTCondor job for this script", default='20000')
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
MaxSegments=int(args.MaxSegments)
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
input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RVx1c_'+BatchID+'_Fit_Seeds.pkl'
output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+BatchID+'_'+str(0)+'/'+pfx+'_'+BatchID+'_'+o+'_'+str(i)+sfx
print(UF.TimeStamp(), "Modules Have been imported successfully...")
print(UF.TimeStamp(),'Loading pre-selected data from ',input_file_location)
base_data=UF.PickleOperations(input_file_location,'r','N/A')[0]
base_data_list=[]
for b in base_data:
   base_data_mini_list = [b.Header[0],b.Header[1],b.Fit]
   base_data_list.append(base_data_mini_list)
data = pd.DataFrame(base_data_list,columns=['Track_1','Track_2','Seed_CNN_Fit'])

SeedStart=i*MaxSegments
SeedEnd=min(len(data),(i+1)*MaxSegments)
seeds=data.loc[SeedStart:SeedEnd-1]
seeds_1=seeds.drop(['Track_2','Seed_CNN_Fit'],axis=1)
seeds_1=seeds_1.rename(columns={"Track_1": "Track_ID"})
seeds_2=seeds.drop(['Track_1','Seed_CNN_Fit'],axis=1)
seeds_2=seeds_2.rename(columns={"Track_2": "Track_ID"})
seed_list=result = pd.concat([seeds_1,seeds_2])
seed_list=seed_list.sort_values(['Track_ID'])
seed_list.drop_duplicates(subset="Track_ID",keep='first',inplace=True)
data_l=pd.merge(seed_list,data , how="inner", left_on=["Track_ID"], right_on=["Track_1"] )
data_l=data_l.drop(['Track_ID'],axis=1)
data_r=pd.merge(seed_list, data , how="inner", left_on=["Track_ID"], right_on=["Track_2"] )
data_r=data_r.drop(['Track_ID'],axis=1)
data=pd.concat([data_l,data_r])
data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(data['Track_1'], data['Track_2'])]
data.drop_duplicates(subset="Seed_ID",keep='first',inplace=True)
data.drop(["Seed_ID"],axis=1,inplace=True)
del seeds_1
del seeds_2
del seed_list
del data_l
del data_r
data=data.values.tolist()
seeds=seeds.values.tolist()
for rows in seeds:
    for i in range(4):
       rows.append([])
for seed in seeds:
        for dt in data:
           if (seed[0]==dt[0] and seed[1]!=dt[1]):
              seed[3].append(dt[1])
              seed[5].append(dt[2])
           elif (seed[0]==dt[1] and seed[1]!=dt[0]):
              seed[3].append(dt[0])
              seed[5].append(dt[2])
           if ((seed[1]==dt[0]) and seed[0]!=dt[1]):
              seed[4].append(dt[1])
              seed[6].append(dt[2])
           elif (seed[1]==dt[1] and seed[0]!=dt[0]):
              seed[4].append(dt[0])
              seed[6].append(dt[2])
        CommonSets = list(set(seed[3]).intersection(seed[4]))
        LinkStrength=0.0
        for CS in CommonSets:
            Lindex=seed[3].index(CS)
            Rindex=seed[4].index(CS)
            LinkStrength+=seed[5][Lindex]
            del seed[5][Lindex]
            del seed[3][Lindex]
            del seed[6][Rindex]
            del seed[4][Rindex]
        UnlinkStrength=sum(seed[6])+sum(seed[5])
        seed.append(CommonSets)
        CommonSetsNo= len(CommonSets)
        OrthogonalSets=(len(seed[3])+len(seed[4]))
        del seed[3:8]
        seed.append(CommonSetsNo)
        seed.append(OrthogonalSets)
        seed.append(LinkStrength)
        seed.append(UnlinkStrength)
Header=[['Track_1','Track_2','Seed_CNN_Fit', 'Links', 'AntiLinks', 'Link_Strength', 'AntiLink_Strenth']]
UF.LogOperations(output_file_location,'w', Header)
UF.LogOperations(output_file_location,'a', seeds)

print(UF.TimeStamp(), "Seed link analysis is finished...")