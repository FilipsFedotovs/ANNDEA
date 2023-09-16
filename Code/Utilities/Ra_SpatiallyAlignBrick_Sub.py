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
parser.add_argument('--PlateZ',help="The Z coordinate of the starting plate", default='-36820.0')
parser.add_argument('--i',help="Set number", default='1')
parser.add_argument('--j',help="Subset number", default='1')
parser.add_argument('--k',help="Subset number", default='1')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
parser.add_argument('--ValMinHits',help="Maximum allowed longitudinal gap value between segments", default='8000',type=int)
parser.add_argument('--MinHits',help="Maximum allowed transverse gap value between segments per SLG length", default='1000',type=int)
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')
parser.add_argument('--BatchID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--Size',help="A size of the volume of the local alignment", default='20000')
parser.add_argument('--OptBound',help="A bound of the optimisation", default='20000')
parser.add_argument('--PY',help="Python libraries directory location", default='.')

######################################## Set variables  #############################################################
args = parser.parse_args()
i=int(args.i)    #This is just used to name the output file
j=int(args.j)  #The subset helps to determine what portion of the track list is used to create the Seeds
k=int(args.k)
p=args.p
o=args.o
sfx=args.sfx
pfx=args.pfx
Size=float(args.Size)
OptBound=float(args.OptBound)
BatchID=args.BatchID
MinHits=args.MinHits
ValMinHits=args.ValMinHits
########################################     Preset framework parameters    #########################################
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
input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/R_'+BatchID+'_HITS_'+str(j)+'_'+str(k)+'.csv'
output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+BatchID+'_'+str(i)+'/'+pfx+'_'+BatchID+'_RawSeeds_'+str(i)+'_'+str(j)+'_'+str(k)+sfx
print(UF.TimeStamp(), "Modules Have been imported successfully...")
print(UF.TimeStamp(),'Loading pre-selected data from ',input_file_location)
data=pd.read_csv(input_file_location,header=0,
                    usecols=['x','y','z','Rec_Seg_ID','Hit_ID'])[['Rec_Seg_ID','Hit_ID','x','y','z']]
final_rows=len(data)
print(UF.TimeStamp(),'The cleaned data has',final_rows,'hits')
print(UF.TimeStamp(),'Removing tracks which have less than',ValMinHits,'hits...')
track_no_data=data.groupby(['Rec_Seg_ID'],as_index=False).count()
track_no_data=track_no_data.drop(['Hit_ID','y','z'],axis=1)
track_no_data=track_no_data.rename(columns={'x': "Track_No"})
new_combined_data=pd.merge(data, track_no_data, how="left", on=['Rec_Seg_ID'])
new_combined_data=new_combined_data.drop(['Hit_ID'],axis=1)
new_combined_data=new_combined_data.sort_values(['Rec_Seg_ID','z'],ascending=[1,1])
new_combined_data['Plate_ID']=new_combined_data['z'].astype(int)
train_data = new_combined_data[new_combined_data.Track_No >= MinHits]
validation_data = new_combined_data[new_combined_data.Track_No >= ValMinHits]
validation_data = validation_data[validation_data.Track_No < MinHits]
print(train_data)
print(validation_data)

#
# UF.LogOperations(output_file_location,'a',result_list) #Writing the remaining data into the csv
# UF.LogOperations(output_result_location,'w',[])
# print(UF.TimeStamp(), "Reconstruction seed generation is finished...")
# #End of the script



