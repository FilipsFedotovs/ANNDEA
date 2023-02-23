#This script connects hits in the data to produce tracks
#Tracking Module of the ANNDEA package
#Made by Filips Fedotovs

########################################    Import libraries    #############################################
import csv
csv_reader=open('../config',"r")
config = list(csv.reader(csv_reader))
for c in config:
    if c[0]=='AFS_DIR':
        AFS_DIR=c[1]
    if c[0]=='EOS_DIR':
        EOS_DIR=c[1]
    if c[0]=='PY_DIR':
        PY_DIR=c[1]
csv_reader.close()
import sys
if PY_DIR!='': #Temp solution - the decision was made to move all libraries to EOS drive as AFS get locked during heavy HTCondor submission loads
    sys.path=['',PY_DIR]
    sys.path.append('/usr/lib64/python36.zip')
    sys.path.append('/usr/lib64/python3.6')
    sys.path.append('/usr/lib64/python3.6/lib-dynload')
    sys.path.append('/usr/lib64/python3.6/site-packages')
    sys.path.append('/usr/lib/python3.6/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')
import pandas as pd #We use Panda for a routine data processing
import argparse
class bcolors:   #We use it for the interface
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print('                                                                                                                                    ')
print('                                                                                                                                    ')
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################        Initialising ANNDEA Hit Tracking module              #####################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script prepares training data for training the tracking model')
parser.add_argument('--RecBatchID',help="Give this reconstruction batch an ID", default='Test_Batch')

######################################## Parsing argument values  #############################################################
args = parser.parse_args()
RecBatchID=args.RecBatchID


sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import UtilityFunctions as UF #This is where we keep routine utility functions
import Parameters as PM #This is where we keep framework global parameters



#Non standard processes (that don't follow the general pattern) have been coded here
print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' Using the results from previous steps to map merged trackIDs to the original reconstruction file')
input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_RTr_OUTPUT.csv'
print(UF.TimeStamp(),'Loading the file ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
Data=pd.read_csv(input_file_location,header=0)
New_Data=Data[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z,PM.Hit_ID]]
Data.drop([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID'],axis=1,inplace=True)
New_Data=New_Data.dropna()
New_Data=New_Data.sort_values([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z],ascending=[1,1,1])
New_Data[RecBatchID+'_Brick_ID'] = New_Data[RecBatchID+'_Brick_ID'].astype(str)
New_Data[RecBatchID+'_Track_ID'] = New_Data[RecBatchID+'_Track_ID'].astype(str)
New_Data['Rec_Seg_ID'] = New_Data[RecBatchID+'_Brick_ID'] + '-' + New_Data[RecBatchID+'_Track_ID']
New_Data.drop_duplicates(subset=[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z],keep='first',inplace=True)
compress_data=New_Data.drop([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z],axis=1)
compress_data['Hit_No']= compress_data[PM.Hit_ID]
compress_data=compress_data.groupby(by=['Rec_Seg_ID'])['Hit_No'].count().reset_index()
New_Data=pd.merge(New_Data, compress_data, how="left", on=['Rec_Seg_ID'])
New_Data = New_Data[New_Data.Hit_No >= PM.MinHitsTrack]
New_Data.drop([PM.z,'Rec_Seg_ID'],axis=1,inplace=True)
Data=pd.merge(Data,New_Data,how='left', on=[PM.Hit_ID])
output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_RTr_OUTPUT_CLEAN.csv'
Data.to_csv(output_file_location,index=False)
print(UF.TimeStamp(), bcolors.OKGREEN+"The cleaned tracked data has been written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)




