#This simple script prepares 2-segment track seeds for the initial CNN/GNN union
# Part of ANNDEA package
#Made by Filips Fedotovs
#Current version 1.0

########################################    Import libraries    #############################################
import argparse
import pandas as pd #We use Panda for a routine data processing
import math #We use it for data manipulation
import gc  #Helps to clear memory
import numpy as np




######################################## Set variables  #############################################################
#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--PlateZ',help="The Z coordinate of the starting plate", default='-36820.0')
parser.add_argument('--Set',help="Set number", default='1')
parser.add_argument('--Subset',help="Subset number", default='1')
parser.add_argument('--MaxSLG',help="Maximum allowed longitudinal gap value between segments", default='8000')
parser.add_argument('--MaxSTG',help="Maximum allowed transverse gap value between segments per SLG length", default='1000')
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')
parser.add_argument('--TrainSampleID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--MaxSegments',help="A maximum number of track combinations that will be used in a particular HTCondor job for this script", default='20000')
######################################## Set variables  #############################################################
args = parser.parse_args()
PlateZ=float(args.PlateZ)   #The coordinate of the st plate in the current scope
Set=args.Set    #This is just used to name the output file
Subset=int(args.Subset)  #The subset helps to determine what portion of the track list is used to create the Seeds
MaxSLG=float(args.MaxSLG)
MaxSTG=float(args.MaxSTG)
TrainSampleID=args.TrainSampleID
########################################     Preset framework parameters    #########################################
MaxRecords=10000000 #A set parameter that helps to manage memory load of this script (Please do not exceed 10000000)
MaxSegments=int(args.MaxSegments)
#Loading Directory locations
EOS_DIR=args.EOS
AFS_DIR=args.AFS

import UtilityFunctions as UF #This is where we keep routine utility functions

#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/ANNADEA/Data/TRAIN_SET/MUTr1_'+TrainSampleID+'_TRACK_SEGMENTS.csv'
output_file_location=EOS_DIR+'/ANNADEA/Data/TRAIN_SET/MUTr1a_'+TrainSampleID+'_RawTrackSeeds_'+str(Set)+'_'+str(Subset)+'.csv'
output_result_location=EOS_DIR+'/ANNADEA/Data/TRAIN_SET/MUTr1a_'+TrainSampleID+'_RawTrackSeeds_'+str(Set)+'_'+str(Subset)+'_RES.csv'
print(UF.TimeStamp(), "Modules Have been imported successfully...")
print(UF.TimeStamp(),'Loading pre-selected data from ',input_file_location)
data=pd.read_csv(input_file_location)



print(UF.TimeStamp(),'Creating segment combinations... ')
data_header = data.groupby('Rec_Seg_ID')['z'].min()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
data_header=data_header.reset_index()

data_end_header = data.groupby('Rec_Seg_ID')['z'].max()  #Keeping only ending hits for the each track record (we do not require the full information about track in this script)
data_end_header=data_end_header.reset_index()
data_end_header=data_end_header.rename(columns={"z": "e_z"})
data_header=pd.merge(data_header, data_end_header, how="inner", on=["Rec_Seg_ID"]) #Shrinking the Track data so just a star hit for each track is present.
#Doing a plate region cut for the Main Data
#data_header.drop(data_header.index[data_header['e_z'] > (PlateZ+MaxSLG)], inplace = True) #Not applicable for TSU
data_header.drop(data_header.index[data_header['z'] < PlateZ], inplace = True)
Records=len(data_header.axes[0])
print(UF.TimeStamp(),'There are total of ', Records, 'tracks in the data set')

Cut=math.ceil(MaxRecords/Records) #Even if use only a max of 20000 track on the right join we cannot perform the full outer join due to the memory limitations, we do it in a small 'cuts'
Steps=math.ceil(MaxSegments/Cut)  #Calculating number of cuts
data_s=pd.merge(data, data_header, how="inner", on=["Rec_Seg_ID","z"]) #Shrinking the Track data so just a star hit for each track is present.
data_s.drop(['e_z'],axis=1,inplace=True)
data_e=pd.merge(data, data_header, how="inner", left_on=["Rec_Seg_ID","z"], right_on=["Rec_Seg_ID","e_z"]) #Shrinking the Track data so just a star hit for each track is present.
data_e=data_e.rename(columns={"x": "e_x"})
data_e=data_e.rename(columns={"y": "e_y"})
data_e.drop(['z_x'],axis=1,inplace=True)
data_e.drop(['z_y'],axis=1,inplace=True)
data=pd.merge(data_s, data_e, how="inner", on=["Rec_Seg_ID",'MC_Mother_Track_ID']) #Combining datasets so for each track we know its starting and ending coordinates
del data_e
del data_s
gc.collect()

#What section of data will we cut?
StartDataCut=Subset*MaxSegments
EndDataCut=(Subset+1)*MaxSegments

#Specifying the right join

r_data=data.rename(columns={"Rec_Seg_ID": "Segment_2"})
r_data=r_data.rename(columns={'MC_Mother_Track_ID': "Mother_2"})
r_data.drop(r_data.index[r_data['z'] != PlateZ], inplace = True)

Records=len(r_data.axes[0])
print(UF.TimeStamp(),'There are  ', Records, 'segments in the starting plate')

r_data=r_data.iloc[StartDataCut:min(EndDataCut,Records)]


Records=len(r_data.axes[0])
print(UF.TimeStamp(),'However we will only attempt  ', Records, 'track segments in the starting plate')
r_data.drop(['y'],axis=1,inplace=True)
r_data.drop(['x'],axis=1,inplace=True)
r_data.drop(['z'],axis=1,inplace=True)
data.drop(['e_y'],axis=1,inplace=True)
data.drop(['e_x'],axis=1,inplace=True)
data.drop(['e_z'],axis=1,inplace=True)
data.drop(data.index[data['z'] <= PlateZ], inplace = True)
data=data.rename(columns={"Rec_Seg_ID": "Segment_1"})
data=data.rename(columns={'MC_Mother_Track_ID': "Mother_1"})
data['join_key'] = 'join_key'
r_data['join_key'] = 'join_key'

result_list=[]  #We will keep the result in list rather then Panda Dataframe to save memory

#Downcasting Panda Data frame data types in order to save memory
data["x"] = pd.to_numeric(data["x"],downcast='float')
data["y"] = pd.to_numeric(data["y"],downcast='float')
data["z"] = pd.to_numeric(data["z"],downcast='float')


r_data["e_x"] = pd.to_numeric(r_data["e_x"],downcast='float')
r_data["e_y"] = pd.to_numeric(r_data["e_y"],downcast='float')
r_data["e_z"] = pd.to_numeric(r_data["e_z"],downcast='float')

#Cleaning memory
del data_header
gc.collect()

#Creating csv file for the results
UF.LogOperations(output_file_location,'w',result_list)
#This is where we start

for i in range(0,Steps):
  r_temp_data=r_data.iloc[0:min(Cut,len(r_data.axes[0]))] #Taking a small slice of the data
  r_data.drop(r_data.index[0:min(Cut,len(r_data.axes[0]))],inplace=True) #Shrinking the right join dataframe
  merged_data=pd.merge(data, r_temp_data, how="inner", on=['join_key']) #Merging Tracks to check whether they could form a seed
  merged_data['SLG']=merged_data['z']-merged_data['e_z'] #Calculating the Euclidean distance between Track start hits
  merged_data['STG']=np.sqrt((merged_data['x']-merged_data['e_x'])**2+((merged_data['y']-merged_data['e_y'])**2)) #Calculating the Euclidean distance between Track start hits
  merged_data['DynamicCut']=MaxSTG+(merged_data['SLG']*0.96)
  merged_data.drop(merged_data.index[merged_data['SLG'] < 0], inplace = True) #Dropping the Seeds that are too far apart
  merged_data.drop(merged_data.index[merged_data['SLG'] > MaxSLG], inplace = True) #Dropping the track segment combinations where the length of the gap between segments is too large
  merged_data.drop(merged_data.index[merged_data['STG'] > merged_data['DynamicCut']], inplace = True)
  print(merged_data)
  exit()
  merged_data.drop(['y','z','x','e_x','e_y','e_z','join_key','STG','SLG','DynamicCut'],axis=1,inplace=True) #Removing the information that we don't need anymore
  if merged_data.empty==False:
    merged_data.drop(merged_data.index[merged_data['Segment_1'] == merged_data['Segment_2']], inplace = True) #Removing the cases where Seed tracks are the same
    merged_data['Track_Type']=(merged_data['Mother_1']==merged_data['Mother_2'])
    merged_data.drop(['Mother_1'],axis=1,inplace=True)
    merged_data.drop(['Mother_2'],axis=1,inplace=True)
    merged_list = merged_data.values.tolist() #Convirting the result to List data type
    result_list+=merged_list #Adding the result to the list
  if len(result_list)>=2000000: #Once the list gets too big we dump the results into csv to save memory
      UF.LogOperations(output_file_location,'a',result_list) #Write to the csv
      #Clearing the memory
      del result_list
      result_list=[]
      gc.collect()

UF.LogOperations(output_file_location,'a',result_list) #Writing the remaining data into the csv
UF.LogOperations(output_result_location,'w',[])
print(UF.TimeStamp(), "Train seed generation is finished...")
#End of the script



