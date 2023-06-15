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
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
parser.add_argument('--MaxDST',help="Maximum allowed longitudinal gap value between segments", default='8000')
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')
parser.add_argument('--BatchID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--MaxSegments',help="A maximum number of track combinations that will be used in a particular HTCondor job for this script", default='20000')
parser.add_argument('--VetoVertex',help="Skip Invalid Mother_IDs", default="[]")
parser.add_argument('--PY',help="Python libraries directory location", default='.')

######################################## Set variables  #############################################################
args = parser.parse_args()
PlateZ=float(args.PlateZ)   #The coordinate of the st plate in the current scope
i=int(args.i)    #This is just used to name the output file
j=int(args.j)  #The subset helps to determine what portion of the track list is used to create the Seeds
p=args.p
o=args.o
sfx=args.sfx
pfx=args.pfx
MaxDST=float(args.MaxDST)
BatchID=args.BatchID

########################################     Preset framework parameters    #########################################
MaxRecords=10000000 #A set parameter that helps to manage memory load of this script (Please do not exceed 10000000)
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
import ast
VetoVertex=ast.literal_eval(args.VetoVertex)
#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MVx1_'+BatchID+'_TRACK_SEGMENTS.csv'
output_file_location=EOS_DIR+'/'+p+'/Temp_'+pfx+'_'+BatchID+'_'+str(i)+'/'+pfx+'_'+BatchID+'_RawSeeds_'+str(i)+'_'+str(j)+sfx
output_result_location=EOS_DIR+'/'+p+'/Temp_'+pfx+'_'+BatchID+'_'+str(i)+'/'+pfx+'_'+BatchID+'_'+o+'_'+str(i)+'_'+str(j)+sfx
print(UF.TimeStamp(), "Modules Have been imported successfully...")
print(UF.TimeStamp(),'Loading pre-selected data from ',input_file_location)
data=pd.read_csv(input_file_location,header=0,
                    usecols=['x','y','z','Rec_Seg_ID','MC_VX_ID'])

print(UF.TimeStamp(),'Creating segment combinations... ')
data_header = data.groupby('Rec_Seg_ID')['z'].min()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
data_header=data_header.reset_index()

#Doing a plate region cut for the Main Data
data_header.drop(data_header.index[data_header['z'] > (PlateZ+MaxDST)], inplace = True) #Not applicable for TSU
data_header.drop(data_header.index[data_header['z'] < PlateZ], inplace = True)

Records=len(data_header.axes[0])
print(UF.TimeStamp(),'There are total of ', Records, 'tracks in the data set')

Cut=math.ceil(MaxRecords/Records) #Even if use only a max of 20000 track on the right join we cannot perform the full outer join due to the memory limitations, we do it in a small 'cuts'
Steps=math.ceil(MaxSegments/Cut)  #Calculating number of cuts
data=pd.merge(data, data_header, how="inner", on=["Rec_Seg_ID","z"]) #Shrinking the Track data so just a star hit for each track is present.

#What section of data will we cut?
StartDataCut=j*MaxSegments
EndDataCut=(j+1)*MaxSegments

#Specifying the right join

r_data=data.rename(columns={"x": "r_x"})
r_data=r_data.rename(columns={'MC_VX_ID': "Mother_2"})
r_data.drop(r_data.index[r_data['z'] != PlateZ], inplace = True)

Records=len(r_data.axes[0])
print(UF.TimeStamp(),'There are  ', Records, 'segments in the starting plate')

r_data=r_data.iloc[StartDataCut:min(EndDataCut,Records)]

Records=len(r_data.axes[0])
print(UF.TimeStamp(),'However we will only attempt  ', Records, 'track segments in the starting plate')
r_data=r_data.rename(columns={"y": "r_y"})
r_data=r_data.rename(columns={"z": "r_z"})
data=data.rename(columns={"Rec_Seg_ID": "Track_1"})
data=data.rename(columns={'MC_VX_ID': "Mother_1"})
r_data=r_data.rename(columns={"Rec_Seg_ID": "Track_2"})
r_data=r_data.rename(columns={'MC_VX_ID': "Mother_2"})
data['join_key'] = 'join_key'
r_data['join_key'] = 'join_key'

result_list=[]  #We will keep the result in list rather then Panda Dataframe to save memory

#Downcasting Panda Data frame data types in order to save memory
data["x"] = pd.to_numeric(data["x"],downcast='float')
data["y"] = pd.to_numeric(data["y"],downcast='float')
data["z"] = pd.to_numeric(data["z"],downcast='float')


r_data["r_x"] = pd.to_numeric(r_data["r_x"],downcast='float')
r_data["r_y"] = pd.to_numeric(r_data["r_y"],downcast='float')
r_data["r_z"] = pd.to_numeric(r_data["r_z"],downcast='float')

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
  merged_data['separation']=np.sqrt(((merged_data['x']-merged_data['r_x'])**2)+((merged_data['y']-merged_data['r_y'])**2)+((merged_data['z']-merged_data['r_z'])**2)) #Calculating the Euclidean distance between Track start hits
  merged_data.drop(merged_data.index[merged_data['separation'] >= MaxDST], inplace = True) #Dropping the track segment combinations where the length of the gap between segments is too large
  merged_data.drop(['y','z','x','r_x','r_y','r_z','join_key','separation'],axis=1,inplace=True) #Removing the information that we don't need anymore
  if merged_data.empty==False:
    merged_data.drop(merged_data.index[merged_data['Track_1'] == merged_data['Track_2']], inplace = True) #Removing the cases where Seed tracks are the same
    merged_data['Seed_Type']=True
    if len(VetoVertex)>=1:
      for n in VetoVertex:
        merged_data['Seed_Type']=((merged_data['Mother_1']==merged_data['Mother_2']) & (merged_data['Mother_1'].str.contains(str('-'+n))==False) & (merged_data['Seed_Type']==True))
    else:
        merged_data['Seed_Type']=(merged_data['Mother_1']==merged_data['Mother_2'])
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



