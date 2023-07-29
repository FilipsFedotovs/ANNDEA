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
parser.add_argument('--j',help="Subset number", default='1')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')
parser.add_argument('--BatchID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--MaxSegments',help="A maximum number of track combinations that will be used in a particular HTCondor job for this script", default='20000')
parser.add_argument('--VetoVertex',help="Skip Invalid Mother_IDs", default="[]")
parser.add_argument('--PY',help="Python libraries directory location", default='.')
args = parser.parse_args()


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
import ast

######################################## Set variables  #############################################################

i=int(args.i)    #This is just used to name the output file
p=args.p
o=args.o
sfx=args.sfx
pfx=args.pfx
BatchID=args.BatchID
VetoVertex=ast.literal_eval(args.VetoVertex)
########################################     Preset framework parameters    #########################################
MaxRecords=10000000 #A set parameter that helps to manage memory load of this script (Please do not exceed 10000000)
MaxSegments=int(args.MaxSegments)

#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/EVx1_'+BatchID+'_VERTEX_SEGMENTS.csv'

output_result_location=EOS_DIR+p+'/Temp_'+pfx+'_'+BatchID+'_'+str(0)+'/'+pfx+'_'+BatchID+'_'+o+'_'+str(i)+sfx
output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+BatchID+'_'+str(0)+'/'+pfx+'_'+BatchID+'_RawSeeds_'+str(i)+sfx

print(UF.TimeStamp(), "Modules Have been imported successfully...")
print(UF.TimeStamp(),'Loading pre-selected data from ',input_file_location)
data=pd.read_csv(input_file_location,header=0,
                    usecols=['Rec_Seg_ID','MC_Vertex_ID'])

data.drop_duplicates(subset="Rec_Seg_ID",keep='first',inplace=True)
print(UF.TimeStamp(),'Creating segment combinations... ')

#Doing a plate region cut for the Main Data
Records=len(data)
print(UF.TimeStamp(),'There are total of ', Records, 'tracks in the data set')
Cut=math.ceil(MaxRecords/Records) #Even if use only a max of 20000 track on the right join we cannot perform the full outer join due to the memory limitations, we do it in a small 'cuts'
Steps=math.ceil(MaxSegments/Cut)  #Calculating number of cuts
StartDataCut=i*MaxSegments
EndDataCut=(i+1)*MaxSegments


#What section of data will we cut?


#Specifying the right join

r_data=data.rename(columns={"Rec_Seg_ID": "Segment_2"})
r_data=r_data.iloc[StartDataCut:min(EndDataCut,Records)]
Records=len(r_data)
print(UF.TimeStamp(),'However we will only attempt  ', Records, 'track segments in the starting plate')
data=data.rename(columns={"Rec_Seg_ID": "Segment_1"})

result_list=[]  #We will keep the result in list rather then Panda Dataframe to save memory
#Downcasting Panda Data frame data types in order to save memory

#Creating csv file for the results
UF.LogOperations(output_file_location,'w',result_list)
#This is where we start

for i in range(0,Steps):
  r_temp_data=r_data.iloc[0:min(Cut,len(r_data.axes[0]))] #Taking a small slice of the data
  r_data.drop(r_data.index[0:min(Cut,len(r_data.axes[0]))],inplace=True) #Shrinking the right join dataframe
  merged_data=pd.merge(data, r_temp_data, how="inner", on=['MC_Vertex_ID']) #Merging Tracks to check whether they could form a seed

  if merged_data.empty==False:
    merged_data.drop(merged_data.index[merged_data['Segment_1'] == merged_data['Segment_2']], inplace = True) #Removing the cases where Seed tracks are the same
    merged_data['Seed_Type']=True
    if len(VetoVertex)>=1:
      for n in VetoVertex:
        merged_data['Seed_Type']=((merged_data['MC_Vertex_ID'].str.contains(str('-'+n))==False) & (merged_data['Seed_Type']==True))
    else:
        merged_data['Seed_Type']=True
    merged_data.drop(merged_data.index[merged_data['Seed_Type'] == False], inplace = True)
    merged_data.drop(['MC_Vertex_ID'],axis=1,inplace=True)
    merged_data.drop(['Seed_Type'],axis=1,inplace=True)
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
print(UF.TimeStamp(), "Eval seed generation is finished...")
#End of the script



