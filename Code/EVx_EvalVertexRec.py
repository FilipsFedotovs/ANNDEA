# Part of <Work in progress package> package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
csv_reader=open('../config',"r")
config = list(csv.reader(csv_reader))
#Loading Directory locations
for c in config:
    if c[0]=='AFS_DIR':
        AFS_DIR=c[1]
    if c[0]=='EOS_DIR':
        EOS_DIR=c[1]
    if c[0]=='PY_DIR':
        PY_DIR=c[1]
csv_reader.close()
import sys
# if PY_DIR!='': #Temp solution - the decision was made to move all libraries to EOS drive as AFS get locked during heavy HTCondor submission loads
#     sys.path=['',PY_DIR]
#     sys.path.append('/usr/lib64/python36.zip')
#     sys.path.append('/usr/lib64/python3.6')
#     sys.path.append('/usr/lib64/python3.6/lib-dynload')
#     sys.path.append('/usr/lib64/python3.6/site-packages')
#     sys.path.append('/usr/lib/python3.6/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')
import argparse
import ast
import pandas as pd #We use Panda for a routine data processing
pd.options.mode.chained_assignment = None #Disable annoying Pandas warnings
import os

class bcolors:   #We use it for the interface (Colour fonts and so on)
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

import UtilityFunctions as UF #This is where we keep routine utility functions
import Parameters as PM #This is where we keep framework global parameters

#Setting the parser
parser = argparse.ArgumentParser(description='This script compares the ouput of the previous step with the output of ANNDEA reconstructed data to calculate reconstruction performance.')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_FEDRA_Raw_UR.csv')
parser.add_argument('--TrackID',help="Name of the control track", default="['Brick_ID','FEDRA_Track_ID']")
parser.add_argument('--MCCategories',help="What MC categories present in the MC data would you like to split by?", default="[]")
parser.add_argument('--RecNames',help="What Names would you like to assign to the reconstruction methods that generated the tracks?", default="['FEDRA']")
parser.add_argument('--Xmin',help="This option restricts data to only those events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option restricts data to only those events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option restricts data to only those events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option restricts data to only those events that have tracks with hits y-coordinates that are below this value", default='0')
parser.add_argument('--MinHitsTrack',help="What is the minimum number of hits per track?", default=PM.MinHitsTrack)
parser.add_argument('--RemoveTracksZ',help="This option enables to remove particular tracks of starting Z-coordinate", default='[]')
parser.add_argument('--VertexID',help="", default='[]')
args = parser.parse_args()

######################################## Welcome message  #############################################################
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################     Initialising Tracking  Quality     Evaluation module ########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(),bcolors.BOLD+'Stage 1:'+bcolors.ENDC+' Preparing the input data...')
######################################## Set variables  #############################################################
TrackID=ast.literal_eval(args.TrackID)
VertexID=ast.literal_eval(args.VertexID)
MCCategories=ast.literal_eval(args.MCCategories)
RecNames=ast.literal_eval(args.RecNames)
input_file_location=args.f
MinHitsTrack=int(args.MinHitsTrack)
RemoveTracksZ=ast.literal_eval(args.RemoveTracksZ)

Xmin,Xmax,Ymin,Ymax=float(args.Xmin),float(args.Xmax),float(args.Ymin),float(args.Ymax)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0 #We don't slice data if all values are set to zero simultaneousy (which is the default setting)
ofn=(args.f[(args.f.rfind('/'))+1:-4])

print(UF.TimeStamp(), 'Loading ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
columns_to_extract=[PM.x,PM.y,PM.z,PM.Hit_ID,PM.MC_Event_ID,PM.MC_Track_ID,PM.MC_VX_ID,TrackID[0],TrackID[1]]
for col in VertexID:
    columns_to_extract+=col
columns_to_extract+=MCCategories
columns_to_extract = list(dict.fromkeys(columns_to_extract))
if os.path.isfile(input_file_location)!=True:
                     print(UF.TimeStamp(), bcolors.FAIL+"Critical fail: file",input_file_location,'is missing, please check that the name/path is correct...'+bcolors.ENDC)
                     exit()

raw_data=pd.read_csv(input_file_location,header=0,usecols=columns_to_extract)[columns_to_extract]
#raw_data=raw_data.drop(raw_data.index[(raw_data['MC_Event_ID'] != '31-102000')])

total_rows=len(raw_data.axes[0])
print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')

if SliceData:
           print(UF.TimeStamp(),'Slicing the data...')
           raw_data=raw_data.drop(raw_data.index[(raw_data[PM.x] > Xmax) | (raw_data[PM.x] < Xmin) | (raw_data[PM.y] > Ymax) | (raw_data[PM.y] < Ymin)])
           final_rows=len(raw_data.axes[0])
           print(UF.TimeStamp(),'The sliced raw data has ',final_rows,' hits')
raw_data.drop([PM.x,PM.y],axis=1,inplace=True)

raw_data[PM.MC_Event_ID] = raw_data[PM.MC_Event_ID].astype(str)
raw_data[PM.MC_VX_ID] = raw_data[PM.MC_VX_ID].astype(str)
raw_data[PM.MC_Track_ID] = raw_data[PM.MC_Track_ID].astype(str)
raw_data[TrackID[0]] = raw_data[TrackID[0]].astype(str)
raw_data[TrackID[1]] = raw_data[TrackID[1]].astype(str)
raw_data[PM.Hit_ID] = raw_data[PM.Hit_ID].astype(str)
raw_data['MC_Mother_Track_ID'] =  raw_data[PM.MC_Event_ID] + '-' + raw_data[PM.MC_Track_ID]
raw_data['MC_Vx'] = raw_data[PM.MC_Event_ID] + '-' + raw_data[PM.MC_VX_ID]
raw_data['Track_ID'] = raw_data[TrackID[0]] + '-' + raw_data[TrackID[1]]

#raw_data.drop([PM.MC_Event_ID,PM.MC_Track_ID,PM.MC_VX_ID,],axis=1,inplace=True)
for rn in range(len(RecNames)):
    raw_data[VertexID[rn][0]] = raw_data[VertexID[rn][0]].astype(str)
    raw_data[VertexID[rn][1]] = raw_data[VertexID[rn][1]].astype(str)
    raw_data[RecNames[rn]] = raw_data[VertexID[rn][0]] + '-' + raw_data[VertexID[rn][1]]
    raw_data.drop([VertexID[rn][0],VertexID[rn][1]],axis=1,inplace=True)
    if len(RemoveTracksZ)>0:
            print(UF.TimeStamp(),'Removing tracks based on start point')
            TracksZdf = pd.DataFrame(RemoveTracksZ, columns = ['Bad_z'], dtype=float)
            data_aggregated=raw_data.groupby(['Track_ID'])[PM.z].min().reset_index()
            data_aggregated=data_aggregated.rename(columns={PM.z: "PosBad_Z"})
            raw_data=pd.merge(raw_data, data_aggregated, how="left", on=['Track_ID'])
            raw_data=pd.merge(raw_data, TracksZdf, how="left", left_on=["PosBad_Z"], right_on=['Bad_z'])
            raw_data=raw_data[raw_data['Bad_z'].isnull()]
            raw_data=raw_data.drop(['Bad_z', 'PosBad_Z'],axis=1)


print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' Analyzing track reconstruction metrics...')
raw_data_mc=raw_data[['MC_Vx','MC_Mother_Track_ID',PM.Hit_ID]+MCCategories].groupby(by=['MC_Vx','MC_Mother_Track_ID']+MCCategories)[PM.Hit_ID].nunique().reset_index()

raw_data_mc.drop(raw_data_mc.index[(raw_data_mc[PM.Hit_ID] < MinHitsTrack)],inplace=True)
raw_data_mc_save=pd.merge(raw_data,raw_data_mc[['MC_Mother_Track_ID']],how='inner', on =['MC_Mother_Track_ID'])


raw_data_mc=raw_data_mc.groupby(by=['MC_Vx']+MCCategories)['MC_Mother_Track_ID'].nunique().reset_index()
raw_data_mc.drop(raw_data_mc.index[(raw_data_mc['MC_Mother_Track_ID'] < 2)],inplace=True)

for n in PM.VetoVertex:
     raw_data_mc.drop(raw_data_mc.index[(raw_data_mc['MC_Vx'].str.contains(str('-'+n)))],inplace=True)

mc_data_tot=raw_data_mc['MC_Vx'].nunique()
print(UF.TimeStamp(),'Total number of MC vertices is:',mc_data_tot)
raw_data_mc.drop(['MC_Mother_Track_ID'],axis=1,inplace=True)
data_mc=pd.merge(raw_data_mc_save[['MC_Vx','MC_Mother_Track_ID',PM.Hit_ID]],raw_data_mc,how='inner', on =['MC_Vx'])

data_mc_light=data_mc.drop_duplicates(subset=['MC_Mother_Track_ID','MC_Vx']+MCCategories,keep='first')[['MC_Mother_Track_ID','MC_Vx']+MCCategories]
data_mc_left=data_mc_light[['MC_Mother_Track_ID','MC_Vx']].rename(columns={'MC_Mother_Track_ID': 'MC_Mother_Track_ID_l'})
data_mc_tot_merged=pd.merge(data_mc_light,data_mc_left, how='inner', on=['MC_Vx'])

data_mc_tot_merged.drop(data_mc_tot_merged.index[data_mc_tot_merged['MC_Mother_Track_ID'] == data_mc_tot_merged['MC_Mother_Track_ID_l']], inplace = True)
data_mc_tot_merged["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(data_mc_tot_merged['MC_Mother_Track_ID'], data_mc_tot_merged['MC_Mother_Track_ID_l'])]
data_mc_tot_merged=data_mc_tot_merged.drop_duplicates(subset=["Seed_ID"],keep='first')
data_mc_tot_merged.drop(['MC_Mother_Track_ID','MC_Mother_Track_ID_l'],axis=1,inplace=True)
print(UF.TimeStamp(),'Total number of unique MC vertex tracks combinations is:',len(data_mc_tot_merged))
data_mc_light_no=data_mc_light[['MC_Vx','MC_Mother_Track_ID']].rename(columns={'MC_Mother_Track_ID':'MC_Mother_Track_No'})
data_mc_light_no=data_mc_light_no.groupby(by=['MC_Vx'])['MC_Mother_Track_No'].nunique().reset_index()
data_mc_light_no.drop(data_mc_light_no.index[(data_mc_light_no['MC_Mother_Track_No'] < 2)],inplace=True)
print(UF.TimeStamp(),'Total number of MC genuine reconstructed vertices is:',len(data_mc_light_no))
data_mc_light=pd.merge(data_mc_light,data_mc_light_no,how='inner', on=['MC_Vx'])
data_mc_distribution=data_mc_light
data_mc_distribution['Analysis']='Analysis'
data_mc_distribution=data_mc_distribution.groupby(by=['Analysis']+MCCategories)['MC_Vx'].nunique().reset_index()

for RN in RecNames:
  #raw_data_rec=raw_data.drop(raw_data.index[(raw_data[RN] == 'nan-nan')])
  raw_data_rec = raw_data[raw_data[RN].str.contains("nan") == False]
  rec_data_rec=raw_data_rec[raw_data_rec[RN].str.contains("--")==False]
  rec_data_tot=raw_data_rec[RN].nunique()
  raw_data_rec=raw_data_rec[['Track_ID',PM.Hit_ID,RN]]
  raw_data_temp_rec=raw_data_rec[['Track_ID',PM.Hit_ID,RN]].rename(columns={PM.Hit_ID: 'Track_ID_Size'})
  raw_data_temp_rec=raw_data_temp_rec.groupby(by=['Track_ID'])['Track_ID_Size'].nunique().reset_index()
  raw_data_temp_rec.drop(raw_data_temp_rec.index[(raw_data_temp_rec['Track_ID_Size'] < MinHitsTrack)],inplace=True)

  data_rec=pd.merge(raw_data_rec[['Track_ID',PM.Hit_ID,RN]],raw_data_temp_rec,how='inner', on =['Track_ID'])
  data_rec=data_rec[data_rec[RN].str.contains("--")==False]
  data_rec=pd.merge(data_rec,data_mc,how='inner', on =[PM.Hit_ID])
  data_rec=data_rec.rename(columns={PM.Hit_ID: 'Track_ID_Overlap'})
  data_rec=data_rec.groupby(by=['Track_ID','Track_ID_Size','MC_Mother_Track_ID',RN])['Track_ID_Overlap'].nunique().reset_index()
  data_rec.drop(data_rec.index[(data_rec['Track_ID_Overlap'] < 2)],inplace=True)
  data_rec.sort_values(by=[RN,'Track_ID','Track_ID_Overlap'], ascending=[1,1,0],inplace=True)
  data_rec.drop_duplicates(subset=[RN,'Track_ID'],keep='first',inplace=True)
  data_rec.drop(['Track_ID_Overlap', 'Track_ID_Size', 'Track_ID'],axis=1, inplace=True)
  data_rec.rename(columns={'MC_Mother_Track_ID': RN+'_Matched_Track_ID'},inplace=True)
  data_rec=data_rec.drop_duplicates(subset=[RN+'_Matched_Track_ID',RN],keep='first')

  #The Track Matching is complete


  data_rec_left=data_rec.rename(columns={RN+'_Matched_Track_ID': RN+'_Matched_Track_ID_l'})

  data_rec_tot_merged=pd.merge(data_rec,data_rec_left, how='inner', on=[RN])
  data_rec_tot_merged.drop(data_rec_tot_merged.index[data_rec_tot_merged[RN+'_Matched_Track_ID'] == data_rec_tot_merged[RN+'_Matched_Track_ID_l']], inplace = True)
  data_rec_tot_merged["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(data_rec_tot_merged[RN+'_Matched_Track_ID'], data_rec_tot_merged[RN+'_Matched_Track_ID_l'])]
  data_rec_tot_merged=data_rec_tot_merged.drop_duplicates(subset=["Seed_ID"],keep='first')

  data_rec_tot_merged.drop([RN+'_Matched_Track_ID',RN+'_Matched_Track_ID_l'],axis=1,inplace=True)
  print(UF.TimeStamp(),'Total number of unique '+RN+' vertex tracks combinations is:',len(data_rec_tot_merged))
  data_rec_mc_merged=pd.merge(data_mc_tot_merged,data_rec_tot_merged, how='inner', on=['Seed_ID'])
  print(UF.TimeStamp(),'Total number of unique matched vertex tracks combinations is:',len(data_rec_mc_merged))
  Recall=len(data_rec_mc_merged)/len(data_mc_tot_merged)
  Precision=len(data_rec_mc_merged)/len(data_rec_tot_merged)
  print(UF.TimeStamp(),'Overall efficiency:',bcolors.BOLD+str(round(Recall,2)*100), '%'+bcolors.ENDC)
  print(UF.TimeStamp(),'Overall Purity:',bcolors.BOLD+str(round(Precision,2)*100), '%'+bcolors.ENDC)



  data_rec_light_no=data_rec.rename(columns={RN+'_Matched_Track_ID':RN+'_Matched_Track_No'})
  data_rec_light_no=data_rec_light_no.groupby(by=[RN])[RN+'_Matched_Track_No'].nunique().reset_index()
  data_rec_light_no.drop(data_rec_light_no.index[(data_rec_light_no[RN+'_Matched_Track_No'] < 2)],inplace=True)
  print(UF.TimeStamp(),'Total number of '+RN+' genuine reconstructed vertices is:',len(data_rec_light_no))
  data_rec_light=pd.merge(data_rec,data_rec_light_no,how='inner', on=[RN])
  data_rec_light=data_rec_light.rename(columns={RN+'_Matched_Track_ID':'MC_Mother_Track_ID'})


  data_mc_rec_combined=pd.merge(data_mc_light,data_rec_light,how='left',on=['MC_Mother_Track_ID'])
  data_mc_rec_combined=data_mc_rec_combined.groupby(by=['MC_Vx',RN,'MC_Mother_Track_No',RN+'_Matched_Track_No']+MCCategories)['MC_Mother_Track_ID'].nunique().reset_index()
  data_mc_rec_combined.drop(data_mc_rec_combined.index[(data_mc_rec_combined['MC_Mother_Track_ID'] < 2)],inplace=True)
  data_mc_rec_combined.sort_values(by=['MC_Vx','MC_Mother_Track_ID'], ascending=[1,0],inplace=True)
  data_mc_rec_combined.drop_duplicates(subset=['MC_Vx'],keep='first',inplace=True)
  data_rec_distribution=data_mc_rec_combined
  data_rec_distribution['Analysis']='Analysis'
  data_rec_distribution=data_rec_distribution[[RN,'Analysis']+MCCategories].groupby(by=['Analysis']+MCCategories)[RN].nunique().reset_index()
  data_mc_distribution=pd.merge(data_mc_distribution,data_rec_distribution,how='left', on=['Analysis']+MCCategories)
  data_mc_distribution[RN]=data_mc_distribution[RN].fillna(0)
  data_mc_distribution[RN+'_Ovr_Eff']=data_mc_distribution[RN]/data_mc_distribution['MC_Vx']

  data_mc_rec_combined.drop([RN,'MC_Vx'],axis=1,inplace=True)
  data_mc_rec_combined=data_mc_rec_combined.groupby(by=['Analysis']+MCCategories).sum().reset_index()
  data_mc_rec_combined=data_mc_rec_combined.rename(columns={RN+'_Matched_Track_No':RN+'_Mult'})
  data_mc_rec_combined=data_mc_rec_combined.rename(columns={'MC_Mother_Track_No':RN+'_MC_Mult'})
  data_mc_rec_combined=data_mc_rec_combined.rename(columns={'MC_Mother_Track_ID':RN+'_True_Mult'})
  data_mc_rec_combined[RN+'_Avg_Eff']=data_mc_rec_combined[RN+'_True_Mult']/data_mc_rec_combined[RN+'_MC_Mult']
  data_mc_rec_combined[RN+'_Avg_Prt']=data_mc_rec_combined[RN+'_True_Mult']/data_mc_rec_combined[RN+'_Mult']
  data_mc_distribution=pd.merge(data_mc_distribution,data_mc_rec_combined,how='left', on=['Analysis']+MCCategories)
data_mc_distribution.drop(['Analysis'],axis=1,inplace=True)
print(UF.TimeStamp(), bcolors.OKGREEN+'Recombination metrics are ready and listed bellow:'+bcolors.ENDC)
print(data_mc_distribution.to_string())
print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(),bcolors.BOLD+'Stage 4:'+bcolors.ENDC+' Writing the output...')
output_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/'+ofn+'_'+RN+'_ETr_rec_stats.csv'
data_mc_distribution.to_csv(output_file_location,index=False)
print(UF.TimeStamp(), bcolors.OKGREEN+"The track reconstruction stats for further analysis are written there:"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
#End of the script




