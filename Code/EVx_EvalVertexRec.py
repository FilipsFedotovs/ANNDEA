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
if PY_DIR!='': #Temp solution - the decision was made to move all libraries to EOS drive as AFS get locked during heavy HTCondor submission loads
    sys.path=['',PY_DIR]
    sys.path.append('/usr/lib64/python36.zip')
    sys.path.append('/usr/lib64/python3.6')
    sys.path.append('/usr/lib64/python3.6/lib-dynload')
    sys.path.append('/usr/lib64/python3.6/site-packages')
    sys.path.append('/usr/lib/python3.6/site-packages')
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
parser.add_argument('--TrackID',help="Name of the control track", default="[['Brick_ID','FEDRA_Track_ID']]")
parser.add_argument('--SkipRcmb',help="Skip recombination calculations (to reduce CPU load)", default="Y")
parser.add_argument('--MCCategories',help="What MC categories present in the MC data would you like to split by?", default="[]")
parser.add_argument('--RecNames',help="What Names would you like to assign to the reconstruction methods that generated the tracks?", default="['FEDRA']")
parser.add_argument('--Xmin',help="This option restricts data to only those events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option restricts data to only those events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option restricts data to only those events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option restricts data to only those events that have tracks with hits y-coordinates that are below this value", default='0')
parser.add_argument('--MinHitsTrack',help="What is the minimum number of hits per track?", default=PM.MinHitsTrack)
parser.add_argument('--RemoveTracksZ',help="This option enables to remove particular tracks of starting Z-coordinate", default='[]')
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
MCCategories=ast.literal_eval(args.MCCategories)
RecNames=ast.literal_eval(args.RecNames)
input_file_location=args.f
SkipRcmb=args.SkipRcmb=='N'
MinHitsTrack=int(args.MinHitsTrack)
RemoveTracksZ=ast.literal_eval(args.RemoveTracksZ)

Xmin,Xmax,Ymin,Ymax=float(args.Xmin),float(args.Xmax),float(args.Ymin),float(args.Ymax)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0 #We don't slice data if all values are set to zero simultaneousy (which is the default setting)
ofn=(args.f[(args.f.rfind('/'))+1:-4])

print(UF.TimeStamp(), 'Loading ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
columns_to_extract=[PM.x,PM.y,PM.z,PM.Hit_ID,PM.MC_Event_ID,PM.MC_Track_ID,PM.MC_VX_ID]
for col in TrackID:
    columns_to_extract+=col
columns_to_extract+=MCCategories
if os.path.isfile(input_file_location)!=True:
                     print(UF.TimeStamp(), bcolors.FAIL+"Critical fail: file",input_file_location,'is missing, please check that the name/path is correct...'+bcolors.ENDC)
                     exit()

raw_data=pd.read_csv(input_file_location,header=0,usecols=columns_to_extract)[columns_to_extract]
raw_data=raw_data.drop(raw_data.index[(raw_data['MC_Event_ID'] != '31-96')])
#raw_data=raw_data.drop(raw_data.index[(raw_data['MotherPDG'] != 14)])
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
raw_data[PM.Hit_ID] = raw_data[PM.Hit_ID].astype(str)
raw_data['MC_Mother_Track_ID'] = raw_data[PM.MC_Event_ID] + '-' + raw_data[PM.MC_Track_ID]
raw_data['MC_Mother_Vertex_ID'] = raw_data[PM.MC_Event_ID] + '-' + raw_data[PM.MC_VX_ID]
raw_data.drop([PM.MC_Event_ID,PM.MC_Track_ID,PM.MC_VX_ID],axis=1,inplace=True)
for rn in range(len(RecNames)):
    raw_data[TrackID[rn][0]] = raw_data[TrackID[rn][0]].astype(str)
    raw_data[TrackID[rn][1]] = raw_data[TrackID[rn][1]].astype(str)
    raw_data[TrackID[rn][2]] = raw_data[TrackID[rn][2]].astype(str)
    raw_data[RecNames[rn]] = raw_data[TrackID[rn][0]] + '-' + raw_data[TrackID[rn][1]]
    raw_data[RecNames[rn] + '-VX'] = raw_data[TrackID[rn][0]] + '-' + raw_data[TrackID[rn][2]]
    raw_data.drop([TrackID[rn][0],TrackID[rn][1],TrackID[rn][2]],axis=1,inplace=True)
    if len(RemoveTracksZ)>0:
            print(UF.TimeStamp(),'Removing tracks based on start point')
            TracksZdf = pd.DataFrame(RemoveTracksZ, columns = ['Bad_z'], dtype=float)
            data_aggregated=raw_data.groupby([RecNames[rn]])[PM.z].min().reset_index()
            data_aggregated=data_aggregated.rename(columns={PM.z: "PosBad_Z"})
            raw_data=pd.merge(raw_data, data_aggregated, how="left", on=[RecNames[rn]])
            raw_data=pd.merge(raw_data, TracksZdf, how="left", left_on=["PosBad_Z"], right_on=['Bad_z'])
            raw_data=raw_data[raw_data['Bad_z'].isnull()]
            raw_data=raw_data.drop(['Bad_z', 'PosBad_Z'],axis=1)
if SkipRcmb:
    print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
    print(UF.TimeStamp(),bcolors.BOLD+'Stage 2:'+bcolors.ENDC+' Calculating recombination metrics...')

    eval_data_comb=raw_data[['MC_Mother_Track_ID',PM.Hit_ID]]
    eval_data_comb=pd.merge(eval_data_comb,eval_data_comb[[PM.Hit_ID,'MC_Mother_Track_ID']].rename(columns={PM.Hit_ID: "Right_Hit"}),how='inner', on = ['MC_Mother_Track_ID'])

    eval_data_comb.drop(eval_data_comb.index[eval_data_comb[PM.Hit_ID] == eval_data_comb["Right_Hit"]], inplace = True)
    eval_data_comb["Hit_Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(eval_data_comb[PM.Hit_ID], eval_data_comb["Right_Hit"])]
    eval_data_comb.drop_duplicates(subset="Hit_Seed_ID",keep='first',inplace=True)
    eval_data_comb.drop(['MC_Mother_Track_ID',PM.Hit_ID,'Right_Hit'],axis=1,inplace=True)
    TruthHitSeedCount=len(eval_data_comb)


    print(UF.TimeStamp(),'Total 2-hit combinations are expected according to Monte Carlo:',TruthHitSeedCount)
    for RN in RecNames:
        print(UF.TimeStamp(),'Creating '+RN+' recombination metrics...')
        rec_data_comb=raw_data[[RN,PM.Hit_ID]]
        rec_data_comb.drop(rec_data_comb.index[(rec_data_comb[RN] == 'nan-nan')],inplace=True)
        rec_data_comb=rec_data_comb[rec_data_comb[RN].str.contains("nan")==False]
        rec_data_comb=pd.merge(rec_data_comb,rec_data_comb[[PM.Hit_ID,RN]].rename(columns={PM.Hit_ID: "Right_Hit"}),how='inner', on = [RN])
        rec_data_comb.drop(rec_data_comb.index[rec_data_comb[PM.Hit_ID] == rec_data_comb["Right_Hit"]], inplace = True)
        rec_data_comb["Hit_Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(rec_data_comb[PM.Hit_ID], rec_data_comb["Right_Hit"])]
        rec_data_comb.drop_duplicates(subset="Hit_Seed_ID",keep='first',inplace=True)
        rec_data_comb.drop([RN,PM.Hit_ID,'Right_Hit'],axis=1,inplace=True)
        RecHitSeedCount=len(rec_data_comb)
        OverlapHitSeedCount=len(pd.merge(eval_data_comb,rec_data_comb,how='inner', on=["Hit_Seed_ID"]))
        if TruthHitSeedCount==0:
            Recall=0
        else:
            Recall=round((float(OverlapHitSeedCount)/float(TruthHitSeedCount))*100,2)
        if OverlapHitSeedCount==0:
            Precision=0
        else:
            Precision=round((float(OverlapHitSeedCount)/float(RecHitSeedCount))*100,2)
        print(UF.TimeStamp(), bcolors.OKGREEN+'Recombination metrics for ',bcolors.BOLD+RN+bcolors.ENDC,' are ready and listed bellow:'+bcolors.ENDC)
        print(UF.TimeStamp(),'Total 2-hit combinations were reconstructed by '+RN+':',RecHitSeedCount)
        print(UF.TimeStamp(),'Correct combinations were reconstructed by '+RN+':',OverlapHitSeedCount)
        print(UF.TimeStamp(),'Therefore the recall of the '+RN+': is' ,bcolors.BOLD+str(Recall), '%'+bcolors.ENDC)
        print(UF.TimeStamp(),'And the precision of the '+RN+': is',bcolors.BOLD+str(Precision), '%'+bcolors.ENDC)

print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' Analyzing track reconstruction metrics...')

raw_data_mc=raw_data[['MC_Mother_Vertex_ID','MC_Mother_Track_ID',PM.Hit_ID]+MCCategories].groupby(by=['MC_Mother_Vertex_ID','MC_Mother_Track_ID']+MCCategories)[PM.Hit_ID].nunique().reset_index()
raw_data_mc.drop(raw_data_mc.index[(raw_data_mc[PM.Hit_ID] < MinHitsTrack)],inplace=True)
raw_data_mc=raw_data[['MC_Mother_Vertex_ID','MC_Mother_Track_ID']+MCCategories].groupby(by=['MC_Mother_Vertex_ID']+MCCategories)['MC_Mother_Track_ID'].nunique().reset_index()
raw_data_mc.drop(raw_data_mc.index[(raw_data_mc['MC_Mother_Track_ID'] < 2)],inplace=True)
for n in PM.VetoVertex:
     raw_data_mc.drop(raw_data_mc.index[(raw_data_mc['MC_Mother_Vertex_ID'].str.contains(str('-'+n)))],inplace=True)
mc_data_tot=raw_data_mc['MC_Mother_Vertex_ID'].nunique()
print(UF.TimeStamp(),'Total number of MC verteces is:',mc_data_tot)
data_mc=pd.merge(raw_data[['MC_Mother_Vertex_ID','MC_Mother_Track_ID',PM.Hit_ID]],raw_data_mc,how='inner', on =['MC_Mother_Vertex_ID'])
print(raw_data_mc)
for RN in RecNames:
  #raw_data_rec=raw_data.drop(raw_data.index[(raw_data[RN] == 'nan-nan')])
  raw_data_rec = raw_data[raw_data[RN].str.contains("nan") == False]
  print(raw_data_rec)
  exit()
  raw_data_rec=raw_data_rec[[RN,PM.Hit_ID]]
  raw_data_temp_rec=raw_data_rec[[RN,PM.Hit_ID]].rename(columns={PM.Hit_ID: RN+'_Size'})
  raw_data_temp_rec=raw_data_temp_rec.groupby(by=[RN])[RN+'_Size'].nunique().reset_index()
  raw_data_temp_rec.drop(raw_data_temp_rec.index[(raw_data_temp_rec[RN+'_Size'] < MinHitsTrack)],inplace=True)

  rec_data_tot=raw_data_temp_rec[RN].nunique()
  data_rec=pd.merge(raw_data_rec[[RN,PM.Hit_ID]],raw_data_temp_rec,how='inner', on =[RN])
  data_rec=pd.merge(data_rec,data_mc,how='inner', on =[PM.Hit_ID])
  data_rec=data_rec.rename(columns={PM.Hit_ID: RN+'_Overlap'})
  data_rec=data_rec.groupby(by=[RN,RN+'_Size','MC_Mother_Track_ID'])[RN+'_Overlap'].nunique().reset_index()
  data_rec.drop(data_rec.index[(data_rec[RN+'_Overlap'] < 2)],inplace=True)
  data_temp_rec=data_rec[[RN,'MC_Mother_Track_ID']].rename(columns={RN: RN+'_Segmentation'})


  data_temp_rec=data_temp_rec.groupby(by=['MC_Mother_Track_ID'])[RN+'_Segmentation'].nunique().reset_index()

  data_rec=pd.merge(data_rec,data_temp_rec,how='inner', on =['MC_Mother_Track_ID'])

  data_rec.sort_values(by=[RN,RN+'_Overlap'], ascending=[1,0],inplace=True)

  data_rec.drop_duplicates(subset=[RN],keep='first',inplace=True)

  data_rec.drop([RN],axis=1,inplace=True)
  rec_data_mtch=data_rec['MC_Mother_Track_ID'].nunique()
  raw_data_mc_loc=pd.merge(raw_data_mc,data_rec,how='left', on =['MC_Mother_Track_ID'])
  print(UF.TimeStamp(), bcolors.OKGREEN+'Recombination metrics for ',bcolors.BOLD+RN+bcolors.ENDC,bcolors.OKGREEN+' are ready and listed bellow:'+bcolors.ENDC)
  print(UF.TimeStamp(),'Total number of reconstructed tracks :',bcolors.BOLD+str(rec_data_tot)+bcolors.ENDC)
  print(UF.TimeStamp(),'But the number of those tracks matched to MC tracks is:',bcolors.BOLD+str(rec_data_mtch)+bcolors.ENDC)
  if raw_data_mc_loc["MC_Mother_Track_Size"].sum()>0:
    Recall=raw_data_mc_loc[RN+'_Overlap'].sum()/raw_data_mc_loc["MC_Mother_Track_Size"].sum()
  else:
    Recall=0
  if raw_data_mc_loc[RN+'_Size'].sum()>0:
    Precision=raw_data_mc_loc[RN+'_Overlap'].sum()/raw_data_mc_loc[RN+'_Size'].sum()
  else:
      Precision=0
  Segmentation=raw_data_mc_loc[RN+'_Segmentation'].mean()
  print(UF.TimeStamp(),'Average track reconstruction efficiency:',bcolors.BOLD+str(round(Recall,2)*100), '%'+bcolors.ENDC)
  print(UF.TimeStamp(),'Average track reconstruction purity:',bcolors.BOLD+str(round(Precision,2)*100), '%'+bcolors.ENDC)
  print(UF.TimeStamp(),'Average track segmentation:',bcolors.BOLD+str(round(Segmentation,2))+bcolors.ENDC)
  print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
  print(UF.TimeStamp(),bcolors.BOLD+'Stage 4:'+bcolors.ENDC+' Writing the output...')
  output_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/'+ofn+'_'+RN+'_ETr_rec_stats.csv'
  raw_data_mc_loc.to_csv(output_file_location,index=False)
  print(UF.TimeStamp(), bcolors.OKGREEN+"The track reconstruction stats for further analysis are written there:"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
#End of the script




