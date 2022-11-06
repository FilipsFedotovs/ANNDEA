# Part of <Work in progress package> package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
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


import UtilityFunctions as UF #This is where we keep routine utility functions
import Parameters as PM #This is where we keep framework global parameters

#Setting the parser
parser = argparse.ArgumentParser(description='This script compares the ouput of the previous step with the output of ANNDEA reconstructed data to calculate reconstruction performance.')
parser.add_argument('--f',help="Please enter the full path directory there the files are located", default='/eos/user/a/aiulian/sim_fedra/mergeneutrinos_260422_1e2nu_1e5mu/newtracking_110522/')
parser.add_argument('--TestBricks',help="What Names would you like to assign to the reconstruction methods that generated the tracks?", default="[11]")
parser.add_argument('--Gap',help="Offset along z?", default="50000")

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

def MotherIDNorm(row):
        if row['Z']>=row['Fiducial_Cut_z_LB'] and row['Z']<=row['Fiducial_Cut_z_UB']:
          return row['MCMotherID']
        else:
          return -2

input_file_location=args.f
no_quadrants=1
no_brick_layers=2
columns_to_extract=['ID','x','y','z','TX','TY','MCEvent','MCTrack','MCMotherID','P','MotherPDG','PdgCode', 'ProcID', 'FEDRATrackID']
gap=int(args.Gap)
data=None
for q in range(1,no_quadrants+1):
    for bl in range(1,no_brick_layers+1):
        input_file=input_file_location+'b0000'+str(bl)+str(q)+'/brick'+str(bl)+str(q)+'.csv'
        print(UF.TimeStamp(), 'Loading ',bcolors.OKBLUE+input_file+bcolors.ENDC)
        new_data=pd.read_csv(input_file,header=0,usecols=columns_to_extract)
        input_vx_file=input_file_location+'b0000'+str(bl)+str(q)+'/brick'+str(bl)+str(q)+'_vertices.csv'
        new_vx_data=pd.read_csv(input_vx_file,header=0)
        new_data=pd.merge(new_data,new_vx_data,how='left',on=['FEDRATrackID'])
        new_data['Brick_ID']=str(bl)+str(q)
        new_data['Z']=(new_data['z']+(bl*(77585+gap)))-gap
        new_data.drop(['z'],axis=1,inplace=True)
        new_data['MC_Event_ID']=str(bl)+str(q)+'-'+new_data['MCEvent'].astype(str)
        new_data['MC_Track']=new_data['MCEvent'].astype(str)+'-'+new_data['MCTrack'].astype(str)
        new_data['Fiducial_Cut_z_LB']=new_data['Z'].min()
        new_data['Fiducial_Cut_z_UB']=new_data['Z'].max()
        data=pd.concat([data,new_data])
        #new_data=pd.read_csv(input_file,header=0)
data_agg=data.groupby(by=['MC_Track'])['Z'].min().reset_index()
data_agg=data_agg.rename(columns={'Z': 'MC_Track_Start_Z'})
data=pd.merge(data,data_agg,how='inner',on=['MC_Track'])
data['MC_Mother_ID']=data.apply(MotherIDNorm,axis=1)
data=data.rename(columns={'ID': 'Hit_ID'})
data=data.rename(columns={'TX': 'tx'})
data=data.rename(columns={'TY': 'ty'})
data=data.rename(columns={'MCTrack': 'MC_Track_ID'})
data=data.rename(columns={'PdgCode': 'PDG_ID'})
data=data.rename(columns={'VertexS': 'FEDRA_Vertex_ID'})
data=data.rename(columns={'VertexE': 'FEDRA_Secondary_Vertex_ID'})
data=data.rename(columns={'Z': 'z'})
data.drop(['MCEvent','MCMotherID','MC_Track_Start_Z'],axis=1,inplace=True)
print(data)
exit()
#
#
#
#
# if SliceData:
#            print(UF.TimeStamp(),'Slicing the data...')
#            raw_data=raw_data.drop(raw_data.index[(raw_data[PM.x] > Xmax) | (raw_data[PM.x] < Xmin) | (raw_data[PM.y] > Ymax) | (raw_data[PM.y] < Ymin)])
#            final_rows=len(raw_data.axes[0])
#            print(UF.TimeStamp(),'The sliced raw data has ',final_rows,' hits')
# raw_data.drop([PM.x,PM.y,PM.z],axis=1,inplace=True)
# raw_data[PM.MC_Event_ID] = raw_data[PM.MC_Event_ID].astype(str)
# raw_data[PM.MC_Track_ID] = raw_data[PM.MC_Track_ID].astype(str)
# raw_data[PM.Hit_ID] = raw_data[PM.Hit_ID].astype(str)
# raw_data['MC_Mother_Track_ID'] = raw_data[PM.MC_Event_ID] + '-' + raw_data[PM.MC_Track_ID]
# raw_data.drop([PM.MC_Event_ID,PM.MC_Track_ID],axis=1,inplace=True)
#
# for rn in range(len(RecNames)):
#     raw_data[TrackID[rn][0]] = raw_data[TrackID[rn][0]].astype(str)
#     raw_data[TrackID[rn][1]] = raw_data[TrackID[rn][1]].astype(str)
#     raw_data[RecNames[rn]] = raw_data[TrackID[rn][0]] + '-' + raw_data[TrackID[rn][1]]
#     raw_data.drop([TrackID[rn][0],TrackID[rn][1]],axis=1,inplace=True)
# print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
# print(UF.TimeStamp(),bcolors.BOLD+'Stage 2:'+bcolors.ENDC+' Calculating recombination metrics...')
#
#
# eval_data_comb=raw_data[['MC_Mother_Track_ID',PM.Hit_ID]]
# eval_data_comb=pd.merge(eval_data_comb,eval_data_comb[[PM.Hit_ID,'MC_Mother_Track_ID']].rename(columns={PM.Hit_ID: "Right_Hit"}),how='inner', on = ['MC_Mother_Track_ID'])
#
# eval_data_comb.drop(eval_data_comb.index[eval_data_comb[PM.Hit_ID] == eval_data_comb["Right_Hit"]], inplace = True)
# eval_data_comb["Hit_Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(eval_data_comb[PM.Hit_ID], eval_data_comb["Right_Hit"])]
# eval_data_comb.drop_duplicates(subset="Hit_Seed_ID",keep='first',inplace=True)
# eval_data_comb.drop(['MC_Mother_Track_ID',PM.Hit_ID,'Right_Hit'],axis=1,inplace=True)
# TruthHitSeedCount=len(eval_data_comb)
#
#
# print(UF.TimeStamp(),'Total 2-hit combinations are expected according to Monte Carlo:',TruthHitSeedCount)
# for RN in RecNames:
#     print(UF.TimeStamp(),'Creating '+RN+' recombination metrics...')
#     rec_data_comb=raw_data[[RN,PM.Hit_ID]]
#     rec_data_comb.drop(rec_data_comb.index[(rec_data_comb[RN] == 'nan-nan')],inplace=True)
#     rec_data_comb=pd.merge(rec_data_comb,rec_data_comb[[PM.Hit_ID,RN]].rename(columns={PM.Hit_ID: "Right_Hit"}),how='inner', on = [RN])
#     rec_data_comb.drop(rec_data_comb.index[rec_data_comb[PM.Hit_ID] == rec_data_comb["Right_Hit"]], inplace = True)
#     rec_data_comb["Hit_Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(rec_data_comb[PM.Hit_ID], rec_data_comb["Right_Hit"])]
#     rec_data_comb.drop_duplicates(subset="Hit_Seed_ID",keep='first',inplace=True)
#     rec_data_comb.drop([RN,PM.Hit_ID,'Right_Hit'],axis=1,inplace=True)
#     RecHitSeedCount=len(rec_data_comb)
#     OverlapHitSeedCount=len(pd.merge(eval_data_comb,rec_data_comb,how='inner', on=["Hit_Seed_ID"]))
#     if TruthHitSeedCount==0:
#         Recall=0
#     else:
#         Recall=round((float(OverlapHitSeedCount)/float(TruthHitSeedCount))*100,2)
#     if OverlapHitSeedCount==0:
#         Precision=0
#     else:
#         Precision=round((float(OverlapHitSeedCount)/float(RecHitSeedCount))*100,2)
#     print(UF.TimeStamp(), bcolors.OKGREEN+'Recombination metrics for ',bcolors.BOLD+RN+bcolors.ENDC,' are ready and listed bellow:'+bcolors.ENDC)
#     print(UF.TimeStamp(),'Total 2-hit combinations were reconstructed by '+RN+':',RecHitSeedCount)
#     print(UF.TimeStamp(),'Correct combinations were reconstructed by '+RN+':',OverlapHitSeedCount)
#     print(UF.TimeStamp(),'Therefore the recall of the '+RN+': is' ,bcolors.BOLD+str(Recall), '%'+bcolors.ENDC)
#     print(UF.TimeStamp(),'And the precision of the '+RN+': is',bcolors.BOLD+str(Precision), '%'+bcolors.ENDC)
#
#
#
# print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
# print(UF.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' Analyzing track reconstruction metrics...')
#
#
# raw_data_mc=raw_data.groupby(by=['MC_Mother_Track_ID']+MCCategories)[PM.Hit_ID].nunique().reset_index()
# raw_data_mc.drop(raw_data_mc.index[(raw_data_mc[PM.Hit_ID] < 2)],inplace=True)
# raw_data_mc.rename(columns={PM.Hit_ID: "MC_Mother_Track_Size"},inplace=True)
# mc_data_tot=raw_data_mc['MC_Mother_Track_ID'].nunique()
# print(UF.TimeStamp(),'Total number of MC tracks is:',mc_data_tot)
# data_mc=pd.merge(raw_data[['MC_Mother_Track_ID',PM.Hit_ID]],raw_data_mc,how='inner', on =['MC_Mother_Track_ID'])
# for RN in RecNames:
#   raw_data_rec=raw_data.drop(raw_data.index[(raw_data[RN] == 'nan-nan')])
#   raw_data_rec=raw_data_rec[[RN,PM.Hit_ID]]
#   raw_data_temp_rec=raw_data_rec[[RN,PM.Hit_ID]].rename(columns={PM.Hit_ID: RN+'_Size'})
#   raw_data_temp_rec=raw_data_temp_rec.groupby(by=[RN])[RN+'_Size'].nunique().reset_index()
#   raw_data_temp_rec.drop(raw_data_temp_rec.index[(raw_data_temp_rec[RN+'_Size'] < 2)],inplace=True)
#   rec_data_tot=raw_data_temp_rec[RN].nunique()
#   data_rec=pd.merge(raw_data_rec[[RN,PM.Hit_ID]],raw_data_temp_rec,how='inner', on =[RN])
#   data_rec=pd.merge(data_rec,data_mc,how='inner', on =[PM.Hit_ID])
#   data_rec=data_rec.rename(columns={PM.Hit_ID: RN+'_Overlap'})
#   data_rec=data_rec.groupby(by=[RN,RN+'_Size','MC_Mother_Track_ID'])[RN+'_Overlap'].nunique().reset_index()
#   data_rec.drop(data_rec.index[(data_rec[RN+'_Overlap'] < 2)],inplace=True)
#   data_temp_rec=data_rec[[RN,'MC_Mother_Track_ID']].rename(columns={RN: RN+'_Segmentation'})
#   data_temp_rec=data_temp_rec.groupby(by=['MC_Mother_Track_ID'])[RN+'_Segmentation'].nunique().reset_index()
#   data_rec=pd.merge(data_rec,data_temp_rec,how='inner', on =['MC_Mother_Track_ID'])
#   data_rec.sort_values(by=[RN,RN+'_Overlap'], ascending=[1,0],inplace=True)
#   data_rec.drop_duplicates(subset=[RN],keep='first',inplace=True)
#   data_rec.drop([RN],axis=1,inplace=True)
#   rec_data_mtch=data_rec['MC_Mother_Track_ID'].nunique()
#   raw_data_mc=pd.merge(raw_data_mc,data_rec,how='left', on =['MC_Mother_Track_ID'])
#   print(UF.TimeStamp(), bcolors.OKGREEN+'Recombination metrics for ',bcolors.BOLD+RN+bcolors.ENDC,bcolors.OKGREEN+' are ready and listed bellow:'+bcolors.ENDC)
#   print(UF.TimeStamp(),'Total number of reconstructed tracks :',bcolors.BOLD+str(rec_data_tot)+bcolors.ENDC)
#   print(UF.TimeStamp(),'But the number of those tracks matched to MC tracks is:',bcolors.BOLD+str(rec_data_mtch)+bcolors.ENDC)
#   if raw_data_mc["MC_Mother_Track_Size"].sum()>0:
#     Recall=raw_data_mc[RN+'_Overlap'].sum()/raw_data_mc["MC_Mother_Track_Size"].sum()
#   else:
#     Recall=0
#   if raw_data_mc[RN+'_Size'].sum()>0:
#     Precision=raw_data_mc[RN+'_Overlap'].sum()/raw_data_mc[RN+'_Size'].sum()
#   else:
#       Precision=0
#   Segmentation=raw_data_mc[RN+'_Segmentation'].mean()
#   print(UF.TimeStamp(),'Average track reconstruction efficiency:',bcolors.BOLD+str(round(Recall,2)*100), '%'+bcolors.ENDC)
#   print(UF.TimeStamp(),'Average track reconstruction purity:',bcolors.BOLD+str(round(Precision,2)*100), '%'+bcolors.ENDC)
#   print(UF.TimeStamp(),'Average track segmentation:',bcolors.BOLD+str(round(Segmentation,2))+bcolors.ENDC)
# print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
# print(UF.TimeStamp(),bcolors.BOLD+'Stage 4:'+bcolors.ENDC+' Writing the output...')
# output_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/EH_TRACK_REC_STATS.csv'
# raw_data_mc.to_csv(output_file_location,index=False)
# print(UF.TimeStamp(), bcolors.OKGREEN+"The track reconstruction stats for further analysis are written there:"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
# print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
# #End of the script




