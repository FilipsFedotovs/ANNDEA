#This simple connects hits in the data to produce tracks
#Tracking Module of the ANNDEA package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import sys
import pandas as pd #We use Panda for a routine data processing
import math #We use it for data manipulation
import numpy as np
import os
from tabulate import tabulate
import time
from alive_progress import alive_bar
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
parser.add_argument('--Mode', help='Script will continue from the last checkpoint, unless you want to start from the scratch, then type "Reset"',default='')
parser.add_argument('--ModelName',help="WHat GNN model would you like to use?", default='MH_GNN_5FTR_4_120_4_120')
parser.add_argument('--Patience',help="How many checks to do before resubmitting the job?", default='15')
parser.add_argument('--Log',help="Would you like to log the performance: No, MC, Kalman? (Only available if you have MC Truth or Kalman track reconstruction data)", default='No')
parser.add_argument('--RecBatchID',help="Give this reconstruction batch an ID", default='SHIP_UR_v1')
parser.add_argument('--LocalSub',help="Local submission?", default='N')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_FEDRA_Raw_UR.csv')
parser.add_argument('--Xmin',help="This option restricts data to only those events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option restricts data to only those events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option restricts data to only those events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option restricts data to only those events that have tracks with hits y-coordinates that are below this value", default='0')
parser.add_argument('--Z_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along z-axis. (In order to avoid segmentation this value should be more than 1)", default='2')
parser.add_argument('--Y_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along y-axis. (In order to avoid segmentation this value should be more than 1)", default='2')
parser.add_argument('--X_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along x-axis. (In order to avoid segmentation this value should be more than 1)", default='2')

######################################## Parsing argument values  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
Log=args.Log.upper()
ModelName=args.ModelName
RecBatchID=args.RecBatchID
Patience=int(args.Patience)
LocalSub=(args.LocalSub=='Y')
input_file_location=args.f
Xmin,Xmax,Ymin,Ymax=float(args.Xmin),float(args.Xmax),float(args.Ymin),float(args.Ymax)
Z_overlap,Y_overlap,X_overlap=int(args.Z_overlap),int(args.Y_overlap),int(args.X_overlap)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0 #We don't slice data if all values are set to zero simultaneousy (which is the default setting)

#Loading Directory locations
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
if PY_DIR!='':
     sys.path=[PY_DIR]
     sys.path.append('/usr/lib64/python36.zip')
     sys.path.append('/usr/lib64/python3.6')
     sys.path.append('/usr/lib64/python3.6/lib-dynload')
sys.path.append(AFS_DIR+'/Code/Utilities')
import UtilityFunctions as UF #This is where we keep routine utility functions
import Parameters as PM #This is where we keep framework global parameters

#Establishing paths
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
print(UF.TimeStamp(),bcolors.BOLD+'Preparation 1/3:'+bcolors.ENDC+' Setting up metafiles...')
#Loading the model meta file
print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+Model_Meta_Path+bcolors.ENDC)
if os.path.isfile(Model_Meta_Path):
       Model_Meta_Raw=UF.PickleOperations(Model_Meta_Path, 'r', 'N/A')
       print(Model_Meta_Raw[1])
       Model_Meta=Model_Meta_Raw[0]
       stepX=Model_Meta.stepX
       stepY=Model_Meta.stepY
       stepZ=Model_Meta.stepZ
       cut_dt=Model_Meta.cut_dt
       cut_dr=Model_Meta.cut_dr
else:
       print(UF.TimeStamp(),bcolors.WARNING+'Fail! No existing model meta files have been found, exiting now'+bcolors.ENDC)

########################################     Phase 1 - Create compact source file    #########################################
print(UF.TimeStamp(),bcolors.BOLD+'Preparation 2/3:'+bcolors.ENDC+' Preparing the source data...')
required_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1_'+RecBatchID+'_hits.csv'
if os.path.isfile(required_file_location)==False or Mode=='RESET':
         print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
         data=pd.read_csv(input_file_location,
                     header=0,
                     usecols=[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty])[[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty]]
         total_rows=len(data.axes[0])
         data[PM.Hit_ID] = data[PM.Hit_ID].astype(str)
         print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
         print(UF.TimeStamp(),'Removing unreconstructed hits...')
         data=data.dropna()
         final_rows=len(data.axes[0])
         print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
         data[PM.Hit_ID] = data[PM.Hit_ID].astype(int)
         data[PM.Hit_ID] = data[PM.Hit_ID].astype(str)
         if SliceData:
              print(UF.TimeStamp(),'Slicing the data...')
              data=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
              final_rows=len(data.axes[0])
              print(UF.TimeStamp(),'The sliced data has ',final_rows,' hits')
         data=data.rename(columns={PM.x: "x"})
         data=data.rename(columns={PM.y: "y"})
         data=data.rename(columns={PM.z: "z"})
         data=data.rename(columns={PM.tx: "tx"})
         data=data.rename(columns={PM.ty: "ty"})
         data=data.rename(columns={PM.Hit_ID: "Hit_ID"})
         data.to_csv(required_file_location,index=False)
         print(UF.TimeStamp(), bcolors.OKGREEN+"The segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+required_file_location+bcolors.ENDC)

if Log!='NO':
  required_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/ETr1_'+RecBatchID+'_hits.csv'
  if os.path.isfile(required_file_location)==False or Mode=='RESET':
     print(UF.TimeStamp(),'Creating Evaluation file...')
     print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
     data=pd.read_csv(input_file_location,
                 header=0,
                 usecols=[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.MC_Track_ID,PM.MC_Event_ID])[[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.MC_Track_ID,PM.MC_Event_ID]]

     total_rows=len(data.axes[0])
     print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
     print(UF.TimeStamp(),'Removing unreconstructed hits...')
     data=data.dropna()
     final_rows=len(data.axes[0])
     print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
     data[PM.MC_Event_ID] = data[PM.MC_Event_ID].astype(str)
     data[PM.MC_Track_ID] = data[PM.MC_Track_ID].astype(str)
     data[PM.Hit_ID] = data[PM.Hit_ID].astype(str)
     data['MC_Mother_Track_ID'] = data[PM.MC_Event_ID] + '-' + data[PM.MC_Track_ID]
     data=data.drop([PM.MC_Event_ID],axis=1)
     data=data.drop([PM.MC_Track_ID],axis=1)
     if SliceData:
          print(UF.TimeStamp(),'Slicing the data...')
          data=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
          final_rows=len(data.axes[0])
          print(UF.TimeStamp(),'The sliced data has ',final_rows,' hits')
     print(UF.TimeStamp(),'Removing tracks which have less than',PM.MinHitsTrack,'hits...')
     track_no_data=data.groupby(['MC_Mother_Track_ID'],as_index=False).count()
     track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty,PM.Hit_ID],axis=1)
     track_no_data=track_no_data.rename(columns={PM.x: "MC_Track_No"})
     new_combined_data=pd.merge(data, track_no_data, how="left", on=['MC_Mother_Track_ID'])
     new_combined_data = new_combined_data[new_combined_data.MC_Track_No >= PM.MinHitsTrack]
     new_combined_data = new_combined_data.drop(["MC_Track_No"],axis=1)
     new_combined_data=new_combined_data.sort_values(['MC_Mother_Track_ID',PM.z],ascending=[1,1])
     grand_final_rows=len(new_combined_data.axes[0])
     print(UF.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
     new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
     new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
     new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
     new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
     new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
     new_combined_data=new_combined_data.rename(columns={PM.Hit_ID: "Hit_ID"})
     new_combined_data.to_csv(required_file_location,index=False)
     print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
     print(UF.TimeStamp(), bcolors.OKGREEN+"The track segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+required_file_location+bcolors.ENDC)

if Log=='KALMAN':
   required_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/KTr1_'+RecBatchID+'_hits.csv'
   if os.path.isfile(required_file_location)==False or Mode=='RESET':
        print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
        data=pd.read_csv(input_file_location,
                    header=0,
                    usecols=[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.Rec_Track_ID,PM.Rec_Track_Domain])[[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.Rec_Track_ID,PM.Rec_Track_Domain]]

        total_rows=len(data.axes[0])
        print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
        print(UF.TimeStamp(),'Removing unreconstructed hits...')
        data=data.dropna()
        final_rows=len(data.axes[0])
        print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
        data[PM.Rec_Track_Domain] = data[PM.Rec_Track_Domain].astype(str)
        data[PM.Rec_Track_ID] = data[PM.Rec_Track_ID].astype(str)
        data[PM.Hit_ID] = data[PM.Hit_ID].astype(str)
        data['Rec_Track_ID'] = data[PM.Rec_Track_Domain] + '-' + data[PM.Rec_Track_ID]
        data=data.drop([PM.Rec_Track_Domain],axis=1)
        data=data.drop([PM.Rec_Track_ID],axis=1)
        if SliceData:
             print(UF.TimeStamp(),'Slicing the data...')
             data=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
             final_rows=len(data.axes[0])
             print(UF.TimeStamp(),'The sliced data has ',final_rows,' hits')
        print(UF.TimeStamp(),'Removing tracks which have less than',PM.MinHitsTrack,'hits...')
        track_no_data=data.groupby(['Rec_Track_ID'],as_index=False).count()
        track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty,PM.Hit_ID],axis=1)
        track_no_data=track_no_data.rename(columns={PM.x: "Rec_Track_No"})
        new_combined_data=pd.merge(data, track_no_data, how="left", on=['Rec_Track_ID'])
        new_combined_data = new_combined_data[new_combined_data.Rec_Track_No >= PM.MinHitsTrack]
        new_combined_data = new_combined_data.drop(["Rec_Track_No"],axis=1)
        new_combined_data=new_combined_data.sort_values(['Rec_Track_ID',PM.z],ascending=[1,1])
        grand_final_rows=len(new_combined_data.axes[0])
        print(UF.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
        new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
        new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
        new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
        new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
        new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
        new_combined_data=new_combined_data.rename(columns={PM.Hit_ID: "Hit_ID"})
        new_combined_data.to_csv(required_file_location,index=False)
        print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(), bcolors.OKGREEN+"The track segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+required_file_location+bcolors.ENDC)

# ########################################     Preset framework parameters    #########################################
input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1_'+RecBatchID+'_hits.csv'
print(UF.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
data=pd.read_csv(input_file_location,header=0,usecols=['z','x','y'])
print(UF.TimeStamp(),'Analysing data... ',bcolors.ENDC)
z_offset=data['z'].min()
data['z']=data['z']-z_offset
z_max=data['z'].max()
if Z_overlap==1:
    Zsteps=math.ceil((z_max)/stepZ)
else:
    Zsteps=(math.ceil((z_max)/stepZ)*(Z_overlap))-1
y_offset=data['y'].min()
x_offset=data['x'].min()
data['x']=data['x']-x_offset
data['y']=data['y']-y_offset
x_max=data['x'].max()
y_max=data['y'].max()
FreshStart=True

if X_overlap==1:
    Xsteps=math.ceil((x_max)/stepX)
else:
    Xsteps=(math.ceil((x_max)/stepX)*(X_overlap))-1

if Y_overlap==1:
    Ysteps=math.ceil((y_max)/stepY)
else:
    Ysteps=(math.ceil((y_max)/stepY)*(Y_overlap))-1

#Defining handy functions to make the code little cleaner

def CheckStatus():
    #Let's check at what stage are we
    print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
    print(UF.TimeStamp(),bcolors.BOLD+'Preparation 3/3:'+bcolors.ENDC+' Working out the scope of the upcoming work...')
    #First of all lets check that the output of reconstruction is completed
    required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_RTr_OUTPUT.pkl'
    if os.path.isfile(required_output_file_location):
        return 5
    else:
        #Reconstruction output hasn't been produced - lets check the previous step result and so on
        required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1d_'+RecBatchID+'_hit_cluster_rec_x_set.pkl'
        if os.path.isfile(required_output_file_location):
           return 4
        else:
           bad_pop=0
           with alive_bar(Xsteps,force_tty=True, title='Checking the Y-shift results from HTCondor') as bar:
             for i in range(0,Xsteps):
                  required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1c_'+RecBatchID+'_hit_cluster_rec_y_set_' +str(i)+'.pkl'
                  bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                  bar()
                  if os.path.isfile(required_output_file_location)!=True:
                     bad_pop+=1
           if bad_pop==0:
               return 3
           else:
                bad_pop=0
                with alive_bar(Ysteps*Xsteps,force_tty=True, title='Checking the Z-shift results from HTCondor') as bar:
                        for j in range(0,Ysteps):
                             for i in range(0,Xsteps):
                                  required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1b_'+RecBatchID+'_hit_cluster_rec_z_set_'+str(j)+'_' +str(i)+'.pkl'
                                  bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                                  bar()
                                  if os.path.isfile(required_output_file_location)!=True:
                                     bad_pop+=1
                if bad_pop==0:
                   return 2
                else:
                    bad_pop=0
                    with alive_bar(Zsteps*Ysteps*Xsteps,force_tty=True, title='Checking the results from HTCondor') as bar:
                        for k in range(0,Zsteps):
                            for j in range(0,Ysteps):
                                 for i in range(0,Xsteps):
                                      required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1a_'+RecBatchID+'_hit_cluster_rec_set_'+str(k)+'_' +str(j)+'_' +str(i)+'.pkl'
                                      bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                                      bar()
                                      if os.path.isfile(required_output_file_location)!=True:
                                         bad_pop+=1
                    if bad_pop==0:
                       return 1
    return 0

def AutoPilot0(wait_min, interval_min, max_interval_tolerance):
     print(UF.TimeStamp(),'Going on an autopilot mode for ',wait_min, 'minutes while checking HTCondor every',interval_min,'min',bcolors.ENDC)
     wait_sec=wait_min*60
     interval_sec=interval_min*60
     intervals=int(math.ceil(wait_sec/interval_sec))
     for interval in range(1,intervals+1):
         time.sleep(interval_sec)
         bad_pop=[]
         print(UF.TimeStamp(),"Scheduled job checkup...") #Progress display
         with alive_bar(Zsteps*Ysteps*Xsteps,force_tty=True, title='Checking the results from HTCondor') as bar:
             for k in range(0,Zsteps):
               for j in range(0,Ysteps):
                   for i in range(0,Xsteps):
                      required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1a_'+RecBatchID+'_hit_cluster_rec_set_'+str(k)+'_' +str(j)+'_' +str(i)+'.pkl'
                      bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                      bar()
                      OptionHeader = [' --Z_ID ',' --Y_ID ', ' --X_ID ', ' --stepZ ', ' --stepY ', ' --stepX ', ' --EOS ', " --AFS ", " --zOffset ", " --yOffset ", " --xOffset ", ' --cut_dt ', ' --cut_dr ', ' --ModelName ', ' --Log ',' --RecBatchID ',' --Z_overlap ',' --Y_overlap ',' --X_overlap ']
                      OptionLine = [k, j, i, stepZ,stepY,stepX, EOS_DIR, AFS_DIR, z_offset, y_offset, x_offset, cut_dt,cut_dr, ModelName ,Log,RecBatchID,Z_overlap,Y_overlap,X_overlap]
                      SHName = AFS_DIR + '/HTCondor/SH/SH_RTr1a_'+ RecBatchID+'_' + str(k) + '_' + str(j) + '_' + str(i) +'.sh'
                      SUBName = AFS_DIR + '/HTCondor/SUB/SUB_RTr1a_'+ RecBatchID+'_'+ str(k) + '_' + str(j) + '_' + str(i) + '.sub'
                      MSGName = AFS_DIR + '/HTCondor/MSG/MSG_RTr1a_' + RecBatchID+'_' + str(k) + '_' + str(j) + '_' + str(i)
                      ScriptName = AFS_DIR + '/Code/Utilities/RTr1a_ReconstructHits_Sub.py '
                      if os.path.isfile(required_output_file_location)!=True:
                         bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-RTr1-'+RecBatchID, False,False])
         if len(bad_pop)>0:
               print(UF.TimeStamp(),bcolors.WARNING+'Autopilot status update: There are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
               if interval%max_interval_tolerance==0:
                  for bp in bad_pop:
                      UF.SubmitJobs2Condor(bp,LocalSub)
                  print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
         else:
              return True
     return False
def AutoPilot1(wait_min, interval_min, max_interval_tolerance):
     print(UF.TimeStamp(),'Going on an autopilot mode for ',wait_min, 'minutes while checking HTCondor every',interval_min,'min',bcolors.ENDC)
     wait_sec=wait_min*60
     interval_sec=interval_min*60
     intervals=int(math.ceil(wait_sec/interval_sec))
     for interval in range(1,intervals+1):
         time.sleep(interval_sec)
         bad_pop=[]
         print(UF.TimeStamp(),"Scheduled job checkup...") #Progress display
         with alive_bar(Ysteps*Xsteps,force_tty=True, title='Checking the Z-shift results from HTCondor') as bar:
            for j in range(0,Ysteps):
                 for i in range(0,Xsteps):
                      required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1b_'+RecBatchID+'_hit_cluster_rec_z_set_'+str(j)+'_' +str(i)+'.pkl'
                      bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                      bar()
                      OptionHeader = [' --Z_ID_Max ',' --Y_ID ', ' --X_ID ', ' --EOS ', " --AFS ", ' --RecBatchID ']
                      OptionLine = [Zsteps, j, i, EOS_DIR, AFS_DIR, RecBatchID]
                      SHName = AFS_DIR + '/HTCondor/SH/SH_RTr1b_'+ RecBatchID+ '_' + str(j) + '_' + str(i) +'.sh'
                      SUBName = AFS_DIR + '/HTCondor/SUB/SUB_RTr1b_'+ RecBatchID+ '_' + str(j) + '_' + str(i) + '.sub'
                      MSGName = AFS_DIR + '/HTCondor/MSG/MSG_RTr1b_' + RecBatchID+ '_' + str(j) + '_' + str(i)
                      ScriptName = AFS_DIR + '/Code/Utilities/RTr1b_LinkSegmentsZ_Sub.py '
                      if os.path.isfile(required_output_file_location)!=True:
                         bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-RTr1-'+RecBatchID, False,False])
         if len(bad_pop)>0:
               print(UF.TimeStamp(),bcolors.WARNING+'Autopilot status update: There are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
               if interval%max_interval_tolerance==0:
                  for bp in bad_pop:
                      UF.SubmitJobs2Condor(bp,LocalSub)
                  print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
         else:
              return True
     return False
def AutoPilot2(wait_min, interval_min, max_interval_tolerance):
     print(UF.TimeStamp(),'Going on an autopilot mode for ',wait_min, 'minutes while checking HTCondor every',interval_min,'min',bcolors.ENDC)
     wait_sec=wait_min*60
     interval_sec=interval_min*60
     intervals=int(math.ceil(wait_sec/interval_sec))
     for interval in range(1,intervals+1):
         time.sleep(interval_sec)
         bad_pop=[]
         print(UF.TimeStamp(),"Scheduled job checkup...") #Progress display
         with alive_bar(Ysteps*Xsteps,force_tty=True, title='Checking the Y-shift results from HTCondor') as bar:
                   for i in range(0,Xsteps):
                      required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1c_'+RecBatchID+'_hit_cluster_rec_y_set_' +str(i)+'.pkl'
                      bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                      bar()
                      OptionHeader = [' --Y_ID_Max ', ' --X_ID ', ' --EOS ', " --AFS ", ' --RecBatchID ']
                      OptionLine = [Ysteps, i, EOS_DIR, AFS_DIR, RecBatchID]
                      SHName = AFS_DIR + '/HTCondor/SH/SH_RTr1c_'+ RecBatchID+ '_'  + str(i) +'.sh'
                      SUBName = AFS_DIR + '/HTCondor/SUB/SUB_RTr1c_'+ RecBatchID+ '_' + str(i) + '.sub'
                      MSGName = AFS_DIR + '/HTCondor/MSG/MSG_RTr1c_' + RecBatchID+ '_' + str(i)
                      ScriptName = AFS_DIR + '/Code/Utilities/RTr1c_LinkSegmentsY_Sub.py '
                      if os.path.isfile(required_output_file_location)!=True:
                         bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-RTr1-'+RecBatchID, False,False])
         if len(bad_pop)>0:
               print(UF.TimeStamp(),bcolors.WARNING+'Autopilot status update: There are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
               if interval%max_interval_tolerance==0:
                  for bp in bad_pop:
                      UF.SubmitJobs2Condor(bp,LocalSub)
                  print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
         else:
              return True
     return False
def AutoPilot3(wait_min, interval_min, max_interval_tolerance):
     print(UF.TimeStamp(),'Going on an autopilot mode for ',wait_min, 'minutes while checking HTCondor every',interval_min,'min',bcolors.ENDC)
     wait_sec=wait_min*60
     interval_sec=interval_min*60
     intervals=int(math.ceil(wait_sec/interval_sec))
     for interval in range(1,intervals+1):
         time.sleep(interval_sec)
         bad_pop=[]
         print(UF.TimeStamp(),"Scheduled job checkup...") #Progress display
         required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1d_'+RecBatchID+'_hit_cluster_rec_x_set.pkl'
         OptionHeader = [' --X_ID_Max ', ' --EOS ', " --AFS ", ' --RecBatchID ']
         OptionLine = [Xsteps, EOS_DIR, AFS_DIR, RecBatchID]
         SHName = AFS_DIR + '/HTCondor/SH/SH_RTr1d_'+RecBatchID+'.sh'
         SUBName = AFS_DIR + '/HTCondor/SUB/SUB_RTr1d_'+RecBatchID+'.sub'
         MSGName = AFS_DIR + '/HTCondor/MSG/MSG_RTr1d_' + RecBatchID
         ScriptName = AFS_DIR + '/Code/Utilities/RTr1d_LinkSegmentsX_Sub.py '
         if os.path.isfile(required_output_file_location)!=True:
            bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-RTr1-'+RecBatchID, False,False])
         if len(bad_pop)>0:
               print(UF.TimeStamp(),bcolors.WARNING+'Autopilot status update: There are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
               if interval%max_interval_tolerance==0:
                  for bp in bad_pop:
                      UF.SubmitJobs2Condor(bp,LocalSub)
                  print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
         else:
              return True
     return False

if Mode=='RESET':
    print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
    HTCondorTag="SoftUsed == \"ANNDEA-RTr-"+RecBatchID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'RTr1_'+RecBatchID, ['RTr1a','RTr1b','RTr1c','RTr1d',RecBatchID+'_RTr_OUTPUT.pkl'], HTCondorTag)
    status=0
    FreshStart=False
else:
    status=CheckStatus()

print(UF.TimeStamp(),'There are 5 stages (0-4) of this script',status,bcolors.ENDC)
print(UF.TimeStamp(),'Current status has a code',status,bcolors.ENDC)


while status<5:
    if status==0:
        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.BOLD+'Stage 0:'+bcolors.ENDC+' Sending hit cluster to the HTCondor, so the model assigns weights between hits')

        for k in range(0,Zsteps):
                for j in range(0,Ysteps):
                     for i in range(0,Xsteps):
                          required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1a_'+RecBatchID+'_hit_cluster_rec_set_'+str(k)+'_' +str(j)+'_' +str(i)+'.pkl'
                          OptionHeader = [' --Z_ID ',' --Y_ID ', ' --X_ID ', ' --stepZ ', ' --stepY ', ' --stepX ', ' --EOS ', " --AFS ", " --zOffset ", " --yOffset ", " --xOffset ", ' --cut_dt ', ' --cut_dr ', ' --ModelName ', ' --Log ',' --RecBatchID ',' --Z_overlap ',' --Y_overlap ',' --X_overlap ']
                          OptionLine = [k, j, i, stepZ,stepY,stepX, EOS_DIR, AFS_DIR, z_offset, y_offset, x_offset, cut_dt,cut_dr, ModelName ,Log,RecBatchID,Z_overlap,Y_overlap,X_overlap]
                          SHName = AFS_DIR + '/HTCondor/SH/SH_RTr1a_'+ RecBatchID+'_' + str(k) + '_' + str(j) + '_' + str(i) +'.sh'
                          SUBName = AFS_DIR + '/HTCondor/SUB/SUB_RTr1a_'+ RecBatchID+'_'+ str(k) + '_' + str(j) + '_' + str(i) + '.sub'
                          MSGName = AFS_DIR + '/HTCondor/MSG/MSG_RTr1a_' + RecBatchID+'_' + str(k) + '_' + str(j) + '_' + str(i)
                          ScriptName = AFS_DIR + '/Code/Utilities/RTr1a_ReconstructHits_Sub.py '
                          if os.path.isfile(required_output_file_location)!=True:
                             bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-RTr1-'+RecBatchID, False,False])
        print(bad_pop[0])
        exit()
        if len(bad_pop)==0:
             FreshStart=False
             print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 0 has successfully completed'+bcolors.ENDC)
             status=1


        if FreshStart:
            if (Zsteps*Xsteps*Ysteps)==len(bad_pop):
                 print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
                 print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
                 print(bcolors.BOLD+'If you would like to wait please enter enter the maximum wait time in minutes'+bcolors.ENDC)
                 print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
                 UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
                 print(UF.TimeStamp(),'Submitting jobs to HTCondor... ',bcolors.ENDC)
                 if UserAnswer=='E':
                      print(UF.TimeStamp(),'OK, exiting now then')
                      exit()
                 if UserAnswer=='R':
                     for k in range(0,Zsteps):
                         for j in range(0,Ysteps):
                               OptionHeader = [' --Z_ID ',' --Y_ID ', ' --X_ID ', ' --stepZ ', ' --stepY ', ' --stepX ', ' --EOS ', " --AFS ", " --zOffset ", " --yOffset ", " --xOffset ", ' --cut_dt ', ' --cut_dr ', ' --ModelName ', ' --Log ',' --RecBatchID ',' --Z_overlap ',' --Y_overlap ',' --X_overlap ']
                               OptionLine = [k, j, '$1', stepZ,stepY,stepX, EOS_DIR, AFS_DIR, z_offset, y_offset, x_offset, cut_dt,cut_dr, ModelName ,Log,RecBatchID,Z_overlap,Y_overlap,X_overlap]
                               SHName = AFS_DIR + '/HTCondor/SH/SH_RTr1a_'+ RecBatchID+'_' + str(k) + '_' + str(j) + '.sh'
                               SUBName = AFS_DIR + '/HTCondor/SUB/SUB_RTr1a_'+ RecBatchID+'_'+ str(k) + '_' + str(j) + '.sub'
                               MSGName = AFS_DIR + '/HTCondor/MSG/MSG_RTr1a_' + RecBatchID+'_' + str(k) + '_' + str(j)
                               ScriptName = AFS_DIR + '/Code/Utilities/RTr1a_ReconstructHits_Sub.py '
                               UF.SubmitJobs2Condor([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, Xsteps, 'ANNDEA-RTr1-'+RecBatchID, False,False],LocalSub)
                 else:
                    if AutoPilot0(600,10,Patience):
                        FreshStart=False
                        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 0 has successfully completed'+bcolors.ENDC)
                        status=1
                    else:
                        print(UF.TimeStamp(),bcolors.FAIL+'Stage 0 is uncompleted...'+bcolors.ENDC)
                        status=6
                        break

            elif len(bad_pop)>0:
                   print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to wait please enter enter the maximum wait time in minutes'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
                   UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
                   if UserAnswer=='E':
                       print(UF.TimeStamp(),'OK, exiting now then')
                       exit()
                   if UserAnswer=='R':
                      for bp in bad_pop:
                           UF.SubmitJobs2Condor(bp,LocalSub)
                      print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
                      if AutoPilot0(600,10,Patience):
                          FreshStart=False
                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 0 has successfully completed'+bcolors.ENDC)
                          status=1
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage 0 is uncompleted...'+bcolors.ENDC)
                          status=6
                          break
                   else:
                      if AutoPilot0(int(UserAnswer),10,Patience):
                          FreshStart=False
                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 0 has successfully completed'+bcolors.ENDC)
                          status=1
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage 0 is uncompleted...'+bcolors.ENDC)
                          status=6
                          break
        else:
            if (Zsteps*Xsteps*Ysteps)==len(bad_pop):
                 print(UF.TimeStamp(),'Submitting jobs to HTCondor... ',bcolors.ENDC)
                 for k in range(0,Zsteps):
                         for j in range(0,Ysteps):
                               OptionHeader = [' --Z_ID ',' --Y_ID ', ' --X_ID ', ' --stepZ ', ' --stepY ', ' --stepX ', ' --EOS ', " --AFS ", " --zOffset ", " --yOffset ", " --xOffset ", ' --cut_dt ', ' --cut_dr ', ' --ModelName ', ' --Log ',' --RecBatchID ',' --Z_overlap ',' --Y_overlap ',' --X_overlap ']
                               OptionLine = [k, j, '$1', stepZ,stepY,stepX, EOS_DIR, AFS_DIR, z_offset, y_offset, x_offset, cut_dt,cut_dr, ModelName ,Log,RecBatchID,Z_overlap,Y_overlap,X_overlap]
                               SHName = AFS_DIR + '/HTCondor/SH/SH_RTr1a_'+ RecBatchID+'_' + str(k) + '_' + str(j) + '.sh'
                               SUBName = AFS_DIR + '/HTCondor/SUB/SUB_RTr1a_'+ RecBatchID+'_'+ str(k) + '_' + str(j) + '.sub'
                               MSGName = AFS_DIR + '/HTCondor/MSG/MSG_RTr1a_' + RecBatchID+'_' + str(k) + '_' + str(j)
                               ScriptName = AFS_DIR + '/Code/Utilities/RTr1a_ReconstructHits_Sub.py '
                               UF.SubmitJobs2Condor([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, Xsteps, 'ANNDEA-RTr1-'+RecBatchID, False,False],LocalSub)
                               print(UF.TimeStamp(),'Waiting ',str(GentleOnCondor),' minutes to relieve congestion on HTCondor...',bcolors.ENDC)
                               time.sleep(60*GentleOnCondor)

                 if AutoPilot0(600,10,Patience):
                        FreshStart=False
                        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 0 has successfully completed'+bcolors.ENDC)
                        status=1
                 else:
                     print(UF.TimeStamp(),bcolors.FAIL+'Stage 0 is uncompleted...'+bcolors.ENDC)
                     status=6
                     break

            elif len(bad_pop)>0:
                      for bp in bad_pop:
                           UF.SubmitJobs2Condor(bp,LocalSub)
                      if AutoPilot0(600,10,Patience):
                          FreshStart=False
                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 0 has successfully completed'+bcolors.ENDC)
                          status=1
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage 0 is uncompleted...'+bcolors.ENDC)
                          status=6
                          break

    if status==1:
        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.BOLD+'Stage 1:'+bcolors.ENDC+' Sending hit cluster to the HTCondor, so the reconstructed clusters can be merged along z-axis')
        bad_pop=[]
        with alive_bar(Ysteps*Xsteps,force_tty=True, title='Checking the Z-shift results from HTCondor') as bar:
            for j in range(0,Ysteps):
                 for i in range(0,Xsteps):
                      required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1b_'+RecBatchID+'_hit_cluster_rec_z_set_'+str(j)+'_' +str(i)+'.pkl'
                      bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                      bar()
                      OptionHeader = [' --Z_ID_Max ',' --Y_ID ', ' --X_ID ', ' --EOS ', " --AFS ", ' --RecBatchID ']
                      OptionLine = [Zsteps, j, i, EOS_DIR, AFS_DIR, RecBatchID]
                      SHName = AFS_DIR + '/HTCondor/SH/SH_RTr1b_'+ RecBatchID+ '_' + str(j) + '_' + str(i) +'.sh'
                      SUBName = AFS_DIR + '/HTCondor/SUB/SUB_RTr1b_'+ RecBatchID+ '_' + str(j) + '_' + str(i) + '.sub'
                      MSGName = AFS_DIR + '/HTCondor/MSG/MSG_RTr1b_' + RecBatchID+ '_' + str(j) + '_' + str(i)
                      ScriptName = AFS_DIR + '/Code/Utilities/RTr1b_LinkSegmentsZ_Sub.py '
                      if os.path.isfile(required_output_file_location)!=True:
                         bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-RTr1-'+RecBatchID, False,False])

        if FreshStart:
            if (Xsteps*Ysteps)==len(bad_pop):
                 print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
                 print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
                 print(bcolors.BOLD+'If you would like to wait please enter enter the maximum wait time in minutes'+bcolors.ENDC)
                 print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
                 UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
                 print(UF.TimeStamp(),'Submitting jobs to HTCondor... ',bcolors.ENDC)
                 if UserAnswer=='E':
                      print(UF.TimeStamp(),'OK, exiting now then')
                      exit()
                 if UserAnswer=='R':
                         for j in range(0,Ysteps):
                               ptionHeader = [' --Z_ID_Max ',' --Y_ID ', ' --X_ID ', ' --EOS ', " --AFS ", ' --RecBatchID ']
                               OptionLine = [Zsteps, j, '$1', EOS_DIR, AFS_DIR, RecBatchID]
                               SHName = AFS_DIR + '/HTCondor/SH/SH_RTr1b_'+ RecBatchID+ '_' + str(j) + '.sh'
                               SUBName = AFS_DIR + '/HTCondor/SUB/SUB_RTr1b_'+ RecBatchID+'_' + str(j) + '.sub'
                               MSGName = AFS_DIR + '/HTCondor/MSG/MSG_RTr1b_' + RecBatchID+ '_' + str(j)
                               ScriptName = AFS_DIR + '/Code/Utilities/RTr1b_LinkSegmentsZ_Sub.py '
                               UF.SubmitJobs2Condor([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, Xsteps, 'ANNDEA-RTr1-'+RecBatchID, False,False],LocalSub)
                 else:

                    if AutoPilot1(120,10,Patience):
                       print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 1 has successfully completed'+bcolors.ENDC)
                       FreshStart=False
                       status=2
                    else:
                        print(UF.TimeStamp(),bcolors.FAIL+'Stage 1 is uncompleted...'+bcolors.ENDC)
                        status=6
                        break

            elif len(bad_pop)>0:
                   print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to wait please enter enter the maximum wait time in minutes'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
                   UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
                   if UserAnswer=='E':
                       print(UF.TimeStamp(),'OK, exiting now then')
                       exit()
                   if UserAnswer=='R':
                      for bp in bad_pop:
                           UF.SubmitJobs2Condor(bp,LocalSub)
                      print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
                      if AutoPilot1(600,10,Patience):
                          FreshStart=False
                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 1 has successfully completed'+bcolors.ENDC)
                          status=2
                      else:
                            print(UF.TimeStamp(),bcolors.FAIL+'Stage 1 is uncompleted...'+bcolors.ENDC)
                            status=6
                            break
                   else:
                      if AutoPilot1(int(UserAnswer),10,Patience):
                          FreshStart=False
                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 1 has successfully completed'+bcolors.ENDC)
                          status=2
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage 1 is uncompleted...'+bcolors.ENDC)
                          status=6
                          break

            elif len(bad_pop)==0:
                FreshStart=False
                print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 1 has successfully completed'+bcolors.ENDC)
                status=2
        else:
            if (Xsteps*Ysteps)==len(bad_pop):
                 print(UF.TimeStamp(),'Submitting jobs to HTCondor... ',bcolors.ENDC)
                 for j in range(0,Ysteps):
                               ptionHeader = [' --Z_ID_Max ',' --Y_ID ', ' --X_ID ', ' --EOS ', " --AFS ", ' --RecBatchID ']
                               OptionLine = [Zsteps, j, '$1', EOS_DIR, AFS_DIR, RecBatchID]
                               SHName = AFS_DIR + '/HTCondor/SH/SH_RTr1b_'+ RecBatchID+ '_' + str(j) + '.sh'
                               SUBName = AFS_DIR + '/HTCondor/SUB/SUB_RTr1b_'+ RecBatchID+'_' + str(j) + '.sub'
                               MSGName = AFS_DIR + '/HTCondor/MSG/MSG_RTr1b_' + RecBatchID+ '_' + str(j)
                               ScriptName = AFS_DIR + '/Code/Utilities/RTr1b_LinkSegmentsZ_Sub.py '
                               UF.SubmitJobs2Condor([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, Xsteps, 'ANNDEA-RTr1-'+RecBatchID, False,False],LocalSub)

                 if AutoPilot1(600,10,Patience):
                       FreshStart=False
                       print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 1 has successfully completed'+bcolors.ENDC)
                       status=2
                 else:
                     print(UF.TimeStamp(),bcolors.FAIL+'Stage 1 is uncompleted...'+bcolors.ENDC)
                     status=6
                     break

            elif len(bad_pop)>0:
                      for bp in bad_pop:
                           UF.SubmitJobs2Condor(bp,LocalSub)
                      if AutoPilot1(600,10,Patience):
                          FreshStart=False
                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 1 has successfully completed'+bcolors.ENDC)
                          status=2
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage 1 is uncompleted...'+bcolors.ENDC)
                          status=6
                          break

            elif len(bad_pop)==0:
                FreshStart=False
                print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 1 has successfully completed'+bcolors.ENDC)
                status=2

    if status==2:
        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.BOLD+'Stage 2:'+bcolors.ENDC+' Sending hit cluster to the HTCondor, so the reconstructed clusters can be merged along y-axis')
        bad_pop=[]
        with alive_bar(Xsteps,force_tty=True, title='Checking the Y-shift results from HTCondor') as bar:
            for i in range(0,Xsteps):
                      required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1c_'+RecBatchID+'_hit_cluster_rec_y_set_' +str(i)+'.pkl'
                      bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                      bar()
                      OptionHeader = [' --Y_ID_Max ', ' --X_ID ', ' --EOS ', " --AFS ", ' --RecBatchID ']
                      OptionLine = [Ysteps, i, EOS_DIR, AFS_DIR, RecBatchID]
                      SHName = AFS_DIR + '/HTCondor/SH/SH_RTr1c_'+ RecBatchID+ '_'  + str(i) +'.sh'
                      SUBName = AFS_DIR + '/HTCondor/SUB/SUB_RTr1c_'+ RecBatchID+ '_' + str(i) + '.sub'
                      MSGName = AFS_DIR + '/HTCondor/MSG/MSG_RTr1c_' + RecBatchID+ '_' + str(i)
                      ScriptName = AFS_DIR + '/Code/Utilities/RTr1c_LinkSegmentsY_Sub.py '
                      if os.path.isfile(required_output_file_location)!=True:
                         bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-RTr1-'+RecBatchID, False,False])

        if FreshStart:
            if (Xsteps)==len(bad_pop):
                 print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
                 print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
                 print(bcolors.BOLD+'If you would like to wait please enter enter the maximum wait time in minutes'+bcolors.ENDC)
                 print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
                 UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
                 print(UF.TimeStamp(),'Submitting jobs to HTCondor... ',bcolors.ENDC)
                 if UserAnswer=='E':
                      print(UF.TimeStamp(),'OK, exiting now then')
                      exit()
                 if UserAnswer=='R':
                               OptionHeader = [' --Y_ID_Max ', ' --X_ID ', ' --EOS ', " --AFS ", ' --RecBatchID ']
                               OptionLine = [Ysteps, '$1', EOS_DIR, AFS_DIR, RecBatchID]
                               SHName = AFS_DIR + '/HTCondor/SH/SH_RTr1c_'+ RecBatchID+ '.sh'
                               SUBName = AFS_DIR + '/HTCondor/SUB/SUB_RTr1c_'+ RecBatchID+'.sub'
                               MSGName = AFS_DIR + '/HTCondor/MSG/MSG_RTr1c_' + RecBatchID
                               ScriptName = AFS_DIR + '/Code/Utilities/RTr1c_LinkSegmentsY_Sub.py '
                               UF.SubmitJobs2Condor([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, Xsteps, 'ANNDEA-RTr1-'+RecBatchID, False,False],LocalSub)
                 else:
                    if AutoPilot2(600,10,Patience):
                        FreshStart=False
                        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
                        status=3
                    else:
                        print(UF.TimeStamp(),bcolors.FAIL+'Stage 2 is uncompleted...'+bcolors.ENDC)
                        status=6
                        break

            elif len(bad_pop)>0:
                   print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to wait please enter enter the maximum wait time in minutes'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
                   UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
                   if UserAnswer=='E':
                       print(UF.TimeStamp(),'OK, exiting now then')
                       exit()
                   if UserAnswer=='R':
                      for bp in bad_pop:
                           UF.SubmitJobs2Condor(bp,LocalSub)
                      print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
                      if AutoPilot2(600,10,Patience):
                         FreshStart=False
                         print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
                         status=3
                      else:
                         print(UF.TimeStamp(),bcolors.FAIL+'Stage 2 is uncompleted...'+bcolors.ENDC)
                         status=6
                         break
                   else:

                      if AutoPilot2(int(UserAnswer),10,Patience):
                         FreshStart=False
                         print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
                         status=3
                      else:
                         print(UF.TimeStamp(),bcolors.FAIL+'Stage 2 is uncompleted...'+bcolors.ENDC)
                         status=6
                         break

            elif len(bad_pop)==0:
                FreshStart=False
                print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
                status=3
        else:
            if (Xsteps)==len(bad_pop):
                 print(UF.TimeStamp(),'Submitting jobs to HTCondor... ',bcolors.ENDC)
                 OptionHeader = [' --Y_ID_Max ', ' --X_ID ', ' --EOS ', " --AFS ", ' --RecBatchID ']
                 OptionLine = [Ysteps, '$1', EOS_DIR, AFS_DIR, RecBatchID]
                 SHName = AFS_DIR + '/HTCondor/SH/SH_RTr1c_'+ RecBatchID+ '.sh'
                 SUBName = AFS_DIR + '/HTCondor/SUB/SUB_RTr1c_'+ RecBatchID+'.sub'
                 MSGName = AFS_DIR + '/HTCondor/MSG/MSG_RTr1c_' + RecBatchID
                 ScriptName = AFS_DIR + '/Code/Utilities/RTr1c_LinkSegmentsY_Sub.py '
                 UF.SubmitJobs2Condor([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, Xsteps, 'ANNDEA-RTr-'+RecBatchID, False,False],LocalSub)
                 if AutoPilot2(600,10,Patience):
                        FreshStart=False
                        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
                        status=3
                 else:
                     print(UF.TimeStamp(),bcolors.FAIL+'Stage 2 is uncompleted...'+bcolors.ENDC)
                     status=6
                     break

            elif len(bad_pop)>0:
                      for bp in bad_pop:
                           UF.SubmitJobs2Condor(bp,LocalSub)
                      print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
                      if AutoPilot2(600,10,Patience):
                         FreshStart=False
                         print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
                         status=3
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage 2 is uncompleted...'+bcolors.ENDC)
                          status=6
                          break
            elif len(bad_pop)==0:
                FreshStart=False
                print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
                status=3



    if status==3:
        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' Sending hit cluster to the HTCondor, so the reconstructed clusters can be merged along x-axis')
        bad_pop=[]
        required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1d_'+RecBatchID+'_hit_cluster_rec_x_set.pkl'
        OptionHeader = [' --X_ID_Max ', ' --EOS ', " --AFS ", ' --RecBatchID ']
        OptionLine = [Xsteps, EOS_DIR, AFS_DIR, RecBatchID]
        SHName = AFS_DIR + '/HTCondor/SH/SH_RTr1d_'+RecBatchID+'.sh'
        SUBName = AFS_DIR + '/HTCondor/SUB/SUB_RTr1d_'+RecBatchID+'.sub'
        MSGName = AFS_DIR + '/HTCondor/MSG/MSG_RTr1d_' + RecBatchID
        ScriptName = AFS_DIR + '/Code/Utilities/RTr1d_LinkSegmentsX_Sub.py '
        if os.path.isfile(required_output_file_location)!=True:
            bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-RTr-'+RecBatchID, False,False])
        if FreshStart:
            if len(bad_pop)>0:
                   print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor job remaining'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to wait please enter enter the maximum wait time in minutes'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
                   UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
                   if UserAnswer=='E':
                       print(UF.TimeStamp(),'OK, exiting now then')
                       exit()
                   if UserAnswer=='R':
                      for bp in bad_pop:
                           UF.SubmitJobs2Condor(bp,LocalSub)
                      print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
                      if AutoPilot3(600,10,Patience):
                         FreshStart=False
                         print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 3 has successfully completed'+bcolors.ENDC)
                         status=4
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage 3 is uncompleted...'+bcolors.ENDC)
                          status=6
                          break

                   else:
                      if AutoPilot3(int(UserAnswer),10,Patience):
                         FreshStart=False
                         print(UF.TimeStamp(),bcolors.BOLD+'Stage 3 has successfully completed'+bcolors.ENDC)
                         status=4
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage 3 is uncompleted...'+bcolors.ENDC)
                          status=6
                          break

            elif len(bad_pop)==0:
                FreshStart=False
                print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 3 has successfully completed'+bcolors.ENDC)
                status=4

        else:
            if len(bad_pop)>0:
                      for bp in bad_pop:
                           UF.SubmitJobs2Condor(bp,LocalSub)
                      print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
                      if AutoPilot3(600,10,Patience):
                         FreshStart=False
                         print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 3 has successfully completed'+bcolors.ENDC)
                         status=4
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage 3 is uncompleted...'+bcolors.ENDC)
                          status=6
                          break

            elif len(bad_pop)==0:
                FreshStart=False
                print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 3 has successfully completed'+bcolors.ENDC)
                status=4

    if status==4:
       print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
       print(UF.TimeStamp(),bcolors.BOLD+'Stage 4:'+bcolors.ENDC+' Using the results from previous steps to map merged trackIDs to the original reconstruction file')
       try:
           FirstFile=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1d_'+RecBatchID+'_hit_cluster_rec_x_set.pkl'
           print(UF.TimeStamp(),'Loading the object ',bcolors.OKBLUE+FirstFile+bcolors.ENDC)
           FirstFileRaw=UF.PickleOperations(FirstFile,'r', 'N/A')
           FirstFile=FirstFileRaw[0]
           TrackMap=FirstFile.RecTracks
           input_file_location=args.f
           print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
           Data=pd.read_csv(input_file_location,header=0)
           Data[PM.Hit_ID] = Data[PM.Hit_ID].astype(str)
           if SliceData:
                  CutData=Data.drop(Data.index[(Data[PM.x] > Xmax) | (Data[PM.x] < Xmin) | (Data[PM.y] > Ymax) | (Data[PM.y] < Ymin)])
                  OtherData=Data.drop(Data.index[(Data[PM.x] <= Xmax) | (Data[PM.x] >= Xmin) | (Data[PM.y] <= Ymax) | (Data[PM.y] >= Ymin)])
           try:
             CutData.drop(['ANN_Brick_ID','ANN_Track_ID'],axis=1,inplace=True) #Removing old ANNDEA reconstruction results so we can overwrite with the new ones
           except Exception as e: print(UF.TimeStamp(), bcolors.WARNING+str(e)+bcolors.ENDC)

           CutData=pd.merge(CutData,TrackMap,how='left', left_on=[PM.Hit_ID], right_on=['HitID'])
           CutData.drop(['HitID'],axis=1,inplace=True)
           Data=pd.concat([CutData,OtherData])
           output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_RTr_OUTPUT.csv'
           Data.to_csv(output_file_location,index=False)
           print(UF.TimeStamp(), bcolors.OKGREEN+"The tracked data has been written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)

           if Log!='NO':
                        #Calculating recombination accuracy stats, useful for diagnostics but it does allow to determine the high level of tracking - there is a dedicated module for that
                        print(UF.TimeStamp(),'Since the logging was requested, ANN average recombination performance across the clusters is being calculated...')
                        fake_results_1=[]
                        fake_results_2=[]
                        fake_results_3=[]
                        fake_results_4=[]
                        fake_results_5=[]
                        fake_results_6=[]
                        fake_results_7=[]
                        truth_results_1=[]
                        truth_results_2=[]
                        truth_results_3=[]
                        truth_results_4=[]
                        truth_results_5=[]
                        truth_results_6=[]
                        truth_results_7=[]
                        precision_results_1=[]
                        precision_results_2=[]
                        precision_results_3=[]
                        precision_results_4=[]
                        precision_results_5=[]
                        precision_results_6=[]
                        precision_results_7=[]
                        recall_results_1=[]
                        recall_results_2=[]
                        recall_results_3=[]
                        recall_results_4=[]
                        recall_results_5=[]
                        recall_results_6=[]
                        recall_results_7=[]
                        with alive_bar(Zsteps*Ysteps*Xsteps,force_tty=True, title='Collecting all clusters together...') as bar:
                            for k in range(0,Zsteps):
                                for j in range(0,Ysteps):
                                    for i in range(0,Xsteps):
                                        bar()
                                        input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1a_'+RecBatchID+'_hit_cluster_rec_set_'+str(k)+'_' +str(j)+'_' +str(i)+'.pkl'
                                        cluster_data_raw=UF.PickleOperations(input_file_location, 'r', 'N/A')
                                        cluster_data=cluster_data_raw[0]
                                        result_temp=cluster_data.RecStats
                                        fake_results_1.append(int(result_temp[1][0]))
                                        fake_results_2.append(int(result_temp[1][1]))
                                        fake_results_3.append(int(result_temp[1][2]))
                                        fake_results_4.append(int(result_temp[1][3]))
                                        fake_results_5.append(int(result_temp[1][4]))
                                        fake_results_6.append(int(result_temp[1][5]))
                                        fake_results_7.append(int(result_temp[1][6]))
                                        truth_results_1.append(int(result_temp[2][0]))
                                        truth_results_2.append(int(result_temp[2][1]))
                                        truth_results_3.append(int(result_temp[2][2]))
                                        truth_results_4.append(int(result_temp[2][3]))
                                        truth_results_5.append(int(result_temp[2][4]))
                                        truth_results_6.append(int(result_temp[2][5]))
                                        truth_results_7.append(int(result_temp[2][6]))
                                        if (int(result_temp[2][6])+int(result_temp[1][6]))>0:
                                            precision_results_1.append(int(result_temp[2][0])/(int(result_temp[2][0])+int(result_temp[1][0])))
                                            precision_results_2.append(int(result_temp[2][1])/(int(result_temp[2][1])+int(result_temp[1][1])))
                                            precision_results_3.append(int(result_temp[2][2])/(int(result_temp[2][2])+int(result_temp[1][2])))
                                            precision_results_4.append(int(result_temp[2][3])/(int(result_temp[2][3])+int(result_temp[1][3])))
                                            precision_results_5.append(int(result_temp[2][4])/(int(result_temp[2][4])+int(result_temp[1][4])))
                                            precision_results_6.append(int(result_temp[2][5])/(int(result_temp[2][5])+int(result_temp[1][5])))
                                            precision_results_7.append(int(result_temp[2][6])/(int(result_temp[2][6])+int(result_temp[1][6])))
                                        if int(result_temp[2][2])>0:
                                            recall_results_1.append(int(result_temp[2][0])/(int(result_temp[2][2])))
                                            recall_results_2.append(int(result_temp[2][1])/(int(result_temp[2][2])))
                                            recall_results_3.append(int(result_temp[2][2])/(int(result_temp[2][2])))
                                            recall_results_4.append(int(result_temp[2][3])/(int(result_temp[2][2])))
                                            recall_results_5.append(int(result_temp[2][4])/(int(result_temp[2][2])))
                                            recall_results_6.append(int(result_temp[2][5])/(int(result_temp[2][2])))
                                            recall_results_7.append(int(result_temp[2][6])/(int(result_temp[2][2])))
                                        else:
                                               continue
                                        label=result_temp[0]
                                        label.append('Original # of valid Combinations')
                            print(UF.TimeStamp(),bcolors.OKGREEN+'ANN reconstruction results have been compiled and presented bellow:'+bcolors.ENDC)
                            print(tabulate([[label[0], np.average(fake_results_1), np.average(truth_results_1), np.sum(truth_results_1)/(np.sum(fake_results_1)+np.sum(truth_results_1)), np.std(precision_results_1), np.average(recall_results_1), np.std(recall_results_1)], \
                            [label[1], np.average(fake_results_2), np.average(truth_results_2), np.sum(truth_results_2)/(np.sum(fake_results_2)+np.sum(truth_results_2)), np.std(precision_results_2), np.average(recall_results_2), np.std(recall_results_2)], \
                            [label[2], np.average(fake_results_3), np.average(truth_results_3), np.sum(truth_results_3)/(np.sum(fake_results_3)+np.sum(truth_results_3)), np.std(precision_results_3), np.average(recall_results_3), np.std(recall_results_3)], \
                            [label[3], np.average(fake_results_4), np.average(truth_results_4), np.sum(truth_results_4)/(np.sum(fake_results_4)+np.sum(truth_results_4)), np.std(precision_results_4), np.average(recall_results_4), np.std(recall_results_4)],\
                            [label[4], np.average(fake_results_5), np.average(truth_results_5), np.sum(truth_results_5)/(np.sum(fake_results_5)+np.sum(truth_results_5)), np.std(precision_results_5), np.average(recall_results_5), np.std(recall_results_5)], \
                            [label[5], np.average(fake_results_6), np.average(truth_results_6), np.sum(truth_results_6)/(np.sum(fake_results_6)+np.sum(truth_results_6)), np.std(precision_results_6), np.average(recall_results_6), np.std(recall_results_6)], \
                            [label[6], np.average(fake_results_7), np.average(truth_results_7), np.sum(truth_results_7)/(np.sum(fake_results_7)+np.sum(truth_results_3)), np.std(precision_results_7), np.average(recall_results_7), np.std(recall_results_7)]], \
                            headers=['Step', 'Avg # Fake edges', 'Avg # of Genuine edges', 'Avg precision', 'Precision std','Avg recall', 'Recall std' ], tablefmt='orgtbl'))

           if Log=='KALMAN':
                        print(UF.TimeStamp(),'Since the logging was requested, Kalman filter average recombination performance across the clusters is being calculated...')
                        fake_results_1=[]
                        fake_results_2=[]
                        fake_results_3=[]
                        fake_results_4=[]
                        truth_results_1=[]
                        truth_results_2=[]
                        truth_results_3=[]
                        truth_results_4=[]
                        precision_results_1=[]
                        precision_results_2=[]
                        precision_results_3=[]
                        precision_results_4=[]
                        recall_results_1=[]
                        recall_results_2=[]
                        recall_results_3=[]
                        recall_results_4=[]
                        with alive_bar(Zsteps*Ysteps*Xsteps,force_tty=True, title='Collating the results') as bar:
                            for k in range(0,Zsteps):
                                for j in range(0,Ysteps):
                                    for i in range(0,Xsteps):
                                        bar()
                                        input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1a_'+RecBatchID+'_hit_cluster_rec_set_'+str(k)+'_' +str(j)+'_' +str(i)+'.pkl'
                                        cluster_data_raw=UF.PickleOperations(input_file_location, 'r', 'N/A')
                                        cluster_data=cluster_data_raw[0]
                                        result_temp=cluster_data.KalmanRecStats
                                        fake_results_1.append(int(result_temp[1][0]))
                                        fake_results_2.append(int(result_temp[1][1]))
                                        fake_results_3.append(int(result_temp[1][2]))
                                        fake_results_4.append(int(result_temp[1][3]))
                                        truth_results_1.append(int(result_temp[2][0]))
                                        truth_results_2.append(int(result_temp[2][1]))
                                        truth_results_3.append(int(result_temp[2][2]))
                                        truth_results_4.append(int(result_temp[2][3]))
                                        if (int(result_temp[2][3])+int(result_temp[1][3]))>0:
                                            precision_results_1.append(int(result_temp[2][0])/(int(result_temp[2][0])+int(result_temp[1][0])))
                                            precision_results_2.append(int(result_temp[2][1])/(int(result_temp[2][1])+int(result_temp[1][1])))
                                            precision_results_3.append(int(result_temp[2][2])/(int(result_temp[2][2])+int(result_temp[1][2])))
                                            precision_results_4.append(int(result_temp[2][3])/(int(result_temp[2][3])+int(result_temp[1][3])))
                                        if int(result_temp[2][2])>0:
                                            recall_results_1.append(int(result_temp[2][0])/(int(result_temp[2][2])))
                                            recall_results_2.append(int(result_temp[2][1])/(int(result_temp[2][2])))
                                            recall_results_3.append(int(result_temp[2][2])/(int(result_temp[2][2])))
                                            recall_results_4.append(int(result_temp[2][3])/(int(result_temp[2][2])))
                                        else:
                                               continue
                                        label=result_temp[0]
                                        label.append('Original # of valid Combinations')

                        print(UF.TimeStamp(),bcolors.OKGREEN+'Fedra reconstruction results have been compiled and presented bellow:'+bcolors.ENDC)
                        print(tabulate([[label[0], np.average(fake_results_1), np.average(truth_results_1), np.sum(truth_results_1)/(np.sum(fake_results_1)+np.sum(truth_results_1)), np.std(precision_results_1), np.average(recall_results_1), np.std(recall_results_1)], \
                        [label[1], np.average(fake_results_2), np.average(truth_results_2), np.sum(truth_results_2)/(np.sum(fake_results_2)+np.sum(truth_results_2)), np.std(precision_results_2), np.average(recall_results_2), np.std(recall_results_2)], \
                        [label[2], np.average(fake_results_3), np.average(truth_results_3), np.sum(truth_results_3)/(np.sum(fake_results_3)+np.sum(truth_results_3)), np.std(precision_results_3), np.average(recall_results_3), np.std(recall_results_3)], \
                        [label[3], np.average(fake_results_4), np.average(truth_results_4), np.sum(truth_results_4)/(np.sum(fake_results_4)+np.sum(truth_results_4)), np.std(precision_results_4), np.average(recall_results_4), np.std(recall_results_4)]],\
                        headers=['Step', 'Avg # Fake edges', 'Avg # of Genuine edges', 'Avg precision', 'Precision std','Avg recall', 'Recall std' ], tablefmt='orgtbl'))
           print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 4 has successfully completed'+bcolors.ENDC)
           status=5
       except Exception as e:
           print(UF.TimeStamp(),bcolors.FAIL+'Stage 4 is uncompleted due to...',+e+bcolors.ENDC)
           status=6
           break
if status==5:
    print(UF.TimeStamp(), bcolors.OKGREEN+"Reconstruction has been completed"+bcolors.ENDC)
    exit()
else:
    print(UF.TimeStamp(), bcolors.FAIL+"Reconstruction has not been completed as one of the processes has timed out. Please run the script again (without Reset Mode)."+bcolors.ENDC)
    exit()



