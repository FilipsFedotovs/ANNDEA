#This simple connects hits in the data to produce tracks
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
parser.add_argument('--ModelName',help="WHat GNN model would you like to use?", default='MH_GNN_5FTR_5_80_5_80')
parser.add_argument('--Patience',help="How many checks to do before resubmitting the job?", default='30')
parser.add_argument('--SubPause',help="How long to wait in minutes after submitting 10000 jobs?", default='60')
parser.add_argument('--SubGap',help="How long to wait in minutes after submitting 10000 jobs?", default='10000')
parser.add_argument('--Log',help="Would you like to log the performance: No, MC, Kalman? (Only available if you have MC Truth or Kalman track reconstruction data)", default='No')
parser.add_argument('--RecBatchID',help="Give this reconstruction batch an ID", default='Test_Batch')
parser.add_argument('--LocalSub',help="Local submission?", default='N')
parser.add_argument('--ForceStatus',help="Would you like the program run from specific status number? (Only for advance users)", default='0')
parser.add_argument('--RequestExtCPU',help="Would you like to request extra CPUs?", default='N')
parser.add_argument('--JobFlavour',help="Specifying the length of the HTCondor job walltime. Currently at 'workday' which is 8 hours.", default='workday')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/eos/experiment/ship/ANNDEA/Data')
parser.add_argument('--Xmin',help="This option restricts data to only those events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option restricts data to only those events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option restricts data to only those events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option restricts data to only those events that have tracks with hits y-coordinates that are below this value", default='0')
parser.add_argument('--Z_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along z-axis. (In order to avoid segmentation this value should be more than 1)", default='3')
parser.add_argument('--Y_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along y-axis. (In order to avoid segmentation this value should be more than 1)", default='2')
parser.add_argument('--X_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along x-axis. (In order to avoid segmentation this value should be more than 1)", default='2')

######################################## Parsing argument values  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
Log=args.Log.upper()
ModelName=args.ModelName
RecBatchID=args.RecBatchID
Patience=int(args.Patience)
SubPause=int(args.SubPause)*60
SubGap=int(args.SubGap)
LocalSub=(args.LocalSub=='Y')
if LocalSub:
   time_int=0
else:
    time_int=10
JobFlavour=args.JobFlavour
RequestExtCPU=(args.RequestExtCPU=='Y')
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
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
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
Program=[]
#Calculating the number of volumes that will be sent to HTCondor for reconstruction. Account for overlap if specified.
if X_overlap==1:
    Xsteps=math.ceil((x_max)/stepX)
else:
    Xsteps=(math.ceil((x_max)/stepX)*(X_overlap))-1

if Y_overlap==1:
    Ysteps=math.ceil((y_max)/stepY)
else:
    Ysteps=(math.ceil((y_max)/stepY)*(Y_overlap))-1

#Defining handy functions to make the code little cleaner

#The function bellow helps to monitor the HTCondor jobs and keep the submission flow
def AutoPilot(wait_min, interval_min, max_interval_tolerance,program):
     print(UF.TimeStamp(),'Going on an autopilot mode for ',wait_min, 'minutes while checking HTCondor every',interval_min,'min',bcolors.ENDC)
     wait_sec=wait_min*60
     interval_sec=interval_min*60
     intervals=int(math.ceil(wait_sec/interval_sec))
     for interval in range(1,intervals+1):
         time.sleep(interval_sec)
         print(UF.TimeStamp(),"Scheduled job checkup...") #Progress display
         bad_pop=UF.CreateCondorJobs(program[1][0],
                                    program[1][1],
                                    program[1][2],
                                    program[1][3],
                                    program[1][4],
                                    program[1][5],
                                    program[1][6],
                                    program[1][7],
                                    program[1][8],
                                    program[2],
                                    program[3],
                                    program[1][9],
                                    False)
         if len(bad_pop)>0:
               print(UF.TimeStamp(),bcolors.WARNING+'Autopilot status update: There are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
               if interval%max_interval_tolerance==0:
                  for bp in bad_pop:
                      UF.SubmitJobs2Condor(bp,program[5],RequestExtCPU,JobFlavour)
                  print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
         else:
              return True,False
     return False,False
#The function bellow helps to automate the submission process
def StandardProcess(program,status,freshstart):
        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.BOLD+'Stage '+str(status)+':'+bcolors.ENDC+str(program[status][0]))
        batch_sub=program[status][4]>1
        bad_pop=UF.CreateCondorJobs(program[status][1][0],
                                    program[status][1][1],
                                    program[status][1][2],
                                    program[status][1][3],
                                    program[status][1][4],
                                    program[status][1][5],
                                    program[status][1][6],
                                    program[status][1][7],
                                    program[status][1][8],
                                    program[status][2],
                                    program[status][3],
                                    program[status][1][9],
                                    False)
        if len(bad_pop)==0:
             print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+' has successfully completed'+bcolors.ENDC)
             return True,False


        elif (program[status][4])==len(bad_pop):
                 bad_pop=UF.CreateCondorJobs(program[status][1][0],
                                    program[status][1][1],
                                    program[status][1][2],
                                    program[status][1][3],
                                    program[status][1][4],
                                    program[status][1][5],
                                    program[status][1][6],
                                    program[status][1][7],
                                    program[status][1][8],
                                    program[status][2],
                                    program[status][3],
                                    program[status][1][9],
                                    batch_sub)
                 print(UF.TimeStamp(),'Submitting jobs to HTCondor... ',bcolors.ENDC)
                 _cnt=0
                 for bp in bad_pop:
                          if _cnt>SubGap:
                              print(UF.TimeStamp(),'Pausing submissions for  ',str(int(SubPause/60)), 'minutes to relieve congestion...',bcolors.ENDC)
                              time.sleep(SubPause)
                              _cnt=0
                          UF.SubmitJobs2Condor(bp,program[status][5],RequestExtCPU,JobFlavour)
                          _cnt+=bp[6]
                 if program[status][5]:
                    print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+' has successfully completed'+bcolors.ENDC)
                    return True,False
                 elif AutoPilot(600,time_int,Patience,program[status]):
                        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+' has successfully completed'+bcolors.ENDC)
                        return True,False
                 else:
                        print(UF.TimeStamp(),bcolors.FAIL+'Stage '+str(status)+' is uncompleted...'+bcolors.ENDC)
                        return False,False


        elif len(bad_pop)>0:
            if freshstart:
                   print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to wait please enter enter the maximum wait time in minutes'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
                   UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
                   if UserAnswer=='E':
                       print(UF.TimeStamp(),'OK, exiting now then')
                       exit()
                   if UserAnswer=='R':
                      _cnt=0
                      for bp in bad_pop:
                           if _cnt>SubGap:
                              print(UF.TimeStamp(),'Pausing submissions for  ',str(int(SubPause/60)), 'minutes to relieve congestion...',bcolors.ENDC)
                              time.sleep(SubPause)
                              _cnt=0
                           UF.SubmitJobs2Condor(bp,program[status][5],RequestExtCPU,JobFlavour)
                           _cnt+=bp[6]
                      print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
                      if program[status][5]:
                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+' has successfully completed'+bcolors.ENDC)
                          return True,False
                      elif AutoPilot(600,time_int,Patience,program[status]):
                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+ 'has successfully completed'+bcolors.ENDC)
                          return True,False
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage '+str(status)+' is uncompleted...'+bcolors.ENDC)
                          return False,False
                   else:
                      if program[status][5]:
                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+' has successfully completed'+bcolors.ENDC)
                          return True,False
                      elif AutoPilot(int(UserAnswer),time_int,Patience,program[status]):
                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+ 'has successfully completed'+bcolors.ENDC)
                          return True,False
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage '+str(status)+' is uncompleted...'+bcolors.ENDC)
                          return False,False
            else:
                      _cnt=0
                      for bp in bad_pop:
                           if _cnt>SubGap:
                              print(UF.TimeStamp(),'Pausing submissions for  ',str(int(SubPause/60)), 'minutes to relieve congestion...',bcolors.ENDC)
                              time.sleep(SubPause)
                              _cnt=0
                           UF.SubmitJobs2Condor(bp,program[status][5],RequestExtCPU,JobFlavour)
                           _cnt+=bp[6]
                      if program[status][5]:
                           print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+' has successfully completed'+bcolors.ENDC)
                           return True,False
                      elif AutoPilot(600,time_int,Patience,program[status]):
                           print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+ 'has successfully completed'+bcolors.ENDC)
                           return True,False
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage '+str(status)+' is uncompleted...'+bcolors.ENDC)
                          return False,False

#If we chose reset mode we do a full cleanup.
# #Reconstructing a single brick can cause in gereation of 100s of thousands of files - need to make sure that we remove them.
if Mode=='RESET':
    print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
    HTCondorTag="SoftUsed == \"ANNDEA-RTr1a-"+RecBatchID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'RTr1_'+RecBatchID, ['RTr1a','RTr1b','RTr1c','RTr1d',RecBatchID+'_RTr_OUTPUT.csv'], HTCondorTag)
    FreshStart=False
if Mode=='CLEANUP':
    Status=5
else:
    Status=int(args.ForceStatus)
################ Set the execution sequence for the script
###### Stage 0
prog_entry=[]
job_sets=[]
for i in range(0,Xsteps):
                job_sets.append(Ysteps)
prog_entry.append(' Sending hit cluster to the HTCondor, so the model assigns weights between hits')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','hit_cluster_rec_set','RTr1a','.csv',RecBatchID,job_sets,'RTr1a_ReconstructTracks_Sub.py'])
prog_entry.append([' --stepZ ', ' --stepY ', ' --stepX ', " --zOffset ", " --yOffset ", " --xOffset ", ' --cut_dt ', ' --cut_dr ', ' --ModelName ', ' --Log ',' --Z_overlap ',' --Y_overlap ',' --X_overlap ', ' --Z_ID_Max '])
prog_entry.append([stepZ,stepY,stepX,z_offset, y_offset, x_offset, cut_dt,cut_dr, ModelName ,Log,Z_overlap,Y_overlap,X_overlap, Zsteps])
prog_entry.append(Xsteps*Ysteps)
prog_entry.append(LocalSub)
Program.append(prog_entry)

if Mode=='RESET':
   print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
#Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))

###### Stage 1
prog_entry=[]
job_sets=Xsteps
prog_entry.append(' Sending hit cluster to the HTCondor, so the reconstructed clusters can be merged along y-axis')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','hit_cluster_rec_y_set','RTr1b','.csv',RecBatchID,job_sets,'RTr1b_LinkSegmentsY_Sub.py'])
prog_entry.append([' --Y_ID_Max ', ' --i '])
prog_entry.append([Ysteps,Xsteps])
prog_entry.append(Xsteps)
prog_entry.append(LocalSub)
Program.append(prog_entry)
if Mode=='RESET':
   print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
#Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))

###### Stage 2
prog_entry=[]
job_sets=1
prog_entry.append(' Sending hit cluster to the HTCondor, so the reconstructed clusters can be merged along x-axis')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','hit_cluster_rec_x_set','RTr1c','.csv',RecBatchID,job_sets,'RTr1c_LinkSegmentsX_Sub.py'])
prog_entry.append([' --X_ID_Max '])
prog_entry.append([Xsteps])
prog_entry.append(1)
prog_entry.append(True) #This part we can execute locally, no need for HTCondor
Program.append(prog_entry)
if Mode=='RESET':
   print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
#Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))

###### Stage 3
Program.append('Custom')


print(UF.TimeStamp(),'There are '+str(len(Program)+1)+' stages (0-'+str(len(Program)+1)+') of this script',bcolors.ENDC)
print(UF.TimeStamp(),'Current stage has a code',Status,bcolors.ENDC)
while Status<len(Program):
    if Program[Status]!='Custom':
        #Standard process here
        Result=StandardProcess(Program,Status,FreshStart)
        if Result[0]:
            FreshStart=Result[1]
            if int(args.ForceStatus)==0:
                Status+=1
                continue
            else:
                exit()
        else:
            Status=len(Program)+1
            break

    elif Status==3:
       #Non standard processes (that don't follow the general pattern) have been coded here
       print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
       print(UF.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' Using the results from previous steps to map merged trackIDs to the original reconstruction file')
       try:
           #Read the output with hit- ANN Track map
           FirstFile=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RTr1c_'+RecBatchID+'_0'+'/RTr1c_'+RecBatchID+'_hit_cluster_rec_x_set_0.csv'
           print(UF.TimeStamp(),'Loading the file ',bcolors.OKBLUE+FirstFile+bcolors.ENDC)
           TrackMap=pd.read_csv(FirstFile,header=0)
           input_file_location=args.f
           print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
           #Reading the original file with Raw hits
           Data=pd.read_csv(input_file_location,header=0)
           Data[PM.Hit_ID] = Data[PM.Hit_ID].astype(str)

           if SliceData: #If we want to perform reconstruction on the fraction of the Brick
                  CutData=Data.drop(Data.index[(Data[PM.x] > Xmax) | (Data[PM.x] < Xmin) | (Data[PM.y] > Ymax) | (Data[PM.y] < Ymin)]) #The focus area where we reconstruct
                  OtherData=Data.drop(Data.index[(Data[PM.x] <= Xmax) | (Data[PM.x] >= Xmin) | (Data[PM.y] <= Ymax) | (Data[PM.y] >= Ymin)]) #The rest of the volume
           else:
               CutData=Data #If we reconstruct the whole brick we jsut take the whole data. No need to separate.

           CutData.drop(['ANN_Brick_ID','ANN_Track_ID'],axis=1,inplace=True,errors='ignore') #Removing old ANNDEA reconstruction results so we can overwrite with the new ones
           #Map reconstructed ANN tracks to hits in the Raw file - this is in essesne the final output of the Tracking.
           print(CutData)
           print(TrackMap)
           CutData=pd.merge(CutData,TrackMap,how='left', left_on=[PM.Hit_ID], right_on=['HitID'])
           CutData.drop(['HitID'],axis=1,inplace=True) #Make sure that HitID is not the Hit ID name in the raw data.
           if SliceData:
            Data=pd.concat([CutData,OtherData]) #If we slice the data we do the of Reconstructed and Unreconstructed subset of the brick.
           else:
            Data=CutData
           output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_RTr_OUTPUT.csv' #Final output. We can use this file for further operations
           Data.to_csv(output_file_location,index=False)
           print(UF.TimeStamp(), bcolors.OKGREEN+"The tracked data has been written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
           #The code bellow displays the Cluster level reconstruction stats. Only available if MC and FEDRA (Optional) rec stats are available. Is not recomended for big jobs.
           if Log!='NO':
                        #Calculating ANNDEA volume level recombination accuracy stats, useful for diagnostics but it does not allow to determine the high level of tracking - there is a dedicated module for that
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
                                        input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RTr1a_'+RecBatchID+str(i)+'/RTr1a_'+RecBatchID+'_hit_cluster_rec_set_'+str(k)+'_' +str(j)+'_' +str(i)+'.pkl'
                                        cluster_data_raw=UF.PickleOperations(input_file_location, 'r', 'N/A')
                                        cluster_data=cluster_data_raw[0]
                                        result_temp=cluster_data.RecStats #If we enable the logging during the reconstruction, they will be recorded there.
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
                                        #Calculating precision by using formula: Precision = TP/(FP+TP)
                                        if (int(result_temp[2][6])+int(result_temp[1][6]))>0: #Avoiding division by zero
                                            precision_results_1.append(int(result_temp[2][0])/(int(result_temp[2][0])+int(result_temp[1][0])))
                                            precision_results_2.append(int(result_temp[2][1])/(int(result_temp[2][1])+int(result_temp[1][1])))
                                            precision_results_3.append(int(result_temp[2][2])/(int(result_temp[2][2])+int(result_temp[1][2])))
                                            precision_results_4.append(int(result_temp[2][3])/(int(result_temp[2][3])+int(result_temp[1][3])))
                                            precision_results_5.append(int(result_temp[2][4])/(int(result_temp[2][4])+int(result_temp[1][4])))
                                            precision_results_6.append(int(result_temp[2][5])/(int(result_temp[2][5])+int(result_temp[1][5])))
                                            precision_results_7.append(int(result_temp[2][6])/(int(result_temp[2][6])+int(result_temp[1][6])))
                                        #Calculating recall by using formula: Precision = TP/(FN+TP)
                                        if int(result_temp[2][2])>0: #Avoiding division by zero
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
           print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 4 has successfully completed'+bcolors.ENDC)
           Status=5
       except Exception as e:
          print(UF.TimeStamp(),bcolors.FAIL+'Stage 4 is uncompleted due to...',+e+bcolors.ENDC)
          Status=6
          break
if Status==5:

    print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
    exit()
    HTCondorTag="SoftUsed == \"ANNDEA-RTr1a-"+RecBatchID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'RTr1a_'+RecBatchID, ['RTr1_'+RecBatchID,RecBatchID+'_RTr_OUTPUT.csv'], HTCondorTag)
    HTCondorTag="SoftUsed == \"ANNDEA-RTr1b-"+RecBatchID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'RTr1b_'+RecBatchID, ['RTr1_'+RecBatchID,RecBatchID+'_RTr_OUTPUT.csv'], HTCondorTag)
    for p in Program:
        if p!='Custom':
           print(UF.TimeStamp(),UF.ManageTempFolders(p,'Delete'))
    print(UF.TimeStamp(), bcolors.OKGREEN+"Reconstruction has been completed"+bcolors.ENDC)
    exit()
else:
    print(UF.TimeStamp(), bcolors.FAIL+"Reconstruction has not been completed as one of the processes has timed out or --ForceStatus!=0 option was chosen. Please run the script again (without Reset Mode)."+bcolors.ENDC)
    exit()



