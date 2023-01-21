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
import csv
import argparse
import pandas as pd #We use Panda for a routine data processing
import math #We use it for data manipulation
import numpy as np
import os
import time
import ast
from alive_progress import alive_bar
import random
import gc
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
print(bcolors.HEADER+"#########     Initialising ANNDEA Track Union Training Sample Generation module          ###############"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script prepares training data for training the tracking model')
parser.add_argument('--Mode', help='Script will continue from the last checkpoint, unless you want to start from the scratch, then type "Reset"',default='')
parser.add_argument('--ModelName',help="WHat GNN model would you like to use?", default="['MH_GNN_5FTR_4_120_4_120']")
parser.add_argument('--Patience',help="How many checks to do before resubmitting the job?", default='15')
parser.add_argument('--TrainSampleID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_FEDRA_Raw_UR.csv')
parser.add_argument('--Xmin',help="This option restricts data to only those events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option restricts data to only those events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option restricts data to only those events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option restricts data to only those events that have tracks with hits y-coordinates that are below this value", default='0')
parser.add_argument('--Samples',help="How many samples? Please enter the number or ALL if you want to use all data", default='ALL')
parser.add_argument('--LabelRatio',help="What is the desired proportion of genuine seeds in the training/validation sets", default='0.5')
parser.add_argument('--TrainSampleSize',help="Maximum number of samples per Training file", default='50000')
parser.add_argument('--ForceStatus',help="Would you like the program run from specific status number? (Only for advance users)", default='0')
parser.add_argument('--RequestExtCPU',help="Would you like to request extra CPUs?", default='N')
parser.add_argument('--JobFlavour',help="Specifying the length of the HTCondor job walltime. Currently at 'workday' which is 8 hours.", default='workday')
parser.add_argument('--LocalSub',help="Local submission?", default='N')
parser.add_argument('--SubPause',help="How long to wait in minutes after submitting 10000 jobs?", default='60')

######################################## Parsing argument values  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
ModelName=ast.literal_eval(args.ModelName)
TrainSampleID=args.TrainSampleID
Patience=int(args.Patience)
TrainSampleSize=int(args.TrainSampleSize)
input_file_location=args.f
JobFlavour=args.JobFlavour
SubPause=int(args.SubPause)*60
RequestExtCPU=(args.RequestExtCPU=='Y')
Xmin,Xmax,Ymin,Ymax=float(args.Xmin),float(args.Xmax),float(args.Ymin),float(args.Ymax)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0 #We don't slice data if all values are set to zero simultaneousy (which is the default setting)
LocalSub=(args.LocalSub=='Y')
if LocalSub:
   time_int=0
else:
    time_int=10
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

#Establishing paths
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
TrainSampleOutputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
required_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MUTr1_'+TrainSampleID+'_TRACK_SEGMENTS.csv'


########################################     Phase 1 - Create compact source file    #########################################
print(UF.TimeStamp(),bcolors.BOLD+'Stage 0:'+bcolors.ENDC+' Preparing the source data...')

if os.path.isfile(required_file_location)==False or Mode=='RESET':
        print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
        data=pd.read_csv(input_file_location,
                    header=0,
                    usecols=[PM.Rec_Track_ID,PM.Rec_Track_Domain,
                            PM.x,PM.y,PM.z,PM.tx,PM.ty,
                            PM.MC_Track_ID,PM.MC_Event_ID])
        total_rows=len(data.axes[0])
        print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
        print(UF.TimeStamp(),'Removing unreconstructed hits...')
        data=data.dropna()
        final_rows=len(data.axes[0])
        print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
        data[PM.MC_Event_ID] = data[PM.MC_Event_ID].astype(str)
        data[PM.MC_Track_ID] = data[PM.MC_Track_ID].astype(str)
        data[PM.Rec_Track_ID] = data[PM.Rec_Track_ID].astype(str)
        data[PM.Rec_Track_Domain] = data[PM.Rec_Track_Domain].astype(str)
        data['Rec_Seg_ID'] = data[PM.Rec_Track_Domain] + '-' + data[PM.Rec_Track_ID]
        data['MC_Mother_Track_ID'] = data[PM.MC_Event_ID] + '-' + data[PM.MC_Track_ID]
        data=data.drop([PM.Rec_Track_ID],axis=1)
        data=data.drop([PM.Rec_Track_Domain],axis=1)
        data=data.drop([PM.MC_Event_ID],axis=1)
        data=data.drop([PM.MC_Track_ID],axis=1)
        compress_data=data.drop([PM.x,PM.y,PM.z,PM.tx,PM.ty],axis=1)
        compress_data['MC_Mother_Track_No']= compress_data['MC_Mother_Track_ID']
        compress_data=compress_data.groupby(by=['Rec_Seg_ID','MC_Mother_Track_ID'])['MC_Mother_Track_No'].count().reset_index()
        compress_data=compress_data.sort_values(['Rec_Seg_ID','MC_Mother_Track_No'],ascending=[1,0])
        compress_data.drop_duplicates(subset='Rec_Seg_ID',keep='first',inplace=True)
        data=data.drop(['MC_Mother_Track_ID'],axis=1)
        compress_data=compress_data.drop(['MC_Mother_Track_No'],axis=1)
        data=pd.merge(data, compress_data, how="left", on=['Rec_Seg_ID'])

        if SliceData:
             print(UF.TimeStamp(),'Slicing the data...')
             ValidEvents=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
             ValidEvents.drop([PM.x,PM.y,PM.z,PM.tx,PM.ty,'MC_Mother_Track_ID'],axis=1,inplace=True)
             ValidEvents.drop_duplicates(subset='Rec_Seg_ID',keep='first',inplace=True)
             data=pd.merge(data, ValidEvents, how="inner", on=['Rec_Seg_ID'])
             final_rows=len(data.axes[0])
             print(UF.TimeStamp(),'The sliced data has ',final_rows,' hits')

        output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MUTr1_'+TrainSampleID+'_TRACK_SEGMENTS.csv'
        print(UF.TimeStamp(),'Removing tracks which have less than',PM.MinHitsTrack,'hits...')
        track_no_data=data.groupby(['MC_Mother_Track_ID','Rec_Seg_ID'],as_index=False).count()
        track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty],axis=1)
        track_no_data=track_no_data.rename(columns={PM.x: "Rec_Seg_No"})
        new_combined_data=pd.merge(data, track_no_data, how="left", on=['Rec_Seg_ID','MC_Mother_Track_ID'])
        new_combined_data = new_combined_data[new_combined_data.Rec_Seg_No >= PM.MinHitsTrack]
        new_combined_data = new_combined_data.drop(["Rec_Seg_No"],axis=1)
        new_combined_data=new_combined_data.sort_values(['Rec_Seg_ID',PM.x],ascending=[1,1])
        grand_final_rows=len(new_combined_data.axes[0])
        print(UF.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
        new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
        new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
        new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
        new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
        new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
        new_combined_data.to_csv(output_file_location,index=False)
        data=new_combined_data[['Rec_Seg_ID','z']]
        print(UF.TimeStamp(),'Analysing the data sample in order to understand how many jobs to submit to HTCondor... ',bcolors.ENDC)
        data = data.groupby('Rec_Seg_ID')['z'].min()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
        data=data.reset_index()
        data = data.groupby('z')['Rec_Seg_ID'].count()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
        data=data.reset_index()
        data=data.sort_values(['z'],ascending=True)
        data['Sub_Sets']=np.ceil(data['Rec_Seg_ID']/PM.MaxSegments)
        data['Sub_Sets'] = data['Sub_Sets'].astype(int)
        data = data.values.tolist()
        print(UF.TimeStamp(), bcolors.OKGREEN+"The track segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
        Meta=UF.TrainingSampleMeta(TrainSampleID)
        Meta.IniTrackSeedMetaData(PM.MaxSLG,PM.MaxSTG,PM.MaxDOCA,PM.MaxAngle,data,PM.MaxSegments,PM.VetoMotherTrack,PM.MaxSeeds)
        Meta.UpdateStatus(1)
        print(UF.PickleOperations(TrainSampleOutputMeta,'w', Meta)[1])
        print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 0 has successfully completed'+bcolors.ENDC)
elif os.path.isfile(TrainSampleOutputMeta)==True:
    print(UF.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+TrainSampleOutputMeta+bcolors.ENDC)
    MetaInput=UF.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
MaxSLG=Meta.MaxSLG
MaxSTG=Meta.MaxSTG
MaxDOCA=Meta.MaxDOCA
MaxAngle=Meta.MaxAngle
JobSets=Meta.JobSets
MaxSegments=Meta.MaxSegments
MaxSeeds=Meta.MaxSeeds
VetoMotherTrack=Meta.VetoMotherTrack
TotJobs=0
for j in range(0,len(JobSets)):
          for sj in range(0,int(JobSets[j][2])):
              TotJobs+=1

# ########################################     Preset framework parameters    #########################################
FreshStart=True
Program=[]
exit()
#Defining handy functions to make the code little cleaner
def UpdateStatus(status):
    Meta.UpdateStatus(status)
    print(UF.PickleOperations(TrainSampleOutputMeta,'w', Meta)[1])
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
                          if _cnt>PM.SubPauseGap:
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
                      for bp in bad_pop:
                           UF.SubmitJobs2Condor(bp,program[status][5],RequestExtCPU,JobFlavour)
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
                      for bp in bad_pop:
                           UF.SubmitJobs2Condor(bp,program[status][5],RequestExtCPU,JobFlavour)
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
    HTCondorTag="SoftUsed == \"ANNDEA-MUTr1a-"+TrainSampleID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'MUTr1_'+TrainSampleID, ['MUTr1a','MUTr1b','MUTr1c','MUTr1d',TrainSampleID+'_MUTr_OUTPUT.pkl'], HTCondorTag)
    FreshStart=False
if Mode=='CLEANUP':
    Status=5
else:
    Status=int(args.ForceStatus)
################ Set the execution sequence for the script

exit()
###### Stage 0
prog_entry=[]
job_sets=[]
for i in range(len(JobSets)):
                job_sets.append(int(JobSets[i][2]))
TotJobs=0

if type(job_sets) is int:
                        TotJobs=job_sets
elif type(job_sets[0]) is int:
                        TotJobs=np.sum(job_sets)
elif type(job_sets[0][0]) is int:
                        for lp in job_sets:
                            TotJobs+=np.sum(lp)
prog_entry.append(' Sending hit cluster to the HTCondor, so the model assigns weights between hits')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','RawSeedsRes','MUTr1a','.csv',TrainSampleID,job_sets,'MUTr1a_GenerateRawSelectedSeeds_Sub.py'])
prog_entry.append([ " --MaxSegments ", " --MaxSLG "," --MaxSTG "," --VetoMotherTrack "])
prog_entry.append([MaxSegments, MaxSLG, MaxSTG,'"'+str(VetoMotherTrack)+'"'])
prog_entry.append(TotJobs)
prog_entry.append(LocalSub)
Program.append(prog_entry)
if Mode=='RESET':
   print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
#Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))
exit()
###### Stage 1
Program.append('Custom')


prog_entry=[]
job_sets=[]
JobSet=[]
for i in range(len(JobSets)):
         JobSet.append(int(JobSets[i][2]))
prog_entry.append(' Sending hit cluster to the HTCondor, so the reconstructed clusters can be merged along z-axis')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','hit_cluster_rec_z_set','RTr1b','.pkl',RecBatchID,job_sets,'RTr1b_LinkSegmentsZ_Sub.py'])
prog_entry.append([' --Z_ID_Max ',' --j ', ' --i '])
prog_entry.append([Zsteps,Ysteps,Xsteps])
prog_entry.append(Xsteps*Ysteps)
prog_entry.append(LocalSub)
Program.append(prog_entry)
if Mode=='RESET':
   print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
#Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))
###### Stage 2
prog_entry=[]
job_sets=Xsteps
prog_entry.append(' Sending hit cluster to the HTCondor, so the reconstructed clusters can be merged along y-axis')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','hit_cluster_rec_y_set','RTr1c','.pkl',RecBatchID,job_sets,'RTr1c_LinkSegmentsY_Sub.py'])
prog_entry.append([' --Y_ID_Max ', ' --i '])
prog_entry.append([Ysteps,Xsteps])
prog_entry.append(Xsteps)
prog_entry.append(LocalSub)
Program.append(prog_entry)
if Mode=='RESET':
   print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
#Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))

###### Stage 3
prog_entry=[]
job_sets=1
prog_entry.append(' Sending hit cluster to the HTCondor, so the reconstructed clusters can be merged along x-axis')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','hit_cluster_rec_x_set','RTr1d','.pkl',RecBatchID,job_sets,'RTr1d_LinkSegmentsX_Sub.py'])
prog_entry.append([' --X_ID_Max '])
prog_entry.append([Xsteps])
prog_entry.append(1)
prog_entry.append(True) #This part we can execute locally, no need for HTCondor
Program.append(prog_entry)
if Mode=='RESET':
   print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
#Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))

###### Stage 4
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

    elif Status==1:
       #Non standard processes (that don't follow the general pattern) have been coded here
        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.BOLD+'Stage 1:'+bcolors.ENDC+' Collecting and de-duplicating the results from stage 1')
        min_i=0
        for i in range(0,len(JobSets)): #//Temporarily measure to save space
                   test_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MUTr1a_'+TrainSampleID+'_SelectedSeeds_'+str(i)+'_'+str(0)+'_'+str(0)+'.csv'
                   if os.path.isfile(test_file_location):
                        min_i=max(0,i-1)
        with alive_bar(len(JobSets)-min_i,force_tty=True, title='Checking the results from HTCondor') as bar:
            for i in range(min_i,len(JobSets)): #//Temporarily measure to save space
                bar.text = f'-> Analysing set : {i}...'
                bar()
                Meta=UF.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')[0]
                MaxSLG=Meta.MaxSLG
                JobSets=Meta.JobSets
                if len(Meta.JobSets[i])>3:
                   Meta.JobSets[i]=Meta.JobSets[i][:4]
                   Meta.JobSets[i][3]=[]
                else:
                   Meta.JobSets[i].append([])
                for j in range(0,int(JobSets[i][2])):

                   output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MUTr1a_'+TrainSampleID+'_RawSeeds_'+str(i)+'_'+str(j)+'.csv'

                   if os.path.isfile(output_file_location)==False:
                      Meta.JobSets[j].append(0)
                      continue #Skipping because not all jobs necesseraly produce the required file (if statistics are too low)
                   else:
                    result=pd.read_csv(output_file_location,names = ['Segment_1','Segment_2', 'Seed_Type'])
                    Records=len(result)
                    print(UF.TimeStamp(),'Set',str(i),'and subset', str(j), 'contains', Records, 'seeds',bcolors.ENDC)
                    result["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(result['Segment_1'], result['Segment_2'])]
                    result.drop_duplicates(subset="Seed_ID",keep='first',inplace=True)
                    result.drop(result.index[result['Segment_1'] == result['Segment_2']], inplace = True)
                    result.drop(["Seed_ID"],axis=1,inplace=True)
                    Records_After_Compression=len(result)
                    if Records>0:
                      Compression_Ratio=int((Records_After_Compression/Records)*100)
                    else:
                      Compression_Ratio=0
                    print(UF.TimeStamp(),'Set',str(i),'and subset', str(j), 'compression ratio is ', Compression_Ratio, ' %',bcolors.ENDC)
                    fractions=int(math.ceil(Records_After_Compression/MaxSeeds))
                    Meta.JobSets[i][3].append(fractions)
                    for k in range(0,fractions):
                     new_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MUTr1a_'+TrainSampleID+'_SelectedSeeds_'+str(i)+'_'+str(j)+'_'+str(k)+'.csv'
                     result[(k*MaxSeeds):min(Records_After_Compression,((k+1)*MaxSeeds))].to_csv(new_output_file_location,index=False)
                print(UF.PickleOperations(TrainSampleOutputMeta,'w', Meta)[1])
        FreshStart=False
        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
        status=3
if Status==5:
    print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
    HTCondorTag="SoftUsed == \"ANNDEA-RTr1a-"+RecBatchID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'RTr1_'+RecBatchID, ['RTr1_'+RecBatchID,RecBatchID+'_RTr_OUTPUT.pkl'], HTCondorTag)
    for p in Program:
        if p!='Custom':
           print(UF.TimeStamp(),UF.ManageTempFolders(p,'Delete'))
    print(UF.TimeStamp(), bcolors.OKGREEN+"Reconstruction has been completed"+bcolors.ENDC)
    exit()
else:
    print(UF.TimeStamp(), bcolors.FAIL+"Reconstruction has not been completed as one of the processes has timed out or --ForceStatus!=0 option was chosen. Please run the script again (without Reset Mode)."+bcolors.ENDC)
    exit()



