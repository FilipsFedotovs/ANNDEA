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
import UtilityFunctions as UF #This is where we keep routine utility functions
import Parameters as PM #This is where we keep framework global parameters
import pandas as pd #We use Panda for a routine data processing
pd.options.mode.chained_assignment = None #Silence annoying warnings
import math #We use it for data manipulation
import numpy as np
import os
import time
from alive_progress import alive_bar
import argparse
import ast
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
parser.add_argument('--SubPause',help="How long to wait in minutes after submitting 10000 jobs?", default='60')
parser.add_argument('--SubGap',help="How long to wait in minutes after submitting 10000 jobs?", default='10000')
parser.add_argument('--LocalSub',help="Local submission?", default='N')
parser.add_argument('--RequestExtCPU',help="Would you like to request extra CPUs? How Many?", default=1)
parser.add_argument('--JobFlavour',help="Specifying the length of the HTCondor job wall time. Currently at 'workday' which is 8 hours.", default='workday')
parser.add_argument('--TrackID',help="What track name is used?", default='ANN_Track_ID')
parser.add_argument('--BrickID',help="What brick ID name is used?", default='ANN_Brick_ID')
parser.add_argument('--Mode', help='Script will continue from the last checkpoint, unless you want to start from the scratch, then type "Reset"',default='')
parser.add_argument('--ModelName',help="What  models would you like to use?", default="[]")
parser.add_argument('--Patience',help="How many checks to do before resubmitting the job?", default='15')
parser.add_argument('--RecBatchID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_Rec_Raw_UR.csv')
parser.add_argument('--Xmin',help="This option restricts data to only those events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option restricts data to only those events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option restricts data to only those events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option restricts data to only those events that have tracks with hits y-coordinates that are below this value", default='0')
parser.add_argument('--Log',help="Would you like to log the performance of this reconstruction? (Only available with MC data)", default='N')
parser.add_argument('--Acceptance',help="What is the ANN fit acceptance?", default='0.5')
parser.add_argument('--CalibrateAcceptance',help="Would you like to recalibrate the acceptance?", default='N')
parser.add_argument('--ReqMemory',help="Specifying the length of the HTCondor job walltime. Currently at 'workday' which is 8 hours.", default='2 GB')

######################################## Parsing argument values  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
RecBatchID=args.RecBatchID
Patience=int(args.Patience)
TrackID=args.TrackID
BrickID=args.BrickID
SubPause=int(args.SubPause)*60
SubGap=int(args.SubGap)
LocalSub=(args.LocalSub=='Y')
if LocalSub:
   time_int=0
else:
    time_int=10
JobFlavour=args.JobFlavour
RequestExtCPU=int(args.RequestExtCPU)
ReqMemory=args.ReqMemory
Xmin,Xmax,Ymin,Ymax=float(args.Xmin),float(args.Xmax),float(args.Ymin),float(args.Ymax)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0 #We don't slice data if all values are set to zero simultaneousy (which is the default setting)
ModelName=ast.literal_eval(args.ModelName)
Patience=int(args.Patience)
Acceptance=float(args.Acceptance)
CalibrateAcceptance=(args.CalibrateAcceptance=='Y')
initial_input_file_location=args.f
Log=args.Log=='Y'
Xmin,Xmax,Ymin,Ymax=float(args.Xmin),float(args.Xmax),float(args.Ymin),float(args.Ymax)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0 #We don't slice data if all values are set to zero simultaneousy (which is the default setting)
FreshStart=True
#Establishing paths
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
if ModelName[0]!='Blank':
    EOSsubModelMetaDIR=EOSsubDIR+'/'+'Models/'+ModelName[0]+'_Meta'
else:
    EOSsubModelMetaDIR=EOSsubDIR+'/'+'Models/'+ModelName[1]+'_Meta'
RecOutputMeta=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_info.pkl'
required_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1_'+RecBatchID+'_TRACK_SEGMENTS.csv'
required_eval_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/EUTr1_'+RecBatchID+'_TRACK_SEGMENTS.csv'
########################################     Phase 1 - Create compact source file    #########################################
print(UF.TimeStamp(),bcolors.BOLD+'Stage 0:'+bcolors.ENDC+' Preparing the source data...')

if Log and (os.path.isfile(required_eval_file_location)==False or Mode=='RESET'):
    if os.path.isfile(EOSsubModelMetaDIR)==False:
              print(UF.TimeStamp(), bcolors.FAIL+"Fail to proceed further as the model file "+EOSsubModelMetaDIR+ " has not been found..."+bcolors.ENDC)
              exit()
    else:
           print(UF.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+EOSsubModelMetaDIR+bcolors.ENDC)
           MetaInput=UF.PickleOperations(EOSsubModelMetaDIR,'r', 'N/A')
           Meta=MetaInput[0]
           MinHitsTrack=Meta.MinHitsTrack
    print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+initial_input_file_location+bcolors.ENDC)
    if BrickID=='':
        ColUse=[TrackID,PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.MC_Track_ID,PM.MC_Event_ID]
    else:
        ColUse=[TrackID,BrickID,PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.MC_Track_ID,PM.MC_Event_ID]
    
    data=pd.read_csv(initial_input_file_location,
                header=0,
                usecols=ColUse)
    if BrickID=='':
        data[BrickID]='D'
    total_rows=len(data)
    print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
    print(UF.TimeStamp(),'Removing unreconstructed hits...')
    data=data.dropna()
    final_rows=len(data)
    print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
    data[PM.MC_Event_ID] = data[PM.MC_Event_ID].astype(str)
    data[PM.MC_Track_ID] = data[PM.MC_Track_ID].astype(str)
    
    data[BrickID] = data[BrickID].astype(str)
    data[TrackID] = data[TrackID].astype(str)
    data['Rec_Seg_ID'] = data[TrackID] + '-' + data[BrickID]
    data['MC_Mother_Track_ID'] = data[PM.MC_Event_ID] + '-' + data[PM.MC_Track_ID]
    data=data.drop([TrackID],axis=1)
    data=data.drop([BrickID],axis=1)
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
    print(UF.TimeStamp(),'Removing tracks which have less than',MinHitsTrack,'hits...')
    track_no_data=data.groupby(['MC_Mother_Track_ID','Rec_Seg_ID'],as_index=False).count()
    track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty],axis=1)
    track_no_data=track_no_data.rename(columns={PM.x: "Rec_Seg_No"})
    new_combined_data=pd.merge(data, track_no_data, how="left", on=['Rec_Seg_ID','MC_Mother_Track_ID'])
    new_combined_data = new_combined_data[new_combined_data.Rec_Seg_No >= MinHitsTrack]
    new_combined_data = new_combined_data.drop(["Rec_Seg_No"],axis=1)
    new_combined_data=new_combined_data.sort_values(['Rec_Seg_ID',PM.x],ascending=[1,1])
    grand_final_rows=len(new_combined_data)
    print(UF.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
    new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
    new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
    new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
    new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
    new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
    new_combined_data.to_csv(required_eval_file_location,index=False)
    print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
    print(UF.TimeStamp(), bcolors.OKGREEN+"The track segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+required_eval_file_location+bcolors.ENDC)
    print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)

if os.path.isfile(required_file_location)==False or Mode=='RESET':
        if os.path.isfile(EOSsubModelMetaDIR)==False:
              print(UF.TimeStamp(), bcolors.FAIL+"Fail to proceed further as the model file "+EOSsubModelMetaDIR+ " has not been found..."+bcolors.ENDC)
              exit()
        else:
           print(UF.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+EOSsubModelMetaDIR+bcolors.ENDC)
           MetaInput=UF.PickleOperations(EOSsubModelMetaDIR,'r', 'N/A')
           Meta=MetaInput[0]
           MaxSLG=Meta.MaxSLG
           MaxSTG=Meta.MaxSTG
           MaxDOCA=Meta.MaxDOCA
           MaxAngle=Meta.MaxAngle
           MaxSegments=PM.MaxSegments
           MaxSeeds=PM.MaxSeeds
           VetoMotherTrack=PM.VetoMotherTrack
           MinHitsTrack=Meta.MinHitsTrack
        print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+initial_input_file_location+bcolors.ENDC)
        if BrickID=='':
            ColUse=[TrackID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
        else:
            ColUse=[TrackID,BrickID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
        data=pd.read_csv(initial_input_file_location,
                    header=0,
                    usecols=ColUse)
        if BrickID=='':
            data[BrickID]='D'
        total_rows=len(data.axes[0])
        print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
        print(UF.TimeStamp(),'Removing unreconstructed hits...')
        data=data.dropna()
        final_rows=len(data.axes[0])
        print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
        data[BrickID] = data[BrickID].astype(str)
        data[TrackID] = data[TrackID].astype(str)

        data['Rec_Seg_ID'] = data[TrackID] + '-' + data[BrickID]
        data=data.drop([TrackID],axis=1)
        data=data.drop([BrickID],axis=1)
        if SliceData:
             print(UF.TimeStamp(),'Slicing the data...')
             ValidEvents=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
             ValidEvents.drop([PM.x,PM.y,PM.z,PM.tx,PM.ty],axis=1,inplace=True)
             ValidEvents.drop_duplicates(subset="Rec_Seg_ID",keep='first',inplace=True)
             data=pd.merge(data, ValidEvents, how="inner", on=['Rec_Seg_ID'])
             final_rows=len(data.axes[0])
             print(UF.TimeStamp(),'The sliced data has ',final_rows,' hits')
        print(UF.TimeStamp(),'Removing tracks which have less than',MinHitsTrack,'hits...')
        track_no_data=data.groupby(['Rec_Seg_ID'],as_index=False).count()
        track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty],axis=1)
        track_no_data=track_no_data.rename(columns={PM.x: "Track_No"})
        new_combined_data=pd.merge(data, track_no_data, how="left", on=["Rec_Seg_ID"])
        new_combined_data = new_combined_data[new_combined_data.Track_No >= MinHitsTrack]
        new_combined_data = new_combined_data.drop(['Track_No'],axis=1)
        new_combined_data=new_combined_data.sort_values(['Rec_Seg_ID',PM.x],ascending=[1,1])
        grand_final_rows=len(new_combined_data.axes[0])
        print(UF.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
        new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
        new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
        new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
        new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
        new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
        new_combined_data.to_csv(required_file_location,index=False)
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
        print(UF.TimeStamp(), bcolors.OKGREEN+"The track segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+required_file_location+bcolors.ENDC)
        Meta=UF.TrainingSampleMeta(RecBatchID)
        Meta.IniTrackSeedMetaData(MaxSLG,MaxSTG,MaxDOCA,MaxAngle,data,MaxSegments,VetoMotherTrack,MaxSeeds,MinHitsTrack)
        Meta.UpdateStatus(0)
        print(UF.PickleOperations(RecOutputMeta,'w', Meta)[1])
        print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 0 has successfully completed'+bcolors.ENDC)
elif os.path.isfile(RecOutputMeta)==True:
    print(UF.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+RecOutputMeta+bcolors.ENDC)
    MetaInput=UF.PickleOperations(RecOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
MaxSLG=Meta.MaxSLG
MaxSTG=Meta.MaxSTG
MaxDOCA=Meta.MaxDOCA
MaxAngle=Meta.MaxAngle
JobSets=Meta.JobSets
MaxSegments=Meta.MaxSegments
MaxSeeds=Meta.MaxSeeds
VetoMotherTrack=Meta.VetoMotherTrack
MinHitsTrack=Meta.MinHitsTrack

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
                                    False,
                                    program[6])
         if len(bad_pop)>0:
               print(UF.TimeStamp(),bcolors.WARNING+'Autopilot status update: There are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
               if interval%max_interval_tolerance==0:
                  for bp in bad_pop:
                      UF.SubmitJobs2Condor(bp,program[5],RequestExtCPU,JobFlavour,ReqMemory)
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
                                    False,
                                    program[status][6])


        if len(bad_pop)==0:
             print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+' has successfully completed'+bcolors.ENDC)
             UpdateStatus(status+1)
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
                                    batch_sub,
                                    program[status][6])
                 print(UF.TimeStamp(),'Submitting jobs to HTCondor... ',bcolors.ENDC)
                 _cnt=0
                 for bp in bad_pop:
                          if _cnt>SubGap:
                              print(UF.TimeStamp(),'Pausing submissions for  ',str(int(SubPause/60)), 'minutes to relieve congestion...',bcolors.ENDC)
                              time.sleep(SubPause)
                              _cnt=0
                          UF.SubmitJobs2Condor(bp,program[status][5],RequestExtCPU,JobFlavour,ReqMemory)
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
            # if freshstart:
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
                           UF.SubmitJobs2Condor(bp,program[status][5],RequestExtCPU,JobFlavour,ReqMemory)
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

def UpdateStatus(status):
    Meta.UpdateStatus(status)
    print(UF.PickleOperations(RecOutputMeta,'w', Meta)[1])

if Mode=='RESET':
    print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
    HTCondorTag="SoftUsed == \"ANNDEA-EUTr1a-"+RecBatchID+"\""
    UF.EvalCleanUp(AFS_DIR, EOS_DIR, 'EUTr1a_'+RecBatchID, ['EUTr1a','EUTr1b'], HTCondorTag)
    HTCondorTag="SoftUsed == \"ANNDEA-RUTr1a-"+RecBatchID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'RUTr1a_'+RecBatchID, ['RUTr1a',RecBatchID+'_REC_LOG.csv'], HTCondorTag)
    FreshStart=False
    UpdateStatus(0)
    Status=0
else:
    print(UF.TimeStamp(),'Analysing the current script status...',bcolors.ENDC)
    Status=Meta.Status[-1]

print(UF.TimeStamp(),'Current status is ',Status,bcolors.ENDC)
################ Set the execution sequence for the script
Program=[]

if Log:
    ###### Stage 0
    prog_entry=[]
    job_sets=[]
    prog_entry.append(' Sending eval seeds to HTCondor...')
    print(UF.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+initial_input_file_location+bcolors.ENDC)
    data=pd.read_csv(required_eval_file_location,header=0,usecols=['Rec_Seg_ID'])
    print(UF.TimeStamp(),'Analysing data... ',bcolors.ENDC)
    data.drop_duplicates(subset="Rec_Seg_ID",keep='first',inplace=True)  #Keeping only starting hits for each track record (we do not require the full information about track in this script)
    Records=len(data.axes[0])
    Sets=int(np.ceil(Records/MaxSegments))
    prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TEST_SET/','RawSeedsRes','EUTr1a','.csv',RecBatchID,Sets,'EUTr1a_GenerateRawSelectedSeeds_Sub.py'])
    prog_entry.append([" --MaxSegments ", " --VetoMotherTrack "])
    prog_entry.append([MaxSegments, '"'+str(VetoMotherTrack)+'"'])
    prog_entry.append(Sets)
    prog_entry.append(LocalSub)
    prog_entry.append(['',''])
    if Mode=='RESET':
        print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
    #Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
    print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))
    Program.append(prog_entry)
    # ###### Stage 1
    Program.append('Custom - PickE')

else:
    UpdateStatus(0)
    Status=0

if Mode=='CLEANUP':
    UpdateStatus(19)
    Status=19

# ###### Stage 2
prog_entry=[]
job_sets=[]
JobSet=[]
for i in range(len(JobSets)):
    JobSet.append(int(JobSets[i][2]))
TotJobs=0

if type(JobSet) is int:
            TotJobs=JobSet
elif type(JobSet[0]) is int:
            TotJobs=np.sum(JobSet)
elif type(JobSet[0][0]) is int:
            for lp in JobSet:
                TotJobs+=np.sum(lp)
prog_entry.append(' Sending tracks to the HTCondor, so track segment combinations can be formed...')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','RawSeedsRes','RUTr1a','.csv',RecBatchID,JobSet,'RUTr1a_GenerateRawSelectedSeeds_Sub.py'])
prog_entry.append([ " --MaxSegments ", " --MaxSLG "," --MaxSTG "])
prog_entry.append([MaxSegments, MaxSLG, MaxSTG])
prog_entry.append(np.sum(JobSet))
prog_entry.append(LocalSub)
prog_entry.append([" --PlateZ ",JobSets])
if Mode=='RESET':
        print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
    #Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))
Program.append(prog_entry)

Program.append('Custom - PickR')

####### Stage 4
for md in ModelName:
    Program.append(md)

Program.append('Custom - RemoveOverlap')

Program.append('Custom - PerformMerging')

Program.append('Custom - TrackMapping')
while Status<len(Program):
    if Program[Status][:6]!='Custom' and (Program[Status] in ModelName)==False:
        #Standard process here
        Result=StandardProcess(Program,Status,FreshStart)
        if Result[0]:
             FreshStart=Result[1]
        else:
             Status=20
             break
    elif Program[Status]=='Custom - PickE':
        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.BOLD+'Status ',Status,': Collecting and de-duplicating the results from previous stage')
        print(UF.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+initial_input_file_location+bcolors.ENDC)
        data=pd.read_csv(required_eval_file_location,header=0,usecols=['Rec_Seg_ID'])
        print(UF.TimeStamp(),'Analysing data... ',bcolors.ENDC)
        data.drop_duplicates(subset="Rec_Seg_ID",keep='first',inplace=True)  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
        Records=len(data.axes[0])
        Sets=int(np.ceil(Records/MaxSegments))
        with alive_bar(Sets,force_tty=True, title='Analysing data...') as bar:
            for i in range(Sets): #//Temporarily measure to save space
                    bar.text = f'-> Analysing set : {i}...'
                    bar()
                    if i==0:
                       output_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/Temp_EUTr1a'+'_'+RecBatchID+'_'+str(0)+'/EUTr1a_'+RecBatchID+'_RawSeeds_'+str(i)+'.csv'
                       result=pd.read_csv(output_file_location,names = ['Segment_1','Segment_2'])
                       print(UF.TimeStamp(),'Set',str(i), 'contains', len(result), 'seeds',bcolors.ENDC)
                    else:
                        output_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/Temp_EUTr1a'+'_'+RecBatchID+'_'+str(0)+'/EUTr1a_'+RecBatchID+'_RawSeeds_'+str(i)+'.csv'
                        new_result=pd.read_csv(output_file_location,names = ['Segment_1','Segment_2'])
                        print(UF.TimeStamp(),'Set',str(i), 'contains', len(new_result), 'seeds',bcolors.ENDC)
                        result=pd.concat([result,new_result])

        Records=len(result)
        result["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(result['Segment_1'], result['Segment_2'])]
        result.drop_duplicates(subset="Seed_ID",keep='first',inplace=True)
        result.drop(result.index[result['Segment_1'] == result['Segment_2']], inplace = True)
        result.drop(["Seed_ID"],axis=1,inplace=True)
        Records_After_Compression=len(result)
        if Records>0:
                      Compression_Ratio=int((Records_After_Compression/Records)*100)
        else:
                      Compression_Ratio=0
        print(UF.TimeStamp(),'Set',str(i), 'compression ratio is ', Compression_Ratio, ' %',bcolors.ENDC)
        new_output_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/EUTr1b_'+RecBatchID+'_SEED_TRUTH_COMBINATIONS.csv'
        result.to_csv(new_output_file_location,index=False)
        eval_no=len(result)
        rec_data=pd.read_csv(required_file_location,header=0,
                    usecols=['Rec_Seg_ID'])
        rec_data.drop_duplicates(subset="Rec_Seg_ID",keep='first',inplace=True)
        rec_data.drop_duplicates(keep='first',inplace=True)
        rec_no=len(rec_data)
        rec_no=(rec_no**2)-rec_no-eval_no
        UF.LogOperations(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv', 'w', [['Step_No','Step_Desc','Fake_Seeds','Truth_Seeds','Precision','Recall'],[1,'Initial Sampling',rec_no,eval_no,eval_no/(rec_no+eval_no),1.0]])
        print(UF.TimeStamp(), bcolors.OKGREEN+"The process log has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv'+bcolors.ENDC)
        FreshStart=False
        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage',Status,' has successfully completed'+bcolors.ENDC)
        UpdateStatus(Status+1)
    elif Program[Status]=='Custom - PickR':
        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.BOLD+'Stage '+str(Status)+':'+bcolors.ENDC+' Collecting and de-duplicating the results from stage 2')
        min_i=0
        #for i in range(0,len(JobSets)): #//Temporarily measure to save space || Update 13/08/23 - I have commented it out as it creates more problems than solves it
        #           test_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RUTr1a'+'_'+RecBatchID+'_'+str(i)+'/RUTr1a_'+RecBatchID+'_SelectedSeeds_'+str(i)+'_'+str(0)+'_'+str(0)+'.csv'
        #           if os.path.isfile(test_file_location):
        #                min_i=max(0,i-1)
        print(UF.TimeStamp(),'Analysing the data sample in order to understand how many jobs to submit to HTCondor... ',bcolors.ENDC)
        data=pd.read_csv(required_file_location,header=0,
                    usecols=['z','Rec_Seg_ID'])
        data = data.groupby('Rec_Seg_ID')['z'].min()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
        data=data.reset_index()
        data = data.groupby('z')['Rec_Seg_ID'].count()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
        data=data.reset_index()
        data=data.sort_values(['z'],ascending=True)
        data['Sub_Sets']=np.ceil(data['Rec_Seg_ID']/PM.MaxSegments)
        data['Sub_Sets'] = data['Sub_Sets'].astype(int)
        JobSets = data.values.tolist()
        with alive_bar(len(JobSets)-min_i,force_tty=True, title='Checking the results from HTCondor') as bar:
            for i in range(min_i,len(JobSets)): #//Temporarily measure to save space
                bar.text = f'-> Analysing set : {i}...'
                bar()
                Meta=UF.PickleOperations(RecOutputMeta,'r', 'N/A')[0]
                MaxSLG=Meta.MaxSLG
                JobSets=Meta.JobSets
                if len(Meta.JobSets[i])>3:
                   Meta.JobSets[i]=Meta.JobSets[i][:4]
                   Meta.JobSets[i][3]=[]
                else:
                   Meta.JobSets[i].append([])
                for j in range(0,int(JobSets[i][2])):
                   output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RUTr1a'+'_'+RecBatchID+'_'+str(i)+'/RUTr1a_'+RecBatchID+'_RawSeeds_'+str(i)+'_'+str(j)+'.csv'

                   if os.path.isfile(output_file_location)==False:
                      Meta.JobSets[j].append(0)
                      continue #Skipping because not all jobs necessarily produce the required file (if statistics are too low)
                   else:
                    result=pd.read_csv(output_file_location,names = ['Segment_1','Segment_2'])
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
                     new_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RUTr1a'+'_'+RecBatchID+'_'+str(i)+'/RUTr1a_'+RecBatchID+'_SelectedSeeds_'+str(i)+'_'+str(j)+'_'+str(k)+'.csv'
                     result[(k*MaxSeeds):min(Records_After_Compression,((k+1)*MaxSeeds))].to_csv(new_output_file_location,index=False)
                print(UF.PickleOperations(RecOutputMeta,'w', Meta)[1])
        if Log:
         try:
             print(UF.TimeStamp(),'Initiating the logging...')
             eval_data_file=EOS_DIR+'/ANNDEA/Data/TEST_SET/EUTr1b_'+RecBatchID+'_SEED_TRUTH_COMBINATIONS.csv'
             eval_data=pd.read_csv(eval_data_file,header=0,usecols=['Segment_1','Segment_2'])
             eval_data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(eval_data['Segment_1'], eval_data['Segment_2'])]
             eval_data.drop(['Segment_1'],axis=1,inplace=True)
             eval_data.drop(['Segment_2'],axis=1,inplace=True)
             eval_no=0
             rec_no=0
             with alive_bar(len(JobSets),force_tty=True, title='Preparing data for the log...') as bar:
                 for i in range(0,len(Meta.JobSets)):
                    bar()
                    rec=None
                    for j in range(0,int(Meta.JobSets[i][2])):
                        for k in range(0,Meta.JobSets[i][3][j]):
                          new_input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RUTr1a'+'_'+RecBatchID+'_'+str(i)+'/RUTr1a_'+RecBatchID+'_SelectedSeeds_'+str(i)+'_'+str(j)+'_'+str(k)+'.csv'
                          if os.path.isfile(new_input_file_location)==False:
                                break
                          else:
                             rec_new=pd.read_csv(new_input_file_location,usecols = ['Segment_1','Segment_2'])
                             rec_new["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(rec_new['Segment_1'], rec_new['Segment_2'])]
                             rec_new.drop(['Segment_1'],axis=1,inplace=True)
                             rec_new.drop(['Segment_2'],axis=1,inplace=True)
                             rec = pd.concat([rec, rec_new], ignore_index=True)
                             rec.drop_duplicates(subset="Seed_ID",keep='first',inplace=True)
                    try:
                        rec_eval=pd.merge(eval_data, rec, how="inner", on=['Seed_ID'])

                        eval_no+=len(rec_eval)
                        rec_no+=(len(rec)-len(rec_eval))
                    except:
                        continue
             UF.LogOperations(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv', 'a', [[2,'SLG and STG cuts',rec_no,eval_no,eval_no/(rec_no+eval_no),eval_no/len(eval_data)]])
             print(UF.TimeStamp(), bcolors.OKGREEN+"The log has been created successfully at "+bcolors.ENDC, bcolors.OKBLUE+EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv'+bcolors.ENDC)
         except:
             print(UF.TimeStamp(), bcolors.WARNING+'Log creation has failed'+bcolors.ENDC)
        FreshStart=False
        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(Status)+' has successfully completed'+bcolors.ENDC)
        UpdateStatus(Status+1)
    elif Program[Status]=='Custom - RemoveOverlap':
        input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1c_'+RecBatchID+'_Fit_Seeds.pkl'
        print(UF.TimeStamp(), "Loading the fit track seeds from the file",bcolors.OKBLUE+input_file_location+bcolors.ENDC)
        base_data=UF.PickleOperations(input_file_location,'r','N/A')[0]
        print(UF.TimeStamp(), bcolors.OKGREEN+"Loading is successful, there are "+str(len(base_data))+" fit seeds..."+bcolors.ENDC)
        with alive_bar(len(base_data),force_tty=True, title="Stripping non-z information from seeds...") as bar:
            for tr in range(len(base_data)):
                bar()
                for t in range(len(base_data[tr].Hits)):
                    for h in range(len(base_data[tr].Hits[t])):

                        base_data[tr].Hits[t][h]=base_data[tr].Hits[t][h][2] #Remove scaling factors
        base_data=[tr for tr in base_data if tr.Fit >= Acceptance]
        print(UF.TimeStamp(), bcolors.OKGREEN+"The refining was successful, "+str(len(base_data))+" track seeds remain..."+bcolors.ENDC)
        output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1c_'+RecBatchID+'_Fit_Filtered_Seeds.pkl'
        print(UF.PickleOperations(output_file_location,'w',base_data)[0])
        #no_iter=int(math.ceil(float(len(base_data)/float(MaxSegments))))
        UpdateStatus(Status+1)
    elif Program[Status]=='Custom - TrackMapping':
                raw_name=initial_input_file_location[:-4]
                for l in range(len(raw_name)-1,0,-1):
                    if raw_name[l]=='/':
                        print(l,raw_name)
                        break
                raw_name=raw_name[l+1:]
                final_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+raw_name+'_'+RecBatchID+'_MERGED.csv'
                print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
                print(UF.TimeStamp(),bcolors.BOLD+'Stage '+str(Status)+':'+bcolors.ENDC+' Taking the list of seeds previously generated by Stage '+str(Status-1)+' and mapping them to the input data')
                print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+initial_input_file_location+bcolors.ENDC)
                required_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1_'+RecBatchID+'_TRACK_SEGMENTS.csv'
                data=pd.read_csv(args.f,header=0)
                if BrickID=='':
                    ColUse=[PM.Hit_ID,TrackID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
                else:
                    ColUse=[PM.Hit_ID,TrackID,BrickID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
                data=data[ColUse]
                if BrickID=='':
                    data[BrickID]='D'
               
                print(UF.TimeStamp(),'Loading mapped data from',bcolors.OKBLUE+EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1e_'+RecBatchID+'_Union_Tracks.csv'+bcolors.ENDC)
                map_data=pd.read_csv(EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1e_'+RecBatchID+'_Union_Tracks.csv',header=0)
                total_rows=len(data.axes[0])
                print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
                print(UF.TimeStamp(),'Removing unreconstructed hits...')
                data.dropna(subset=[TrackID],inplace=True)
                final_rows=len(data)
                print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
                data[TrackID] = data[TrackID].astype(str)
                data[BrickID] = data[BrickID].astype(str)
                if os.path.isfile(EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1e_'+RecBatchID+'_Mapped_Tracks_Temp.csv')==False:
                    data['Rec_Seg_ID'] = data[TrackID] + '-' + data[BrickID]
                    print(UF.TimeStamp(),'Resolving duplicated hits...')
                    selected_combined_data=pd.merge(data, map_data, how="left", left_on=["Rec_Seg_ID"], right_on=['Old_Track_ID'])
                    Hit_Map_Stats=selected_combined_data[['Temp_Track_Quarter','Temp_Track_ID',PM.z,PM.Hit_ID]] #Calculating the stats
                    Hit_Map_Stats=Hit_Map_Stats.groupby(['Temp_Track_Quarter','Temp_Track_ID']).agg({PM.z:pd.Series.nunique,PM.Hit_ID: pd.Series.nunique}).reset_index() #Calculate the number fo unique plates and hits
                    Ini_No_Tracks=len(Hit_Map_Stats)
                    print(UF.TimeStamp(),bcolors.WARNING+'The initial number of tracks is '+ str(Ini_No_Tracks)+bcolors.ENDC)
                    Hit_Map_Stats=Hit_Map_Stats.rename(columns={PM.z: "No_Plates",PM.Hit_ID:"No_Hits"}) #Renaming the columns so they don't interfere once we join it back to the hit map
                    Hit_Map_Stats=Hit_Map_Stats[Hit_Map_Stats.No_Plates >= PM.MinHitsTrack]
                    Prop_No_Tracks=len(Hit_Map_Stats)
                    print(UF.TimeStamp(),bcolors.WARNING+'After dropping single hit tracks, left '+ str(Prop_No_Tracks)+' tracks...'+bcolors.ENDC)
                    selected_combined_data=pd.merge(selected_combined_data,Hit_Map_Stats,how='inner',on = ['Temp_Track_Quarter','Temp_Track_ID']) #Join back to the hit map
                    Good_Tracks=selected_combined_data[selected_combined_data.No_Plates == selected_combined_data.No_Hits] #For all good tracks the number of hits matches the number of plates, we won't touch them
                    Good_Tracks=Good_Tracks[['Temp_Track_Quarter','Temp_Track_ID',PM.Hit_ID]] #Just strip off the information that we don't need anymore
                    Bad_Tracks=selected_combined_data[selected_combined_data.No_Plates < selected_combined_data.No_Hits] #These are the bad guys. We need to remove this extra hits
                    Bad_Tracks=Bad_Tracks[['Temp_Track_Quarter','Temp_Track_ID',PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.Hit_ID]]
                    #Id the problematic plates
                    Bad_Tracks_Stats=Bad_Tracks[['Temp_Track_Quarter','Temp_Track_ID',PM.z,PM.Hit_ID]]
                    Bad_Tracks_Stats=Bad_Tracks_Stats.groupby(['Temp_Track_Quarter','Temp_Track_ID',PM.z]).agg({PM.Hit_ID: pd.Series.nunique}).reset_index() #Which plates have double hits?
                    Bad_Tracks_Stats=Bad_Tracks_Stats.rename(columns={PM.Hit_ID: "Problem"}) #Renaming the columns, so they don't interfere once we join it back to the hit map
                    Bad_Tracks=pd.merge(Bad_Tracks,Bad_Tracks_Stats,how='inner',on = ['Temp_Track_Quarter','Temp_Track_ID',PM.z])
                    Bad_Tracks.sort_values(['Temp_Track_Quarter','Temp_Track_ID',PM.z],ascending=[0,0,1],inplace=True)
                    Bad_Tracks_Head=Bad_Tracks[['Temp_Track_Quarter','Temp_Track_ID']]
                    Bad_Tracks_Head.drop_duplicates(inplace=True)
                    Bad_Tracks_List=Bad_Tracks.values.tolist() #I find it is much easier to deal with tracks in list format when it comes to fitting
                    Bad_Tracks_Head=Bad_Tracks_Head.values.tolist()
                    Bad_Track_Pool=[]
                    #Bellow we build the track representatation that we can use to fit slopes
                    with alive_bar(len(Bad_Tracks_Head),force_tty=True, title='Building track representations...') as bar:
                                for bth in Bad_Tracks_Head:
                                   bar()
                                   bth.append([])
                                   bt=0
                                   trigger=False
                                   while bt<(len(Bad_Tracks_List)):
                                       if (bth[0]==Bad_Tracks_List[bt][0] and bth[1]==Bad_Tracks_List[bt][1]):
                                          if Bad_Tracks_List[bt][8]==1: #We only build polynomials for hits in a track that do not have duplicates - these are 'trusted hits'
                                             bth[2].append(Bad_Tracks_List[bt][2:-2])
                                          del Bad_Tracks_List[bt]
                                          bt-=1
                                          trigger=True
                                       elif trigger:
                                           break
                                       else:
                                           continue
                                       bt+=1

                    with alive_bar(len(Bad_Tracks_Head),force_tty=True, title='Fitting the tracks...') as bar:
                     for bth in Bad_Tracks_Head:
                       bar()
                       if len(bth[2])==1: #Only one trusted hit - In these cases whe we take only tx and ty slopes of the single base track. Polynomial of the first degree and the equations of the line are x=ax+tx*z and y=ay+ty*z
                           x=bth[2][0][0]
                           z=bth[2][0][2]
                           tx=bth[2][0][3]
                           ax=x-tx*z
                           bth.append(ax) #Append x intercept
                           bth.append(tx) #Append x slope
                           bth.append(0) #Append a placeholder slope (for polynomial case)
                           y=bth[2][0][1]
                           ty=bth[2][0][4]
                           ay=y-ty*z
                           bth.append(ay) #Append x intercept
                           bth.append(ty) #Append x slope
                           bth.append(0) #Append a placeholder slope (for polynomial case)
                           del(bth[2])
                       elif len(bth[2])==2: #Two trusted hits - In these cases whe we fit a polynomial of the first degree and the equations of the line are x=ax+tx*z and y=ay+ty*z
                           x,y,z=[],[],[]
                           x=[bth[2][0][0],bth[2][1][0]]
                           y=[bth[2][0][1],bth[2][1][1]]
                           z=[bth[2][0][2],bth[2][1][2]]
                           tx=np.polyfit(z,x,1)[0]
                           ax=np.polyfit(z,x,1)[1]
                           ty=np.polyfit(z,y,1)[0]
                           ay=np.polyfit(z,y,1)[1]
                           bth.append(ax) #Append x intercept
                           bth.append(tx) #Append x slope
                           bth.append(0) #Append a placeholder slope (for polynomial case)
                           bth.append(ay) #Append x intercept
                           bth.append(ty) #Append x slope
                           bth.append(0) #Append a placeholder slope (for polynomial case)
                           del(bth[2])
                       elif len(bth[2])==0:
                           del(bth)
                           continue
                       else: #Three pr more trusted hits - In these cases whe we fit a polynomial of the second degree and the equations of the line are x=ax+(t1x*z)+(t2x*z*z) and y=ay+(t1y*z)+(t2y*z*z)
                           x,y,z=[],[],[]
                           for i in bth[2]:
                               x.append(i[0])
                           for j in bth[2]:
                               y.append(j[1])
                           for k in bth[2]:
                               z.append(k[2])
                           t2x=np.polyfit(z,x,2)[0]
                           t1x=np.polyfit(z,x,2)[1]
                           ax=np.polyfit(z,x,2)[2]
                           t2y=np.polyfit(z,y,2)[0]
                           t1y=np.polyfit(z,y,2)[1]
                           ay=np.polyfit(z,y,2)[2]
                           bth.append(ax) #Append x intercept
                           bth.append(t1x) #Append x slope
                           bth.append(t2x) #Append a placeholder slope (for polynomial case)
                           bth.append(ay) #Append x intercept
                           bth.append(t1y) #Append x slope
                           bth.append(t2y) #Append a placeholder slope (for polynomial case)
                           del(bth[2])

                    #Once we get coefficients for all tracks we convert them back to Pandas dataframe and join back to the data
                    Bad_Tracks_Head=pd.DataFrame(Bad_Tracks_Head, columns = ['Temp_Track_Quarter','Temp_Track_ID','ax','t1x','t2x','ay','t1y','t2y'])
                    print(UF.TimeStamp(),'Removing problematic hits...')
                    Bad_Tracks=pd.merge(Bad_Tracks,Bad_Tracks_Head,how='inner',on = ['Temp_Track_Quarter','Temp_Track_ID'])
                    print(UF.TimeStamp(),'Calculating x and y coordinates of the fitted line for all plates in the track...')
                    #Calculating x and y coordinates of the fitted line for all plates in the track
                    Bad_Tracks['new_x']=Bad_Tracks['ax']+(Bad_Tracks[PM.z]*Bad_Tracks['t1x'])+((Bad_Tracks[PM.z]**2)*Bad_Tracks['t2x'])
                    Bad_Tracks['new_y']=Bad_Tracks['ay']+(Bad_Tracks[PM.z]*Bad_Tracks['t1y'])+((Bad_Tracks[PM.z]**2)*Bad_Tracks['t2y'])
                    #Calculating how far hits deviate from the fit polynomial
                    print(UF.TimeStamp(),'Calculating how far hits deviate from the fit polynomial...')
                    Bad_Tracks['d_x']=Bad_Tracks[PM.x]-Bad_Tracks['new_x']
                    Bad_Tracks['d_y']=Bad_Tracks[PM.y]-Bad_Tracks['new_y']
                    Bad_Tracks['d_r']=Bad_Tracks['d_x']**2+Bad_Tracks['d_y']**2
                    Bad_Tracks['d_r'] = Bad_Tracks['d_r'].astype(float)
                    Bad_Tracks['d_r']=np.sqrt(Bad_Tracks['d_r']) #Absolute distance
                    Bad_Tracks=Bad_Tracks[['Temp_Track_Quarter','Temp_Track_ID',PM.z,PM.Hit_ID,'d_r']]
                    #Sort the tracks and their hits by Track ID, Plate and distance to the perfect line
                    print(UF.TimeStamp(),'Sorting the tracks and their hits by Track ID, Plate and distance to the perfect line...')
                    Bad_Tracks.sort_values(['Temp_Track_Quarter','Temp_Track_ID',PM.z,'d_r'],ascending=[0,0,1,1],inplace=True)
                    before=len(Bad_Tracks)
                    print(UF.TimeStamp(),'Before de-duplicattion we had ',before,' hits involving problematic tracks.')
                    #If there are two hits per plate we will keep the one which is closer to the line
                    Bad_Tracks.drop_duplicates(subset=['Temp_Track_Quarter','Temp_Track_ID',PM.z],keep='first',inplace=True)
                    after=len(Bad_Tracks)
                    print(UF.TimeStamp(),'Now their number was dropped to ',after,' hits.')
                    Bad_Tracks=Bad_Tracks[['Temp_Track_Quarter','Temp_Track_ID',PM.Hit_ID]]
                    Good_Tracks=pd.concat([Good_Tracks,Bad_Tracks]) #Combine all ANNDEA tracks together
                    Good_Tracks.to_csv(EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1e_'+RecBatchID+'_Mapped_Tracks_Temp.csv',index=False)
                    data.drop(["Rec_Seg_ID"],axis=1,inplace=True)
                else:
                    Good_Tracks=pd.read_csv(EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1e_'+RecBatchID+'_Mapped_Tracks_Temp.csv')
                print(UF.TimeStamp(),'Mapping data...')
                data=pd.read_csv(args.f,header=0)
                new_combined_data=pd.merge(data, Good_Tracks, how="left", on=[PM.Hit_ID])
                if BrickID!='':
            
                    new_combined_data['Temp_Track_Quarter'] = new_combined_data['Temp_Track_Quarter'].fillna(new_combined_data[BrickID])
                else:
                    new_combined_data['Temp_Track_Quarter'] = new_combined_data['Temp_Track_Quarter'].fillna('D')
                new_combined_data['Temp_Track_ID'] = new_combined_data['Temp_Track_ID'].fillna(new_combined_data[TrackID])

                new_combined_data=new_combined_data.rename(columns={'Temp_Track_Quarter': RecBatchID+'_Brick_ID','Temp_Track_ID': RecBatchID+'_Track_ID'})
                new_combined_data.to_csv(final_output_file_location,index=False)
                print(UF.TimeStamp(), bcolors.OKGREEN+"The merged track data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+final_output_file_location+bcolors.ENDC)
                print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
                UpdateStatus(Status+1)
    elif Program[Status]=='Custom - PerformMerging':
         print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
         print(UF.TimeStamp(),bcolors.BOLD+'Stage '+str(Status)+':'+bcolors.ENDC+' Taking the list of seeds previously generated by Stage 2, converting them into Emulsion Objects and doing more rigorous selection')
         print(UF.TimeStamp(), "Starting the script from the scratch")
         input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1c_'+RecBatchID+'_Fit_Filtered_Seeds.pkl'
         print(UF.TimeStamp(), "Loading fit track seeds from the file",bcolors.OKBLUE+input_file_location+bcolors.ENDC)
         base_data=UF.PickleOperations(input_file_location,'r','N/A')[0]
         print(UF.TimeStamp(), "Stripping off the seeds with low acceptance...")
         base_data=[tr for tr in base_data if tr.Fit >= Acceptance]
         print(UF.TimeStamp(), 'Ok starting the final merging of the remaining tracks')
         InitialDataLength=len(base_data)
         if CalibrateAcceptance:
            print(UF.TimeStamp(),'Calibrating the acceptance...')
            eval_data_file=EOS_DIR+'/ANNDEA/Data/TEST_SET/EUTr1b_'+RecBatchID+'_SEED_TRUTH_COMBINATIONS.csv'
            eval_data=pd.read_csv(eval_data_file,header=0,usecols=['Segment_1','Segment_2'])
            eval_data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(eval_data['Segment_1'], eval_data['Segment_2'])]
            eval_data.drop(['Segment_1'],axis=1,inplace=True)
            eval_data.drop(['Segment_2'],axis=1,inplace=True)
            eval_data['True']=1
            csv_out=[]
            for Tr in base_data:
                  csv_out.append([Tr.Header[0],Tr.Header[1],Tr.Fit])
            rec_data = pd.DataFrame(csv_out, columns = ['Segment_1','Segment_2','Fit'])
            rec_data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(rec_data['Segment_1'], rec_data['Segment_2'])]
            rec_data.drop(['Segment_1'],axis=1,inplace=True)
            rec_data.drop(['Segment_2'],axis=1,inplace=True)
            combined_data=pd.merge(rec_data,eval_data,how='left',on='Seed_ID')
            combined_data=combined_data.fillna(0)
            combined_data.drop(['Seed_ID'],axis=1,inplace=True)
            print(combined_data)
            TP = combined_data['True'].sum()
            P = combined_data['True'].count()
            Min_Acceptance=round(combined_data['Fit'].min(),2)
            FP=P-TP
            Ini_Precision=TP/P
            F1=(2*(Ini_Precision))/(Ini_Precision+1.0)
            iterations=int((1.0-Min_Acceptance)/0.01)
            for i in range(1,iterations):
                cut_off=Min_Acceptance+(i*0.01)
                print('Cutoff at:',cut_off)
                cut_data=combined_data.drop(combined_data.index[combined_data['Fit'] < cut_off])
                tp = cut_data['True'].sum()
                p=cut_data['True'].count()
                precision=tp/p
                recall=tp/TP
                f1=(2*(precision*recall))/(precision+recall)
                print('Cutoff at:',cut_off,'; Precision:', precision, '; Recall:', recall, '; F1:', f1)
            exit()
         SeedCounter=0
         SeedCounterContinue=True
         with alive_bar(len(base_data),force_tty=True, title='Merging the track segments...') as bar:
             while SeedCounterContinue:
                 if SeedCounter==len(base_data):
                                   SeedCounterContinue=False
                                   break
                 SubjectSeed=base_data[SeedCounter]


                 for ObjectSeed in base_data[SeedCounter+1:]:
                          if MaxSLG>=0:
                            if SubjectSeed.InjectDistantTrackSeed(ObjectSeed):
                                base_data.pop(base_data.index(ObjectSeed))
                          else:
                            if SubjectSeed.InjectTrackSeed(ObjectSeed):
                                base_data.pop(base_data.index(ObjectSeed))
                 SeedCounter+=1
                 bar()
         print(str(InitialDataLength), "segment pairs from different files were merged into", str(len(base_data)), 'tracks...')
         for v in range(0,len(base_data)):
             base_data[v].AssignANNTrUID(v)

         output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1e_'+RecBatchID+'_Union_Tracks.pkl'
         output_csv_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1e_'+RecBatchID+'_Union_Tracks.csv'
         csv_out=[['Old_Track_ID','Temp_Track_Quarter','Temp_Track_ID']]
         for Tr in base_data:
             for TH in Tr.Header:
                 csv_out.append([TH,RecBatchID,Tr.UTrID])

         print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
         print(UF.TimeStamp(), "Saving the results into the file",bcolors.OKBLUE+output_csv_location+bcolors.ENDC)
         UF.LogOperations(output_csv_location,'w', csv_out)
         print(UF.TimeStamp(), "Saving the results into the file",bcolors.OKBLUE+output_file_location+bcolors.ENDC)
         print(UF.PickleOperations(output_file_location,'w',base_data)[1])
         if args.Log=='Y':
            print(UF.TimeStamp(),'Initiating the logging...')
            eval_data_file=EOS_DIR+'/ANNDEA/Data/TEST_SET/EUTr1b_'+RecBatchID+'_SEED_TRUTH_COMBINATIONS.csv'
            eval_data=pd.read_csv(eval_data_file,header=0,usecols=['Segment_1','Segment_2'])
            eval_data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(eval_data['Segment_1'], eval_data['Segment_2'])]
            eval_data.drop(['Segment_1'],axis=1,inplace=True)
            eval_data.drop(['Segment_2'],axis=1,inplace=True)
            rec_list=[]
            rec_1 = pd.DataFrame(csv_out, columns = ['Segment_1','Q','Track_ID'])
            rec_1['Q']=rec_1['Q'].astype(str)
            rec_1['Track_ID']=rec_1['Track_ID'].astype(str)
            rec_1['New_Track_ID']=rec_1['Q']+'-'+rec_1['Track_ID']
            rec_1.drop(['Q'],axis=1,inplace=True)
            rec_1.drop(['Track_ID'],axis=1,inplace=True)
            rec_2 = pd.DataFrame(csv_out, columns = ['Segment_2','Q','Track_ID'])
            rec_2['Q']=rec_2['Q'].astype(str)
            rec_2['Track_ID']=rec_2['Track_ID'].astype(str)
            rec_2['New_Track_ID']=rec_2['Q']+'-'+rec_2['Track_ID']
            rec_2.drop(['Q'],axis=1,inplace=True)
            rec_2.drop(['Track_ID'],axis=1,inplace=True)
            rec=pd.merge(rec_1, rec_2, how="inner", on=['New_Track_ID'])
            rec.drop(['New_Track_ID'],axis=1,inplace=True)
            rec.drop(rec.index[rec['Segment_1'] == rec['Segment_2']], inplace = True)
            rec["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(rec['Segment_1'], rec['Segment_2'])]
            rec.drop(['Segment_1'],axis=1,inplace=True)
            rec.drop(['Segment_2'],axis=1,inplace=True)
            rec.drop_duplicates(subset=['Seed_ID'], keep='first', inplace=True)
            rec_eval=pd.merge(eval_data, rec, how="inner", on=['Seed_ID'])
            eval_no=len(rec_eval)
            rec_no=(len(rec)-len(rec_eval))
            UF.LogOperations(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv', 'a', [[4+len(ModelName),'Track Seed Merging',rec_no,eval_no,eval_no/(rec_no+eval_no),eval_no/len(eval_data)]])
            print(UF.TimeStamp(), bcolors.OKGREEN+"The log data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv'+bcolors.ENDC)
         UpdateStatus(Status+1)
    else:
        for md in range(len(ModelName)):
            if Program[Status]==ModelName[md]:
                if md==0:
                    prog_entry=[]
                    job_sets=[]
                    JobSet=[]
                    TotJobs=0
                    Program_Dummy=[]
                    Meta=UF.PickleOperations(RecOutputMeta,'r', 'N/A')[0]
                    JobSets=Meta.JobSets
                    for i in range(len(JobSets)):
                        JobSet.append([])
                        for j in range(len(JobSets[i][3])):
                                JobSet[i].append(JobSets[i][3][j])
                    if type(JobSet) is int:
                                TotJobs=JobSet
                    elif type(JobSet[0]) is int:
                                TotJobs=np.sum(JobSet)
                    elif type(JobSet[0][0]) is int:
                                for lp in JobSet:
                                    TotJobs+=np.sum(lp)
                    prog_entry.append(' Sending tracks to the HTCondor, so track segment combination pairs can be formed...')
                    prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','RefinedSeeds','RUTr1'+ModelName[md],'.pkl',RecBatchID,JobSet,'RUTr1b_RefineSeeds_Sub.py'])
                    prog_entry.append([" --MaxSTG ", " --MaxSLG ", " --MaxDOCA ", " --MaxAngle "," --ModelName "," --FirstTime "])
                    prog_entry.append([MaxSTG, MaxSLG, MaxDOCA, MaxAngle,'"'+ModelName[md]+'"', 'True'])
                    prog_entry.append(TotJobs)
                    prog_entry.append(LocalSub)
                    prog_entry.append(['',''])
                    for dum in range(0,Status):
                        Program_Dummy.append('DUM')
                    Program_Dummy.append(prog_entry)
                    if Mode=='RESET':
                        print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
                    #Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
                    print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))
                    Result=StandardProcess(Program_Dummy,Status,FreshStart)
                    if Result:
                        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
                        print(UF.TimeStamp(),bcolors.BOLD+'Stage '+str(Status)+':'+bcolors.ENDC+' Analysing the fitted seeds')
                        JobSet=[]
                        for i in range(len(JobSets)):
                             JobSet.append([])
                             for j in range(len(JobSets[i][3])):
                                 JobSet[i].append(JobSets[i][3][j])
                        base_data = None
                        with alive_bar(len(JobSets),force_tty=True, title='Checking the results from HTCondor') as bar:
                         for i in range(0,len(JobSet)):
                                bar()
                                for j in range(len(JobSet[i])):
                                         for k in range(JobSet[i][j]):
                                              required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RUTr1'+ModelName[md]+'_'+RecBatchID+'_'+str(i)+'/RUTr1'+ModelName[md]+'_'+RecBatchID+'_RefinedSeeds_'+str(i)+'_'+str(j)+'_'+str(k)+'.pkl'
                                              new_data=UF.PickleOperations(required_output_file_location,'r','N/A')[0]
                                              print(UF.TimeStamp(),'Set',str(i)+'_'+str(j)+'_'+str(k), 'contains', len(new_data), 'seeds',bcolors.ENDC)
                                              if base_data == None:
                                                    base_data = new_data
                                              else:
                                                    base_data+=new_data
                        Records=len(base_data)
                        print(UF.TimeStamp(),'The output contains', Records, 'raw images',bcolors.ENDC)

                        base_data=list(set(base_data))
                        Records_After_Compression=len(base_data)
                        if Records>0:
                                              Compression_Ratio=int((Records_After_Compression/Records)*100)
                        else:
                                              CompressionRatio=0
                        print(UF.TimeStamp(),'The output compression ratio is ', Compression_Ratio, ' %',bcolors.ENDC)
                else:
                    prog_entry=[]
                    TotJobs=0
                    Program_Dummy=[]
                    keep_testing=True
                    TotJobs=0
                    while keep_testing:
                        test_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RUTr1'+ModelName[md-1]+'_'+RecBatchID+'_0/RUTr1'+str(ModelName[md])+'_'+RecBatchID+'_Input_Seeds_'+str(TotJobs)+'.pkl'
                        if os.path.isfile(test_file_location):
                            TotJobs+=1
                        else:
                            keep_testing=False
                    prog_entry.append(' Sending tracks to the HTCondor, so track segment combination pairs can be formed...')
                    prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','OutputSeeds','RUTr1'+ModelName[md],'.pkl',RecBatchID,TotJobs,'RUTr1b_RefineSeeds_Sub.py'])

                    prog_entry.append([" --MaxSTG ", " --MaxSLG ", " --MaxDOCA ", " --MaxAngle "," --ModelName "," --FirstTime "])
                    prog_entry.append([MaxSTG, MaxSLG, MaxDOCA, MaxAngle,'"'+ModelName[md]+'"', ModelName[md-1]])
                    prog_entry.append(TotJobs)
                    prog_entry.append(LocalSub)
                    prog_entry.append(['',''])
                    for dum in range(0,Status):
                        Program_Dummy.append('DUM')
                    Program_Dummy.append(prog_entry)
                    if Mode=='RESET':
                        print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
                    #Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
                    print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))
                    Result=StandardProcess(Program_Dummy,Status,FreshStart)
                    if Result:
                        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
                        print(UF.TimeStamp(),bcolors.BOLD+'Stage '+str(Status)+':'+bcolors.ENDC+' Analysing the fitted seeds')
                        base_data = None
                        with alive_bar(len(JobSets),force_tty=True, title='Checking the results from HTCondor') as bar:
                         for i in range(0,TotJobs):
                                              bar()
                                              required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RUTr1'+ModelName[md]+'_'+RecBatchID+'_0/RUTr1'+ModelName[md]+'_'+RecBatchID+'_OutputSeeds_'+str(i)+'.pkl'
                                              new_data=UF.PickleOperations(required_output_file_location,'r','N/A')[0]
                                              print(UF.TimeStamp(),'Set',str(i), 'contains', len(new_data), 'seeds',bcolors.ENDC)
                                              if base_data == None:
                                                    base_data = new_data
                                              else:
                                                    base_data+=new_data
                        Records=len(base_data)
                        print(UF.TimeStamp(),'The output contains', Records, 'fit images',bcolors.ENDC)

                        base_data=list(set(base_data))
                        Records_After_Compression=len(base_data)
                        if Records>0:
                                              Compression_Ratio=int((Records_After_Compression/Records)*100)
                        else:
                                              CompressionRatio=0
                        print(UF.TimeStamp(),'The output compression ratio is ', Compression_Ratio, ' %',bcolors.ENDC)

                if md==len(ModelName)-1:
                        output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1c_'+RecBatchID+'_Fit_Seeds.pkl'
                        print(UF.PickleOperations(output_file_location,'w',base_data)[1])
                else:
                        output_split=int(np.ceil(Records_After_Compression/PM.MaxSegments))
                        for os_itr in range(output_split):
                            output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RUTr1'+ModelName[md]+'_'+RecBatchID+'_0/RUTr1'+str(ModelName[md+1])+'_'+RecBatchID+'_Input_Seeds_'+str(os_itr)+'.pkl'
                            print(UF.PickleOperations(output_file_location,'w',base_data[os_itr*PM.MaxSegments:(os_itr+1)*PM.MaxSegments])[1])
                if Log:
                             print(UF.TimeStamp(),'Initiating the logging...')
                             eval_data_file=EOS_DIR+'/ANNDEA/Data/TEST_SET/EUTr1b_'+RecBatchID+'_SEED_TRUTH_COMBINATIONS.csv'
                             eval_data=pd.read_csv(eval_data_file,header=0,usecols=['Segment_1','Segment_2'])
                             eval_data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(eval_data['Segment_1'], eval_data['Segment_2'])]
                             eval_data.drop(['Segment_1'],axis=1,inplace=True)
                             eval_data.drop(['Segment_2'],axis=1,inplace=True)
                             rec_no=0
                             eval_no=0
                             rec_list=[]
                             for rd in base_data:
                                 rec_list.append([rd.Header[0],rd.Header[1]])
                             del base_data
                             rec = pd.DataFrame(rec_list, columns = ['Segment_1','Segment_2'])
                             rec["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(rec['Segment_1'], rec['Segment_2'])]
                             rec.drop(['Segment_1'],axis=1,inplace=True)
                             rec.drop(['Segment_2'],axis=1,inplace=True)
                             rec_eval=pd.merge(eval_data, rec, how="inner", on=['Seed_ID'])
                             eval_no=len(rec_eval)
                             rec_no=(len(rec)-len(rec_eval))
                             UF.LogOperations(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv', 'a', [[3+md,ModelName[md],rec_no,eval_no,eval_no/(rec_no+eval_no),eval_no/len(eval_data)]])
                             print(UF.TimeStamp(), bcolors.OKGREEN+"The log data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv'+bcolors.ENDC)
                del new_data
                print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(Status)+' has successfully completed'+bcolors.ENDC)
                UpdateStatus(Status+1)
    print(UF.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+RecOutputMeta+bcolors.ENDC)
    MetaInput=UF.PickleOperations(RecOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
    Status=Meta.Status[-1]

if Status<20:
    #Removing the temp files that were generated by the process
    print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
    HTCondorTag="SoftUsed == \"ANNDEA-RUTr1-"+RecBatchID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'RUTr1_'+RecBatchID, ['RUTr1'], HTCondorTag)
    HTCondorTag="SoftUsed == \"ANNDEA-RUTr1a-"+RecBatchID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'RUTr1a_'+RecBatchID, [], HTCondorTag)
    HTCondorTag="SoftUsed == \"ANNDEA-RUTr1b-"+RecBatchID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'RUTr1b_'+RecBatchID, [], HTCondorTag)
    HTCondorTag="SoftUsed == \"ANNDEA-RUTr1c-"+RecBatchID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'RUTr1c_'+RecBatchID, [], HTCondorTag)
    HTCondorTag="SoftUsed == \"ANNDEA-RUTr1e-"+RecBatchID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'RUTr1e_'+RecBatchID, [], HTCondorTag)
    for p in Program:
        if p[:6]!='Custom' and (p in ModelName)==False:
           print(UF.TimeStamp(),UF.ManageTempFolders(p,'Delete'))
    for md in range(len(ModelName)):
                print(md)
                if md==0:
                    prog_entry=[]
                    job_sets=[]
                    JobSet=[]
                    TotJobs=0
                    Program_Dummy=[]
                    Meta=UF.PickleOperations(RecOutputMeta,'r', 'N/A')[0]
                    JobSets=Meta.JobSets
                    for i in range(len(JobSets)):
                        JobSet.append([])
                        for j in range(len(JobSets[i][3])):
                                JobSet[i].append(JobSets[i][3][j])
                    if type(JobSet) is int:
                                TotJobs=JobSet
                    elif type(JobSet[0]) is int:
                                TotJobs=np.sum(JobSet)
                    elif type(JobSet[0][0]) is int:
                                for lp in JobSet:
                                    TotJobs+=np.sum(lp)
                    prog_entry.append(' Blank placeholder')
                    prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','RefinedSeeds','RUTr1'+ModelName[md],'.pkl',RecBatchID,JobSet,'RUTr1b_RefineSeeds_Sub.py'])
                    prog_entry.append([" --MaxSTG ", " --MaxSLG ", " --MaxDOCA ", " --MaxAngle "," --ModelName "," --FirstTime "])
                    prog_entry.append([MaxSTG, MaxSLG, MaxDOCA, MaxAngle,'"'+ModelName[md]+'"', 'True'])
                    prog_entry.append(TotJobs)
                    prog_entry.append(LocalSub)
                    prog_entry.append(['',''])

                    print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
                    #Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
                else:
                    prog_entry=[]
                    TotJobs=0
                    Program_Dummy=[]
                    keep_testing=True
                    TotJobs=0
                    while keep_testing:
                        test_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RUTr1'+ModelName[md-1]+'_'+RecBatchID+'_0/RUTr1'+str(ModelName[md])+'_'+RecBatchID+'_Input_Seeds_'+str(TotJobs)+'.pkl'
                        if os.path.isfile(test_file_location):
                            TotJobs+=1
                        else:
                            keep_testing=False
                    prog_entry.append(' Blank placeholder')
                    prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','OutputSeeds','RUTr1'+ModelName[md],'.pkl',RecBatchID,TotJobs,'RUTr1b_RefineSeeds_Sub.py'])
                    prog_entry.append([" --MaxSTG ", " --MaxSLG ", " --MaxDOCA ", " --MaxAngle "," --ModelName "," --FirstTime "])
                    prog_entry.append([MaxSTG, MaxSLG, MaxDOCA, MaxAngle,'"'+ModelName[md]+'"', ModelName[md-1]])
                    prog_entry.append(TotJobs)
                    prog_entry.append(LocalSub)
                    prog_entry.append(['',''])
                    print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete')) #Deleting a specific set of created folders
    print(UF.TimeStamp(), bcolors.OKGREEN+"Segment merging has been completed"+bcolors.ENDC)
else:
    print(UF.TimeStamp(), bcolors.FAIL+"Segment merging has not been completed as one of the processes has timed out. Please run the script again (without Reset Mode)."+bcolors.ENDC)
    exit()



