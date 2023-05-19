#This simple connects hits in the data to produce tracks
#Tracking Module of the ANNDEA package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
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

#Loading Directory locations
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

#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script prepares training data for training the tracking model')
parser.add_argument('--Mode', help='Script will continue from the last checkpoint, unless you want to start from the scratch, then type "Reset"',default='')
parser.add_argument('--Patience',help="How many checks to do before resubmitting the job?", default='15')
parser.add_argument('--TrainSampleID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_FEDRA_Raw_UR.csv')
parser.add_argument('--Xmin',help="This option restricts data to only those events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option restricts data to only those events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option restricts data to only those events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option restricts data to only those events that have tracks with hits y-coordinates that are below this value", default='0')
parser.add_argument('--Samples',help="How many samples? Please enter the number or ALL if you want to use all data", default='ALL')
parser.add_argument('--TrainSampleSize',help="Maximum number of samples per Training file", default='50000')
parser.add_argument('--LocalSub',help="Local submission?", default='N')
parser.add_argument('--ClassHeaders',help="What class headers to use?", default="['EM Background']")
parser.add_argument('--ClassNames',help="What class headers to use?", default="[['Flag','ProcID']]")
parser.add_argument('--ClassValues',help="What class values to use?", default="[['11','-11'],['8']]")
parser.add_argument('--TrackID',help="What track name is used?", default='ANN_Track_ID')
parser.add_argument('--BrickID',help="What brick ID name is used?", default='ANN_Brick_ID')
parser.add_argument('--ReqMemory',help="How uch memory to request?", default='2 GB')
parser.add_argument('--RequestExtCPU',help="Would you like to request extra CPUs?", default=1)
parser.add_argument('--JobFlavour',help="Specifying the length of the HTCondor job walltime. Currently at 'workday' which is 8 hours.", default='workday')
parser.add_argument('--RemoveTracksZ',help="This option enables to remove particular tracks of starting Z-coordinate", default='[]')
parser.add_argument('--MinHitsTrack',help="What is the minimum number of hits per track?", default=PM.MinHitsTrack)
parser.add_argument('--SubPause',help="How long to wait in minutes after submitting 10000 jobs?", default='60')
parser.add_argument('--SubGap',help="How long to wait in minutes after submitting 10000 jobs?", default='10000')
parser.add_argument('--PY',help="Python libraries directory location", default='.')
parser.add_argument('--ForceStatus',help="Would you like the program run from specific status number? (Only for advance users)", default='')
######################################## Parsing argument values  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
ClassHeaders=ast.literal_eval(args.ClassHeaders)
ClassNames=ast.literal_eval(args.ClassNames)
ClassValues=ast.literal_eval(args.ClassValues)
TrackID=args.TrackID
BrickID=args.BrickID
TrainSampleID=args.TrainSampleID
Patience=int(args.Patience)
TrainSampleSize=int(args.TrainSampleSize)
input_file_location=args.f
Xmin,Xmax,Ymin,Ymax=float(args.Xmin),float(args.Xmax),float(args.Ymin),float(args.Ymax)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0 #We don't slice data if all values are set to zero simultaneousy (which is the default setting)
RequestExtCPU=int(args.RequestExtCPU)
SubPause=int(args.SubPause)*60
SubGap=int(args.SubGap)
ReqMemory=args.ReqMemory
JobFlavour=args.JobFlavour
RemoveTracksZ=ast.literal_eval(args.RemoveTracksZ)
MinHitsTrack=int(args.MinHitsTrack)
LocalSub=(args.LocalSub=='Y')
if LocalSub:
   time_int=0
else:
    time_int=10
#Establishing paths
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
TrainSampleOutputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
required_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MCTr1_'+TrainSampleID+'_TRACKS.csv'
ColumnsToImport=[TrackID,BrickID,PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.MC_Track_ID,PM.MC_Event_ID]
ExtraColumns=[]
for i in ClassNames:
    for j in i:
        if (j in ColumnsToImport)==False:
            ColumnsToImport.append(j)
        if (j in ExtraColumns)==False:
                ExtraColumns.append(j)

########################################     Phase 1 - Create compact source file    #########################################
print(UF.TimeStamp(),bcolors.BOLD+'Stage 0:'+bcolors.ENDC+' Preparing the source data...')
if Mode.upper() == 'CLEANUP':
   Status=4
   MetaInput=UF.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')
   Meta=MetaInput[0]
elif os.path.isfile(required_file_location)==False or Mode=='RESET':
        print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
        data=pd.read_csv(input_file_location,
                    header=0,
                    usecols=ColumnsToImport)
        total_rows=len(data.axes[0])
        print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
        print(UF.TimeStamp(),'Removing unreconstructed hits...')
        data=data.dropna()
        final_rows=len(data.axes[0])
        print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
        data[PM.MC_Event_ID] = data[PM.MC_Event_ID].astype(str)
        for i in ExtraColumns:
            data[i]=data[i].astype(str)
        data[PM.MC_Track_ID] = data[PM.MC_Track_ID].astype(str)
        data[TrackID] = data[TrackID].astype(str)
        data[BrickID] = data[BrickID].astype(str)
        data[PM.MC_Event_ID] = data[PM.MC_Event_ID].astype(str)
        data['Rec_Seg_ID'] = data[BrickID] + '-' + data[TrackID]
        data['MC_Mother_Track_ID'] = data[PM.MC_Event_ID] + '-' + data[PM.MC_Track_ID]
        data=data.drop([TrackID],axis=1)
        data=data.drop([BrickID],axis=1)
        data=data.drop([PM.MC_Event_ID],axis=1)
        data=data.drop([PM.MC_Track_ID],axis=1)
        if len(RemoveTracksZ)>0:
            print(UF.TimeStamp(),'Removing tracks based on start point')
            TracksZdf = pd.DataFrame(RemoveTracksZ, columns = ['Bad_z'], dtype=float)
            data_aggregated=data.groupby(['Rec_Seg_ID'])['z'].min().reset_index()
            data_aggregated=data_aggregated.rename(columns={'z': "PosBad_Z"})
            data=pd.merge(data, data_aggregated, how="left", on=['Rec_Seg_ID'])
            data=pd.merge(data, TracksZdf, how="left", left_on=["PosBad_Z"], right_on=['Bad_z'])
            data=data[data['Bad_z'].isnull()]
            data=data.drop(['Bad_z', 'PosBad_Z'],axis=1)
        final_rows=len(data.axes[0])
        print(UF.TimeStamp(),'After removing tracks that start at the specific plates we have',final_rows,' hits left')
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
             ValidEvents.drop([PM.x,PM.y,PM.z,PM.tx,PM.ty,'MC_Mother_Track_ID']+ExtraColumns,axis=1,inplace=True)
             ValidEvents.drop_duplicates(subset='Rec_Seg_ID',keep='first',inplace=True)
             data=pd.merge(data, ValidEvents, how="inner", on=['Rec_Seg_ID'])
             final_rows=len(data.axes[0])
             print(UF.TimeStamp(),'The sliced data has ',final_rows,' hits')

        output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MCTr1_'+TrainSampleID+'_TRACKS.csv'
        print(UF.TimeStamp(),'Removing tracks which have less than',MinHitsTrack,'hits...')
        track_no_data=data.groupby(['MC_Mother_Track_ID','Rec_Seg_ID']+ExtraColumns,as_index=False).count()
        track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty],axis=1)
        track_no_data=track_no_data.rename(columns={PM.x: "Rec_Seg_No"})
        new_combined_data=pd.merge(data, track_no_data, how="left", on=['Rec_Seg_ID','MC_Mother_Track_ID']+ExtraColumns)
        new_combined_data = new_combined_data[new_combined_data.Rec_Seg_No >= MinHitsTrack]
        new_combined_data = new_combined_data.drop(["Rec_Seg_No"],axis=1)
        new_combined_data=new_combined_data.sort_values(['Rec_Seg_ID',PM.x],ascending=[1,1])
        grand_final_rows=len(new_combined_data.axes[0])
        print(UF.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
        new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
        new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
        new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
        new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
        new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
        new_combined_data.drop(['MC_Mother_Track_ID'],axis=1,inplace=True)
        new_combined_data.to_csv(output_file_location,index=False)
        data=new_combined_data[['Rec_Seg_ID']]
        print(UF.TimeStamp(),'Analysing the data sample in order to understand how many jobs to submit to HTCondor... ',bcolors.ENDC)
        data.drop_duplicates(subset='Rec_Seg_ID',keep='first',inplace=True)
        data = data.values.tolist()
        no_submissions=math.ceil(len(data)/PM.MaxSegments)
        print(UF.TimeStamp(), bcolors.OKGREEN+"The track segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
        Meta=UF.TrainingSampleMeta(TrainSampleID)
        Meta.IniTrackMetaData(ClassHeaders,ClassNames,ClassValues,PM.MaxSegments,no_submissions,MinHitsTrack)
        Meta.UpdateStatus(0)
        print(UF.PickleOperations(TrainSampleOutputMeta,'w', Meta)[1])
        print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 0 has successfully completed'+bcolors.ENDC)
elif os.path.isfile(TrainSampleOutputMeta)==True:
    print(UF.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+TrainSampleOutputMeta+bcolors.ENDC)
    MetaInput=UF.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]

ClassHeaders=Meta.ClassHeaders
ClassNames=Meta.ClassNames
ClassValues=Meta.ClassValues
JobSets=Meta.JobSets
MaxSegments=Meta.MaxSegments
TotJobs=JobSets


########################################     Preset framework parameters    #########################################
FreshStart=True
Program=[]
def UpdateStatus(status):
    Meta.UpdateStatus(status)
    print(UF.PickleOperations(TrainSampleOutputMeta,'w', Meta)[1])

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
                      HTCondorTag="SoftUsed == \"ANNDEA-"+program[status][1][5]+"-"+TrainSampleID+"\""
                      UF.TrainCleanUp(AFS_DIR, EOS_DIR, program[status][1][5]+'_'+TrainSampleID, [], HTCondorTag)
                      for bp in bad_pop:
                           UF.SubmitJobs2Condor(bp,program[status][5],RequestExtCPU,JobFlavour,ReqMemory)
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
                      HTCondorTag="SoftUsed == \"ANNDEA-"+program[status][1][5]+"-"+TrainSampleID+"\""
                      UF.TrainCleanUp(AFS_DIR, EOS_DIR, program[status][1][5]+'_'+TrainSampleID, [], HTCondorTag)
                      for bp in bad_pop:
                           UF.SubmitJobs2Condor(bp,program[status][5],RequestExtCPU,JobFlavour,ReqMemory)
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
    HTCondorTag="SoftUsed == \"ANNDEA-MCTr1a-"+TrainSampleID+"\""
    UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MCTr1a_'+TrainSampleID, [], HTCondorTag)
    FreshStart=False
    UpdateStatus(0)
    Status=0
else:
    print(UF.TimeStamp(),'Analysing the current script status...',bcolors.ENDC)
    Status=Meta.Status[-1]

###### Stage 0
prog_entry=[]
prog_entry.append(UF.TimeStamp()+bcolors.BOLD+'Stage 1:'+bcolors.ENDC+' Sending hit cluster to the HTCondor, so tack segment combination pairs can be formed...')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','IDseeds','MCTr1a','.pkl',TrainSampleID,JobSets,'MCTr1a_GenerateRawTrackSamples_Sub.py'])
prog_entry.append([ " --MaxSegments ", " --ClassNames "," --ClassValues "])
prog_entry.append([MaxSegments,'"'+str(ClassNames)+'"','"'+str(ClassValues)+'"'])
prog_entry.append(JobSets)
prog_entry.append(LocalSub)
prog_entry.append(['',''])
Program.append(prog_entry)
if Mode=='RESET':
   print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
#Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))

###### Stage 1
Program.append('Custom')
###### Stage 2
Program.append('Custom')

print(UF.TimeStamp(),'There are ',len(Program),' stages of this script', bcolors.ENDC)
print(UF.TimeStamp(),'Current status has a stage',Status+1,bcolors.ENDC)

while Status<len(Program):
      if Program[Status]!='Custom':
        #Standard process here
        Result=StandardProcess(Program,Status,FreshStart)
        print(Result)
        if Result[0]:
            FreshStart=Result[1]
            if args.ForceStatus=='':
                Status+=1
                UpdateStatus(Status)
                continue
            else:
                exit()
        else:
            Status=len(Program)+1
            UpdateStatus(Status)
            break
      if Program[Status]=='Custom':
          if Status==1:
            print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
            print(UF.TimeStamp(),bcolors.BOLD+'Stage 2:'+bcolors.ENDC+' Collecting and de-duplicating the results from stage 1')
            for i in range(JobSets):
                    req_file=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/Temp_MCTr1a_'+TrainSampleID+'_0/MCTr1a_'+TrainSampleID+'_IDseeds_'+str(i)+'.pkl'
                    output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/Temp_MCTr1a_'+TrainSampleID+'_0/MCTr1b_'+TrainSampleID+'_SelectedTrackSamples_'+str(i)+'.pkl'
                    base_data=UF.PickleOperations(req_file,'r', 'N/A')[0]
                    ExtractedData=[]
                    min_len=len([j for j in base_data if j.Label==0])
                    for j in range(len(ClassHeaders)+1):
                        if len([k for k in base_data if k.Label==j])!=0:
                           ExtractedData.append([k for k in base_data if k.Label==j])
                           min_len=min(len([k for k in base_data if k.Label==j]),min_len)
                    TotalData=[]
                    for s in range(len(ExtractedData)):
                        TotalData+=random.sample(ExtractedData[s],min_len)
                    print(UF.PickleOperations(output_file_location,'w', TotalData)[1])
            print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
            UpdateStatus(Status+1)
            Status+=1
            continue

          if Status==2:
              print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
              print(UF.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' Taking the list of seeds previously generated by Stage 2, converting them into Emulsion Objects and doing more rigorous selection')
              TotalData=[]
              for i in range(JobSets):
                    req_file=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/Temp_MCTr1a_'+TrainSampleID+'_0/MCTr1b_'+TrainSampleID+'_SelectedTrackSamples_'+str(i)+'.pkl'
                    base_data=UF.PickleOperations(req_file,'r', 'N/A')[0]
                    TotalData+=base_data
              ValidationSampleSize=int(round(min((len(TotalData)*float(PM.valRatio)),PM.MaxValSampleSize),0))
              random.shuffle(TotalData)
              output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_TRACK_OUTPUT.pkl'
              print(UF.PickleOperations(output_file_location,'w', TotalData[:ValidationSampleSize])[1])
              TotalData=TotalData[ValidationSampleSize:]
              No_Train_Files=int(math.ceil(len(TotalData)/TrainSampleSize))
              for i in range(0,No_Train_Files):
                  output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_TRACK_OUTPUT_'+str(i+1)+'.pkl'
                  print(UF.PickleOperations(output_file_location,'w', TotalData[(i*TrainSampleSize):min(len(TotalData),((i+1)*TrainSampleSize))])[1])
              print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
              UpdateStatus(Status + 1)
              Status+=1
              continue
      print(UF.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+TrainSampleOutputMeta+bcolors.ENDC)
      print(UF.TimeStamp(),'Would you like to delete Temporary files?')
      user_response=input()
      if user_response=='y' or user_response=='Y':
               UpdateStatus(Status+1)
               Status+=1
               continue
      else:
               print(UF.TimeStamp(), bcolors.OKGREEN+"Train sample generation has been completed"+bcolors.ENDC)
               exit()
      MetaInput=UF.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')
      Meta=MetaInput[0]
      Status=Meta.Status[-1]
if Status==3:
        for p in Program:
            if p!='Custom':
              print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
              print(UF.TimeStamp(),UF.ManageTempFolders(p,'Delete'))
              HTCondorTag="SoftUsed == \"ANNDEA-MCTr1a-"+TrainSampleID+"\""
              UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MCTr1a_'+TrainSampleID, ['MCTr1a', 'MCTr1_'+TrainSampleID], HTCondorTag)
        print(UF.TimeStamp(), bcolors.OKGREEN+"Train sample generation has been completed"+bcolors.ENDC)
        exit()
else:
      print(UF.TimeStamp(), bcolors.FAIL+"Reconstruction has not been completed as one of the processes has timed out. Please run the script again (without Reset Mode)."+bcolors.ENDC)
      exit()



