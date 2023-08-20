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
sys.path.append(AFS_DIR+'/Code/Utilities')
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
csv_reader.close()
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import UtilityFunctions as UF #This is where we keep routine utility functions
import Parameters as PM #This is where we keep framework global parameters

#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script prepares training data for training the tracking model')
parser.add_argument('--Mode', help='Script will continue from the last checkpoint, unless you want to start from the scratch, then type "Reset"',default='')
parser.add_argument('--ModelName',help="WHat GNN model would you like to use?", default="['MH_GNN_5FTR_4_120_4_120']")
parser.add_argument('--Patience',help="How many checks to do before resubmitting the job?", default='15')
parser.add_argument('--TrainSampleID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--TrackID',help="What track name is used?", default='ANN_Track_ID')
parser.add_argument('--BrickID',help="What brick ID name is used?", default='ANN_Brick_ID')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_FEDRA_Raw_UR.csv')
parser.add_argument('--Xmin',help="This option restricts data to only those events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option restricts data to only those events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option restricts data to only those events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option restricts data to only those events that have tracks with hits y-coordinates that are below this value", default='0')
parser.add_argument('--Samples',help="How many samples? Please enter the number or ALL if you want to use all data", default='ALL')
parser.add_argument('--LabelRatio',help="What is the desired proportion of genuine seeds in the training/validation sets", default='0.5')
parser.add_argument('--TrainSampleSize',help="Maximum number of samples per Training file", default='50000')
parser.add_argument('--ForceStatus',help="Would you like the program run from specific status number? (Only for advance users)", default='')
parser.add_argument('--RequestExtCPU',help="Would you like to request extra CPUs?", default=1)
parser.add_argument('--JobFlavour',help="Specifying the length of the HTCondor job wall time. Currently at 'workday' which is 8 hours.", default='workday')
parser.add_argument('--LocalSub',help="Local submission?", default='N')
parser.add_argument('--SubPause',help="How long to wait in minutes after submitting 10000 jobs?", default='60')
parser.add_argument('--SubGap',help="How long to wait in minutes after submitting 10000 jobs?", default='10000')
parser.add_argument('--MinHitsTrack',help="What is the minimum number of hits per track?", default=PM.MinHitsTrack)
parser.add_argument('--MaxDST',help="Maximum distance between vertex seed track starting points", default='4000')
parser.add_argument('--MaxVXT',help="Minimum distance from vertex origin to closest starting hit of any track in the seed", default='4000')
parser.add_argument('--MaxDOCA',help="Maximum DOCA allowed", default='200')
parser.add_argument('--MaxAngle',help="Maximum magnitude of angle allowed", default='3.6')
parser.add_argument('--ReqMemory',help="How much memory to request?", default='2 GB')
parser.add_argument('--FiducialVolumeCut',help="Limits on the vx, y, z coordinates of the vertex origin", default='[]')
parser.add_argument('--RemoveTracksZ',help="This option enables to remove particular tracks of starting Z-coordinate", default='[]')
parser.add_argument('--ExcludeClassNames',help="What class headers to use?", default="[]")
parser.add_argument('--ExcludeClassValues',help="What class values to use?", default="[[]]")
######################################## Parsing argument values  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
MinHitsTrack=int(args.MinHitsTrack)
ModelName=ast.literal_eval(args.ModelName)
TrainSampleID=args.TrainSampleID
TrackID=args.TrackID
BrickID=args.BrickID
Patience=int(args.Patience)
TrainSampleSize=int(args.TrainSampleSize)
input_file_location=args.f
JobFlavour=args.JobFlavour
SubPause=int(args.SubPause)*60
SubGap=int(args.SubGap)
MaxDST=float(args.MaxDST)
MaxVXT=float(args.MaxVXT)
MaxDOCA=float(args.MaxDOCA)
MaxAngle=float(args.MaxAngle)
RequestExtCPU=int(args.RequestExtCPU)
ReqMemory=args.ReqMemory
FiducialVolumeCut=ast.literal_eval(args.FiducialVolumeCut)
ExcludeClassNames=ast.literal_eval(args.ExcludeClassNames)
ExcludeClassValues=ast.literal_eval(args.ExcludeClassValues)
ColumnsToImport=[TrackID,BrickID,PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.MC_Event_ID,PM.MC_VX_ID]
ExtraColumns=[]
BanDF=['-']
BanDF=pd.DataFrame(BanDF, columns=['Exclude'])
for i in range(len(ExcludeClassNames)):
        df=pd.DataFrame(ExcludeClassValues[i], columns=[ExcludeClassNames[i]])
        df['Exclude']='-'
        BanDF=pd.merge(BanDF,df,how='inner',on=['Exclude'])

        if (ExcludeClassNames[i] in ExtraColumns)==False:
                ExtraColumns.append(ExcludeClassNames[i])

RemoveTracksZ=ast.literal_eval(args.RemoveTracksZ)
Xmin,Xmax,Ymin,Ymax=float(args.Xmin),float(args.Xmax),float(args.Ymin),float(args.Ymax)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0 #We don't slice data if all values are set to zero simultaneously (which is the default setting)
LocalSub=(args.LocalSub=='Y')
if LocalSub:
   time_int=0
else:
    time_int=10


#Establishing paths
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
TrainSampleOutputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
required_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MVx1_'+TrainSampleID+'_TRACK_SEGMENTS.csv'


########################################     Phase 1 - Create compact source file    #########################################
print(UF.TimeStamp(),bcolors.BOLD+'Stage -1:'+bcolors.ENDC+' Preparing the source data...')

if os.path.isfile(required_file_location)==False or Mode=='RESET':
        print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
        data=pd.read_csv(input_file_location,
                    header=0,
                    usecols=ColumnsToImport+ExtraColumns)
        total_rows=len(data)
        print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
        print(UF.TimeStamp(),'Removing unreconstructed hits...')
        if len(ExtraColumns)>0:
            for c in ExtraColumns:
                data[c] = data[c].astype(str)
            data=pd.merge(data,BanDF,how='left',on=ExtraColumns)
            data=data.fillna('')
        else:
            data['Exclude']=''
        data=data.dropna()
        final_rows=len(data)
        print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
        data[PM.MC_Event_ID] = data[PM.MC_Event_ID].astype(str)
        data[PM.MC_VX_ID] = data[PM.MC_VX_ID].astype(str)
        data[TrackID] = data[TrackID].astype(str)
        data[BrickID] = data[BrickID].astype(str)
        data['Rec_Seg_ID'] = data[TrackID] + '-' + data[BrickID]
        data['MC_VX_ID'] = data[PM.MC_Event_ID] + '-' + data['Exclude'] + data[PM.MC_VX_ID]
        data=data.drop([TrackID],axis=1)
        data=data.drop([BrickID],axis=1)
        data=data.drop([PM.MC_Event_ID],axis=1)
        data=data.drop([PM.MC_VX_ID],axis=1)
        data=data.drop(['Exclude'],axis=1)
        for c in ExtraColumns:
            data=data.drop([c],axis=1)
        compress_data=data.drop([PM.x,PM.y,PM.z,PM.tx,PM.ty],axis=1)
        compress_data['MC_Mother_No']= compress_data['MC_VX_ID']
        compress_data=compress_data.groupby(by=['Rec_Seg_ID','MC_VX_ID'])['MC_Mother_No'].count().reset_index()
        compress_data=compress_data.sort_values(['Rec_Seg_ID','MC_Mother_No'],ascending=[1,0])
        compress_data.drop_duplicates(subset='Rec_Seg_ID',keep='first',inplace=True)
        data=data.drop(['MC_VX_ID'],axis=1)
        compress_data=compress_data.drop(['MC_Mother_No'],axis=1)
        data=pd.merge(data, compress_data, how="left", on=['Rec_Seg_ID'])
        
        if SliceData:
             print(UF.TimeStamp(),'Slicing the data...')
             ValidEvents=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
             ValidEvents.drop([PM.x,PM.y,PM.z,PM.tx,PM.ty,'MC_VX_ID']+ExtraColumns,axis=1,inplace=True)
             ValidEvents.drop_duplicates(subset='Rec_Seg_ID',keep='first',inplace=True)
             data=pd.merge(data, ValidEvents, how="inner", on=['Rec_Seg_ID'])
             final_rows=len(data.axes[0])
             print(UF.TimeStamp(),'The sliced data has ',final_rows,' hits')

        output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MVx1_'+TrainSampleID+'_TRACK_SEGMENTS.csv'
        print(UF.TimeStamp(),'Removing tracks which have less than',MinHitsTrack,'hits...')
        track_no_data=data.groupby(['MC_VX_ID','Rec_Seg_ID'],as_index=False).count()
        track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty],axis=1)
        track_no_data=track_no_data.rename(columns={PM.x: "Rec_Seg_No"})
        new_combined_data=pd.merge(data, track_no_data, how="left", on=['Rec_Seg_ID','MC_VX_ID'])
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
        if len(RemoveTracksZ)>0:
            print(UF.TimeStamp(),'Removing tracks based on start point')
            TracksZdf = pd.DataFrame(RemoveTracksZ, columns = ['Bad_z'], dtype=float)
            data_aggregated=new_combined_data.groupby(['Rec_Seg_ID'])['z'].min().reset_index()
            data_aggregated=data_aggregated.rename(columns={'z': "PosBad_Z"})
            new_combined_data=pd.merge(new_combined_data, data_aggregated, how="left", on=['Rec_Seg_ID'])
            new_combined_data=pd.merge(new_combined_data, TracksZdf, how="left", left_on=["PosBad_Z"], right_on=['Bad_z'])
            new_combined_data=new_combined_data[new_combined_data['Bad_z'].isnull()]
            new_combined_data=new_combined_data.drop(['Bad_z', 'PosBad_Z'],axis=1)
        final_rows=len(new_combined_data.axes[0])
        print(UF.TimeStamp(),'After removing tracks that start at the specific plates we have',final_rows,' hits left')
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
        Meta.IniVertexSeedMetaData(MaxDST,MaxVXT,MaxDOCA,MaxAngle,data,PM.MaxSegments,PM.MaxSeeds,MinHitsTrack,FiducialVolumeCut,ExcludeClassNames,ExcludeClassValues)
        Meta.UpdateStatus(0)
        print(UF.PickleOperations(TrainSampleOutputMeta,'w', Meta)[1])
        print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage -1 has successfully completed'+bcolors.ENDC)
elif os.path.isfile(TrainSampleOutputMeta)==True:
    print(UF.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+TrainSampleOutputMeta+bcolors.ENDC)
    MetaInput=UF.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
MaxDST=Meta.MaxDST
MaxVXT=Meta.MaxVXT
MaxDOCA=Meta.MaxDOCA
MaxAngle=Meta.MaxAngle
JobSets=Meta.JobSets
MaxSegments=Meta.MaxSegments
MaxSeeds=Meta.MaxSeeds
MinHitsTrack=Meta.MinHitsTrack
FiducialVolumeCut=Meta.FiducialVolumeCut
ExcludeClassNames=Meta.ClassNames
ExcludeClassValues=Meta.ClassValues
TotJobs=0
for j in range(0,len(JobSets)):
          for sj in range(0,int(JobSets[j][2])):
              TotJobs+=1

# ########################################     Preset framework parameters    #########################################
FreshStart=True
Program=[]

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
    HTCondorTag="SoftUsed == \"ANNDEA-MVx1a-"+TrainSampleID+"\""
    UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MVx1_'+TrainSampleID, ['MVx1a','MVx1b','MVx1c','MVx1d'], HTCondorTag)
    FreshStart=False
if Mode=='CLEANUP':
    Status=6
elif args.ForceStatus=='':
    Status=Meta.Status[-1]
else:
   Status=int(args.ForceStatus)
################ Set the execution sequence for the script

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

prog_entry.append(' Sending hit cluster to the HTCondor, so tack segment combination pairs can be formed...')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','RawSeedsRes','MVx1a','.csv',TrainSampleID,job_sets,'MVx1a_GenerateRawSelectedSeeds_Sub.py'])
prog_entry.append([ " --MaxSegments ", " --MaxDST "])
prog_entry.append([MaxSegments, MaxDST])
prog_entry.append(TotJobs)
prog_entry.append(LocalSub)
prog_entry.append([" --PlateZ ",JobSets])
Program.append(prog_entry)
if Mode=='RESET':
   print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
#Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))
###### Stage 1
Program.append('Custom')

##### Stage 2
prog_entry=[]
job_sets=[]
JobSets=Meta.JobSets
try:
    for i in range(len(JobSets)):
                 job_sets.append([])
                 for j in range(len(JobSets[i][3])):
                     job_sets[i].append(JobSets[i][3][j])
except:
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

prog_entry.append(' Taking the list of seeds previously generated by Stage 2, converting them into Emulsion Objects and doing more rigorous selection')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','RefinedSeeds','MVx1b','.pkl',TrainSampleID,job_sets,'MVx1b_RefineSeeds_Sub.py'])
prog_entry.append([" --MaxDST ", " --MaxVXT ", " --MaxDOCA ", " --MaxAngle "," --FiducialVolumeCut "])
prog_entry.append([MaxDST, MaxVXT, MaxDOCA, MaxAngle, '"'+str(FiducialVolumeCut)+'"'])
prog_entry.append(TotJobs)
prog_entry.append(LocalSub)
prog_entry.append(['',''])
Program.append(prog_entry)
if Mode=='RESET':
   print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
#Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))


###### Stage 3
Program.append('Custom')
###### Stage 4
Program.append('Custom')
###### Stage 5
Program.append('Custom')

print(UF.TimeStamp(),'There are '+str(len(Program)+1)+' stages (0-'+str(len(Program)+1)+') of this script',bcolors.ENDC)
print(UF.TimeStamp(),'Current stage has a code',Status,bcolors.ENDC)

while Status<len(Program):
    if Program[Status]!='Custom':
        #Standard process here
        Result=StandardProcess(Program,Status,FreshStart)
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
            break

    elif Status==1:
       #Non standard processes (that don't follow the general pattern) have been coded here
        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.BOLD+'Stage 1:'+bcolors.ENDC+' Collecting and de-duplicating the results from stage 1')
        min_i=0
        for i in range(0,len(JobSets)): #//Temporarily measure to save space
                   test_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/Temp_MVx1a_'+TrainSampleID+'_'+str(i)+'/MVx1a_'+TrainSampleID+'_SelectedSeeds_'+str(i)+'_'+str(0)+'_'+str(0)+'.csv'
                   if os.path.isfile(test_file_location):
                        min_i=max(0,i-1)
        with alive_bar(len(JobSets)-min_i,force_tty=True, title='Checking the results from HTCondor') as bar:
            for i in range(min_i,len(JobSets)): #//Temporarily measure to save space
                bar.text = f'-> Analysing set : {i}...'
                bar()
                Meta=UF.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')[0]
                MaxDST=Meta.MaxDST
                JobSets=Meta.JobSets
                if len(Meta.JobSets[i])>3:
                   Meta.JobSets[i]=Meta.JobSets[i][:4]
                   Meta.JobSets[i][3]=[]
                else:
                   Meta.JobSets[i].append([])
                for j in range(0,int(JobSets[i][2])):

                   output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/Temp_MVx1a_'+TrainSampleID+'_'+str(i)+'/MVx1a_'+TrainSampleID+'_RawSeeds_'+str(i)+'_'+str(j)+'.csv'

                   if os.path.isfile(output_file_location)==False:
                      Meta.JobSets[j].append(0)
                      continue #Skipping because not all jobs necesseraly produce the required file (if statistics are too low)
                   else:
                    result=pd.read_csv(output_file_location,names = ['Track_1','Track_2', 'Seed_Type'])
                    Records=len(result)
                    print(UF.TimeStamp(),'Set',str(i),'and subset', str(j), 'contains', Records, 'seeds',bcolors.ENDC)
                    result["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(result['Track_1'], result['Track_2'])]
                    result.drop_duplicates(subset="Seed_ID",keep='first',inplace=True)
                    result.drop(result.index[result['Track_1'] == result['Track_2']], inplace = True)
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
                     new_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/Temp_MVx1a_'+TrainSampleID+'_'+str(i)+'/MVx1a_'+TrainSampleID+'_SelectedSeeds_'+str(i)+'_'+str(j)+'_'+str(k)+'.csv'
                     result[(k*MaxSeeds):min(Records_After_Compression,((k+1)*MaxSeeds))].to_csv(new_output_file_location,index=False)
                print(UF.PickleOperations(TrainSampleOutputMeta,'w', Meta)[1])

        #Part of the program needs to be rewritten
        job_sets=[]
        JobSets=Meta.JobSets
        for i in range(len(JobSets)):
                     job_sets.append([])
                     for j in range(len(JobSets[i][3])):
                         job_sets[i].append(JobSets[i][3][j])
        TotJobs=0
        if type(job_sets) is int:
                                TotJobs=job_sets
        elif type(job_sets[0]) is int:
                                TotJobs=np.sum(job_sets)
        elif type(job_sets[0][0]) is int:
                                for lp in job_sets:
                                    TotJobs+=np.sum(lp)

        prog_entry=[]
        prog_entry.append(' Taking the list of seeds previously generated by Stage 2, converting them into Emulsion Objects and doing more rigorous selection')
        prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','RefinedSeeds','MVx1b','.pkl',TrainSampleID,job_sets,'MVx1b_RefineSeeds_Sub.py'])
        prog_entry.append([" --MaxDST ", " --MaxVXT ", " --MaxDOCA ", " --MaxAngle "," --FiducialVolumeCut "])
        prog_entry.append([MaxDST, MaxVXT, MaxDOCA, MaxAngle, '"'+str(FiducialVolumeCut)+'"'])
        prog_entry.append(TotJobs)
        prog_entry.append(LocalSub)
        prog_entry.append(['',''])
        Program[2]=prog_entry
        #############################
        FreshStart=False
        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 1 has successfully completed'+bcolors.ENDC)
        Status=2
        UpdateStatus(Status)
    elif Status==3:
        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' Analysing the training samples')
        JobSet=[]
        for i in range(len(JobSets)):
             JobSet.append([])
             for j in range(len(JobSets[i][3])):
                 JobSet[i].append(JobSets[i][3][j])
        for i in range(0,len(JobSet)):
             output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MVx1c_'+TrainSampleID+'_CompressedSeeds_'+str(i)+'.pkl'
             if os.path.isfile(output_file_location)==False:
                if os.path.isfile(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MVx1c_'+TrainSampleID+'_Temp_Stats.csv')==False:
                   UF.LogOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MVx1c_'+TrainSampleID+'_Temp_Stats.csv','w', [[0,0]])
                Temp_Stats=UF.LogOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MVx1c_'+TrainSampleID+'_Temp_Stats.csv','r', '_')

                TotalImages=int(Temp_Stats[0][0])
                TrueSeeds=int(Temp_Stats[0][1])
                base_data = None
                for j in range(len(JobSet[i])):
                         for k in range(JobSet[i][j]):
                              required_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/Temp_MVx1b_'+TrainSampleID+'_'+str(i)+'/MVx1b_'+TrainSampleID+'_'+'RefinedSeeds'+'_'+str(i)+'_'+str(j) + '_' + str(k)+'.pkl'
                              new_data=UF.PickleOperations(required_output_file_location,'r','N/A')[0]
                              if base_data == None:
                                    base_data = new_data
                              else:
                                    base_data+=new_data
                try:
                    Records=len(base_data)
                    print(UF.TimeStamp(),'Set',str(i),'contains', Records, 'raw images',bcolors.ENDC)

                    base_data=list(set(base_data))
                    Records_After_Compression=len(base_data)
                    if Records>0:
                              Compression_Ratio=int((Records_After_Compression/Records)*100)
                    else:
                              CompressionRatio=0
                    TotalImages+=Records_After_Compression
                    TrueSeeds+=sum(1 for im in base_data if im.Label == 1)
                    print(UF.TimeStamp(),'Set',str(i),'compression ratio is ', Compression_Ratio, ' %',bcolors.ENDC)
                    print(UF.PickleOperations(output_file_location,'w',base_data)[1])
                except:
                    continue
                del new_data
                UF.LogOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MVx1c_'+TrainSampleID+'_Temp_Stats.csv','w', [[TotalImages,TrueSeeds]])
        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 4 has successfully completed'+bcolors.ENDC)
        Status=4
        UpdateStatus(Status)
        continue
    elif Status==4:
           print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
           print(UF.TimeStamp(),bcolors.BOLD+'Stage 4:'+bcolors.ENDC+' Resampling the results from the previous stage')
           print(UF.TimeStamp(),'Sampling the required number of seeds',bcolors.ENDC)
           Temp_Stats=UF.LogOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MVx1c_'+TrainSampleID+'_Temp_Stats.csv','r', '_')
           TotalImages=int(Temp_Stats[0][0])
           TrueSeeds=int(Temp_Stats[0][1])
           JobSet=[]
           for i in range(len(JobSets)):
             JobSet.append([])
             for j in range(len(JobSets[i][3])):
                 JobSet[i].append(JobSets[i][3][j])
           if args.Samples=='ALL':
               if TrueSeeds<=(float(args.LabelRatio)*TotalImages):
                   RequiredTrueSeeds=TrueSeeds
                   RequiredFakeSeeds=int(round((RequiredTrueSeeds/float(args.LabelRatio))-RequiredTrueSeeds,0))
               else:
                   RequiredFakeSeeds=TotalImages-TrueSeeds
                   RequiredTrueSeeds=int(round((RequiredFakeSeeds/(1.0-float(args.LabelRatio)))-RequiredFakeSeeds,0))

           else:
               NormalisedTotSamples=int(args.Samples)
               if TrueSeeds<=(float(args.LabelRatio)*NormalisedTotSamples):
                   RequiredTrueSeeds=TrueSeeds
                   RequiredFakeSeeds=int(round((RequiredTrueSeeds/float(args.LabelRatio))-RequiredTrueSeeds,0))
               else:
                   RequiredFakeSeeds=NormalisedTotSamples*(1.0-float(args.LabelRatio))
                   RequiredTrueSeeds=int(round((RequiredFakeSeeds/(1.0-float(args.LabelRatio)))-RequiredFakeSeeds,0))
           if TrueSeeds==0:
               TrueSeedCorrection=0
           else:
              TrueSeedCorrection=RequiredTrueSeeds/TrueSeeds
           if TotalImages-TrueSeeds>0:
            FakeSeedCorrection=RequiredFakeSeeds/(TotalImages-TrueSeeds)
           else:
             FakeSeedCorrection=0
           with alive_bar(len(JobSet),force_tty=True, title='Resampling the files...') as bar:
            for i in range(0,len(JobSet)):
              output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MVx1d_'+TrainSampleID+'_SampledCompressedSeeds_'+str(i)+'.pkl'
              input_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MVx1c_'+TrainSampleID+'_CompressedSeeds_'+str(i)+'.pkl'
              bar.text = f'-> Resampling the file : {input_file_location}, exists...'
              bar()
              if os.path.isfile(output_file_location)==False and os.path.isfile(input_file_location):
                  base_data=UF.PickleOperations(input_file_location,'r','N/A')[0]
                  ExtractedTruth=[im for im in base_data if im.Label == 1]
                  ExtractedFake=[im for im in base_data if im.Label == 0]
                  del base_data
                  gc.collect()
                  ExtractedTruth=random.sample(ExtractedTruth,int(round(TrueSeedCorrection*len(ExtractedTruth),0)))
                  ExtractedFake=random.sample(ExtractedFake,int(round(FakeSeedCorrection*len(ExtractedFake),0)))
                  TotalData=[]
                  TotalData=ExtractedTruth+ExtractedFake
                  print(UF.PickleOperations(output_file_location,'w',TotalData)[1])
                  del TotalData
                  del ExtractedTruth
                  del ExtractedFake
                  gc.collect()
           print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 4 has successfully completed'+bcolors.ENDC)
           Status=5
           UpdateStatus(Status)
           continue
    elif Status==5:
           print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
           print(UF.TimeStamp(),bcolors.BOLD+'Stage 5:'+bcolors.ENDC+' Preparing the final output')
           TotalData=[]
           JobSet=[]
           for i in range(len(JobSets)):
             JobSet.append([])
             for j in range(len(JobSets[i][3])):
                 JobSet[i].append(JobSets[i][3][j])

           for i in range(0,len(JobSet)):
               input_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MVx1d_'+TrainSampleID+'_SampledCompressedSeeds_'+str(i)+'.pkl'
               if os.path.isfile(input_file_location):
                  base_data=UF.PickleOperations(input_file_location,'r','N/A')[0]
                  TotalData+=base_data
           del base_data
           gc.collect()
           ValidationSampleSize=int(round(min((len(TotalData)*float(PM.valRatio)),PM.MaxValSampleSize),0))
           random.shuffle(TotalData)
           output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_VERTEX_SEEDS_OUTPUT.pkl'
           print(UF.PickleOperations(output_file_location,'w',TotalData[:ValidationSampleSize])[1])
           TotalData=TotalData[ValidationSampleSize:]
           print(UF.TimeStamp(), bcolors.OKGREEN+"Validation Set has been saved at ",bcolors.OKBLUE+output_file_location+bcolors.ENDC,bcolors.OKGREEN+'file...'+bcolors.ENDC)
           No_Train_Files=int(math.ceil(len(TotalData)/TrainSampleSize))
           with alive_bar(No_Train_Files,force_tty=True, title='Resampling the files...') as bar:
               for SC in range(0,No_Train_Files):
                 output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_VERTEX_SEEDS_OUTPUT_'+str(SC+1)+'.pkl'
                 print(UF.PickleOperations(output_file_location,'w',TotalData[(SC*TrainSampleSize):min(len(TotalData),((SC+1)*TrainSampleSize))])[1])
                 bar.text = f'-> Saving the file : {output_file_location}...'
                 bar()

           print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 5 has successfully completed'+bcolors.ENDC)
           print(UF.TimeStamp(),'Would you like to delete Temporary files?')
           user_response=input()
           if user_response=='y' or user_response=='Y':
               Status=6
               UpdateStatus(Status)
               continue
           else:
               print(UF.TimeStamp(), bcolors.OKGREEN+"Train sample generation has been completed"+bcolors.ENDC)
               exit() 
if Status==6:
           print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
           for p in Program:
            if p!='Custom':
               print(UF.TimeStamp(),UF.ManageTempFolders(p,'Delete'))
           HTCondorTag="SoftUsed == \"ANNDEA-MVx1-"+TrainSampleID+"\""
           UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MVx1_'+TrainSampleID, ['MVx1_'+TrainSampleID], HTCondorTag)
           HTCondorTag="SoftUsed == \"ANNDEA-MVx1c-"+TrainSampleID+"\""
           UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MVx1c_'+TrainSampleID, ['MVx1c_'+TrainSampleID], HTCondorTag)
           HTCondorTag="SoftUsed == \"ANNDEA-MVx1d-"+TrainSampleID+"\""
           UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MVx1d_'+TrainSampleID, ['MVx1d_'+TrainSampleID], HTCondorTag)
           print(UF.TimeStamp(), bcolors.OKGREEN+"Train sample generation has been completed"+bcolors.ENDC)
           exit()
else:
    print(UF.TimeStamp(), bcolors.FAIL+"Reconstruction has not been completed as one of the processes has timed out or --ForceStatus!=0 option was chosen. Please run the script again (without Reset Mode)."+bcolors.ENDC)
    exit()



