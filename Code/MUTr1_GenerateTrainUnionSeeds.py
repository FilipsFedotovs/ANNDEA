#This simple connects hits in the data to produce tracks
#Tracking Module of the ANNADEA package
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


######################################## Parsing argument values  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
ModelName=ast.literal_eval(args.ModelName)
TrainSampleID=args.TrainSampleID
Patience=int(args.Patience)
input_file_location=args.f
Xmin,Xmax,Ymin,Ymax=float(args.Xmin),float(args.Xmax),float(args.Ymin),float(args.Ymax)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0 #We don't slice data if all values are set to zero simultaneousy (which is the default setting)

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
EOSsubDIR=EOS_DIR+'/'+'ANNADEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
TrainSampleOutputMeta=EOS_DIR+'/ANNADEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
required_file_location=EOS_DIR+'/ANNADEA/Data/TRAIN_SET/MUTr1_'+TrainSampleID+'_TRACK_SEGMENTS.csv'
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

        output_file_location=EOS_DIR+'/ANNADEA/Data/TRAIN_SET/MUTr1_'+TrainSampleID+'_TRACK_SEGMENTS.csv'
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
        Meta.IniTrackSeedMetaData(PM.MaxSLG,PM.MaxSTG,PM.MaxDOCA,PM.MaxAngle,data,PM.MaxSegments,PM.VetoMotherTrack)
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
VetoMotherTrack=Meta.VetoMotherTrack
TotJobs=0
for j in range(0,len(JobSets)):
          for sj in range(0,int(JobSets[j][2])):
              TotJobs+=1

########################################     Preset framework parameters    #########################################
print(UF.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
FreshStart=True
status=0
#
# #Defining handy functions to make the code little cleaner
#
# def CheckStatus():
#     #Let's check at what stage are we
#     print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
#     print(UF.TimeStamp(),bcolors.BOLD+'Preparation 3/3:'+bcolors.ENDC+' Working out the scope of the upcoming work...')
#     #First of all lets check that the output of reconstruction is completed
#     required_output_file_location=EOS_DIR+'/ANNADEA/Data/REC_SET/'+RecBatchID+'_RH_OUTPUT.pkl'
#     if os.path.isfile(required_output_file_location):
#         return 5
#     else:
#         #Reconstruction output hasn't been produced - lets check the previous step result and so on
#         required_output_file_location=EOS_DIR+'/ANNADEA/Data/REC_SET/RH1d_'+RecBatchID+'_hit_cluster_rec_x_set.pkl'
#         if os.path.isfile(required_output_file_location):
#            return 4
#         else:
#            bad_pop=0
#            with alive_bar(Xsteps,force_tty=True, title='Checking the Y-shift results from HTCondor') as bar:
#              for i in range(0,Xsteps):
#                   required_output_file_location=EOS_DIR+'/ANNADEA/Data/REC_SET/RH1c_'+RecBatchID+'_hit_cluster_rec_y_set_' +str(i)+'.pkl'
#                   bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
#                   bar()
#                   if os.path.isfile(required_output_file_location)!=True:
#                      bad_pop+=1
#            if bad_pop==0:
#                return 3
#            else:
#                 bad_pop=0
#                 with alive_bar(Ysteps*Xsteps,force_tty=True, title='Checking the Z-shift results from HTCondor') as bar:
#                         for j in range(0,Ysteps):
#                              for i in range(0,Xsteps):
#                                   required_output_file_location=EOS_DIR+'/ANNADEA/Data/REC_SET/RH1b_'+RecBatchID+'_hit_cluster_rec_z_set_'+str(j)+'_' +str(i)+'.pkl'
#                                   bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
#                                   bar()
#                                   if os.path.isfile(required_output_file_location)!=True:
#                                      bad_pop+=1
#                 if bad_pop==0:
#                    return 2
#                 else:
#                     bad_pop=0
#                     with alive_bar(Zsteps*Ysteps*Xsteps,force_tty=True, title='Checking the results from HTCondor') as bar:
#                         for k in range(0,Zsteps):
#                             for j in range(0,Ysteps):
#                                  for i in range(0,Xsteps):
#                                       required_output_file_location=EOS_DIR+'/ANNADEA/Data/REC_SET/RH1a_'+RecBatchID+'_hit_cluster_rec_set_'+str(k)+'_' +str(j)+'_' +str(i)+'.pkl'
#                                       bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
#                                       bar()
#                                       if os.path.isfile(required_output_file_location)!=True:
#                                          bad_pop+=1
#                     if bad_pop==0:
#                        return 1
#     return 0
#
def AutoPilot(wait_min, interval_min, max_interval_tolerance,AFS,EOS,path,o,pfx,sfx,ID,loop_params,OptionHeader,OptionLine,Sub_File,Exception=['',''], Log=False, GPU=False):
     print(UF.TimeStamp(),'Going on an autopilot mode for ',wait_min, 'minutes while checking HTCondor every',interval_min,'min',bcolors.ENDC)
     wait_sec=wait_min*60
     interval_sec=interval_min*60
     intervals=int(math.ceil(wait_sec/interval_sec))
     for interval in range(1,intervals+1):
         time.sleep(interval_sec)
         print(UF.TimeStamp(),"Scheduled job checkup...") #Progress display
         bad_pop=UF.CreateCondorJobs(AFS,EOS,
                                    path,
                                    o,
                                    pfx,
                                    sfx,
                                    ID,
                                    loop_params,
                                    OptionHeader,
                                    OptionLine,
                                    Sub_File,
                                    False,
                                    Exception,
                                    Log,
                                    GPU)
         if len(bad_pop)>0:
               print(UF.TimeStamp(),bcolors.WARNING+'Autopilot status update: There are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
               if interval%max_interval_tolerance==0:
                  for bp in bad_pop:
                      UF.SubmitJobs2Condor(bp)
                  print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
         else:
              return True
     return False

if Mode=='RESET':
    print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
    HTCondorTag="SoftUsed == \"ANNADEA-MUTr-"+TrainSampleID+"\""
    UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MUTr1_'+TrainSampleID, ['MUTr1a','MUTr1b'], HTCondorTag)
    status=0
    FreshStart=False
else:
    #status=CheckStatus()
    print('WIP')
print(UF.TimeStamp(),'There are 5 stages (0-4) of this script',status,bcolors.ENDC)
print(UF.TimeStamp(),'Current status has a code',status,bcolors.ENDC)
#
status=1
while status<4:
      if status==1:
          print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
          print(UF.TimeStamp(),bcolors.BOLD+'Stage 1:'+bcolors.ENDC+' Sending hit cluster to the HTCondor, so tack segment combination pairs can be formed...')
          OptionHeader = [ " --MaxSegments ", " --MaxSLG "," --MaxSTG "," --VetoMotherTrack "]
          OptionLine = [MaxSegments, MaxSLG, MaxSTG,'"'+str(VetoMotherTrack)+'"']
          bad_pop=UF.CreateCondorJobs(AFS_DIR,EOS_DIR,
                                    '/ANNADEA/Data/TRAIN_SET/',
                                    'RawSeedsRes',
                                    'MUTr1a',
                                    '.csv',
                                    TrainSampleID,
                                    [len(JobSets),
                                     int(JobSets[j][2])],
                                    OptionHeader,
                                    OptionLine,
                                    'MUTr1a_GenerateRawSelectedSeeds_Sub.py',
                                    False,
                                    [" --PlateZ ",JobSets])
          if len(bad_pop)==0:
              FreshStart=False
              print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 1 has successfully completed'+bcolors.ENDC)
              status=2


          if FreshStart:
              if (TotJobs)==len(bad_pop):
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
                      bad_pop=UF.CreateCondorJobs(AFS_DIR,EOS_DIR,
                                    '/ANNADEA/Data/TRAIN_SET/',
                                    'RawSeedsRes',
                                    'MUTr1a',
                                    '.csv',
                                    TrainSampleID,
                                    [len(JobSets),
                                     int(JobSets[j][2])],
                                    OptionHeader,
                                    OptionLine,
                                    'MUTr1a_GenerateRawSelectedSeeds_Sub.py',
                                    True,
                                    [" --PlateZ ",JobSets])
                      for bp in bad_pop:
                          UF.SubmitJobs2Condor(bp)
                  else:
                     if AutoPilot(600,10,Patience,AFS_DIR,EOS_DIR,'/ANNADEA/Data/TRAIN_SET/','RawSeedsRes','MUTr1a','.csv',TrainSampleID,[len(JobSets),
                                     int(JobSets[j][2])],OptionHeader,OptionLine,'MUTr1a_GenerateRawSelectedSeeds_Sub.py',[" --PlateZ ",JobSets],False,False):
                         FreshStart=False
                         print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 0 has successfully completed'+bcolors.ENDC)
                         status=2
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
                           UF.SubmitJobs2Condor(bp)
                      print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
                      if AutoPilot(600,10,Patience,AFS_DIR,EOS_DIR,'/ANNADEA/Data/TRAIN_SET/','RawSeedsRes','MUTr1a','.csv',TrainSampleID,[len(JobSets),
                                     int(JobSets[j][2])],OptionHeader,OptionLine,'MUTr1a_GenerateRawSelectedSeeds_Sub.py',[" --PlateZ ",JobSets],False,False):
                          FreshStart=False
                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 1 has successfully completed'+bcolors.ENDC)
                          status=2
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage 1 is uncompleted...'+bcolors.ENDC)
                          status=6
                          break
                   else:
                      if AutoPilot(int(UserAnswer),10,Patience,AFS_DIR,EOS_DIR,'/ANNADEA/Data/TRAIN_SET/','RawSeedsRes','MUTr1a','.csv',TrainSampleID,[len(JobSets),
                                     int(JobSets[j][2])],OptionHeader,OptionLine,'MUTr1a_GenerateRawSelectedSeeds_Sub.py',[" --PlateZ ",JobSets],False,False):
                          FreshStart=False
                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 1 has successfully completed'+bcolors.ENDC)
                          status=2
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage 1 is uncompleted...'+bcolors.ENDC)
                          status=6
                          break
          else:
            if (TotJobs)==len(bad_pop):
                 bad_pop=UF.CreateCondorJobs(AFS_DIR,EOS_DIR,
                                    '/ANNADEA/Data/TRAIN_SET/',
                                    'RawSeedsRes',
                                    'MUTr1a',
                                    '.csv',
                                    TrainSampleID,
                                    [len(JobSets),
                                     int(JobSets[j][2])],
                                    OptionHeader,
                                    OptionLine,
                                    'MUTr1a_GenerateRawSelectedSeeds_Sub.py',
                                    True,
                                    [" --PlateZ ",JobSets])
                 for bp in bad_pop:
                          UF.SubmitJobs2Condor(bp)


                 if AutoPilot(600,10,Patience,AFS_DIR,EOS_DIR,'/ANNADEA/Data/TRAIN_SET/','RawSeedsRes','MUTr1a','.csv',TrainSampleID,[len(JobSets),
                                     int(JobSets[j][2])],OptionHeader,OptionLine,'MUTr1a_GenerateRawSelectedSeeds_Sub.py',[" --PlateZ ",JobSets],False,False):
                        FreshStart=False
                        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 1 has successfully completed'+bcolors.ENDC)
                        status=2
                 else:
                     print(UF.TimeStamp(),bcolors.FAIL+'Stage 1 is uncompleted...'+bcolors.ENDC)
                     status=6
                     break

            elif len(bad_pop)>0:
                      for bp in bad_pop:
                           UF.SubmitJobs2Condor(bp)
                      if AutoPilot(600,10,Patience,AFS_DIR,EOS_DIR,'/ANNADEA/Data/TRAIN_SET/','RawSeedsRes','MUTr1a','.csv',TrainSampleID,[len(JobSets),
                                     int(JobSets[j][2])],OptionHeader,OptionLine,'MUTr1a_GenerateRawSelectedSeeds_Sub.py',[" --PlateZ ",JobSets],False,False):
                          FreshStart=False
                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 1 has successfully completed'+bcolors.ENDC)
                          status=2
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage 1 is uncompleted...'+bcolors.ENDC)
                          status=6
                          break

      if status==2:
        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.BOLD+'Stage 2:'+bcolors.ENDC+' Collecting and de-duplicating the results from stage 1')
        with alive_bar(TotJobs,force_tty=True, title='Checking the results from HTCondor') as bar:
            for j in range(0,len(JobSets)): #//Temporarily measure to save space
                if len(Meta.JobSets[j])>3:
                   Meta.JobSets[j]=Meta.JobSets[j][:4]
                   Meta.JobSets[j][3]=[]
                else:
                   Meta.JobSets[j].append([])
                for sj in range(0,int(JobSets[j][2])):
                   output_file_location=EOS_DIR+'/ANNADEA/Data/TRAIN_SET/MUTr1a_'+TrainSampleID+'_RawSeeds_'+str(j)+'_'+str(sj)+'.csv'
                   bar.text = f'-> Collecting the file : {output_file_location}...'
                   bar()
                   if os.path.isfile(output_file_location)==False:
                      Meta.JobSets[j].append(0)
                      continue #Skipping because not all jobs necesseraly produce the required file (if statistics are too low)
                   else:
                    result=pd.read_csv(output_file_location,names = ['Segment_1','Segment_2', 'Seed_Type'])
                    Records=len(result.axes[0])
                    print(UF.TimeStamp(),'Set',str(j),'and subset', str(sj), 'contains', Records, 'seeds',bcolors.ENDC)
                    result["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(result['Segment_1'], result['Segment_2'])]
                    result.drop_duplicates(subset="Seed_ID",keep='first',inplace=True)
                    result.drop(result.index[result['Segment_1'] == result['Segment_2']], inplace = True)
                    result.drop(["Seed_ID"],axis=1,inplace=True)
                    Records_After_Compression=len(result.axes[0])
                    if Records>0:
                      Compression_Ratio=int((Records_After_Compression/Records)*100)
                    else:
                      Compression_Ratio=0
                    print(UF.TimeStamp(),'Set',str(j),'and subset', str(sj), 'compression ratio is ', Compression_Ratio, ' %',bcolors.ENDC)
                    fractions=int(math.ceil(Records_After_Compression/MaxSegments))
                    Meta.JobSets[j][3].append(fractions)
                    for f in range(0,fractions):
                     new_output_file_location=EOS_DIR+'/ANNADEA/Data/TRAIN_SET/MUTr1a_'+TrainSampleID+'_SelectedSeeds_'+str(j)+'_'+str(sj)+'_'+str(f)+'.csv'
                     result[(f*MaxSegments):min(Records_After_Compression,((f+1)*MaxSegments))].to_csv(new_output_file_location,index=False)
        FreshStart=False
        print(UF.PickleOperations(TrainSampleOutputMeta,'w', Meta)[1])
        JobSets=Meta.JobSets
        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
        status=3
      if status==3:
         print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
         print(UF.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' Taking the list of seeds previously generated by Stage 2, converting them into Emulsion Objects and doing more rigorous selection')
         OptionHeader = [" --MaxSTG ", " --MaxSLG ", " --MaxDOCA ", " --MaxAngle "," --ModelName "]
         OptionLine = [MaxSTG, MaxSLG, MaxDOCA, MaxAngle,'"'+str(ModelName)+'"']
         bad_pop=UF.CreateCondorJobs(AFS_DIR,EOS_DIR,
                                    '/ANNADEA/Data/TRAIN_SET/',
                                    'RefinedSeeds',
                                    'MUTr1b',
                                    '.pkl',
                                    TrainSampleID,
                                    [len(JobSets),
                                     int(JobSets[j][2])],
                                    OptionHeader,
                                    OptionLine,
                                    'MUTr1a_GenerateRawSelectedSeeds_Sub.py',
                                    False,
                                    [" --PlateZ ",JobSets])

         bad_pop=[]
         with alive_bar(TotJobs,force_tty=True, title='Checking the results from HTCondor') as bar:
             for j in range(0,len(JobSets)): #//Temporarily measure to save space
                 for sj in range(0,int(JobSets[j][2])):
                     for f in range(0,int(JobSets[j][3][sj])):
                        required_output_file_location=EOS_DIR+'/ANNADEA/Data/TRAIN_SET/MUTr1b_'+TrainSampleID+'_RefinedSeeds_'+str(j)+'_'+str(sj)+'_'+str(f)+'.pkl'
                        bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                        bar()
                        OptionHeader = [' --Set ', ' --SubSet ', ' --Fraction ', ' --EOS ', " --AFS ", " --MaxSTG ", " --MaxSLG ", " --MaxDOCA ", " --MaxAngle "," --ModelName "]
                        OptionLine = [j, sj, f, EOS_DIR, AFS_DIR, MaxSTG, MaxSLG, MaxDOCA, MaxAngle,'"'+str(ModelName)+'"']
                        SHName = AFS_DIR + '/HTCondor/SH/SH_MUTr1_' + str(j) + '_' + str(sj) + '_' + str(f) +'.sh'
                        SUBName = AFS_DIR + '/HTCondor/SUB/SUB_MUTr1_' + str(j) + '_' + str(sj) +  '_' + str(f) +'.sub'
                        MSGName = AFS_DIR + '/HTCondor/MSG/MSG_MUTr1_' + str(j) + '_' + str(sj) +'_' + str(f)
                        ScriptName = AFS_DIR + '/Code/Utilities/MUTr1b_RefineSeeds_Sub.py '
                        if os.path.isfile(required_output_file_location)!=True:
                           bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'ANNADEA-RH-'+TrainSampleID, False,False])
         print(bad_pop)
         status=6
#
#         if FreshStart:
#             if (Xsteps)==len(bad_pop):
#                  print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
#                  print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
#                  print(bcolors.BOLD+'If you would like to wait please enter enter the maximum wait time in minutes'+bcolors.ENDC)
#                  print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
#                  UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
#                  print(UF.TimeStamp(),'Submitting jobs to HTCondor... ',bcolors.ENDC)
#                  if UserAnswer=='E':
#                       print(UF.TimeStamp(),'OK, exiting now then')
#                       exit()
#                  if UserAnswer=='R':
#                      for i in range(0,Xsteps):
#                                ptionHeader = [' --Y_ID_Max ', ' --X_ID ', ' --EOS ', " --AFS ", ' --RecBatchID ']
#                                OptionLine = [Ysteps, '$1', EOS_DIR, AFS_DIR, RecBatchID]
#                                SHName = AFS_DIR + '/HTCondor/SH/SH_RH1c_'+ RecBatchID+ '.sh'
#                                SUBName = AFS_DIR + '/HTCondor/SUB/SUB_RH1c_'+ RecBatchID+'.sub'
#                                MSGName = AFS_DIR + '/HTCondor/MSG/MSG_RH1c_' + RecBatchID
#                                ScriptName = AFS_DIR + '/Code/Utilities/RH1c_LinkSegmentsY_Sub.py '
#                                UF.SubmitJobs2Condor([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, Xsteps, 'ANNADEA-RH-'+RecBatchID, False,False])
#                  else:
#                     if AutoPilot2(600,10,Patience):
#                         FreshStart=False
#                         print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
#                         status=3
#                     else:
#                         print(UF.TimeStamp(),bcolors.FAIL+'Stage 2 is uncompleted...'+bcolors.ENDC)
#                         status=6
#                         break
#
#             elif len(bad_pop)>0:
#                    print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
#                    print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
#                    print(bcolors.BOLD+'If you would like to wait please enter enter the maximum wait time in minutes'+bcolors.ENDC)
#                    print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
#                    UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
#                    if UserAnswer=='E':
#                        print(UF.TimeStamp(),'OK, exiting now then')
#                        exit()
#                    if UserAnswer=='R':
#                       for bp in bad_pop:
#                            UF.SubmitJobs2Condor(bp)
#                       print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
#                       if AutoPilot2(600,10,Patience):
#                          FreshStart=False
#                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
#                          status=3
#                       else:
#                          print(UF.TimeStamp(),bcolors.FAIL+'Stage 2 is uncompleted...'+bcolors.ENDC)
#                          status=6
#                          break
#                    else:
#
#                       if AutoPilot2(int(UserAnswer),10,Patience):
#                          FreshStart=False
#                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
#                          status=3
#                       else:
#                          print(UF.TimeStamp(),bcolors.FAIL+'Stage 2 is uncompleted...'+bcolors.ENDC)
#                          status=6
#                          break
#
#             elif len(bad_pop)==0:
#                 FreshStart=False
#                 print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
#                 status=3
#         else:
#             if (Xsteps)==len(bad_pop):
#                  print(UF.TimeStamp(),'Submitting jobs to HTCondor... ',bcolors.ENDC)
#                  for i in range(0,Xsteps):
#                                ptionHeader = [' --Y_ID_Max ', ' --X_ID ', ' --EOS ', " --AFS ", ' --RecBatchID ']
#                                OptionLine = [Ysteps, '$1', EOS_DIR, AFS_DIR, RecBatchID]
#                                SHName = AFS_DIR + '/HTCondor/SH/SH_RH1c_'+ RecBatchID+ '.sh'
#                                SUBName = AFS_DIR + '/HTCondor/SUB/SUB_RH1c_'+ RecBatchID+'.sub'
#                                MSGName = AFS_DIR + '/HTCondor/MSG/MSG_RH1c_' + RecBatchID
#                                ScriptName = AFS_DIR + '/Code/Utilities/RH1c_LinkSegmentsY_Sub.py '
#                                UF.SubmitJobs2Condor([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, Xsteps, 'ANNADEA-RH-'+RecBatchID, False,False])
#                  if AutoPilot2(600,10,Patience):
#                         FreshStart=False
#                         print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
#                         status=3
#                  else:
#                      print(UF.TimeStamp(),bcolors.FAIL+'Stage 2 is uncompleted...'+bcolors.ENDC)
#                      status=6
#                      break
#
#             elif len(bad_pop)>0:
#                       for bp in bad_pop:
#                            UF.SubmitJobs2Condor(bp)
#                       print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
#                       if AutoPilot2(600,10,Patience):
#                          FreshStart=False
#                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
#                          status=3
#                       else:
#                           print(UF.TimeStamp(),bcolors.FAIL+'Stage 2 is uncompleted...'+bcolors.ENDC)
#                           status=6
#                           break
#             elif len(bad_pop)==0:
#                 FreshStart=False
#                 print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
#                 status=3
#
#
#
#     if status==3:
#         print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
#         print(UF.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' Sending hit cluster to the HTCondor, so the reconstructed clusters can be merged along x-axis')
#         bad_pop=[]
#         required_output_file_location=EOS_DIR+'/ANNADEA/Data/REC_SET/RH1d_'+RecBatchID+'_hit_cluster_rec_x_set.pkl'
#         OptionHeader = [' --X_ID_Max ', ' --EOS ', " --AFS ", ' --RecBatchID ']
#         OptionLine = [Xsteps, EOS_DIR, AFS_DIR, RecBatchID]
#         SHName = AFS_DIR + '/HTCondor/SH/SH_RH1d_'+RecBatchID+'.sh'
#         SUBName = AFS_DIR + '/HTCondor/SUB/SUB_RH1d_'+RecBatchID+'.sub'
#         MSGName = AFS_DIR + '/HTCondor/MSG/MSG_RH1d_' + RecBatchID
#         ScriptName = AFS_DIR + '/Code/Utilities/RH1d_LinkSegmentsX_Sub.py '
#         if os.path.isfile(required_output_file_location)!=True:
#             bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'ANNADEA-RH-'+RecBatchID, False,False])
#         if FreshStart:
#             if len(bad_pop)>0:
#                    print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor job remaining'+bcolors.ENDC)
#                    print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
#                    print(bcolors.BOLD+'If you would like to wait please enter enter the maximum wait time in minutes'+bcolors.ENDC)
#                    print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
#                    UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
#                    if UserAnswer=='E':
#                        print(UF.TimeStamp(),'OK, exiting now then')
#                        exit()
#                    if UserAnswer=='R':
#                       for bp in bad_pop:
#                            UF.SubmitJobs2Condor(bp)
#                       print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
#                       if AutoPilot3(600,10,Patience):
#                          FreshStart=False
#                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 3 has successfully completed'+bcolors.ENDC)
#                          status=4
#                       else:
#                           print(UF.TimeStamp(),bcolors.FAIL+'Stage 3 is uncompleted...'+bcolors.ENDC)
#                           status=6
#                           break
#
#                    else:
#                       if AutoPilot3(int(UserAnswer),10,Patience):
#                          FreshStart=False
#                          print(UF.TimeStamp(),bcolors.BOLD+'Stage 3 has successfully completed'+bcolors.ENDC)
#                          status=4
#                       else:
#                           print(UF.TimeStamp(),bcolors.FAIL+'Stage 3 is uncompleted...'+bcolors.ENDC)
#                           status=6
#                           break
#
#             elif len(bad_pop)==0:
#                 FreshStart=False
#                 print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 3 has successfully completed'+bcolors.ENDC)
#                 status=4
#
#         else:
#             if len(bad_pop)>0:
#                       for bp in bad_pop:
#                            UF.SubmitJobs2Condor(bp)
#                       print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
#                       if AutoPilot3(600,10,Patience):
#                          FreshStart=False
#                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 3 has successfully completed'+bcolors.ENDC)
#                          status=4
#                       else:
#                           print(UF.TimeStamp(),bcolors.FAIL+'Stage 3 is uncompleted...'+bcolors.ENDC)
#                           status=6
#                           break
#
#             elif len(bad_pop)==0:
#                 FreshStart=False
#                 print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 3 has successfully completed'+bcolors.ENDC)
#                 status=4
#
#     if status==4:
#        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
#        print(UF.TimeStamp(),bcolors.BOLD+'Stage 4:'+bcolors.ENDC+' Using the results from previous steps to map merged trackIDs to the original reconstruction file')
#        try:
#            FirstFile=EOS_DIR+'/ANNADEA/Data/REC_SET/RH1d_'+RecBatchID+'_hit_cluster_rec_x_set.pkl'
#            print(UF.TimeStamp(),'Loading the object ',bcolors.OKBLUE+FirstFile+bcolors.ENDC)
#            FirstFileRaw=UF.PickleOperations(FirstFile,'r', 'N/A')
#            FirstFile=FirstFileRaw[0]
#            TrackMap=FirstFile.RecTracks
#            input_file_location=args.f
#            print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
#            Data=pd.read_csv(input_file_location,header=0)
#            Data[PM.Hit_ID] = Data[PM.Hit_ID].astype(str)
#            if SliceData:
#                   CutData=Data.drop(Data.index[(Data[PM.x] > Xmax) | (Data[PM.x] < Xmin) | (Data[PM.y] > Ymax) | (Data[PM.y] < Ymin)])
#                   OtherData=Data.drop(Data.index[(Data[PM.x] <= Xmax) | (Data[PM.x] >= Xmin) | (Data[PM.y] <= Ymax) | (Data[PM.y] >= Ymin)])
#            try:
#              CutData.drop(['ANN_Brick_ID','ANN_Track_ID'],axis=1,inplace=True)
#            except Exception as e: print(UF.TimeStamp(), bcolors.WARNING+str(e)+bcolors.ENDC)
#
#            CutData=pd.merge(CutData,TrackMap,how='left', left_on=[PM.Hit_ID], right_on=['HitID'])
#            CutData.drop(['HitID'],axis=1,inplace=True)
#            Data=pd.concat([CutData,OtherData])
#            output_file_location=EOS_DIR+'/ANNADEA/Data/REC_SET/'+RecBatchID+'_RH_OUTPUT.csv'
#            Data.to_csv(output_file_location,index=False)
#            print(UF.TimeStamp(), bcolors.OKGREEN+"The tracked data has been written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
#
#            if Log!='NO':
#                         print(UF.TimeStamp(),'Since the logging was requested, ANN average recombination performance across the clusters is being calculated...')
#                         fake_results_1=[]
#                         fake_results_2=[]
#                         fake_results_3=[]
#                         fake_results_4=[]
#                         fake_results_5=[]
#                         fake_results_6=[]
#                         fake_results_7=[]
#                         truth_results_1=[]
#                         truth_results_2=[]
#                         truth_results_3=[]
#                         truth_results_4=[]
#                         truth_results_5=[]
#                         truth_results_6=[]
#                         truth_results_7=[]
#                         precision_results_1=[]
#                         precision_results_2=[]
#                         precision_results_3=[]
#                         precision_results_4=[]
#                         precision_results_5=[]
#                         precision_results_6=[]
#                         precision_results_7=[]
#                         recall_results_1=[]
#                         recall_results_2=[]
#                         recall_results_3=[]
#                         recall_results_4=[]
#                         recall_results_5=[]
#                         recall_results_6=[]
#                         recall_results_7=[]
#                         with alive_bar(Zsteps*Ysteps*Xsteps,force_tty=True, title='Collecting all clusters together...') as bar:
#                             for k in range(0,Zsteps):
#                                 for j in range(0,Ysteps):
#                                     for i in range(0,Xsteps):
#                                         bar()
#                                         input_file_location=EOS_DIR+'/ANNADEA/Data/REC_SET/RH1a_'+RecBatchID+'_hit_cluster_rec_set_'+str(k)+'_' +str(j)+'_' +str(i)+'.pkl'
#                                         cluster_data_raw=UF.PickleOperations(input_file_location, 'r', 'N/A')
#                                         cluster_data=cluster_data_raw[0]
#                                         result_temp=cluster_data.RecStats
#                                         fake_results_1.append(int(result_temp[1][0]))
#                                         fake_results_2.append(int(result_temp[1][1]))
#                                         fake_results_3.append(int(result_temp[1][2]))
#                                         fake_results_4.append(int(result_temp[1][3]))
#                                         fake_results_5.append(int(result_temp[1][4]))
#                                         fake_results_6.append(int(result_temp[1][5]))
#                                         fake_results_7.append(int(result_temp[1][6]))
#                                         truth_results_1.append(int(result_temp[2][0]))
#                                         truth_results_2.append(int(result_temp[2][1]))
#                                         truth_results_3.append(int(result_temp[2][2]))
#                                         truth_results_4.append(int(result_temp[2][3]))
#                                         truth_results_5.append(int(result_temp[2][4]))
#                                         truth_results_6.append(int(result_temp[2][5]))
#                                         truth_results_7.append(int(result_temp[2][6]))
#                                         if (int(result_temp[2][6])+int(result_temp[1][6]))>0:
#                                             precision_results_1.append(int(result_temp[2][0])/(int(result_temp[2][0])+int(result_temp[1][0])))
#                                             precision_results_2.append(int(result_temp[2][1])/(int(result_temp[2][1])+int(result_temp[1][1])))
#                                             precision_results_3.append(int(result_temp[2][2])/(int(result_temp[2][2])+int(result_temp[1][2])))
#                                             precision_results_4.append(int(result_temp[2][3])/(int(result_temp[2][3])+int(result_temp[1][3])))
#                                             precision_results_5.append(int(result_temp[2][4])/(int(result_temp[2][4])+int(result_temp[1][4])))
#                                             precision_results_6.append(int(result_temp[2][5])/(int(result_temp[2][5])+int(result_temp[1][5])))
#                                             precision_results_7.append(int(result_temp[2][6])/(int(result_temp[2][6])+int(result_temp[1][6])))
#                                         if int(result_temp[2][2])>0:
#                                             recall_results_1.append(int(result_temp[2][0])/(int(result_temp[2][2])))
#                                             recall_results_2.append(int(result_temp[2][1])/(int(result_temp[2][2])))
#                                             recall_results_3.append(int(result_temp[2][2])/(int(result_temp[2][2])))
#                                             recall_results_4.append(int(result_temp[2][3])/(int(result_temp[2][2])))
#                                             recall_results_5.append(int(result_temp[2][4])/(int(result_temp[2][2])))
#                                             recall_results_6.append(int(result_temp[2][5])/(int(result_temp[2][2])))
#                                             recall_results_7.append(int(result_temp[2][6])/(int(result_temp[2][2])))
#                                         else:
#                                                continue
#                                         label=result_temp[0]
#                                         label.append('Original # of valid Combinations')
#                             print(UF.TimeStamp(),bcolors.OKGREEN+'ANN reconstruction results have been compiled and presented bellow:'+bcolors.ENDC)
#                             print(tabulate([[label[0], np.average(fake_results_1), np.average(truth_results_1), np.sum(truth_results_1)/(np.sum(fake_results_1)+np.sum(truth_results_1)), np.std(precision_results_1), np.average(recall_results_1), np.std(recall_results_1)], \
#                             [label[1], np.average(fake_results_2), np.average(truth_results_2), np.sum(truth_results_2)/(np.sum(fake_results_2)+np.sum(truth_results_2)), np.std(precision_results_2), np.average(recall_results_2), np.std(recall_results_2)], \
#                             [label[2], np.average(fake_results_3), np.average(truth_results_3), np.sum(truth_results_3)/(np.sum(fake_results_3)+np.sum(truth_results_3)), np.std(precision_results_3), np.average(recall_results_3), np.std(recall_results_3)], \
#                             [label[3], np.average(fake_results_4), np.average(truth_results_4), np.sum(truth_results_4)/(np.sum(fake_results_4)+np.sum(truth_results_4)), np.std(precision_results_4), np.average(recall_results_4), np.std(recall_results_4)],\
#                             [label[4], np.average(fake_results_5), np.average(truth_results_5), np.sum(truth_results_5)/(np.sum(fake_results_5)+np.sum(truth_results_5)), np.std(precision_results_5), np.average(recall_results_5), np.std(recall_results_5)], \
#                             [label[5], np.average(fake_results_6), np.average(truth_results_6), np.sum(truth_results_6)/(np.sum(fake_results_6)+np.sum(truth_results_6)), np.std(precision_results_6), np.average(recall_results_6), np.std(recall_results_6)], \
#                             [label[6], np.average(fake_results_7), np.average(truth_results_7), np.sum(truth_results_7)/(np.sum(fake_results_7)+np.sum(truth_results_3)), np.std(precision_results_7), np.average(recall_results_7), np.std(recall_results_7)]], \
#                             headers=['Step', 'Avg # Fake edges', 'Avg # of Genuine edges', 'Avg precision', 'Precision std','Avg recall', 'Recall std' ], tablefmt='orgtbl'))
#
#            if Log=='KALMAN':
#                         print(UF.TimeStamp(),'Since the logging was requested, FEDRA average recombination performance across the clusters is being calculated...')
#                         fake_results_1=[]
#                         fake_results_2=[]
#                         fake_results_3=[]
#                         fake_results_4=[]
#                         truth_results_1=[]
#                         truth_results_2=[]
#                         truth_results_3=[]
#                         truth_results_4=[]
#                         precision_results_1=[]
#                         precision_results_2=[]
#                         precision_results_3=[]
#                         precision_results_4=[]
#                         recall_results_1=[]
#                         recall_results_2=[]
#                         recall_results_3=[]
#                         recall_results_4=[]
#                         with alive_bar(Zsteps*Ysteps*Xsteps,force_tty=True, title='Collating the results') as bar:
#                             for k in range(0,Zsteps):
#                                 for j in range(0,Ysteps):
#                                     for i in range(0,Xsteps):
#                                         bar()
#                                         input_file_location=EOS_DIR+'/ANNADEA/Data/REC_SET/RH1a_'+RecBatchID+'_hit_cluster_rec_set_'+str(k)+'_' +str(j)+'_' +str(i)+'.pkl'
#                                         cluster_data_raw=UF.PickleOperations(input_file_location, 'r', 'N/A')
#                                         cluster_data=cluster_data_raw[0]
#                                         result_temp=cluster_data.KalmanRecStats
#                                         fake_results_1.append(int(result_temp[1][0]))
#                                         fake_results_2.append(int(result_temp[1][1]))
#                                         fake_results_3.append(int(result_temp[1][2]))
#                                         fake_results_4.append(int(result_temp[1][3]))
#                                         truth_results_1.append(int(result_temp[2][0]))
#                                         truth_results_2.append(int(result_temp[2][1]))
#                                         truth_results_3.append(int(result_temp[2][2]))
#                                         truth_results_4.append(int(result_temp[2][3]))
#                                         if (int(result_temp[2][3])+int(result_temp[1][3]))>0:
#                                             precision_results_1.append(int(result_temp[2][0])/(int(result_temp[2][0])+int(result_temp[1][0])))
#                                             precision_results_2.append(int(result_temp[2][1])/(int(result_temp[2][1])+int(result_temp[1][1])))
#                                             precision_results_3.append(int(result_temp[2][2])/(int(result_temp[2][2])+int(result_temp[1][2])))
#                                             precision_results_4.append(int(result_temp[2][3])/(int(result_temp[2][3])+int(result_temp[1][3])))
#                                         if int(result_temp[2][2])>0:
#                                             recall_results_1.append(int(result_temp[2][0])/(int(result_temp[2][2])))
#                                             recall_results_2.append(int(result_temp[2][1])/(int(result_temp[2][2])))
#                                             recall_results_3.append(int(result_temp[2][2])/(int(result_temp[2][2])))
#                                             recall_results_4.append(int(result_temp[2][3])/(int(result_temp[2][2])))
#                                         else:
#                                                continue
#                                         label=result_temp[0]
#                                         label.append('Original # of valid Combinations')
#
#                         print(UF.TimeStamp(),bcolors.OKGREEN+'Fedra reconstruction results have been compiled and presented bellow:'+bcolors.ENDC)
#                         print(tabulate([[label[0], np.average(fake_results_1), np.average(truth_results_1), np.sum(truth_results_1)/(np.sum(fake_results_1)+np.sum(truth_results_1)), np.std(precision_results_1), np.average(recall_results_1), np.std(recall_results_1)], \
#                         [label[1], np.average(fake_results_2), np.average(truth_results_2), np.sum(truth_results_2)/(np.sum(fake_results_2)+np.sum(truth_results_2)), np.std(precision_results_2), np.average(recall_results_2), np.std(recall_results_2)], \
#                         [label[2], np.average(fake_results_3), np.average(truth_results_3), np.sum(truth_results_3)/(np.sum(fake_results_3)+np.sum(truth_results_3)), np.std(precision_results_3), np.average(recall_results_3), np.std(recall_results_3)], \
#                         [label[3], np.average(fake_results_4), np.average(truth_results_4), np.sum(truth_results_4)/(np.sum(fake_results_4)+np.sum(truth_results_4)), np.std(precision_results_4), np.average(recall_results_4), np.std(recall_results_4)]],\
#                         headers=['Step', 'Avg # Fake edges', 'Avg # of Genuine edges', 'Avg precision', 'Precision std','Avg recall', 'Recall std' ], tablefmt='orgtbl'))
#            print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 4 has successfully completed'+bcolors.ENDC)
#            status=5
#        except Exception as e:
#            print(UF.TimeStamp(),bcolors.FAIL+'Stage 4 is uncompleted due to...',+e+bcolors.ENDC)
#            status=6
#            break
if status==5:
     print(UF.TimeStamp(), bcolors.OKGREEN+"Train sample generation has been completed"+bcolors.ENDC)
     exit()
else:
     print(UF.TimeStamp(), bcolors.FAIL+"Reconstruction has not been completed as one of the processes has timed out. Please run the script again (without Reset Mode)."+bcolors.ENDC)
     exit()



