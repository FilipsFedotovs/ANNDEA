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
import U_UI as UI #This is where we keep routine utility functions
import Parameters as PM #This is where we keep framework global parameters

UI.WelcomeMsg('Initialising ANNDEA Track Union Training Sample Generation module...','Filips Fedotovs (PhD student at UCL), Wenqing Xie (MSc student at UCL)','Please reach out to filips.fedotovs@cern.ch for any queries')
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
parser.add_argument('--ForceStatus',help="Would you like the program run from specific status number? (Only for advance users)", default='N')
parser.add_argument('--RequestExtCPU',help="Would you like to request extra CPUs?", default=1)
parser.add_argument('--JobFlavour',help="Specifying the length of the HTCondor job walltime. Currently at 'workday' which is 8 hours.", default='workday')
parser.add_argument('--LocalSub',help="Local submission?", default='N')
parser.add_argument('--SubPause',help="How long to wait in minutes after submitting 10000 jobs?", default='60')
parser.add_argument('--SubGap',help="How long to wait in minutes after submitting 10000 jobs?", default='10000')
parser.add_argument('--MinHitsTrack',help="What is the minimum number of hits per track?", default=PM.MinHitsTrack)
parser.add_argument('--MaxSLG',help="Maximum allowed longitudinal gap value between segments", default='7000')
parser.add_argument('--MaxSTG',help="Maximum allowed transverse gap value between segments per SLG length", default='160')
parser.add_argument('--MaxDOCA',help="Maximum DOCA allowed", default='100')
parser.add_argument('--MaxAngle',help="Maximum magnitude of angle allowed", default='3.6')
parser.add_argument('--ReqMemory',help="How uch memory to request?", default='2 GB')
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
MaxSLG=float(args.MaxSLG)
MaxSTG=float(args.MaxSTG)
MaxDOCA=float(args.MaxDOCA)
MaxSTG=float(args.MaxSTG)
MaxAngle=float(args.MaxAngle)
ForceStatus=args.ForceStatus
RequestExtCPU=int(args.RequestExtCPU)
ReqMemory=args.ReqMemory
Xmin,Xmax,Ymin,Ymax=float(args.Xmin),float(args.Xmax),float(args.Ymin),float(args.Ymax)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0 #We don't slice data if all values are set to zero simultaneousy (which is the default setting)
LocalSub=(args.LocalSub=='Y')
if LocalSub:
   time_int=0
else:
    time_int=10


#Establishing paths
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
TrainSampleOutputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
required_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MUTr1_'+TrainSampleID+'_TRACK_SEGMENTS.csv'


########################################     Phase 1 - Create compact source file    #########################################
print(UI.TimeStamp(),bcolors.BOLD+'Stage -1:'+bcolors.ENDC+' Preparing the source data...')
if os.path.isfile(required_file_location)==False or Mode=='RESET':
        print(UI.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
        data=pd.read_csv(input_file_location,
                    header=0,
                    usecols=[TrackID,BrickID,
                            PM.x,PM.y,PM.z,PM.tx,PM.ty,
                            PM.MC_Track_ID,PM.MC_Event_ID])
        total_rows=len(data)
        print(UI.TimeStamp(),'The raw data has ',total_rows,' hits')
        print(UI.TimeStamp(),'Removing unreconstructed hits...')
        data=data.dropna()
        final_rows=len(data)
        print(UI.TimeStamp(),'The cleaned data has ',final_rows,' hits')
        data[PM.MC_Event_ID] = data[PM.MC_Event_ID].astype(str)
        data[PM.MC_Track_ID] = data[PM.MC_Track_ID].astype(str)
        data[TrackID] = data[TrackID].astype(str)
        data[BrickID] = data[BrickID].astype(str)
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
             print(UI.TimeStamp(),'Slicing the data...')
             ValidEvents=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
             ValidEvents.drop([PM.x,PM.y,PM.z,PM.tx,PM.ty,'MC_Mother_Track_ID'],axis=1,inplace=True)
             ValidEvents.drop_duplicates(subset='Rec_Seg_ID',keep='first',inplace=True)
             data=pd.merge(data, ValidEvents, how="inner", on=['Rec_Seg_ID'])
             final_rows=len(data.axes[0])
             print(UI.TimeStamp(),'The sliced data has ',final_rows,' hits')

        output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MUTr1_'+TrainSampleID+'_TRACK_SEGMENTS.csv'
        print(UI.TimeStamp(),'Removing tracks which have less than',MinHitsTrack,'hits...')
        track_no_data=data.groupby(['MC_Mother_Track_ID','Rec_Seg_ID'],as_index=False).count()
        track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty],axis=1)
        track_no_data=track_no_data.rename(columns={PM.x: "Rec_Seg_No"})
        new_combined_data=pd.merge(data, track_no_data, how="left", on=['Rec_Seg_ID','MC_Mother_Track_ID'])
        new_combined_data = new_combined_data[new_combined_data.Rec_Seg_No >= MinHitsTrack]
        new_combined_data = new_combined_data.drop(["Rec_Seg_No"],axis=1)
        new_combined_data=new_combined_data.sort_values(['Rec_Seg_ID',PM.x],ascending=[1,1])
        grand_final_rows=len(new_combined_data)
        print(UI.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
        new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
        new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
        new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
        new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
        new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
        new_combined_data.to_csv(output_file_location,index=False)
        data=new_combined_data[['Rec_Seg_ID','z']]
        print(UI.TimeStamp(),'Analysing the data sample in order to understand how many jobs to submit to HTCondor... ',bcolors.ENDC)
        data = data.groupby('Rec_Seg_ID')['z'].min()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
        data=data.reset_index()
        data = data.groupby('z')['Rec_Seg_ID'].count()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
        data=data.reset_index()
        data=data.sort_values(['z'],ascending=True)
        data['Sub_Sets']=np.ceil(data['Rec_Seg_ID']/PM.MaxSegments)
        data['Sub_Sets'] = data['Sub_Sets'].astype(int)
        data = data.values.tolist()
        print(UI.TimeStamp(), bcolors.OKGREEN+"The track segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
        Meta=UI.TrainingSampleMeta(TrainSampleID)
        Meta.IniTrackSeedMetaData(MaxSLG,MaxSTG,MaxDOCA,MaxAngle,data,PM.MaxSegments,PM.VetoMotherTrack,PM.MaxSeeds,MinHitsTrack)
        Meta.UpdateStatus(0)
        print(UI.PickleOperations(TrainSampleOutputMeta,'w', Meta)[1])
        print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
        print(UI.TimeStamp(),bcolors.OKGREEN+'Stage -1 has successfully completed'+bcolors.ENDC)
elif os.path.isfile(TrainSampleOutputMeta)==True:
    print(UI.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+TrainSampleOutputMeta+bcolors.ENDC)
    MetaInput=UI.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')
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
TotJobs=0
for j in range(0,len(JobSets)):
          for sj in range(0,int(JobSets[j][2])):
              TotJobs+=1
# ########################################     Preset framework parameters    #########################################
FreshStart=True
Program=[]

#If we chose reset mode we do a full cleanup.
# #Reconstructing a single brick can cause in gereation of 100s of thousands of files - need to make sure that we remove them.
if Mode=='RESET':
    print(UI.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
    HTCondorTag="SoftUsed == \"ANNDEA-MUTr1a-"+TrainSampleID+"\""
    UI.TrainCleanUp(AFS_DIR, EOS_DIR, 'MUTr1_'+TrainSampleID, ['MUTr1a','MUTr1b','MUTr1c','MUTr1d'], HTCondorTag)
    FreshStart=False
    UI.UpdateStatus(0,Meta,TrainSampleOutputMeta)
    Status=0
else:
    UI.Msg('vanilla','Analysing the current script status...')
    Status=Meta.Status[-1]
if ForceStatus!='N':
    Status=int(ForceStatus)

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
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','RawSeedsRes','MUTr1a','.csv',TrainSampleID,job_sets,'MUTr1a_GenerateRawSelectedSeeds_Sub.py'])
prog_entry.append([ " --MaxSegments ", " --MaxSLG "," --MaxSTG "," --VetoMotherTrack "])
prog_entry.append([MaxSegments, MaxSLG, MaxSTG,'"'+str(VetoMotherTrack)+'"'])
prog_entry.append(TotJobs)
prog_entry.append(LocalSub)
prog_entry.append([" --PlateZ ",JobSets])
prog_entry.append(False)
prog_entry.append(False)
Program.append(prog_entry)
if Mode=='RESET':
   print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Delete'))
#Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Create'))
###### Stage 1
Program.append('Custom')

###### Stage 2
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
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','RefinedSeeds','MUTr1b','.pkl',TrainSampleID,job_sets,'MUTr1b_RefineSeeds_Sub.py'])
prog_entry.append([" --MaxSTG ", " --MaxSLG ", " --MaxDOCA ", " --MaxAngle "," --ModelName "])
prog_entry.append([MaxSTG, MaxSLG, MaxDOCA, MaxAngle,'"'+str(ModelName)+'"'])
prog_entry.append(TotJobs)
prog_entry.append(LocalSub)
prog_entry.append(['',''])
prog_entry.append(False)
prog_entry.append(False)
Program.append(prog_entry)
if Mode=='RESET':
   print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Delete'))
#Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Create'))


###### Stage 3
Program.append('Custom')
###### Stage 4
Program.append('Custom')
###### Stage 5
Program.append('Custom')

print(UI.TimeStamp(),'There are '+str(len(Program)+1)+' stages (0-'+str(len(Program)+1)+') of this script',bcolors.ENDC)
print(UI.TimeStamp(),'Current stage has a code',Status,bcolors.ENDC)

while Status<len(Program):
    if Program[Status]!='Custom':
        #Standard process here
        Result=UI.StandardProcess(Program,Status,SubGap,SubPause,RequestExtCPU,JobFlavour,ReqMemory,time_int,Patience,Meta,TrainSampleOutputMeta)
        if Result[0]:
            FreshStart=Result[1]
            Status+=1
            UI.UpdateStatus(Status,Meta,TrainSampleOutputMeta)
        else:
            Status=len(Program)+1
            break

    elif Status==1:
       #Non standard processes (that don't follow the general pattern) have been coded here
        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
        print(UI.TimeStamp(),bcolors.BOLD+'Stage 1:'+bcolors.ENDC+' Collecting and de-duplicating the results from stage 1')
        min_i=0
        for i in range(0,len(JobSets)): #//Temporarily measure to save space
                   test_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/Temp_MUTr1a_'+TrainSampleID+'_'+str(i)+'/MUTr1a_'+TrainSampleID+'_SelectedSeeds_'+str(i)+'_'+str(0)+'_'+str(0)+'.csv'
                   if os.path.isfile(test_file_location):
                        min_i=max(0,i-1)
        with alive_bar(len(JobSets)-min_i,force_tty=True, title='Checking the results from HTCondor') as bar:
            for i in range(min_i,len(JobSets)): #//Temporarily measure to save space
                bar.text = f'-> Analysing set : {i}...'
                bar()
                Meta=UI.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')[0]
                MaxSLG=Meta.MaxSLG
                JobSets=Meta.JobSets
                if len(Meta.JobSets[i])>3:
                   Meta.JobSets[i]=Meta.JobSets[i][:4]
                   Meta.JobSets[i][3]=[]
                else:
                   Meta.JobSets[i].append([])
                for j in range(0,int(JobSets[i][2])):

                   output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/Temp_MUTr1a_'+TrainSampleID+'_'+str(i)+'/MUTr1a_'+TrainSampleID+'_RawSeeds_'+str(i)+'_'+str(j)+'.csv'

                   if os.path.isfile(output_file_location)==False:
                      Meta.JobSets[j].append(0)
                      continue #Skipping because not all jobs necesseraly produce the required file (if statistics are too low)
                   else:
                    result=pd.read_csv(output_file_location,names = ['Segment_1','Segment_2', 'Seed_Type'])
                    Records=len(result)
                    print(UI.TimeStamp(),'Set',str(i),'and subset', str(j), 'contains', Records, 'seeds',bcolors.ENDC)
                    result["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(result['Segment_1'], result['Segment_2'])]
                    result.drop_duplicates(subset="Seed_ID",keep='first',inplace=True)
                    result.drop(result.index[result['Segment_1'] == result['Segment_2']], inplace = True)
                    result.drop(["Seed_ID"],axis=1,inplace=True)
                    Records_After_Compression=len(result)
                    if Records>0:
                      Compression_Ratio=int((Records_After_Compression/Records)*100)
                    else:
                      Compression_Ratio=0
                    print(UI.TimeStamp(),'Set',str(i),'and subset', str(j), 'compression ratio is ', Compression_Ratio, ' %',bcolors.ENDC)
                    fractions=int(math.ceil(Records_After_Compression/MaxSeeds))
                    Meta.JobSets[i][3].append(fractions)
                    for k in range(0,fractions):
                     new_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/Temp_MUTr1a_'+TrainSampleID+'_'+str(i)+'/MUTr1a_'+TrainSampleID+'_SelectedSeeds_'+str(i)+'_'+str(j)+'_'+str(k)+'.csv'
                     result[(k*MaxSeeds):min(Records_After_Compression,((k+1)*MaxSeeds))].to_csv(new_output_file_location,index=False)
                print(UI.PickleOperations(TrainSampleOutputMeta,'w', Meta)[1])

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
        prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','RefinedSeeds','MUTr1b','.pkl',TrainSampleID,job_sets,'MUTr1b_RefineSeeds_Sub.py'])
        prog_entry.append([" --MaxSTG ", " --MaxSLG ", " --MaxDOCA ", " --MaxAngle "," --ModelName "])
        prog_entry.append([MaxSTG, MaxSLG, MaxDOCA, MaxAngle,'"'+str(ModelName)+'"'])
        prog_entry.append(TotJobs)
        prog_entry.append(LocalSub)
        prog_entry.append(['',''])
        prog_entry.append(False)
        prog_entry.append(False)
        Program[2]=prog_entry
        #############################
        FreshStart=False
        print(UI.TimeStamp(),bcolors.OKGREEN+'Stage 1 has successfully completed'+bcolors.ENDC)
        Status=2
        UI.UpdateStatus(Status)
    elif Status==3:
        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
        print(UI.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' Analysing the training samples')
        JobSet=[]
        for i in range(len(JobSets)):
             JobSet.append([])
             for j in range(len(JobSets[i][3])):
                 JobSet[i].append(JobSets[i][3][j])
        for i in range(0,len(JobSet)):
             output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MUTr1c_'+TrainSampleID+'_CompressedSeeds_'+str(i)+'.pkl'
             if os.path.isfile(output_file_location)==False:
                if os.path.isfile(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MUTr1c_'+TrainSampleID+'_Temp_Stats.csv')==False:
                   UI.LogOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MUTr1c_'+TrainSampleID+'_Temp_Stats.csv','w', [[0,0]])
                Temp_Stats=UI.LogOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MUTr1c_'+TrainSampleID+'_Temp_Stats.csv','r', '_')

                TotalImages=int(Temp_Stats[0][0])
                TrueSeeds=int(Temp_Stats[0][1])
                base_data = None
                for j in range(len(JobSet[i])):
                         for k in range(JobSet[i][j]):
                              required_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/Temp_MUTr1b_'+TrainSampleID+'_'+str(i)+'/MUTr1b_'+TrainSampleID+'_'+'RefinedSeeds'+'_'+str(i)+'_'+str(j) + '_' + str(k)+'.pkl'
                              new_data=UI.PickleOperations(required_output_file_location,'r','N/A')[0]
                              if base_data == None:
                                    base_data = new_data
                              else:
                                    base_data+=new_data
                try:
                    Records=len(base_data)
                    print(UI.TimeStamp(),'Set',str(i),'contains', Records, 'raw images',bcolors.ENDC)

                    base_data=list(set(base_data))
                    Records_After_Compression=len(base_data)
                    if Records>0:
                              Compression_Ratio=int((Records_After_Compression/Records)*100)
                    else:
                              CompressionRatio=0
                    TotalImages+=Records_After_Compression
                    TrueSeeds+=sum(1 for im in base_data if im.Label == 1)
                    print(UI.TimeStamp(),'Set',str(i),'compression ratio is ', Compression_Ratio, ' %',bcolors.ENDC)
                    print(UI.PickleOperations(output_file_location,'w',base_data)[1])
                except:
                    continue
                del new_data
                UI.LogOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MUTr1c_'+TrainSampleID+'_Temp_Stats.csv','w', [[TotalImages,TrueSeeds]])
        print(UI.TimeStamp(),bcolors.OKGREEN+'Stage 4 has successfully completed'+bcolors.ENDC)
        Status=4
        UI.UpdateStatus(Status)
        continue
    elif Status==4:
           print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
           print(UI.TimeStamp(),bcolors.BOLD+'Stage 4:'+bcolors.ENDC+' Resampling the results from the previous stage')
           print(UI.TimeStamp(),'Sampling the required number of seeds',bcolors.ENDC)
           Temp_Stats=UI.LogOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MUTr1c_'+TrainSampleID+'_Temp_Stats.csv','r', '_')
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
              output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MUTr1d_'+TrainSampleID+'_SampledCompressedSeeds_'+str(i)+'.pkl'
              input_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MUTr1c_'+TrainSampleID+'_CompressedSeeds_'+str(i)+'.pkl'
              bar.text = f'-> Resampling the file : {input_file_location}, exists...'
              bar()
              if os.path.isfile(output_file_location)==False and os.path.isfile(input_file_location):
                  base_data=UI.PickleOperations(input_file_location,'r','N/A')[0]
                  ExtractedTruth=[im for im in base_data if im.Label == 1]
                  ExtractedFake=[im for im in base_data if im.Label == 0]
                  del base_data
                  gc.collect()
                  ExtractedTruth=random.sample(ExtractedTruth,int(round(TrueSeedCorrection*len(ExtractedTruth),0)))
                  ExtractedFake=random.sample(ExtractedFake,int(round(FakeSeedCorrection*len(ExtractedFake),0)))
                  TotalData=[]
                  TotalData=ExtractedTruth+ExtractedFake
                  print(UI.PickleOperations(output_file_location,'w',TotalData)[1])
                  del TotalData
                  del ExtractedTruth
                  del ExtractedFake
                  gc.collect()
           print(UI.TimeStamp(),bcolors.OKGREEN+'Stage 4 has successfully completed'+bcolors.ENDC)
           Status=5
           UI.UpdateStatus(Status)
           continue
    elif Status==5:
           print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
           print(UI.TimeStamp(),bcolors.BOLD+'Stage 5:'+bcolors.ENDC+' Preparing the final output')
           TotalData=[]
           JobSet=[]
           for i in range(len(JobSets)):
             JobSet.append([])
             for j in range(len(JobSets[i][3])):
                 JobSet[i].append(JobSets[i][3][j])

           for i in range(0,len(JobSet)):
               input_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MUTr1d_'+TrainSampleID+'_SampledCompressedSeeds_'+str(i)+'.pkl'
               if os.path.isfile(input_file_location):
                  base_data=UI.PickleOperations(input_file_location,'r','N/A')[0]
                  TotalData+=base_data
           del base_data
           gc.collect()
           ValidationSampleSize=int(round(min((len(TotalData)*float(PM.valRatio)),PM.MaxValSampleSize),0))
           random.shuffle(TotalData)
           output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_TRACK_SEEDS_OUTPUT.pkl'
           print(UI.PickleOperations(output_file_location,'w',TotalData[:ValidationSampleSize])[1])
           TotalData=TotalData[ValidationSampleSize:]
           print(UI.TimeStamp(), bcolors.OKGREEN+"Validation Set has been saved at ",bcolors.OKBLUE+output_file_location+bcolors.ENDC,bcolors.OKGREEN+'file...'+bcolors.ENDC)
           No_Train_Files=int(math.ceil(len(TotalData)/TrainSampleSize))
           with alive_bar(No_Train_Files,force_tty=True, title='Resampling the files...') as bar:
               for SC in range(0,No_Train_Files):
                 output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_TRACK_SEEDS_OUTPUT_'+str(SC+1)+'.pkl'
                 print(UI.PickleOperations(output_file_location,'w',TotalData[(SC*TrainSampleSize):min(len(TotalData),((SC+1)*TrainSampleSize))])[1])
                 bar.text = f'-> Saving the file : {output_file_location}...'
                 bar()

           print(UI.TimeStamp(),bcolors.OKGREEN+'Stage 5 has successfully completed'+bcolors.ENDC)
           print(UI.TimeStamp(),'Would you like to delete Temporary files?')
           user_response=input()
           if user_response=='y' or user_response=='Y':
               Status=6
               UI.UpdateStatus(Status)
               continue
           else:
               print(UI.TimeStamp(), bcolors.OKGREEN+"Train sample generation has been completed"+bcolors.ENDC)
               exit() 
if Status==6:
           print(UI.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
           for p in Program:
            if p!='Custom':
               print(UI.TimeStamp(),UI.ManageTempFolders(p,'Delete'))
           HTCondorTag="SoftUsed == \"ANNDEA-MUTr1-"+TrainSampleID+"\""
           UI.TrainCleanUp(AFS_DIR, EOS_DIR, 'MUTr1_'+TrainSampleID, ['MUTr1_'+TrainSampleID], HTCondorTag)
           HTCondorTag="SoftUsed == \"ANNDEA-MUTr1c-"+TrainSampleID+"\""
           UI.TrainCleanUp(AFS_DIR, EOS_DIR, 'MUTr1c_'+TrainSampleID, ['MUTr1c_'+TrainSampleID], HTCondorTag)
           HTCondorTag="SoftUsed == \"ANNDEA-MUTr1d-"+TrainSampleID+"\""
           UI.TrainCleanUp(AFS_DIR, EOS_DIR, 'MUTr1d_'+TrainSampleID, ['MUTr1d_'+TrainSampleID], HTCondorTag)
           print(UI.TimeStamp(), bcolors.OKGREEN+"Train sample generation has been completed"+bcolors.ENDC)
           exit()
else:
    print(UI.TimeStamp(), bcolors.FAIL+"Reconstruction has not been completed as one of the processes has timed out or --ForceStatus!=0 option was chosen. Please run the script again (without Reset Mode)."+bcolors.ENDC)
    exit()



