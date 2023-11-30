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
import U_UI as UI
#import UtilityFunctions as UF #This is where we keep routine utility functions
import Parameters as PM #This is where we keep framework global parameters
import pandas as pd #We use Panda for a routine data processing
pd.options.mode.chained_assignment = None #Silence annoying warnings
import math #We use it for data manipulation
import numpy as np
import os
from alive_progress import alive_bar
import argparse
import ast


UI.WelcomeMsg('Initialising ANNDEA track union module...','Filips Fedotovs (PhD student at UCL), Wenqing Xie (MSc student at UCL)','Please reach out to filips.fedotovs@cern.ch for any queries')


#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script prepares training data for training the tracking model')
parser.add_argument('--SubPause',help="How long to wait in minutes after submitting 10000 jobs?", default='60')
parser.add_argument('--SubGap',help="How long to wait in minutes after submitting 10000 jobs?", default='10000')
parser.add_argument('--LocalSub',help="Local submission?", default='N')
parser.add_argument('--RequestExtCPU',help="Would you like to request extra CPUs? How Many?", default='1')
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
parser.add_argument('--MaxMergeSize',help="Maximum size of the batches at premerging stage?", default='50000')
parser.add_argument('--ForceStatus',help="Local submission?", default='N')

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
MaxMergeSize=int(args.MaxMergeSize)
ForceStatus=args.ForceStatus
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
UI.Msg('status','Stage 0:',' Preparing the source data...')
if Log and (os.path.isfile(required_eval_file_location)==False or Mode=='RESET'):
    if os.path.isfile(EOSsubModelMetaDIR)==False:
              UI.Msg('failed',"Fail to proceed further as the model file "+EOSsubModelMetaDIR+ " has not been found...")
              exit()
    else:
           UI.Msg('location','Loading previously saved data from ',EOSsubModelMetaDIR)
           MetaInput=UI.PickleOperations(EOSsubModelMetaDIR,'r', 'N/A')
           Meta=MetaInput[0]
           MinHitsTrack=Meta.MinHitsTrack
    UI.Msg('location','Loading raw data from',initial_input_file_location)
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
    UI.Msg('result','The raw data has',total_rows,'hits')
    UI.Msg('vanilla','Removing unreconstructed hits...')
    data=data.dropna()
    final_rows=len(data)
    UI.Msg('result','The cleaned data has',final_rows,'hits')
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
         UI.Msg('vanilla','Slicing the data...')
         ValidEvents=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
         ValidEvents.drop([PM.x,PM.y,PM.z,PM.tx,PM.ty,'MC_Mother_Track_ID'],axis=1,inplace=True)
         ValidEvents.drop_duplicates(subset='Rec_Seg_ID',keep='first',inplace=True)
         data=pd.merge(data, ValidEvents, how="inner", on=['Rec_Seg_ID'])
         final_rows=len(data.axes[0])
         UI.Msg('result','The sliced data has',final_rows,'hits')
    UI.Msg('result','Removing tracks which have less than',MinHitsTrack,'hits')
    track_no_data=data.groupby(['MC_Mother_Track_ID','Rec_Seg_ID'],as_index=False).count()
    track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty],axis=1)
    track_no_data=track_no_data.rename(columns={PM.x: "Rec_Seg_No"})
    new_combined_data=pd.merge(data, track_no_data, how="left", on=['Rec_Seg_ID','MC_Mother_Track_ID'])
    new_combined_data = new_combined_data[new_combined_data.Rec_Seg_No >= MinHitsTrack]
    new_combined_data = new_combined_data.drop(["Rec_Seg_No"],axis=1)
    new_combined_data=new_combined_data.sort_values(['Rec_Seg_ID',PM.x],ascending=[1,1])
    grand_final_rows=len(new_combined_data)
    UI.Msg('result','The cleaned data has',grand_final_rows,'hits')
    new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
    new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
    new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
    new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
    new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
    new_combined_data.to_csv(required_eval_file_location,index=False)
    UI.Msg('location',"The track segment data has been created successfully and written to",required_eval_file_location)
if os.path.isfile(required_file_location)==False or Mode=='RESET':
        if os.path.isfile(EOSsubModelMetaDIR)==False:
              UI.Msg('failed',"Fail to proceed further as the model file "+EOSsubModelMetaDIR+ " has not been found...")
              exit()
        else:
           UI.Msg('location','Loading previously saved data from ',EOSsubModelMetaDIR)
           MetaInput=UI.PickleOperations(EOSsubModelMetaDIR,'r', 'N/A')
           Meta=MetaInput[0]
           MaxSLG=Meta.MaxSLG
           MaxSTG=Meta.MaxSTG
           MaxDOCA=Meta.MaxDOCA
           MaxAngle=Meta.MaxAngle
           MaxSegments=PM.MaxSegments
           MaxSeeds=PM.MaxSeeds
           VetoMotherTrack=PM.VetoMotherTrack
           MinHitsTrack=Meta.MinHitsTrack
        UI.Msg('location','Loading raw data from',initial_input_file_location)
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
        UI.Msg('result','The raw data has',total_rows,'hits')
        UI.Msg('vanilla','Removing unreconstructed hits...')
        data=data.dropna()
        final_rows=len(data.axes[0])
        UI.Msg('result','The cleaned data has',final_rows,'hits')
        data[BrickID] = data[BrickID].astype(str)
        data[TrackID] = data[TrackID].astype(str)

        data['Rec_Seg_ID'] = data[TrackID] + '-' + data[BrickID]
        data=data.drop([TrackID],axis=1)
        data=data.drop([BrickID],axis=1)
        if SliceData:
             UI.Msg('vanilla','Slicing the data...')
             ValidEvents=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
             ValidEvents.drop([PM.x,PM.y,PM.z,PM.tx,PM.ty],axis=1,inplace=True)
             ValidEvents.drop_duplicates(subset="Rec_Seg_ID",keep='first',inplace=True)
             data=pd.merge(data, ValidEvents, how="inner", on=['Rec_Seg_ID'])
             final_rows=len(data.axes[0])
             UI.Msg('result','The sliced data has',final_rows,'hits')
        UI.Msg('result','Removing tracks which have less than',MinHitsTrack,'hits')
        track_no_data=data.groupby(['Rec_Seg_ID'],as_index=False).count()
        track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty],axis=1)
        track_no_data=track_no_data.rename(columns={PM.x: "Track_No"})
        new_combined_data=pd.merge(data, track_no_data, how="left", on=["Rec_Seg_ID"])
        new_combined_data = new_combined_data[new_combined_data.Track_No >= MinHitsTrack]
        new_combined_data = new_combined_data.drop(['Track_No'],axis=1)
        new_combined_data=new_combined_data.sort_values(['Rec_Seg_ID',PM.x],ascending=[1,1])
        grand_final_rows=len(new_combined_data.axes[0])
        UI.Msg('result','The cleaned data has',grand_final_rows,'hits')
        new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
        new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
        new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
        new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
        new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
        new_combined_data.to_csv(required_file_location,index=False)
        data=new_combined_data[['Rec_Seg_ID','z']]
        UI.Msg('vanilla','Analysing the data sample in order to understand how many jobs to submit to HTCondor... ')
        data = data.groupby('Rec_Seg_ID')['z'].min()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
        data=data.reset_index()
        data = data.groupby('z')['Rec_Seg_ID'].count()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
        data=data.reset_index()
        data=data.sort_values(['z'],ascending=True)
        data['Sub_Sets']=np.ceil(data['Rec_Seg_ID']/PM.MaxSegments)
        data['Sub_Sets'] = data['Sub_Sets'].astype(int)
        data = data.values.tolist()
        UI.Msg('location',"The track segment data has been created successfully and written to",required_file_location)
        Meta=UI.TrainingSampleMeta(RecBatchID)
        Meta.IniTrackSeedMetaData(MaxSLG,MaxSTG,MaxDOCA,MaxAngle,data,MaxSegments,VetoMotherTrack,MaxSeeds,MinHitsTrack)
        Meta.UpdateStatus(0)
        print(UI.PickleOperations(RecOutputMeta,'w', Meta)[1])
        UI.Msg('completed','Stage 0 has successfully completed')
elif os.path.isfile(RecOutputMeta)==True:
    UI.Msg('location','Loading previously saved data from ',RecOutputMeta)
    MetaInput=UI.PickleOperations(RecOutputMeta,'r', 'N/A')
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

#The function bellow helps to automate the submission process
if Mode=='RESET':
    UI.Msg('vanilla','Performing the cleanup... ')
    HTCondorTag="SoftUsed == \"ANNDEA-EUTr1a-"+RecBatchID+"\""
    UI.EvalCleanUp(AFS_DIR, EOS_DIR, 'EUTr1a_'+RecBatchID, ['EUTr1a','EUTr1b'], HTCondorTag)
    HTCondorTag="SoftUsed == \"ANNDEA-RUTr1a-"+RecBatchID+"\""
    UI.RecCleanUp(AFS_DIR, EOS_DIR, 'RUTr1a_'+RecBatchID, ['RUTr1a',RecBatchID+'_REC_LOG.csv'], HTCondorTag)
    FreshStart=False
    UI.UpdateStatus(0,Meta,RecOutputMeta)
    Status=0
else:
    UI.Msg('vanilla','Analysing the current script status...')
    Status=Meta.Status[-1]
if ForceStatus!='N':
    Status=int(ForceStatus)
UI.Msg('vanilla','Current stage is '+str(Status)+'...')
################ Set the execution sequence for the script
Program=[]
if Log:
    ###### Stage 0
    prog_entry=[]
    job_sets=[]
    prog_entry.append(' Sending eval seeds to HTCondor...')
    UI.Msg('location',"The track segment data has been created successfully and written to",initial_input_file_location)
    data=pd.read_csv(required_eval_file_location,header=0,usecols=['Rec_Seg_ID'])
    UI.Msg('vanilla','Analysing data... ')
    data.drop_duplicates(subset="Rec_Seg_ID",keep='first',inplace=True)  #Keeping only starting hits for each track record (we do not require the full information about track in this script)
    Records=len(data.axes[0])
    Sets=int(np.ceil(Records/MaxSegments))
    prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TEST_SET/','RawSeedsRes','EUTr1a','.csv',RecBatchID,Sets,'EUTr1a_GenerateRawSelectedSeeds_Sub.py'])
    prog_entry.append([" --MaxSegments ", " --VetoMotherTrack "])
    prog_entry.append([MaxSegments, '"'+str(VetoMotherTrack)+'"'])
    prog_entry.append(Sets)
    prog_entry.append(LocalSub)
    prog_entry.append(['',''])
    prog_entry.append(False)
    prog_entry.append(False)
    if Mode=='RESET':
        print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Delete'))
    #Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
    print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Create'))
    Program.append(prog_entry)
    # ###### Stage 1
    Program.append('Custom - PickE')

if Mode=='CLEANUP':
    UI.UpdateStatus(19,Meta,RecOutputMeta)
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
prog_entry.append(False)
prog_entry.append(False)
if Mode=='RESET':
        print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Delete'))
    #Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Create'))
Program.append(prog_entry)
Program.append('Custom - PickR')

####### Stage 4
for md in ModelName:
    Program.append(md)

Program.append('Custom - Merging')

Program.append('Custom - TrackMapping')

while Status<len(Program):
    if Program[Status][:6]!='Custom' and (Program[Status] in ModelName)==False:
        #Standard process here
        Result=UI.StandardProcess(Program,Status,SubGap,SubPause,RequestExtCPU,JobFlavour,ReqMemory,time_int,Patience,Meta,RecOutputMeta)
        if Result[0]:
             FreshStart=Result[1]
        else:
             Status=20
             break
    elif Program[Status]=='Custom - PickE':
        UI.Msg('status','Stage '+str(Status),': Collecting and de-duplicating the results from previous stage'+str(Status-1))
        UI.Msg('location','Loading preselected data from ',initial_input_file_location)
        data=pd.read_csv(required_eval_file_location,header=0,usecols=['Rec_Seg_ID'])
        UI.Msg('vanilla','Analysing data... ')
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
                       print(UI.TimeStamp(),'Set',str(i), 'contains', len(result), 'seeds')
                    else:
                        output_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/Temp_EUTr1a'+'_'+RecBatchID+'_'+str(0)+'/EUTr1a_'+RecBatchID+'_RawSeeds_'+str(i)+'.csv'
                        new_result=pd.read_csv(output_file_location,names = ['Segment_1','Segment_2'])
                        print(UI.TimeStamp(),'Set',str(i), 'contains', len(new_result), 'seeds')
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
        print(UI.TimeStamp(),'Set',str(i), 'compression ratio is ', Compression_Ratio, ' %')
        new_output_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/EUTr1b_'+RecBatchID+'_SEED_TRUTH_COMBINATIONS.csv'
        result.to_csv(new_output_file_location,index=False)
        eval_no=len(result)
        rec_data=pd.read_csv(required_file_location,header=0,
                    usecols=['Rec_Seg_ID'])
        rec_data.drop_duplicates(subset="Rec_Seg_ID",keep='first',inplace=True)
        rec_data.drop_duplicates(keep='first',inplace=True)
        rec_no=len(rec_data)
        rec_no=(rec_no**2)-rec_no-eval_no
        UI.LogOperations(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv', 'w', [['Step_No','Step_Desc','Fake_Seeds','Truth_Seeds','Precision','Recall'],[1,'Initial Sampling',rec_no,eval_no,eval_no/(rec_no+eval_no),1.0]])
        UI.Msg('location',"The process log has been created successfully and written to ",EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv')
        FreshStart=False
        UI.Msg('completed','Stage '+str(Status)+' has successfully completed')
        UI.UpdateStatus(Status+1,Meta,RecOutputMeta)
    elif Program[Status]=='Custom - PickR':
        UI.Msg('status','Stage '+str(Status),': Collecting and de-duplicating the results from previous stage '+str(Status-1)+'...')
        min_i=0
        UI.Msg('vanilla','Analysing the data sample in order to understand how many jobs to submit to HTCondor... ')
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
                Meta=UI.PickleOperations(RecOutputMeta,'r', 'N/A')[0]
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
                    print(UI.TimeStamp(),'Set',str(i),'and subset', str(j), 'contains', Records, 'seeds')
                    result["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(result['Segment_1'], result['Segment_2'])]
                    result.drop_duplicates(subset="Seed_ID",keep='first',inplace=True)
                    result.drop(result.index[result['Segment_1'] == result['Segment_2']], inplace = True)
                    result.drop(["Seed_ID"],axis=1,inplace=True)
                    Records_After_Compression=len(result)
                    if Records>0:
                      Compression_Ratio=int((Records_After_Compression/Records)*100)
                    else:
                      Compression_Ratio=0
                    print(UI.TimeStamp(),'Set',str(i),'and subset', str(j), 'compression ratio is ', Compression_Ratio, ' %')
                    fractions=int(math.ceil(Records_After_Compression/MaxSeeds))
                    Meta.JobSets[i][3].append(fractions)
                    for k in range(0,fractions):
                     new_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RUTr1a'+'_'+RecBatchID+'_'+str(i)+'/RUTr1a_'+RecBatchID+'_SelectedSeeds_'+str(i)+'_'+str(j)+'_'+str(k)+'.csv'
                     result[(k*MaxSeeds):min(Records_After_Compression,((k+1)*MaxSeeds))].to_csv(new_output_file_location,index=False)
                print(UI.PickleOperations(RecOutputMeta,'w', Meta)[1])
        if Log:
         try:
             UI.Msg('vanilla','Initiating the logging...')
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
             UI.LogOperations(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv', 'a', [[2,'SLG and STG cuts',rec_no,eval_no,eval_no/(rec_no+eval_no),eval_no/len(eval_data)]])
             UI.Msg('location',"The log has been created successfully at ",EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv')
         except:
             UI.Msg('failed','Log creation has failed')
        FreshStart=False
        UI.Msg('completed','Stage '+str(Status)+' has successfully completed')
        UI.UpdateStatus(Status+1,Meta,RecOutputMeta)

    elif Program[Status]=='Custom - Merging':
        input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1c_'+RecBatchID+'_Fit_Seeds.pkl'
        UI.Msg('location',"Loading the fit track seeds from the file ",input_file_location)
        base_data=UI.PickleOperations(input_file_location,'r','N/A')[0]
        UI.Msg('success',"Loading is successful, there are "+str(len(base_data))+" fit seeds...")
        with alive_bar(len(base_data),force_tty=True, title="Stripping seeds with low ML acceptance...") as bar:
            for tr in range(len(base_data)):
                bar()
                for t in range(len(base_data[tr].Hits)):
                    for h in range(len(base_data[tr].Hits[t])):
                        base_data[tr].Hits[t][h]=base_data[tr].Hits[t][h][2] #Remove scaling factors
        base_data=[tr for tr in base_data if tr.Fit >= Acceptance]
        output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1d_'+RecBatchID+'_Fit_Filtered_Seeds.pkl'

        UI.Msg('success',"The refining was successful, "+str(len(base_data))+" track seeds remain...")
        print(UI.PickleOperations(output_file_location,'w', base_data)[1])
        if CalibrateAcceptance:
            print(UI.TimeStamp(),'Calibrating the acceptance...')
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
        UI.Msg('status','Stage '+str(Status),': Merging the segment seeds')
        Program_Dummy=[]
        prog_entry=[]
        prog_entry.append(' Sending selected fit seeds to HTCondor for the pre-merging...')
        prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','Fit_Merged_Seeds','RUTr1d','.pkl',RecBatchID,1,'RUTr1d_MergeSeeds_Sub.py'])
        prog_entry.append([" --MaxSLG "])
        prog_entry.append([MaxSLG])
        prog_entry.append(1)
        prog_entry.append(LocalSub)
        prog_entry.append(['',''])
        prog_entry.append(True)
        prog_entry.append(False)
        for dum in range(0,Status):
            Program_Dummy.append('DUM')
        Program_Dummy.append(prog_entry)
        if Mode=='RESET':
            print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Delete'))
        #Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
        print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Create'))
        Result=UI.StandardProcess(Program_Dummy,Status,SubGap,SubPause,4,'nextweek','4 GB',time_int,Patience,Meta,RecOutputMeta)

        if Result:
             input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RUTr1d_'+RecBatchID+'_0/RUTr1d_'+RecBatchID+'_Fit_Merged_Seeds_0.pkl'
             base_data=UI.PickleOperations(input_file_location,'r','N/A')[0]
             for v in range(0,len(base_data)):
                 base_data[v].AssignANNTrUID(v)
             output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1e_'+RecBatchID+'_Union_Tracks.pkl'
             output_csv_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1e_'+RecBatchID+'_Union_Tracks.csv'
             csv_out=[['Old_Track_ID','Temp_Track_Quarter','Temp_Track_ID']]
             for Tr in base_data:
                 for TH in Tr.Header:
                     csv_out.append([TH,RecBatchID,Tr.UTrID])

             UI.Msg('location',"Saving the results into the file",output_csv_location)
             UI.LogOperations(output_csv_location,'w', csv_out)
             UI.Msg('location',"Saving the results into the file",output_file_location)
             print(UI.PickleOperations(output_file_location,'w',base_data)[1])
             if args.Log=='Y':
                print(UI.TimeStamp(),'Initiating the logging...')
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
                UI.LogOperations(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv', 'a', [[4+len(ModelName),'Track Seed Merging',rec_no,eval_no,eval_no/(rec_no+eval_no),eval_no/len(eval_data)]])
                UI.Msg('location',"The log data has been created successfully and written to",EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv')
             UI.UpdateStatus(Status+1,Meta,RecOutputMeta)


    elif Program[Status]=='Custom - TrackMapping':
                raw_name=initial_input_file_location[:-4]
                for l in range(len(raw_name)-1,0,-1):
                    if raw_name[l]=='/':
                        print(l,raw_name)
                        break
                raw_name=raw_name[l+1:]
                final_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+raw_name+'_'+RecBatchID+'_MERGED.csv'
                UI.Msg('status','Stage '+str(Status),': Taking the list of seeds previously generated by Stage '+str(Status-1)+' and mapping them to the input data')
                UI.Msg('location','Loading raw data from',initial_input_file_location)
                required_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1_'+RecBatchID+'_TRACK_SEGMENTS.csv'
                data=pd.read_csv(args.f,header=0)
                if BrickID=='':
                    ColUse=[PM.Hit_ID,TrackID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
                else:
                    ColUse=[PM.Hit_ID,TrackID,BrickID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
                data=data[ColUse]
                if BrickID=='':
                    data[BrickID]='D'
                UI.Msg('location','Loading mapped data from',EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1f_'+RecBatchID+'_Union_Tracks.csv')
                map_data=pd.read_csv(EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1f_'+RecBatchID+'_Union_Tracks.csv',header=0)
                total_rows=len(data.axes[0])
                print(UI.TimeStamp(),'The raw data has ',total_rows,' hits')
                print(UI.TimeStamp(),'Removing unreconstructed hits...')
                data.dropna(subset=[TrackID],inplace=True)
                final_rows=len(data)
                print(UI.TimeStamp(),'The cleaned data has ',final_rows,' hits')
                data[TrackID] = data[TrackID].astype(str)
                data[BrickID] = data[BrickID].astype(str)
                if os.path.isfile(EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1f_'+RecBatchID+'_Mapped_Tracks_Temp.csv')==False:
                    data['Rec_Seg_ID'] = data[TrackID] + '-' + data[BrickID]
                    print(UI.TimeStamp(),'Resolving duplicated hits...')
                    selected_combined_data=pd.merge(data, map_data, how="left", left_on=["Rec_Seg_ID"], right_on=['Old_Track_ID'])
                    Hit_Map_Stats=selected_combined_data[['Temp_Track_Quarter','Temp_Track_ID',PM.z,PM.Hit_ID]] #Calculating the stats
                    Hit_Map_Stats=Hit_Map_Stats.groupby(['Temp_Track_Quarter','Temp_Track_ID']).agg({PM.z:pd.Series.nunique,PM.Hit_ID: pd.Series.nunique}).reset_index() #Calculate the number fo unique plates and hits
                    Ini_No_Tracks=len(Hit_Map_Stats)
                    UI.Msg('result','The initial number of tracks is ',str(Ini_No_Tracks))
                    Hit_Map_Stats=Hit_Map_Stats.rename(columns={PM.z: "No_Plates",PM.Hit_ID:"No_Hits"}) #Renaming the columns so they don't interfere once we join it back to the hit map
                    Hit_Map_Stats=Hit_Map_Stats[Hit_Map_Stats.No_Plates >= PM.MinHitsTrack]
                    Prop_No_Tracks=len(Hit_Map_Stats)
                    UI.Msg('result','After dropping single hit tracks, left ',str(Prop_No_Tracks),' tracks...')
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
                    print(UI.TimeStamp(),'Removing problematic hits...')
                    Bad_Tracks=pd.merge(Bad_Tracks,Bad_Tracks_Head,how='inner',on = ['Temp_Track_Quarter','Temp_Track_ID'])
                    print(UI.TimeStamp(),'Calculating x and y coordinates of the fitted line for all plates in the track...')
                    #Calculating x and y coordinates of the fitted line for all plates in the track
                    Bad_Tracks['new_x']=Bad_Tracks['ax']+(Bad_Tracks[PM.z]*Bad_Tracks['t1x'])+((Bad_Tracks[PM.z]**2)*Bad_Tracks['t2x'])
                    Bad_Tracks['new_y']=Bad_Tracks['ay']+(Bad_Tracks[PM.z]*Bad_Tracks['t1y'])+((Bad_Tracks[PM.z]**2)*Bad_Tracks['t2y'])
                    #Calculating how far hits deviate from the fit polynomial
                    print(UI.TimeStamp(),'Calculating how far hits deviate from the fit polynomial...')
                    Bad_Tracks['d_x']=Bad_Tracks[PM.x]-Bad_Tracks['new_x']
                    Bad_Tracks['d_y']=Bad_Tracks[PM.y]-Bad_Tracks['new_y']
                    Bad_Tracks['d_r']=Bad_Tracks['d_x']**2+Bad_Tracks['d_y']**2
                    Bad_Tracks['d_r'] = Bad_Tracks['d_r'].astype(float)
                    Bad_Tracks['d_r']=np.sqrt(Bad_Tracks['d_r']) #Absolute distance
                    Bad_Tracks=Bad_Tracks[['Temp_Track_Quarter','Temp_Track_ID',PM.z,PM.Hit_ID,'d_r']]
                    #Sort the tracks and their hits by Track ID, Plate and distance to the perfect line
                    print(UI.TimeStamp(),'Sorting the tracks and their hits by Track ID, Plate and distance to the perfect line...')
                    Bad_Tracks.sort_values(['Temp_Track_Quarter','Temp_Track_ID',PM.z,'d_r'],ascending=[0,0,1,1],inplace=True)
                    before=len(Bad_Tracks)
                    print(UI.TimeStamp(),'Before de-duplicattion we had ',before,' hits involving problematic tracks.')
                    #If there are two hits per plate we will keep the one which is closer to the line
                    Bad_Tracks.drop_duplicates(subset=['Temp_Track_Quarter','Temp_Track_ID',PM.z],keep='first',inplace=True)
                    after=len(Bad_Tracks)
                    print(UI.TimeStamp(),'Now their number was dropped to ',after,' hits.')
                    Bad_Tracks=Bad_Tracks[['Temp_Track_Quarter','Temp_Track_ID',PM.Hit_ID]]
                    Good_Tracks=pd.concat([Good_Tracks,Bad_Tracks]) #Combine all ANNDEA tracks together
                    Good_Tracks.to_csv(EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1e_'+RecBatchID+'_Mapped_Tracks_Temp.csv',index=False)
                    data.drop(["Rec_Seg_ID"],axis=1,inplace=True)
                else:
                    Good_Tracks=pd.read_csv(EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1e_'+RecBatchID+'_Mapped_Tracks_Temp.csv')
                print(UI.TimeStamp(),'Mapping data...')
                data=pd.read_csv(args.f,header=0)
                new_combined_data=pd.merge(data, Good_Tracks, how="left", on=[PM.Hit_ID])
                if BrickID!='':
                    new_combined_data['Temp_Track_Quarter'] = new_combined_data['Temp_Track_Quarter'].fillna(new_combined_data[BrickID])
                else:
                    new_combined_data['Temp_Track_Quarter'] = new_combined_data['Temp_Track_Quarter'].fillna('D')
                new_combined_data['Temp_Track_ID'] = new_combined_data['Temp_Track_ID'].fillna(new_combined_data[TrackID])
                new_combined_data=new_combined_data.rename(columns={'Temp_Track_Quarter': RecBatchID+'_Brick_ID','Temp_Track_ID': RecBatchID+'_Track_ID'})
                new_combined_data.to_csv(final_output_file_location,index=False)
                UI.Msg('location',"The merged track data has been created successfully and written to",final_output_file_location)
                UI.UpdateStatus(Status+1,Meta,RecOutputMeta)

    else:
        for md in range(len(ModelName)):
            if Program[Status]==ModelName[md]:
                if md==0:
                    prog_entry=[]
                    job_sets=[]
                    JobSet=[]
                    TotJobs=0
                    Program_Dummy=[]
                    Meta=UI.PickleOperations(RecOutputMeta,'r', 'N/A')[0]
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
                    prog_entry.append(False)
                    prog_entry.append(False)
                    for dum in range(0,Status):
                        Program_Dummy.append('DUM')
                    Program_Dummy.append(prog_entry)
                    if Mode=='RESET':
                        print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Delete'))
                    #Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
                    print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Create'))
                    Result=UI.StandardProcess(Program_Dummy,Status,SubGap,SubPause,RequestExtCPU,JobFlavour,ReqMemory,time_int,Patience,Meta,RecOutputMeta)
                    if Result:
                        UI.Msg('status','Stage '+str(Status),': Analysing the fitted seeds')
                        JobSet=[]
                        for i in range(len(JobSets)):
                             JobSet.append([])
                             for j in range(len(JobSets[i][3])):
                                 JobSet[i].append(JobSets[i][3][j])
                        if md==len(ModelName)-1:
                            base_data = None
                            if Log:
                                rec_list=[]
                            with alive_bar(len(JobSets),force_tty=True, title='Checking the results from HTCondor') as bar:
                             for i in range(0,len(JobSet)):

                                    bar()
                                    for j in range(len(JobSet[i])):
                                             for k in range(JobSet[i][j]):
                                                  required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RUTr1'+ModelName[md]+'_'+RecBatchID+'_'+str(i)+'/RUTr1'+ModelName[md]+'_'+RecBatchID+'_RefinedSeeds_'+str(i)+'_'+str(j)+'_'+str(k)+'.pkl'
                                                  new_data=UI.PickleOperations(required_output_file_location,'r','N/A')[0]
                                                  print(UI.TimeStamp(),'Set',str(i)+'_'+str(j)+'_'+str(k), 'contains', len(new_data), 'seeds')
                                                  if base_data == None:
                                                        base_data = new_data
                                                  else:
                                                        base_data+=new_data
                                                  if Log:
                                                      for rd in new_data:
                                                            rec_list.append([rd.Header[0],rd.Header[1]])
                            Records=len(base_data)
                            print(UI.TimeStamp(),'The output '+str(i)+' contains', Records, 'raw images')
                            base_data=list(set(base_data))
                            Records_After_Compression=len(base_data)
                            if Records>0:
                                                          Compression_Ratio=int((Records_After_Compression/Records)*100)
                            else:
                                                          CompressionRatio=0
                            print(UI.TimeStamp(),'The output '+str(i)+'  compression ratio is ', Compression_Ratio, ' %',)
                            output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1c_'+RecBatchID+'_Fit_Seeds.pkl'
                            print(UI.PickleOperations(output_file_location,'w',base_data)[1])
                            if Log:
                             UI.Msg('vanilla','Initiating the logging...')
                             eval_data_file=EOS_DIR+'/ANNDEA/Data/TEST_SET/EUTr1b_'+RecBatchID+'_SEED_TRUTH_COMBINATIONS.csv'
                             eval_data=pd.read_csv(eval_data_file,header=0,usecols=['Segment_1','Segment_2'])
                             eval_data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(eval_data['Segment_1'], eval_data['Segment_2'])]
                             eval_data.drop(['Segment_1'],axis=1,inplace=True)
                             eval_data.drop(['Segment_2'],axis=1,inplace=True)
                             rec_no=0
                             eval_no=0
                             rec = pd.DataFrame(rec_list, columns = ['Segment_1','Segment_2'])
                             rec["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(rec['Segment_1'], rec['Segment_2'])]
                             rec.drop(['Segment_1'],axis=1,inplace=True)
                             rec.drop(['Segment_2'],axis=1,inplace=True)
                             rec_eval=pd.merge(eval_data, rec, how="inner", on=['Seed_ID'])
                             eval_no=len(rec_eval)
                             rec_no=(len(rec)-len(rec_eval))
                             UI.LogOperations(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv', 'a', [[3+md,ModelName[md],rec_no,eval_no,eval_no/(rec_no+eval_no),eval_no/len(eval_data)]])
                             UI.Msg('location',"The log data has been created successfully and written to",EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv')
                        else:
                            log_rec_no=0
                            with alive_bar(len(JobSets),force_tty=True, title='Checking the results from HTCondor') as bar:
                             for i in range(0,len(JobSet)):
                                    base_data = None
                                    bar()
                                    for j in range(len(JobSet[i])):
                                             for k in range(JobSet[i][j]):
                                                  required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RUTr1'+ModelName[md]+'_'+RecBatchID+'_'+str(i)+'/RUTr1'+ModelName[md]+'_'+RecBatchID+'_RefinedSeeds_'+str(i)+'_'+str(j)+'_'+str(k)+'.pkl'
                                                  new_data=UI.PickleOperations(required_output_file_location,'r','N/A')[0]
                                                  print(UI.TimeStamp(),'Set',str(i)+'_'+str(j)+'_'+str(k), 'contains', len(new_data), 'seeds')
                                                  if base_data == None:
                                                        base_data = new_data
                                                  else:
                                                        base_data+=new_data
                                    if base_data==None:
                                        Records=0
                                    else:
                                        Records=len(base_data)
                                        print(UI.TimeStamp(),'The output '+str(i)+' contains', Records, 'raw images')
                                        base_data=list(set(base_data))
                                        Records_After_Compression=len(base_data)
                                    if Records>0:
                                          Compression_Ratio=int((Records_After_Compression/Records)*100)
                                          output_split=int(np.ceil(Records_After_Compression/PM.MaxSegments))
                                          for os_itr in range(output_split):
                                                output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RUTr1'+ModelName[md]+'_'+RecBatchID+'_0/RUTr1'+str(ModelName[md+1])+'_'+RecBatchID+'_Input_Seeds_'+str(i)+'_'+str(os_itr)+'.pkl'
                                                print(UI.PickleOperations(output_file_location,'w',base_data[os_itr*PM.MaxSegments:(os_itr+1)*PM.MaxSegments])[1])
                                          if Log:
                                                 UI.Msg('vanilla','Initiating the logging for set '+str(i)+' ...')
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
                                                 log_rec_no+=rec_no
                                    else:
                                          CompressionRatio=0
                                          print(UI.TimeStamp(),'The output '+str(i)+'  compression ratio is ', Compression_Ratio, ' %, skipping this step')


                            if Log:
                                         UI.Msg('vanilla','Initiating the logging...')
                                         UI.LogOperations(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv', 'a', [[3+md,ModelName[md],log_rec_no,eval_no,eval_no/(log_rec_no+eval_no),eval_no/len(eval_data)]])
                                         UI.Msg('location',"The log data has been created successfully and written to",EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv')
                else:
                    prog_entry=[]
                    TotJobs=[]
                    NTotJobs=0
                    Program_Dummy=[]
                    for i in range(60):
                        keep_testing=True
                        NJobs=0
                        while keep_testing:
                            test_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RUTr1'+ModelName[md-1]+'_'+RecBatchID+'_0/RUTr1'+str(ModelName[md])+'_'+RecBatchID+'_Input_Seeds_'+str(i)+'_'+str(NJobs)+'.pkl'
                            if os.path.isfile(test_file_location):
                                NJobs+=1
                                NTotJobs+=1
                            else:
                                keep_testing=False
                        TotJobs.append(NJobs)
                    prog_entry.append(' Sending tracks to the HTCondor, so track segment combination pairs can be formed...')
                    prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','OutputSeeds','RUTr1'+ModelName[md],'.pkl',RecBatchID,TotJobs,'RUTr1b_RefineSeeds_Sub.py'])

                    prog_entry.append([" --MaxSTG ", " --MaxSLG ", " --MaxDOCA ", " --MaxAngle "," --ModelName "," --FirstTime "])
                    prog_entry.append([MaxSTG, MaxSLG, MaxDOCA, MaxAngle,'"'+ModelName[md]+'"', ModelName[md-1]])
                    prog_entry.append(NTotJobs)
                    prog_entry.append(LocalSub)
                    prog_entry.append(['',''])
                    prog_entry.append(False)
                    prog_entry.append(False)
                    for dum in range(0,Status):
                        Program_Dummy.append('DUM')
                    Program_Dummy.append(prog_entry)
                    if Mode=='RESET':
                        print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Delete'))
                    #Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
                    print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Create'))
                    Result=UI.StandardProcess(Program_Dummy,Status,SubGap,SubPause,RequestExtCPU,JobFlavour,ReqMemory,time_int,Patience,Meta,RecOutputMeta)
                    if Result:
                        UI.Msg('status','Stage '+str(Status),': Analysing the fitted seeds')
                        base_data = None
                        with alive_bar(len(TotJobs),force_tty=True, title='Checking the results from HTCondor') as bar:
                         for i in range(len(TotJobs)):
                             bar()
                             for j in range(TotJobs[i]):

                                              required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RUTr1'+ModelName[md]+'_'+RecBatchID+'_'+str(i)+'/RUTr1'+ModelName[md]+'_'+RecBatchID+'_OutputSeeds_'+str(i)+'_'+str(j)+'.pkl'
                                              new_data=UI.PickleOperations(required_output_file_location,'r','N/A')[0]
                                              print(UI.TimeStamp(),'Set',str(i),'and subset',str(j),'contains', len(new_data), 'seeds')
                                              if base_data == None:
                                                    base_data = new_data
                                              else:
                                                    base_data+=new_data
                        Records=len(base_data)
                        print(UI.TimeStamp(),'The output contains', Records, 'fit images')
                        base_data=list(set(base_data))
                        Records_After_Compression=len(base_data)
                        if Records>0:
                                              Compression_Ratio=int((Records_After_Compression/Records)*100)
                        else:
                                              CompressionRatio=0
                        print(UI.TimeStamp(),'The output compression ratio is ', Compression_Ratio, ' %')

                        if md==len(ModelName)-1:
                                output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1c_'+RecBatchID+'_Fit_Seeds.pkl'
                                print(UI.PickleOperations(output_file_location,'w',base_data)[1])
                        else:
                                output_split=int(np.ceil(Records_After_Compression/PM.MaxSegments))
                                for os_itr in range(output_split):
                                    output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RUTr1'+ModelName[md]+'_'+RecBatchID+'_0/RUTr1'+str(ModelName[md+1])+'_'+RecBatchID+'_Input_Seeds_'+str(os_itr)+'.pkl'
                                    print(UI.PickleOperations(output_file_location,'w',base_data[os_itr*PM.MaxSegments:(os_itr+1)*PM.MaxSegments])[1])

    UI.Msg('location','Loading previously saved data from ',RecOutputMeta)
    MetaInput=UI.PickleOperations(RecOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
    Status=Meta.Status[-1]

if Status<20:
    #Removing the temp files that were generated by the process
    print(UI.TimeStamp(),'Performing the cleanup... ')
    HTCondorTag="SoftUsed == \"ANNDEA-RUTr1-"+RecBatchID+"\""
    UI.RecCleanUp(AFS_DIR, EOS_DIR, 'RUTr1_'+RecBatchID, ['RUTr1_'+RecBatchID], HTCondorTag)
    HTCondorTag="SoftUsed == \"ANNDEA-RUTr1a-"+RecBatchID+"\""
    UI.RecCleanUp(AFS_DIR, EOS_DIR, 'RUTr1a_'+RecBatchID, [], HTCondorTag)
    HTCondorTag="SoftUsed == \"ANNDEA-RUTr1b-"+RecBatchID+"\""
    UI.RecCleanUp(AFS_DIR, EOS_DIR, 'RUTr1b_'+RecBatchID, [], HTCondorTag)
    HTCondorTag="SoftUsed == \"ANNDEA-RUTr1c-"+RecBatchID+"\""
    UI.RecCleanUp(AFS_DIR, EOS_DIR, 'RUTr1c_'+RecBatchID, [], HTCondorTag)
    HTCondorTag="SoftUsed == \"ANNDEA-RUTr1e-"+RecBatchID+"\""
    UI.RecCleanUp(AFS_DIR, EOS_DIR, 'RUTr1e_'+RecBatchID, [], HTCondorTag)
    for p in Program:
        if p[:6]!='Custom' and (p in ModelName)==False:
           print(UI.TimeStamp(),UI.ManageTempFolders(p,'Delete'))
    for md in range(len(ModelName)):
                if md==0:
                    prog_entry=[]
                    job_sets=[]
                    JobSet=[]
                    TotJobs=0
                    Program_Dummy=[]
                    Meta=UI.PickleOperations(RecOutputMeta,'r', 'N/A')[0]
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
                    print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Delete'))
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
                    print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Delete')) #Deleting a specific set of created folders
    UI.Msg('success',"Segment merging has been completed")
else:
    UI.Msg('failed',"Segment merging has not been completed as one of the processes has timed out. Please run the script again (without Reset Mode).")
    exit()



