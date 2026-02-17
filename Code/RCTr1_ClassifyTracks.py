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
    sys.path.append('/usr/lib64/python39.zip')
    sys.path.append('/usr/lib64/python3.9')
    sys.path.append('/usr/lib64/python3.9/lib-dynload')
    sys.path.append('/usr/lib64/python3.9/site-packages')
    sys.path.append('/usr/lib/python3.9/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')
import U_UI as UI
import Parameters as PM #This is where we keep framework global parameters
import pandas as pd #We use Panda for a routine data processing
pd.options.mode.chained_assignment = None #Silence annoying warnings
import math #We use it for data manipulation
import os
import time
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

UI.WelcomeMsg('Initialising ANNDEA track classification module...','Filips Fedotovs (PhD student at UCL), Wenqing Xie (MSc student at UCL)','Please reach out to filips.fedotovs@cern.ch for any queries')

#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script prepares training data for training the tracking model')
parser.add_argument('--SubPause',help="How long to wait in minutes after submitting 10000 jobs?", default='60')
parser.add_argument('--SubGap',help="How long to wait in minutes after submitting 10000 jobs?", default='10000')
parser.add_argument('--LocalSub',help="Local submission?", default='N')
parser.add_argument('--RequestExtCPU',help="Would you like to request extra CPUs? How Many?", default=1)
parser.add_argument('--JobFlavour',help="Specifying the length of the HTCondor job walltime. Currently at 'workday' which is 8 hours.", default='workday')
parser.add_argument('--TrackID',help="What track name is used?", default='ANN_Track_ID')
parser.add_argument('--BrickID',help="What brick ID name is used?", default='ANN_Brick_ID')
parser.add_argument('--Mode', help='Script will continue from the last checkpoint, unless you want to start from the scratch, then type "Reset"',default='')
parser.add_argument('--ModelName',help="What  model would you like to use?", default="1T_GMM_IC_6_150_4_ANNDEA_ID_model")
parser.add_argument('--Patience',help="How many checks to do before resubmitting the job?", default='15')
parser.add_argument('--RecBatchID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_Rec_Raw_UR.csv')
parser.add_argument('--Xmin',help="This option restricts data to only those events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option restricts data to only those events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option restricts data to only those events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option restricts data to only those events that have tracks with hits y-coordinates that are below this value", default='0')
parser.add_argument('--ReqMemory',help="Specifying the length of the HTCondor job walltime. Currently at 'workday' which is 8 hours.", default='2 GB')
parser.add_argument('--HTCondorLog',help="Local submission?", default=False,type=bool)
######################################## Parsing argument values  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
RecBatchID=args.RecBatchID
Patience=int(args.Patience)
HTCondorLog=args.HTCondorLog
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
ModelName=args.ModelName
Patience=int(args.Patience)
initial_input_file_location=args.f
Xmin,Xmax,Ymin,Ymax=float(args.Xmin),float(args.Xmax),float(args.Ymin),float(args.Ymax)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0 #We don't slice data if all values are set to zero simultaneousy (which is the default setting)
FreshStart=True
#Establishing paths
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
EOSsubModelMetaDIR=EOSsubDIR+'/'+'Models/'+ModelName+'_Meta'
RecOutputMeta=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_info.pkl'
required_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/RCTr1_'+RecBatchID+'_TRACKS.csv'
ColumnsToImport=[TrackID,BrickID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
########################################     Phase 1 - Create compact source file    #########################################

if Mode=='RESET':
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'d',['RCTr1a']))
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'c'))
elif Mode=='CLEANUP':
     print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'d',['RCTr1a']))
     exit()
else:
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'c'))
print(UI.TimeStamp(),bcolors.BOLD+'Stage 0:'+bcolors.ENDC+' Preparing the source data...')

if os.path.isfile(required_file_location)==False or Mode=='RESET':
        if os.path.isfile(EOSsubModelMetaDIR)==False:
              print(UI.TimeStamp(), bcolors.FAIL+"Fail to proceed further as the model file "+EOSsubModelMetaDIR+ " has not been found..."+bcolors.ENDC)
              exit()
        else:
           print(UI.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+EOSsubModelMetaDIR+bcolors.ENDC)
           MetaInput=UI.PickleOperations(EOSsubModelMetaDIR,'r', 'N/A')
           Meta=MetaInput[0]
           ClassHeaders=Meta.ClassHeaders
           ClassValues=Meta.ClassValues
           ClassNames=Meta.ClassNames
           MaxSegments=PM.MaxSegments
           MinHitsTrack=Meta.MinHitsTrack
        print(UI.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+initial_input_file_location+bcolors.ENDC)
        data=pd.read_csv(initial_input_file_location,
                    header=0,
                    usecols=ColumnsToImport)
        total_rows=len(data.axes[0])
        print(UI.TimeStamp(),'The raw data has ',total_rows,' hits')
        print(UI.TimeStamp(),'Removing unreconstructed hits...')
        data=data.dropna()
        final_rows=len(data.axes[0])
        print(UI.TimeStamp(),'The cleaned data has ',final_rows,' hits')
        data[BrickID] = data[BrickID].astype(str)
        data[TrackID] = data[TrackID].astype(str)
        data['Rec_Seg_ID'] = data[TrackID] + '-' + data[BrickID]
        data=data.drop([TrackID],axis=1)
        data=data.drop([BrickID],axis=1)
        RZChoice = input('Would you like to remove tracks based on the starting plate? If no, press "Enter", otherwise type "y", followed by "Enter" : ')
        if RZChoice.upper()=='Y':
            print(UI.TimeStamp(),'Removing tracks based on start point')
            data_aggregated=data.groupby(['Rec_Seg_ID'])[PM.z].min().reset_index()
            data_aggregated_show=data_aggregated.groupby([PM.z]).count().reset_index()
            data_aggregated_show=data_aggregated_show.rename(columns={'Rec_Seg_ID': "No_Tracks"})
            data_aggregated_show['PID']=data_aggregated_show[PM.z].rank(ascending=True).astype(int)
            print('A list of plates and the number of tracks starting on them is listed bellow:')
            print(data_aggregated_show.to_string())
            RPChoice = input('Enter the list of plates separated by comma that you want to remove followed by "Enter" : ')
            if len(RPChoice)>1:
                RPChoice=ast.literal_eval(RPChoice)
            else:
                RPChoice=[int(RPChoice)]
            TracksZdf = pd.DataFrame(RPChoice, columns = ['PID'], dtype=int)
            data_aggregated_show=pd.merge(data_aggregated_show,TracksZdf,how='inner',on='PID')

            data_aggregated_show.drop(['No_Tracks','PID'],axis=1,inplace=True)
            data_aggregated=pd.merge(data_aggregated,data_aggregated_show,how='inner',on=PM.z)
            data_aggregated=data_aggregated.rename(columns={PM.z: 'Tracks_Remove'})

            data=pd.merge(data, data_aggregated, how="left", on=['Rec_Seg_ID'])

            data=data[data['Tracks_Remove'].isnull()]
            data=data.drop(['Tracks_Remove'],axis=1)
        final_rows=len(data.axes[0])
        print(UI.TimeStamp(),'After removing tracks that start at the specific plates we have',final_rows,' hits left')
        RLChoice = input('Would you like to remove tracks based on their length in traverse plates? If no, press "Enter", otherwise type "y", followed by "Enter" : ')
        if RLChoice.upper()=='Y':
            print(UI.TimeStamp(),'Removing tracks based on length')
            data_aggregated=data[['Rec_Seg_ID',PM.z]]
            data_aggregated_l=data_aggregated.groupby(['Rec_Seg_ID'])[PM.z].min().reset_index().rename(columns={PM.z: "min_z"})
            data_aggregated_r=data_aggregated.groupby(['Rec_Seg_ID'])[PM.z].max().reset_index().rename(columns={PM.z: "max_z"})
            data_aggregated=pd.merge(data_aggregated_l,data_aggregated_r,how='inner', on='Rec_Seg_ID')
            data_aggregated_list_z=data[[PM.z]].groupby([PM.z]).count().reset_index()
            data_aggregated_list_z['PID_l']=data_aggregated_list_z[PM.z].rank(ascending=True).astype(int)
            data_aggregated_list_z['PID_r']=data_aggregated_list_z['PID_l']
            data_aggregated=pd.merge(data_aggregated,data_aggregated_list_z[[PM.z,'PID_l']], how='inner', left_on='min_z', right_on=PM.z)
            data_aggregated=pd.merge(data_aggregated,data_aggregated_list_z[[PM.z,'PID_r']], how='inner', left_on='max_z', right_on=PM.z)[['Rec_Seg_ID','PID_l','PID_r']]
            data_aggregated['track_len']=data_aggregated['PID_r']-data_aggregated['PID_l']+1
            data_aggregated=data_aggregated[['Rec_Seg_ID','track_len']]
            data_aggregated_show=data_aggregated.groupby(['track_len']).count().reset_index()
            data_aggregated_show=data_aggregated_show.rename(columns={'Rec_Seg_ID': "No_Tracks"})
            print('Track length distribution:')
            print(data_aggregated_show.to_string())
            RTLChoice = input('Enter the list of track lengths to exclude" : ')
            if len(RTLChoice)>1:
                RTLChoice=ast.literal_eval(RTLChoice)
            else:
                RTLChoice=[int(RTLChoice)]
            TracksLdf = pd.DataFrame(RTLChoice, columns = ['track_len'], dtype=int)
            data_aggregated=pd.merge(data_aggregated,TracksLdf,how='inner',on='track_len')
            data=pd.merge(data, data_aggregated, how="left", on=['Rec_Seg_ID'])
            data=data[data['track_len'].isnull()]
            data=data.drop(['track_len'],axis=1)
        final_rows=len(data.axes[0])
        print(UI.TimeStamp(),'After removing tracks with specific plate lengths we have',final_rows,' hits left')
        if SliceData:
             print(UI.TimeStamp(),'Slicing the data...')
             ValidEvents=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
             ValidEvents.drop([PM.x,PM.y,PM.z,PM.tx,PM.ty],axis=1,inplace=True)
             ValidEvents.drop_duplicates(subset="Rec_Seg_ID",keep='first',inplace=True)
             data=pd.merge(data, ValidEvents, how="inner", on=['Rec_Seg_ID'])
             final_rows=len(data.axes[0])
             print(UI.TimeStamp(),'The sliced data has ',final_rows,' hits')
        print(UI.TimeStamp(),'Removing tracks which have less than',MinHitsTrack,'hits...')
        final_rows=len(data.axes[0])
        print(UI.TimeStamp(),'After removing tracks with number of hits we have',final_rows,' hits left')
        track_no_data=data.groupby(['Rec_Seg_ID'],as_index=False).count()
        track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty],axis=1)
        track_no_data=track_no_data.rename(columns={PM.x: "Track_No"})
        new_combined_data=pd.merge(data, track_no_data, how="left", on=["Rec_Seg_ID"])
        new_combined_data = new_combined_data[new_combined_data.Track_No >= MinHitsTrack]
        new_combined_data = new_combined_data.drop(['Track_No'],axis=1)
        new_combined_data=new_combined_data.sort_values(['Rec_Seg_ID',PM.x],ascending=[1,1])
        grand_final_rows=len(new_combined_data.axes[0])
        print(UI.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
        new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
        new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
        new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
        new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
        new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
        new_combined_data.to_csv(required_file_location,index=False)
        data=new_combined_data[['Rec_Seg_ID']]
        print(UI.TimeStamp(),'Analysing the data sample in order to understand how many jobs to submit to HTCondor... ',bcolors.ENDC)
        data.drop_duplicates(subset='Rec_Seg_ID',keep='first',inplace=True)
        data = data.values.tolist()
        no_submissions=math.ceil(len(data)/MaxSegments)
        print(UI.TimeStamp(), bcolors.OKGREEN+"The track segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+required_file_location+bcolors.ENDC)
        Meta=UI.JobMeta(RecBatchID)
        Meta.UpdateJobMeta(['ClassHeaders','ClassNames','ClassValues','MaxSegments','JobSets','MinHitsTrack'], [ClassHeaders,ClassNames,ClassValues,PM.MaxSegments,no_submissions,MinHitsTrack])
        Meta.UpdateStatus(0)
        print(UI.PickleOperations(RecOutputMeta,'w', Meta)[1])
        print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
        print(UI.TimeStamp(),bcolors.OKGREEN+'Stage 0 has successfully completed'+bcolors.ENDC)

elif os.path.isfile(RecOutputMeta)==True:
    print(UI.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+RecOutputMeta+bcolors.ENDC)
    MetaInput=UI.PickleOperations(RecOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
ClassHeaders=Meta.ClassHeaders
ClassNames=Meta.ClassNames
ClassValues=Meta.ClassValues
JobSets=Meta.JobSets
MaxSegments=Meta.MaxSegments
TotJobs=JobSets
#The function bellow helps to monitor the HTCondor jobs and keep the submission flow

#The function bellow helps to automate the submission process
UI.Msg('vanilla','Analysing the current script status...')
Status=Meta.Status[-1]
UI.Msg('vanilla','Current stage is '+str(Status)+'...')
Program=[]

################ Set the execution sequence for the script
###### Stage 1
prog_entry=[]
prog_entry.append(' Sending tracks to the HTCondor, so tracks can be analysed by Neural Network...')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/'+RecBatchID+'/','ClassifiedTrackSamples','RCTr1a','.pkl',RecBatchID,TotJobs,'RCTr1a_GenerateClassifiedTracks_Sub.py'])
prog_entry.append([ " --MaxSegments ", " --ModelName "])
prog_entry.append([MaxSegments,ModelName])
prog_entry.append(TotJobs)
prog_entry.append(LocalSub)
prog_entry.append('N/A')
prog_entry.append(HTCondorLog)
prog_entry.append(False)

#Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry))
Program.append(prog_entry)
Program.append('Custom')

while Status<len(Program):
    if Program[Status]!='Custom':
        #Standard process here
        Result=UI.StandardProcess(Program,Status,SubGap,SubPause,RequestExtCPU,JobFlavour,ReqMemory,time_int,Patience)
        if Result[0]:
             UI.UpdateStatus(Status+1,Meta,RecOutputMeta)
        else:
             Status=20
             break
    else:
        if Status==1:
            print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
            print(UI.TimeStamp(),bcolors.BOLD+'Stage 2:'+bcolors.ENDC+' Collecting and de-duplicating the results from stage 1')
            req_file=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_RCTr1a_'+RecBatchID+'_0/RCTr1a_'+RecBatchID+'_ClassifiedTrackSamples_0.pkl'
            base_data=UI.PickleOperations(req_file,'r', 'N/A')[0]
            if ClassValues[0][0]!='Reg':
                ExtractedHeader=['Rec_Seg_ID']+base_data[0].ClassHeaders
            else:
                ExtractedHeader=['Rec_Seg_ID', RecBatchID+'_P_Rec']
            ExtractedData=[]
            for i in base_data:
                if ClassValues[0][0]=='Reg':
                    ExtractedData.append(i.Header+i.Class)
                else:
                    ExtractedData.append(i.Header+i.Class)
            for i in range(1,JobSets):
                    req_file=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_RCTr1a_'+RecBatchID+'_0/RCTr1a'+'_'+RecBatchID+'_ClassifiedTrackSamples_'+str(i)+'.pkl'
                    base_data=UI.PickleOperations(req_file,'r', 'N/A')[0]
                    for i in base_data:
                         if ClassValues[0][0]=='Reg':
                            ExtractedData.append(i.Header+[float(i.Class[0])])
                         else:
                            ExtractedData.append(i.Header+i.Class)
            ExtractedData = pd.DataFrame (ExtractedData, columns = ExtractedHeader)
            if ClassValues[0][0]=='Reg':
                ExtractedData[RecBatchID+'_P_Rec']=ExtractedData[RecBatchID+'_P_Rec']*(float(ClassValues[0][2])/2)
                ExtractedData[RecBatchID+'_P_Rec']=ExtractedData[RecBatchID+'_P_Rec']+(float(ClassValues[0][2])/2)
            data=pd.read_csv(args.f,header=0)
            data.drop(base_data[0].ClassHeaders,axis=1,errors='ignore',inplace=True)
            data['Rec_Seg_ID'] = data[TrackID].astype(str) + '-' + data[BrickID].astype(str)
            data=pd.merge(data,ExtractedData,how='left',on=['Rec_Seg_ID'])
            data=data.drop(['Rec_Seg_ID'],axis=1)
            raw_name=initial_input_file_location[:-4]
            for l in range(len(raw_name)-1,0,-1):
                    if raw_name[l]=='/':
                        break
            raw_name=raw_name[l+1:]
            final_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+raw_name+'_'+RecBatchID+'_CLASSIFIED_TRACKS.csv'
            data.to_csv(final_output_file_location,index=False)
            print(UI.TimeStamp(), bcolors.OKGREEN+"The classified track data has been written to"+bcolors.ENDC, bcolors.OKBLUE+final_output_file_location+bcolors.ENDC)
            print(UI.TimeStamp(),bcolors.OKGREEN+'Stage 1 has successfully completed'+bcolors.ENDC)
            UI.UpdateStatus(Status+1,Meta,RecOutputMeta)
    print(UI.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+RecOutputMeta+bcolors.ENDC)
    MetaInput=UI.PickleOperations(RecOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
    Status=Meta.Status[-1]
#
if Status<20:
    #Removing the temp files that were generated by the process
    print(UI.TimeStamp(),'Performing the cleanup... ')
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'d',['RCTr1a']))
    UI.Msg('success',"Segment merging has been completed")
else:
    UI.Msg('failed',"Segment merging has not been completed as one of the processes has timed out. Please run the script again (without Reset Mode).")
    exit()


