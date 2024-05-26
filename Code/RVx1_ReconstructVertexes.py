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
import U_UI as UI #This is where we keep routine utility functions
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

UI.WelcomeMsg('Initialising ANNDEA Vertexing module...','Filips Fedotovs (PhD student at UCL), Leah Wolf (MSc student at UCL), Henry Wilson (MSc student at UCL)','Please reach out to filips.fedotovs@cern.ch for any queries')

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
parser.add_argument('--ModelName',help="What  models would you like to use?", default="[]")
parser.add_argument('--Patience',help="How many checks to do before resubmitting the job?", default='15')
parser.add_argument('--RecBatchID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_Rec_Raw_UR.csv')
parser.add_argument('--Xmin',help="This option restricts data to only those events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option restricts data to only those events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option restricts data to only those events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option restricts data to only those events that have tracks with hits y-coordinates that are below this value", default='0')
parser.add_argument('--Log',help="Would you like to log the performance of this reconstruction? (Only available with MC data)", default='N')
parser.add_argument('--Acceptance',help="What is the ANN fit acceptance?", default='0.662')
parser.add_argument('--LinkAcceptance',help="What is the ANN seed link acceptance?", default='1.2')
parser.add_argument('--CalibrateAcceptance',help="Would you like to recalibrate the acceptance?", default='N')
parser.add_argument('--ReqMemory',help="Specifying the length of the HTCondor job walltime. Currently at 'workday' which is 8 hours.", default='2 GB')
parser.add_argument('--FiducialVolumeCut',help="Limits on the vx, y, z coordinates of the vertex origin", default='[]')
parser.add_argument('--ExcludeClassNames',help="What class headers to use?", default="[]")
parser.add_argument('--ExcludeClassValues',help="What class values to use?", default="[[]]")
parser.add_argument('--ForceStatus',help="Local submission?", default='N')
parser.add_argument('--HTCondorLog',help="Local submission?", default=False,type=bool)
######################################## Parsing argument values  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
RecBatchID=args.RecBatchID
Patience=int(args.Patience)
TrackID=args.TrackID
BrickID=args.BrickID
ForceStatus=args.ForceStatus
HTCondorLog=args.HTCondorLog
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
LinkAcceptance=float(args.LinkAcceptance)
CalibrateAcceptance=(args.CalibrateAcceptance=='Y')
initial_input_file_location=args.f
Log=args.Log=='Y'
FiducialVolumeCut=ast.literal_eval(args.FiducialVolumeCut)
ExcludeClassNames=ast.literal_eval(args.ExcludeClassNames)
ExcludeClassValues=ast.literal_eval(args.ExcludeClassValues)
Xmin,Xmax,Ymin,Ymax=float(args.Xmin),float(args.Xmax),float(args.Ymin),float(args.Ymax)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0 #We don't slice data if all values are set to zero simultaneousy (which is the default setting)

if Mode=='RESET':
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'d',['EVx1a','RVx1a','RVx1b','RVx1c']))
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'c'))
elif Mode=='CLEANUP':
     print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'d',['EVx1a','RVx1a','RVx1b','RVx1c']))
     exit()
else:
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'c'))

#Establishing paths
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
if ModelName[0]!='Blank':
    EOSsubModelMetaDIR=EOSsubDIR+'/'+'Models/'+ModelName[0]+'_Meta'
else:
    EOSsubModelMetaDIR=EOSsubDIR+'/'+'Models/'+ModelName[1]+'_Meta'


RecOutputMeta=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/'+RecBatchID+'_info.pkl'
required_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/RVx1_'+RecBatchID+'_TRACK_SEGMENTS_0.csv'
required_eval_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/'+RecBatchID+'/EVx1_'+RecBatchID+'_TRACK_SEGMENTS.csv'
########################################     Phase 1 - Create compact source file    #########################################
print(UI.TimeStamp(),bcolors.BOLD+'Stage 0:'+bcolors.ENDC+' Preparing the source data...')



if os.path.isfile(required_file_location)==False:
        if os.path.isfile(EOSsubModelMetaDIR)==False:
              print(UI.TimeStamp(), bcolors.FAIL+"Fail to proceed further as the model file "+EOSsubModelMetaDIR+ " has not been found..."+bcolors.ENDC)
              exit()
        else:
           print(UI.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+EOSsubModelMetaDIR+bcolors.ENDC)
           MetaInput=UI.PickleOperations(EOSsubModelMetaDIR,'r', 'N/A')
           Meta=MetaInput[0]
           MaxDST=Meta.MaxDST
           MaxVXT=Meta.MaxVXT
           MaxDOCA=Meta.MaxDOCA
           MaxAngle=Meta.MaxAngle
           MaxSegments=PM.MaxSegments
           MaxSeeds=PM.MaxSeedsPerVxPool
           MinHitsTrack=Meta.MinHitsTrack
        print(UI.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+initial_input_file_location+bcolors.ENDC)
        data=pd.read_csv(initial_input_file_location,
                    header=0,
                    usecols=[TrackID,BrickID,PM.x,PM.y,PM.z,PM.tx,PM.ty])
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
        print(UI.TimeStamp(),'After removing tracks with specific lengths we have',final_rows,' hits left')


        if SliceData:
             print(UI.TimeStamp(),'Slicing the data...')
             ValidEvents=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
             ValidEvents.drop([PM.x,PM.y,PM.z,PM.tx,PM.ty],axis=1,inplace=True)
             ValidEvents.drop_duplicates(subset="Rec_Seg_ID",keep='first',inplace=True)
             data=pd.merge(data, ValidEvents, how="inner", on=['Rec_Seg_ID'])
             final_rows=len(data.axes[0])
             print(UI.TimeStamp(),'The sliced data has ',final_rows,' hits')
        print(UI.TimeStamp(),'Removing tracks which have less than',MinHitsTrack,'hits...')
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
        #--------------------------------------------------------------------
        data_header = new_combined_data.groupby('Rec_Seg_ID')['z'].min()
        data_header=data_header.reset_index()
        UI.Msg('vanilla','Analysing the data sample in order to understand how many jobs to submit to HTCondor... ')
        data=new_combined_data[['Rec_Seg_ID','z']]
        data = data.groupby('Rec_Seg_ID')['z'].min()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
        data=data.reset_index()
        data = data.groupby('z')['Rec_Seg_ID'].count()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
        data=data.reset_index()
        data=data.sort_values(['z'],ascending=True)
        data['Sub_Sets']=np.ceil(data['Rec_Seg_ID']/PM.MaxSegments)
        data['Sub_Sets'] = data['Sub_Sets'].astype(int)
        JobData=data.drop(['Rec_Seg_ID','z'],axis=1)
        CutData=data.drop(['Rec_Seg_ID','Sub_Sets'],axis=1)
        JobData = JobData.values.tolist()
        CutData = CutData.values.tolist()
        JobData=[k for i in JobData for k in i]
        CutData=[k for i in CutData for k in i]
        for i in range(len(CutData)):
          data_temp_header=data_header.drop(data_header.index[data_header['z'] < CutData[i]])
          data_temp_header=data_temp_header.drop(data_temp_header.index[data_temp_header['z'] > CutData[i]+MaxDST])
          data_temp_header=data_temp_header.drop(['z'],axis=1)
          temp_data=pd.merge(new_combined_data, data_temp_header, how="inner", on=["Rec_Seg_ID"]) #Shrinking the Track data so just a star hit for each track is present.
          temp_required_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/RVx1_'+RecBatchID+'_TRACK_SEGMENTS_'+str(i)+'.csv'
          temp_data.to_csv(temp_required_file_location,index=False)
          UI.Msg('location',"The track segment data has been created successfully and written to",temp_required_file_location)

        JobSetList=[]
        for i in range(20):
            JobSetList.append('empty')
        JobSetList[int(Log)]=JobData
        Meta=UI.TrainingSampleMeta(RecBatchID)
        Meta.IniVertexSeedMetaData(MaxDST,MaxVXT,MaxDOCA,MaxAngle,JobSetList,MaxSegments,MaxSeeds,MinHitsTrack,FiducialVolumeCut,ExcludeClassNames,ExcludeClassValues)
        Meta.UpdateStatus(0)
        print(UI.PickleOperations(RecOutputMeta,'w', Meta)[1])
        UI.Msg('completed','Stage 0 has successfully completed')

if Log and (os.path.isfile(required_eval_file_location)==False):
    if os.path.isfile(EOSsubModelMetaDIR)==False:
              print(UI.TimeStamp(), bcolors.FAIL+"Fail to proceed further as the model file "+EOSsubModelMetaDIR+ " has not been found..."+bcolors.ENDC)
              exit()
    else:
           print(UI.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+EOSsubModelMetaDIR+bcolors.ENDC)
           MetaInput=UI.PickleOperations(EOSsubModelMetaDIR,'r', 'N/A')
           Meta=MetaInput[0]
           MinHitsTrack=Meta.MinHitsTrack
           if hasattr(Meta,'ClassNames') and hasattr(Meta,'ClassValues'):
               ExcludeClassNames=Meta.ClassNames
               ExcludeClassValues=Meta.ClassValues
           else:
               ExcludeClassNames=[]
               ExcludeClassValues=[[]]
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
    print(UI.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+initial_input_file_location+bcolors.ENDC)
    data=pd.read_csv(initial_input_file_location,
                header=0,
                usecols=ColumnsToImport+ExtraColumns)
    total_rows=len(data)
    if len(ExtraColumns)>0:
            for c in ExtraColumns:
                data[c] = data[c].astype(str)
            data=pd.merge(data,BanDF,how='left',on=ExtraColumns)
            data=data.fillna('')
    else:
            data['Exclude']=''
    print(UI.TimeStamp(),'The raw data has ',total_rows,' hits')
    print(UI.TimeStamp(),'Removing unreconstructed hits...')
    data=data.dropna()
    final_rows=len(data)
    print(UI.TimeStamp(),'The cleaned data has ',final_rows,' hits')
    data[PM.MC_Event_ID] = data[PM.MC_Event_ID].astype(str)
    data[PM.MC_VX_ID] = data[PM.MC_VX_ID].astype(str)

    data[BrickID] = data[BrickID].astype(str)
    data[TrackID] = data[TrackID].astype(str)
    data['Rec_Seg_ID'] = data[TrackID] + '-' + data[BrickID]
    data['MC_Vertex_ID'] = data[PM.MC_Event_ID] + '-'+ data['Exclude'] + data[PM.MC_VX_ID]

    data=data.drop([TrackID],axis=1)
    data=data.drop([BrickID],axis=1)
    data=data.drop([PM.MC_Event_ID],axis=1)
    data=data.drop([PM.MC_VX_ID],axis=1)
    data=data.drop(['Exclude'],axis=1)

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
    print(UI.TimeStamp(),'After removing tracks with specific lengths we have',final_rows,' hits left')
    compress_data=data.drop([PM.x,PM.y,PM.z,PM.tx,PM.ty],axis=1)
    compress_data['MC_Mother_No']= compress_data['MC_Vertex_ID']
    compress_data=compress_data.groupby(by=['Rec_Seg_ID','MC_Vertex_ID'])['MC_Mother_No'].count().reset_index()
    compress_data=compress_data.sort_values(['Rec_Seg_ID','MC_Mother_No'],ascending=[1,0])
    compress_data.drop_duplicates(subset='Rec_Seg_ID',keep='first',inplace=True)
    data=data.drop(['MC_Vertex_ID'],axis=1)
    compress_data=compress_data.drop(['MC_Mother_No'],axis=1)
    data=pd.merge(data, compress_data, how="left", on=['Rec_Seg_ID'])
    if SliceData:
         print(UI.TimeStamp(),'Slicing the data...')
         ValidEvents=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
         ValidEvents.drop([PM.x,PM.y,PM.z,PM.tx,PM.ty,'MC_Vertex_ID']+ExtraColumns,axis=1,inplace=True)
         ValidEvents.drop_duplicates(subset='Rec_Seg_ID',keep='first',inplace=True)
         data=pd.merge(data, ValidEvents, how="inner", on=['Rec_Seg_ID'])
         final_rows=len(data.axes[0])
         print(UI.TimeStamp(),'The sliced data has ',final_rows,' hits')
    print(UI.TimeStamp(),'Removing tracks which have less than',MinHitsTrack,'hits...')
    track_no_data=data.groupby(['Rec_Seg_ID','MC_Vertex_ID'],as_index=False).count()
    track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty],axis=1)
    track_no_data=track_no_data.rename(columns={PM.x: "Rec_Seg_No"})
    new_combined_data=pd.merge(data, track_no_data, how="left", on=['Rec_Seg_ID','MC_Vertex_ID'])
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
    UI.Msg('vanilla','Analysing evaluation data... ')
    new_combined_data.drop_duplicates(subset="Rec_Seg_ID",keep='first',inplace=True)  #Keeping only starting hits for each track record (we do not require the full information about track in this script)
    Records=len(new_combined_data.axes[0])

    UI.Msg('location','Updating the Meta file ',RecOutputMeta)
    MetaInput=UI.PickleOperations(RecOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
    Sets=int(np.ceil(Records/Meta.MaxSegments))
    Meta.JobSets[0]=Sets
    print(UI.PickleOperations(RecOutputMeta,'w', Meta)[0])
    UI.Msg('location',"The track segment data has been created successfully and written to",required_eval_file_location)

elif os.path.isfile(RecOutputMeta)==True:
    print(UI.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+RecOutputMeta+bcolors.ENDC)
    MetaInput=UI.PickleOperations(RecOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
MaxDST=Meta.MaxDST
MaxVXT=Meta.MaxVXT
MaxDOCA=Meta.MaxDOCA
MaxAngle=Meta.MaxAngle
JobSets=Meta.JobSets
MaxSegments=Meta.MaxSegments
MaxSeeds=PM.MaxSeedsPerVxPool
MinHitsTrack=Meta.MinHitsTrack


#The function bellow helps to automate the submission process
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
    print(UI.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+initial_input_file_location+bcolors.ENDC)
    data=pd.read_csv(required_eval_file_location,header=0,usecols=['Rec_Seg_ID'])
    print(UI.TimeStamp(),'Analysing data... ',bcolors.ENDC)
    data.drop_duplicates(subset="Rec_Seg_ID",keep='first',inplace=True)  #Keeping only starting hits for each track record (we do not require the full information about track in this script)
    Records=len(data.axes[0])
    Sets=int(np.ceil(Records/MaxSegments))
    prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TEST_SET/'+RecBatchID+'/','RawSeedsRes','EVx1a','.csv',RecBatchID,Sets,'EVx1a_GenerateRawSelectedSeeds_Sub.py'])
    prog_entry.append([" --MaxSegments "])
    prog_entry.append([MaxSegments])
    prog_entry.append(Sets)
    prog_entry.append(LocalSub)
    prog_entry.append('N/A')
    prog_entry.append(HTCondorLog)
    prog_entry.append(False)
    print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry))
    Program.append(prog_entry)
    # ###### Stage 1
    Program.append('Custom - Collect Evaluation Results')

###### Stage 2
prog_entry=[]
NJobs=UI.CalculateNJobs(Meta.JobSets[int(Log)])[1]
prog_entry.append(' Sending tracks to the HTCondor, so track segment combinations can be formed...')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/'+RecBatchID+'/','RawSeedsRes','RVx1a','.csv',RecBatchID, Meta.JobSets[int(Log)],'RVx1a_GenerateRawSelectedSeeds_Sub.py'])
prog_entry.append([ " --MaxSegments ", " --MaxDST "])
prog_entry.append([MaxSegments, MaxDST])
prog_entry.append(NJobs)
prog_entry.append(LocalSub)
prog_entry.append('N/A')
prog_entry.append(HTCondorLog)
prog_entry.append(False)
#Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry))
Program.append(prog_entry)
Program.append('Custom - Collect Raw Seeds')

####### Stage 4
for md in ModelName:
     Program.append(md)

Program.append('Custom - LinkAnalysis')

Program.append('Custom - PerformMerging')

Program.append('Custom - VertexMapping')

while Status<len(Program):
    if Program[Status][:6]!='Custom' and (Program[Status] in ModelName)==False:
        #Standard process here
        Result=UI.StandardProcess(Program,Status,SubGap,SubPause,RequestExtCPU,JobFlavour,ReqMemory,time_int,Patience)
        if Result[0]:
            UI.UpdateStatus(Status+1,Meta,RecOutputMeta)
        else:
             Status=20
             break

    elif Program[Status]=='Custom - Collect Evaluation Results':

        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
        UI.Msg('status','Stage '+str(Status),': Collecting and de-duplicating the results from previous stage '+str(Status-1)+'...')
        Sets=Meta.JobSets[0]
        with alive_bar(Sets,force_tty=True, title='Analysing data...') as bar:
            for i in range(Sets): #//Temporarily measure to save space
                    bar.text = f'-> Analysing set : {i}...'
                    bar()
                    if i==0:
                       output_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/'+RecBatchID+'/Temp_EVx1a'+'_'+RecBatchID+'_'+str(0)+'/EVx1a_'+RecBatchID+'_RawSeeds_'+str(i)+'.csv'
                       result=pd.read_csv(output_file_location,names = ['Segment_1','Segment_2'])
                       print(UI.TimeStamp(),'Set',str(i), 'contains', len(result), 'seeds')
                    else:
                        output_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/'+RecBatchID+'/Temp_EVx1a'+'_'+RecBatchID+'_'+str(0)+'/EVx1a_'+RecBatchID+'_RawSeeds_'+str(i)+'.csv'
                        new_result=pd.read_csv(output_file_location,names = ['Segment_1','Segment_2'])
                        print(UI.TimeStamp(),'Set',str(i), 'contains', len(new_result), 'seeds')
                        result=pd.concat([result,new_result])

        Records=len(result)
        result.drop(result.index[result['Segment_1'] == result['Segment_2']], inplace = True)
        result["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(result['Segment_1'], result['Segment_2'])]
        result.drop_duplicates(subset="Seed_ID",keep='first',inplace=True)

        result.drop(["Seed_ID"],axis=1,inplace=True)
        Records_After_Compression=len(result)
        if Records>0:
                      Compression_Ratio=int((Records_After_Compression/Records)*100)
        else:
                      Compression_Ratio=0
        print(UI.TimeStamp(),'Set',str(i), 'compression ratio is ', Compression_Ratio, ' %',bcolors.ENDC)
        new_output_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/'+RecBatchID+'/EVx1b_'+RecBatchID+'_SEED_TRUTH_COMBINATIONS.csv'
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
        UI.Msg('completed','Stage '+str(Status)+' has successfully completed')
        UI.UpdateStatus(Status+1,Meta,RecOutputMeta)
    elif Program[Status]=='Custom - Collect Raw Seeds':
        UI.Msg('status','Stage '+str(Status),': Collecting and de-duplicating the results from previous stage '+str(Status-1)+'...')
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
        Meta=UI.PickleOperations(RecOutputMeta,'r', 'N/A')[0]
        JobSet=Meta.JobSets[int(Log)]
        NewJobSet=[]
        for i in JobSet:
            NewJobSet.append(0)
        with alive_bar(len(JobSet),force_tty=True, title='Checking the results from HTCondor') as bar:
            for i in range(len(JobSet)): #//Temporarily measure to save space
                bar.text = f'-> Analysing set : {i}...'
                bar()
                tot_fractions=0
                if NewJobSet[i]==0:
                    for j in range(0,JobSet[i]):
                        output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_RVx1a'+'_'+RecBatchID+'_'+str(i)+'/RVx1a_'+RecBatchID+'_RawSeeds_'+str(i)+'_'+str(j)+'.csv'
                        result=pd.read_csv(output_file_location,names = ['Segment_1','Segment_2'])
                        Records=len(result)
                        print(UI.TimeStamp(),'Set',str(i),'and subset', str(j), 'contains', Records, 'seeds')
                        fractions=int(math.ceil(Records/MaxSeeds))
                        for k in range(0,fractions):
                         new_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_RVx1a'+'_'+RecBatchID+'_'+str(i)+'/RVx1a_'+RecBatchID+'_SelectedSeeds_'+str(i)+'_'+str(tot_fractions+k)+'.csv'
                         result[(k*MaxSeeds):min(Records,((k+1)*MaxSeeds))].to_csv(new_output_file_location,index=False)
                        tot_fractions+=fractions
                    NewJobSet[i]=tot_fractions
                else:
                    continue
        if Log:
             UI.Msg('vanilla','Initiating the logging...')
             eval_data_file=EOS_DIR+'/ANNDEA/Data/TEST_SET/'+RecBatchID+'/EVx1b_'+RecBatchID+'_SEED_TRUTH_COMBINATIONS.csv'
             eval_data=pd.read_csv(eval_data_file,header=0,usecols=['Segment_1','Segment_2'])
             eval_data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(eval_data['Segment_1'], eval_data['Segment_2'])]
             eval_data.drop(['Segment_1'],axis=1,inplace=True)
             eval_data.drop(['Segment_2'],axis=1,inplace=True)
             NJobs=int(UI.CalculateNJobs(NewJobSet)[1])
             rec=None
             with alive_bar(NJobs,force_tty=True, title='Preparing data for the log...') as bar:
                 for i in range(len(NewJobSet)):
                    bar()
                    if NewJobSet[i]>0:
                        for j in range(NewJobSet[i]):
                             new_input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_RVx1a'+'_'+RecBatchID+'_'+str(i)+'/RVx1a_'+RecBatchID+'_SelectedSeeds_'+str(i)+'_'+str(j)+'.csv'
                             rec_new=pd.read_csv(new_input_file_location,usecols = ['Segment_1','Segment_2'])
                             rec_new["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(rec_new['Segment_1'], rec_new['Segment_2'])]
                             rec_new.drop(['Segment_1'],axis=1,inplace=True)
                             rec_new.drop(['Segment_2'],axis=1,inplace=True)
                             rec = pd.concat([rec, rec_new], ignore_index=True)
                             rec.drop_duplicates(subset="Seed_ID",keep='first',inplace=True)
             if rec is not None:
                 rec_eval=pd.merge(eval_data, rec, how="inner", on=['Seed_ID'])
                 eval_no=len(rec_eval)
                 rec_no=(len(rec)-len(rec_eval))
                 UI.LogOperations(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv', 'a', [[2,'Basic track distance cuts',rec_no,eval_no,eval_no/(rec_no+eval_no),eval_no/len(eval_data)]])
             else:
                 UI.LogOperations(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv', 'a', [[2,'Basic track distance cuts',0,0,0,0]])
             UI.Msg('location',"The log has been created successfully at ",EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv')
        Meta.JobSets[Status+1]=NewJobSet
        print(UI.PickleOperations(RecOutputMeta,'w', Meta)[1])
        UI.Msg('completed','Stage '+str(Status)+' has successfully completed')
        UI.UpdateStatus(Status+1,Meta,RecOutputMeta)
    elif Program[Status]=='Custom - LinkAnalysis':
        input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/RVx1c_'+RecBatchID+'_Fit_Seeds.pkl'
        print(UI.TimeStamp(), "Loading the fit track seeds from the file",bcolors.OKBLUE+input_file_location+bcolors.ENDC)
        base_data=UI.PickleOperations(input_file_location,'r','N/A')[0]
        print(UI.TimeStamp(), bcolors.OKGREEN+"Loading is successful, there are "+str(len(base_data))+" fit seeds..."+bcolors.ENDC)
        prog_entry=[]
        N_Jobs=math.ceil(len(base_data)/MaxSeeds)
        Program_Dummy=[]
        prog_entry.append(' Sending vertexes to the HTCondor, so vertex can be subject to link analysis...')
        prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/'+RecBatchID+'/','AnalyseSeedLinks','RVx1c','.csv',RecBatchID,N_Jobs,'RVx1c_AnalyseSeedLinks_Sub.py'])
        prog_entry.append([" --MaxSegments "])
        prog_entry.append([MaxSeeds])
        prog_entry.append(N_Jobs)
        prog_entry.append(LocalSub)
        prog_entry.append('N/A')
        prog_entry.append(HTCondorLog)
        prog_entry.append(False)
        prog_entry.append(['',''])
        for dum in range(0,Status):
            Program_Dummy.append('DUM')
        Program_Dummy.append(prog_entry)
        print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry))
        Result=UI.StandardProcess(Program_Dummy,Status,SubGap,SubPause,RequestExtCPU,JobFlavour,ReqMemory,time_int,Patience)
        if Result:
            print(UI.TimeStamp(),bcolors.OKGREEN+'All HTCondor Seed Creation jobs have finished'+bcolors.ENDC)
            print(UI.TimeStamp(),'Collating the results...')
            for f in range(N_Jobs):
                 req_file = EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_RVx1c_'+RecBatchID+'_'+str(0)+'/RVx1c_'+RecBatchID+'_AnalyseSeedLinks_'+str(f)+'.csv'
                 progress=round((float(f)/float(N_Jobs))*100,2)
                 print(UI.TimeStamp(),'progress is ',progress,' %', end="\r", flush=True) #Progress display
                 if os.path.isfile(req_file)==False:
                     print(UI.TimeStamp(), bcolors.FAIL+"Critical fail: file",req_file,'is missing, please restart the script with the option "--Mode R"'+bcolors.ENDC)
                 elif os.path.isfile(req_file):
                     if (f)==0:
                         base_data = pd.read_csv(req_file,usecols=['Track_1', 'Track_2','Seed_CNN_Fit','Link_Strength','AntiLink_Strenth'])
                     else:
                         new_data = pd.read_csv(req_file,usecols=['Track_1', 'Track_2','Seed_CNN_Fit','Link_Strength','AntiLink_Strenth'])
                         frames = [base_data, new_data]
                         base_data = pd.concat(frames,ignore_index=True)
                         Records=len(base_data)
        print(UI.TimeStamp(),'The pre-analysed reconstructed set contains', Records, '2-track link-fitted seeds',bcolors.ENDC)
        base_data['Seed_Link_Fit'] = base_data.apply(PM.Seed_Bond_Fit_Acceptance,axis=1)
        base_data['Seed_Index'] = base_data.index
        base_data.drop(base_data.index[base_data['Seed_Link_Fit'] < LinkAcceptance],inplace=True)  # Dropping the seeds that don't pass the link fit threshold
        base_data.drop(base_data.index[base_data['Seed_CNN_Fit'] < Acceptance],inplace=True)  # Dropping the seeds that don't pass the link fit threshold
        if CalibrateAcceptance:
            print(UI.TimeStamp(),'Calibrating the acceptance...')
            eval_data_file=EOS_DIR+'/ANNDEA/Data/TEST_SET/'+RecBatchID+'/EVx1b_'+RecBatchID+'_SEED_TRUTH_COMBINATIONS.csv'
            eval_data=pd.read_csv(eval_data_file,header=0,usecols=['Segment_1','Segment_2'])
            eval_data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(eval_data['Segment_1'], eval_data['Segment_2'])]
            eval_data.drop(['Segment_1'],axis=1,inplace=True)
            eval_data.drop(['Segment_2'],axis=1,inplace=True)
            eval_data['True']=1
            OP=len(eval_data)
            base_data=base_data[['Track_1','Track_2','Seed_CNN_Fit']]
            base_data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(base_data['Track_1'], base_data['Track_2'])]
            base_data.drop(['Track_1'],axis=1,inplace=True)
            base_data.drop(['Track_2'],axis=1,inplace=True)
            combined_data=pd.merge(base_data,eval_data,how='left',on='Seed_ID')
            combined_data=combined_data.fillna(0)
            combined_data.drop(['Seed_ID'],axis=1,inplace=True)
            TP = combined_data['True'].sum()
            P = combined_data['True'].count()
            Min_Acceptance=round(combined_data['Seed_CNN_Fit'].min(),2)
            FP=P-TP
            Ini_Precision=TP/P
            F1=(2*(Ini_Precision))/(Ini_Precision+1.0)
            iterations=int((1.0-Min_Acceptance)/0.01)
            for i in range(1,iterations):
                cut_off=Min_Acceptance+(i*0.01)
                cut_data=combined_data.drop(combined_data.index[combined_data['Seed_CNN_Fit'] < cut_off])
                tp = cut_data['True'].sum()
                p=cut_data['True'].count()
                precision=tp/p
                recall=tp/TP
                o_recall=tp/OP
                f1=(2*(precision*o_recall))/(precision+o_recall)
                print('Cutoff at:',cut_off,'; Precision:', round(precision,3), '; Recall:', round(recall,3), '; Overall recall:', round(o_recall,3), '; F1:', round(f1,3))
            exit()

        
        Records_After_Compression=len(base_data)
        if args.Log=='Y':
          #try:
             print(UI.TimeStamp(),'Initiating the logging...')
             eval_data_file=EOS_DIR+'/ANNDEA/Data/TEST_SET/'+RecBatchID+'/EVx1b_'+RecBatchID+'_SEED_TRUTH_COMBINATIONS.csv'
             eval_data=pd.read_csv(eval_data_file,header=0,usecols=['Segment_1','Segment_2'])
             eval_data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(eval_data['Segment_1'], eval_data['Segment_2'])]
             eval_data.drop(['Segment_1'],axis=1,inplace=True)
             eval_data.drop(['Segment_2'],axis=1,inplace=True)
             eval_no=0
             rec_no=0
             rec=base_data[['Track_1','Track_2']]
             rec["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(rec['Track_1'], rec['Track_2'])]
             rec.drop(['Track_1'],axis=1,inplace=True)
             rec.drop(['Track_2'],axis=1,inplace=True)
             rec_eval=pd.merge(eval_data, rec, how="inner", on=['Seed_ID'])
             rec.to_csv('sd_test.csv')
             eval_no=len(rec_eval)
             rec_no=(len(rec)-len(rec_eval))
             UI.LogOperations(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv', 'a', [[5,'Link Analysis',rec_no,eval_no,eval_no/(rec_no+eval_no),eval_no/len(eval_data)]])
             print(UI.TimeStamp(), bcolors.OKGREEN+"The log has been created successfully at "+bcolors.ENDC, bcolors.OKBLUE+EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv'+bcolors.ENDC)
          #except:
           #    print(UF.TimeStamp(), bcolors.WARNING+'Log creation has failed'+bcolors.ENDC)
        print(UI.TimeStamp(), 'Decorating seed objects in ' + bcolors.ENDC,bcolors.OKBLUE + input_file_location + bcolors.ENDC)
        base_data=base_data.values.tolist()
        new_data=[]
        for b in base_data:
            new_data.append(b[6])
        base_data=new_data
        del new_data
        print(UI.TimeStamp(), 'Loading seed object data from ', bcolors.OKBLUE + input_file_location + bcolors.ENDC)
        object_data = UI.PickleOperations(input_file_location,'r','N/A')[0]
        selected_objects=[]
        for nd in range(len(base_data)):
            selected_objects.append(object_data[base_data[nd]])
            progress = round((float(nd) / float(len(base_data))) * 100, 1)
            print(UI.TimeStamp(), 'Refining the seed objects, progress is ', progress, ' %', end="\r", flush=True)  # Progress display
        del object_data
        del base_data
        output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/RVx1c_'+RecBatchID+'_Link_Fit_Seeds.pkl'
        print(UI.PickleOperations(output_file_location,'w',selected_objects)[0])
        print(UI.TimeStamp(), bcolors.OKGREEN + str(len(selected_objects))+" seed objects are saved in" + bcolors.ENDC,bcolors.OKBLUE + output_file_location + bcolors.ENDC)
        UI.UpdateStatus(Status+1,Meta,RecOutputMeta)
    elif Program[Status]=='Custom - PerformMerging':
        input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/RVx1c_'+RecBatchID+'_Link_Fit_Seeds.pkl'
        print(UI.TimeStamp(), "Loading the fit track seeds from the file",bcolors.OKBLUE+input_file_location+bcolors.ENDC)
        base_data=UI.PickleOperations(input_file_location,'r','N/A')[0]
        original_data_seeds=len(base_data)
        for b in base_data:
            if b.Header[0]=='58862.0' or b.Header[0]=='68496.0':
                print(b.Header)
            if b.Header[1]=='58862.0' or b.Header[1]=='68496.0':
                print(b.Header)
        exit()
        #no_iter = int(math.ceil(float(original_data_seeds / float(PM.MaxSeedsPerVxPool))))
        no_iter=1
        prog_entry=[]
        Program_Dummy=[]
        prog_entry.append('Sending vertices to HTCondor so they can be merged')
        prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/'+RecBatchID+'/','MergeVertices','RVx1d','.pkl',RecBatchID,no_iter,'RVx1d_MergeVertices_Sub.py'])
        prog_entry.append([" --MaxPoolSeeds "])
        prog_entry.append([PM.MaxSeedsPerVxPool])
        prog_entry.append(no_iter)
        prog_entry.append(LocalSub)
        prog_entry.append('N/A')
        prog_entry.append(HTCondorLog)
        prog_entry.append(False)
        prog_entry.append(['',''])
        for dum in range(0,Status):
            Program_Dummy.append('DUM')
        Program_Dummy.append(prog_entry)
        print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry))
        Result=UI.StandardProcess(Program_Dummy,Status,SubGap,SubPause,RequestExtCPU,JobFlavour,ReqMemory,time_int,Patience)
        if Result:
              print(UI.TimeStamp(), bcolors.OKGREEN + 'All HTCondor Seed Creation jobs have finished' + bcolors.ENDC)
              print(UI.TimeStamp(), 'Collating the results...')
              VertexPool=[]
              for i in range(no_iter):
                    progress = round((float(i) / float(no_iter)) * 100, 2)
                    print(UI.TimeStamp(), 'progress is ', progress, ' %', end="\r", flush=True)
                    required_file_location = EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_RVx1d_'+RecBatchID+'_'+str(0)+'/RVx1d_'+RecBatchID+'_MergeVertices_'+str(i)+'.pkl'
                    NewData=UI.PickleOperations(required_file_location,'r','N/A')[0]
                    VertexPool+=NewData
              print(UI.TimeStamp(), 'As a result of the previous operation',str(original_data_seeds),'seeds were merged into',str(len(VertexPool)),'vertices...')

              comp_ratio = round((float(len(VertexPool)) / float(original_data_seeds)) * 100, 2)
              print(UI.TimeStamp(), 'The compression ratio is',comp_ratio, '%...')
              print(UI.TimeStamp(), 'Ok starting the final merging of the remained vertices')
              InitialDataLength=len(VertexPool)
              SeedCounter=0
              SeedCounterContinue=True
              while SeedCounterContinue:
                  if SeedCounter==len(VertexPool):
                                  SeedCounterContinue=False
                                  break
                  progress=round((float(SeedCounter)/float(len(VertexPool)))*100,0)
                  print(UI.TimeStamp(),'progress is ',progress,' %', end="\r", flush=True) #Progress display
                  SubjectSeed=VertexPool[SeedCounter]
                  for ObjectSeed in VertexPool[SeedCounter+1:]:
                              if SubjectSeed.InjectSeed(ObjectSeed):
                                          VertexPool.pop(VertexPool.index(ObjectSeed))
                  SeedCounter+=1
              print(str(InitialDataLength), "vertices from different files were merged into", str(len(VertexPool)), 'vertices with higher multiplicity...')
              for v in range(0,len(VertexPool)):
                    VertexPool[v].AssignANNVxUID(v)
              csv_out=[]
              csv_out=[['Old_Track_ID','Temp_Vertex_Domain','Temp_Vertex_ID']]
              for Vx in VertexPool:
                    for Tr in Vx.Header:
                        csv_out.append([Tr,RecBatchID,Vx.UVxID])
              output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/RVx1e_'+RecBatchID+'_Union_Vertexes.pkl'
              output_csv_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/RVx1e_'+RecBatchID+'_Union_Vertexes.csv'

              print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
              print(UI.TimeStamp(), "Saving the results into the file",bcolors.OKBLUE+output_csv_location+bcolors.ENDC)
              UI.LogOperations(output_csv_location,'w', csv_out)
              print(UI.TimeStamp(), "Saving the results into the file",bcolors.OKBLUE+output_file_location+bcolors.ENDC)
              print(UI.PickleOperations(output_file_location,'w',VertexPool)[1])
 
              if args.Log=='Y':
                  try:
                    print(UI.TimeStamp(),'Initiating the logging...')
                    eval_data_file=EOS_DIR+'/ANNDEA/Data/TEST_SET/'+RecBatchID+'/EVx1b_'+RecBatchID+'_SEED_TRUTH_COMBINATIONS.csv'
                    eval_data=pd.read_csv(eval_data_file,header=0,usecols=['Segment_1','Segment_2'])
                    eval_data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(eval_data['Segment_1'], eval_data['Segment_2'])]
                    eval_data.drop(['Segment_1'],axis=1,inplace=True)
                    eval_data.drop(['Segment_2'],axis=1,inplace=True)
                    rec_no=0
                    eval_no=0
                    rec_list=[]
                    rec_1 = pd.DataFrame(csv_out, columns = ['Track_1','Domain ID','Seed_ID'])
                    rec_2 = pd.DataFrame(csv_out, columns = ['Track_2','Domain ID','Seed_ID'])
                    rec=pd.merge(rec_1, rec_2, how="inner", on=['Seed_ID'])
                    rec.drop(['Seed_ID'],axis=1,inplace=True)
                    rec.drop(rec.index[rec['Track_1'] == rec['Track_2']], inplace = True)
                    rec["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(rec['Track_1'], rec['Track_2'])]
                    rec.drop(['Track_1'],axis=1,inplace=True)
                    rec.drop(['Track_2'],axis=1,inplace=True)
                    rec.drop_duplicates(subset=['Seed_ID'], keep='first', inplace=True)
                    rec_eval=pd.merge(eval_data, rec, how="inner", on=['Seed_ID'])
                    rec.to_csv('vx_test.csv')
                    eval_no=len(rec_eval)
                    rec_no=(len(rec)-len(rec_eval))
                    UI.LogOperations(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv', 'a', [[6,'Vertex Merging',rec_no,eval_no,eval_no/(rec_no+eval_no),eval_no/len(eval_data)]])
                    print(UI.TimeStamp(), bcolors.OKGREEN+"The log has been created successfully at "+bcolors.ENDC, bcolors.OKBLUE+EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv'+bcolors.ENDC)
                  except:
                    print(UI.TimeStamp(), bcolors.WARNING+'Log creation has failed'+bcolors.ENDC)
              UI.UpdateStatus(Status+1,Meta,RecOutputMeta)
    elif Program[Status]=='Custom - VertexMapping':
                raw_name=initial_input_file_location[:-4]
                for l in range(len(raw_name)-1,0,-1):
                    if raw_name[l]=='/':
                        print(l,raw_name)
                        break
                raw_name=raw_name[l+1:]
                final_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+raw_name+'_'+RecBatchID+'.csv'
                final_output_file_location_object=EOS_DIR+'/ANNDEA/Data/REC_SET/'+raw_name+'_'+RecBatchID+'.pkl'
                print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
                print(UI.TimeStamp(),bcolors.BOLD+'Stage '+str(Status)+':'+bcolors.ENDC+' Taking the list of seeds previously generated by Stage '+str(Status-1)+' and mapping them to the input data')
                print(UI.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+initial_input_file_location+bcolors.ENDC)
                required_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/RVx1_'+RecBatchID+'_VERTEX_SEGMENTS.csv'
                print(UI.TimeStamp(),'Loading mapped data from',bcolors.OKBLUE+EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/RVx1e_'+RecBatchID+'_Union_Vertexes.csv'+bcolors.ENDC)
                map_data=pd.read_csv(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/RVx1e_'+RecBatchID+'_Union_Vertexes.csv',header=0)
                Vertex_Object=UI.PickleOperations(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/RVx1e_'+RecBatchID+'_Union_Vertexes.pkl','r', 'N/A')[0]
                print(UI.PickleOperations(final_output_file_location_object,'w',Vertex_Object)[1])

                initial_data=pd.read_csv(initial_input_file_location,header=0)
                initial_data[BrickID] = initial_data[BrickID].astype(str)
                initial_data[TrackID] = initial_data[TrackID].astype(str)
                
                initial_data['Old_Track_ID'] = initial_data[TrackID] + '-' + initial_data[BrickID]

                new_combined_data=pd.merge(initial_data,map_data,how='left',on=['Old_Track_ID'])
                new_combined_data.drop(['Old_Track_ID'],axis=1,inplace=True)
                new_combined_data=new_combined_data.rename(columns={'Temp_Vertex_Domain': RecBatchID+'_Brick_ID','Temp_Vertex_ID': RecBatchID+'_Vertex_ID'})
                new_combined_data.to_csv(final_output_file_location,index=False)
                print(UI.TimeStamp(), bcolors.OKGREEN+"The vertex data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+final_output_file_location+bcolors.ENDC)
                print(UI.TimeStamp(), bcolors.OKGREEN+"The vertex object data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+final_output_file_location_object+bcolors.ENDC)
                print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
                UI.UpdateStatus(Status+1,Meta,RecOutputMeta)
    
    else:
         for md in range(len(ModelName)):
            if Program[Status]==ModelName[md]:
                    prog_entry=[]
                    Program_Dummy=[]
                    Meta=UI.PickleOperations(RecOutputMeta,'r', 'N/A')[0]
                    JobSet=Meta.JobSets[Status]
                    NJobs=UI.CalculateNJobs(JobSet)[1]
                    prog_entry.append(' Sending tracks to the HTCondor, so track combination pairs can be formed...')
                    prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/'+RecBatchID+'/','RefinedSeeds','RVx1'+ModelName[md],'.pkl',RecBatchID,JobSet,'RVx1b_RefineSeeds_Sub.py'])

                    if md==0:
                        prog_entry.append([" --MaxDST ", " --MaxVXT ", " --MaxDOCA ", " --MaxAngle "," --ModelName "," --FirstTime ", " --FiducialVolumeCut "])
                        prog_entry.append([MaxDST, MaxVXT, MaxDOCA, MaxAngle,'"'+ModelName[md]+'"', 'True', '"'+str(FiducialVolumeCut)+'"'])
                    else:
                        prog_entry.append([" --MaxDST ", " --MaxVXT ", " --MaxDOCA ", " --MaxAngle "," --ModelName "," --FirstTime "," --FiducialVolumeCut "])
                        prog_entry.append([MaxDST, MaxVXT, MaxDOCA, MaxAngle,'"'+ModelName[md]+'"', ModelName[md-1], '"'+str(FiducialVolumeCut)+'"'])

                    prog_entry.append(NJobs)
                    prog_entry.append(LocalSub)
                    prog_entry.append('N/A')
                    prog_entry.append(HTCondorLog)
                    prog_entry.append(False)
                    for dum in range(0,Status):
                        Program_Dummy.append('DUM')
                    Program_Dummy.append(prog_entry)
                    #Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
                    print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry))
                    Result=UI.StandardProcess(Program_Dummy,Status,SubGap,SubPause,RequestExtCPU,JobFlavour,ReqMemory,time_int,Patience)
                    if Result:
                        UI.Msg('status','Stage '+str(Status),': Analysing the fitted seeds')
                        JobSet=Meta.JobSets[Status]
                        NJobs=int(UI.CalculateNJobs(JobSet)[1])
                        if md==len(ModelName)-1:
                            with alive_bar(NJobs,force_tty=True, title='Checking the results from HTCondor') as bar:
                             base_data = None

                             for i in range(len(JobSet)):
                                    bar.text = f'-> Analysing set : {i}...'

                                    for j in range(JobSet[i]):
                                                  required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_RVx1'+ModelName[md]+'_'+RecBatchID+'_'+str(i)+'/RVx1'+ModelName[md]+'_'+RecBatchID+'_RefinedSeeds_'+str(i)+'_'+str(j)+'.pkl'
                                                  bar()
                                                  new_data=UI.PickleOperations(required_output_file_location,'r','N/A')[0]
                                                  print(UI.TimeStamp(),'Set',str(i)+'_'+str(j), 'contains', len(new_data), 'seeds')
                                                  if base_data == None:
                                                        base_data = new_data
                                                  else:
                                                        base_data+=new_data
                            Records=len(base_data)
                            print(UI.TimeStamp(),'The output contains', Records, 'raw images')
                            base_data=list(set(base_data))
                            Records_After_Compression=len(base_data)
                            if Records>0:
                                                          Compression_Ratio=int((Records_After_Compression/Records)*100)
                            else:
                                                          CompressionRatio=0
                            print(UI.TimeStamp(),'The output '+str(i)+'  compression ratio is ', Compression_Ratio, ' %',)
                            output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/RVx1c_'+RecBatchID+'_Fit_Seeds.pkl'
                            print(UI.PickleOperations(output_file_location,'w',base_data)[1])
                            if Log:
                             UI.Msg('vanilla','Initiating the logging...')
                             eval_data_file=EOS_DIR+'/ANNDEA/Data/TEST_SET/'+RecBatchID+'/EVx1b_'+RecBatchID+'_SEED_TRUTH_COMBINATIONS.csv'
                             eval_data=pd.read_csv(eval_data_file,header=0,usecols=['Segment_1','Segment_2'])
                             eval_data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(eval_data['Segment_1'], eval_data['Segment_2'])]
                             eval_data.drop(['Segment_1'],axis=1,inplace=True)
                             eval_data.drop(['Segment_2'],axis=1,inplace=True)
                             rec_no=0
                             eval_no=0
                             rec_list=[]
                             for rd in base_data:
                                        rec_list.append([rd.Header[0],rd.Header[1]])
                             rec = pd.DataFrame(rec_list, columns = ['Segment_1','Segment_2'])
                             rec["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(rec['Segment_1'], rec['Segment_2'])]
                             rec.drop(['Segment_1'],axis=1,inplace=True)
                             rec.drop(['Segment_2'],axis=1,inplace=True)
                             rec.drop_duplicates(subset="Seed_ID",keep='first',inplace=True)
                             rec_eval=pd.merge(eval_data, rec, how="inner", on=['Seed_ID'])
                             eval_no=len(rec_eval)
                             rec_no=(len(rec)-len(rec_eval))
                             UI.LogOperations(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv', 'a', [[3+md,ModelName[md],rec_no,eval_no,eval_no/(rec_no+eval_no),eval_no/len(eval_data)]])
                             UI.Msg('location',"The log data has been created successfully and written to",EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv')
                            NewJobSet=1
                        else:
                            NewJobSet=[]
                            for i in JobSet:
                                NewJobSet.append(0)
                            if Log:
                                 rec_list=[]
                            with alive_bar(NJobs,force_tty=True, title='Checking the results from HTCondor') as bar:
                             for i in range(0,len(JobSet)):
                                    base_data = None
                                    bar.text = f'-> Analysing set : {i}...'
                                    bar()
                                    if NewJobSet[i]==0:
                                        for j in range(JobSet[i]):
                                                      required_output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_RVx1'+ModelName[md]+'_'+RecBatchID+'_'+str(i)+'/RVx1'+ModelName[md]+'_'+RecBatchID+'_RefinedSeeds_'+str(i)+'_'+str(j)+'.pkl'
                                                      new_data=UI.PickleOperations(required_output_file_location,'r','N/A')[0]
                                                      print(UI.TimeStamp(),'Set',str(i)+'_'+str(j), 'contains', len(new_data), 'seeds')
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
                                              tot_fractions=int(np.ceil(Records_After_Compression/MaxSeeds))
                                              for j in range(tot_fractions):
                                                    output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_RVx1'+ModelName[md]+'_'+RecBatchID+'_0/RVx1'+str(ModelName[md+1])+'_'+RecBatchID+'_Input_Seeds_'+str(i)+'_'+str(j)+'.pkl'
                                                    print(UI.PickleOperations(output_file_location,'w',base_data[j*MaxSeeds:(j+1)*MaxSeeds])[1])
                                                    if Log:
                                                     for rd in base_data:
                                                         rec_list.append([rd.Header[0],rd.Header[1]])
                                              NewJobSet[i]=tot_fractions
                                        else:
                                              CompressionRatio=0
                                              print(UI.TimeStamp(),'The output '+str(i)+'  compression ratio is ', Compression_Ratio, ' %, skipping this step')
                            if Log:
                                         eval_data_file=EOS_DIR+'/ANNDEA/Data/TEST_SET/'+RecBatchID+'/EVx1b_'+RecBatchID+'_SEED_TRUTH_COMBINATIONS.csv'
                                         eval_data=pd.read_csv(eval_data_file,header=0,usecols=['Segment_1','Segment_2'])
                                         eval_data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(eval_data['Segment_1'], eval_data['Segment_2'])]
                                         eval_data.drop(['Segment_1'],axis=1,inplace=True)
                                         eval_data.drop(['Segment_2'],axis=1,inplace=True)
                                         UI.Msg('vanilla','Initiating the logging...')
                                         rec = pd.DataFrame(rec_list, columns = ['Segment_1','Segment_2'])

                                         rec["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(rec['Segment_1'], rec['Segment_2'])]
                                         rec.drop(['Segment_1'],axis=1,inplace=True)
                                         rec.drop(['Segment_2'],axis=1,inplace=True)
                                         rec.drop_duplicates(subset="Seed_ID",keep='first',inplace=True)
                                         rec_eval=pd.merge(eval_data, rec, how="inner", on=['Seed_ID'])
                                         eval_no=len(rec_eval)
                                         rec_no=(len(rec)-len(rec_eval))
                                         UI.LogOperations(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv', 'a', [[3+md,ModelName[md],rec_no,eval_no,eval_no/(rec_no+eval_no),eval_no/len(eval_data)]])
                                         UI.Msg('location',"The log data has been created successfully and written to",EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv')
                        Meta.JobSets[Status+1]=NewJobSet
                        print(UI.PickleOperations(RecOutputMeta,'w', Meta)[1])
                        UI.Msg('completed','Stage '+str(Status)+' has successfully completed')
                        UI.UpdateStatus(Status+1,Meta,RecOutputMeta)

    print(UI.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+RecOutputMeta+bcolors.ENDC)
    MetaInput=UI.PickleOperations(RecOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
    Status=Meta.Status[-1]

if Status<20:
    #Removing the temp files that were generated by the process
    DelChoice = input('Would you like to delete working files (Y/N)? : ')
    if len(DelChoice)=='Y':
        print(UI.TimeStamp(),'Performing the cleanup... ')
        print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'d',['EVx1a','RVx1a','RVx1b','RVx1c','RVx1d']))
        UI.Msg('success',"Segment merging has been completed")
else:
    UI.Msg('failed',"Segment merging has not been completed as one of the processes has timed out. Please run the script again (without Reset Mode).")
    exit()



