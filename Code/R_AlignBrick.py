#This script aligns the brick based on the reconstructed volumene tracks
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
import U_Alignment as UA
import Parameters as PM #This is where we keep framework global parameters
import pandas as pd #We use Panda for a routine data processing
pd.options.mode.chained_assignment = None #Silence annoying warnings
import math #We use it for data manipulation
import numpy as np
import os
import time
from alive_progress import alive_bar
import argparse

UI.WelcomeMsg('Initialising ECC alignment module...','Filips Fedotovs (PhD student at UCL)', 'Please reach out to filips.fedotovs@cern.ch for any queries')

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
parser.add_argument('--Patience',help="How many checks to do before resubmitting the job?", default='15')
parser.add_argument('--RecBatchID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_Rec_Raw_UR.csv')
parser.add_argument('--Size',help="Split the cross section of the brick in the squares with the size being a length of such a square.", default='0')
parser.add_argument('--ReqMemory',help="Specifying the length of the HTCondor job walltime. Currently at 'workday' which is 8 hours.", default='2 GB')
parser.add_argument('--MinHits',help="What is the minimum number of hits per track?", default=50,type=int)
parser.add_argument('--ValMinHits',help="What is the validation minimum number of hits per track?", default=45,type=int)
parser.add_argument('--SampleSize',help="Would you like to sample for big datasets? By how much?", default=1.0,type=float)
parser.add_argument('--Cycle',help="Number of cycles", default='1')
parser.add_argument('--SpatialOptBound',help="Size", default='200')
parser.add_argument('--AngularOptBound',help="Size", default='2')

######################################## Parsing argument values  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
RecBatchID=args.RecBatchID
Patience=int(args.Patience)
TrackID=args.TrackID
BrickID=args.BrickID
SampleSize=args.SampleSize
MinHits=int(args.MinHits)
ValMinHits=int(args.ValMinHits)
Size=int(args.Size)
SpatialOptBound=args.SpatialOptBound
AngularOptBound=args.AngularOptBound
Cycle=int(args.Cycle)
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
Patience=int(args.Patience)
initial_input_file_location=args.f
FreshStart=True

if Mode=='RESET':
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'d',['Ra','Rb']))
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'c'))
elif Mode=='CLEANUP':
     print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'d',['Ra','Rb']))
     exit()
else:
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'c'))

#Establishing paths
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
RecOutputMeta=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/'+RecBatchID+'_info.pkl'
required_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/R_'+RecBatchID+'_HITS.csv'

########################################     Phase 1 - Create compact source file    #########################################
UI.Msg('status','Stage 0:',' Preparing the source data...')
if os.path.isfile(required_file_location)==False:
        UI.Msg('location','Loading raw data from',initial_input_file_location)
        if BrickID=='':
            ColUse=[TrackID,PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
        else:
            ColUse=[TrackID,BrickID,PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
        data=pd.read_csv(initial_input_file_location,
                    header=0,
                    usecols=ColUse)
        Min_x=data[PM.x].min()
        Max_x=data[PM.x].max()
        Min_y=data[PM.y].min()
        Max_y=data[PM.y].max()
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
        UI.Msg('result','Removing tracks which have less than',ValMinHits,'hits')
        track_no_data=data.groupby(['Rec_Seg_ID'],as_index=False).count()
        track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty,PM.Hit_ID],axis=1)
        track_no_data=track_no_data.rename(columns={PM.x: "Track_No"})
        track_no_data['Random_Factor']=np.random.random(size=len(track_no_data))
        new_combined_data=pd.merge(data, track_no_data, how="left", on=["Rec_Seg_ID"])
        new_combined_data = new_combined_data[new_combined_data.Random_Factor <= SampleSize]
        grand_final_rows=len(new_combined_data.axes[0])
        UI.Msg('result','The resampled data has',grand_final_rows,'hits')
        new_combined_data=new_combined_data.drop(['Random_Factor'],axis=1)
        new_combined_data = new_combined_data[new_combined_data.Track_No >= ValMinHits]
        new_combined_data['Plate_ID']=new_combined_data['z'].astype(int)
        new_combined_data=new_combined_data.sort_values(['Rec_Seg_ID',PM.x],ascending=[1,1])
        grand_final_rows=len(new_combined_data.axes[0])
        UI.Msg('result','The cleaned data has',grand_final_rows,'hits')
        new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
        new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
        new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
        new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
        new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
        new_combined_data=new_combined_data.rename(columns={PM.Hit_ID: "Hit_ID"})
        train_data = new_combined_data[new_combined_data.Track_No >= MinHits]
        validation_data = new_combined_data[new_combined_data.Track_No >= ValMinHits]
        validation_data = validation_data[validation_data.Track_No < MinHits]
        UI.Msg('vanilla','Analysing the data sample in order to understand how many jobs to submit to HTCondor...')
        Sets=new_combined_data.z.unique().size
        x_no=int(math.ceil((Max_x-Min_x)/Size))
        UI.Msg('vanilla','Working out the number of plates to align...')
        plates=train_data[['Plate_ID']].sort_values(['Plate_ID'],ascending=[1])
        plates.drop_duplicates(inplace=True)
        plates=plates.values.tolist() #I find it is much easier to deal with tracks in list format when it comes to fitting
        UI.Msg('result','There are ',len(plates),' plates')
        UI.Msg('result','Initial validation spatial residual value is',round(UA.FitPlate(plates[0][0],0,0,validation_data,'Rec_Seg_ID',True),2),'microns')
        UI.Msg('result','Initial validation angular residual value is',round(UA.FitPlateAngle(plates[0][0],0,0,validation_data,'Rec_Seg_ID',True)*1000,1),'milliradians')

        y_no=int(math.ceil((Max_y-Min_y)/Size))
        for j in range(x_no):
            for k in range(y_no):
                required_temp_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/R_'+RecBatchID+'_HITS_'+str(j)+'_'+str(k)+'.csv'
                x_min_cut=Min_x+(Size*j)
                x_max_cut=Min_x+(Size*(j+1))
                y_min_cut=Min_y+(Size*k)
                y_max_cut=Min_y+(Size*(k+1))
                temp_data=new_combined_data[new_combined_data.x >= x_min_cut]
                temp_data=temp_data[temp_data.x < x_max_cut]
                temp_data=temp_data[temp_data.y >= y_min_cut]
                temp_data=temp_data[temp_data.y < y_max_cut]
                temp_data.to_csv(required_temp_file_location,index=False)
                UI.Msg('location',"The granular hit data has been created successfully and written to",required_temp_file_location)
        JobSets=[]
        new_combined_data.to_csv(required_file_location,index=False)
        UI.Msg('location',"The hit data has been created successfully and written to",required_file_location)
        for i in range(Sets):
            JobSets.append([])
            for j in range(x_no):
                JobSets[i].append(y_no)
        Meta=UI.TrainingSampleMeta(RecBatchID)
        Meta.IniBrickAlignMetaData(Size,ValMinHits,MinHits,SpatialOptBound,AngularOptBound,JobSets,Cycle,plates,[Min_x,Max_x,Min_y,Max_y])
        Meta.UpdateStatus(0)
        print(UI.PickleOperations(RecOutputMeta,'w', Meta)[1])
        UI.Msg('completed','Stage 0 has successfully completed')
elif os.path.isfile(RecOutputMeta)==True:
    UI.Msg('location','Loading previously saved data from ',RecOutputMeta)
    MetaInput=UI.PickleOperations(RecOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
Size=Meta.Size
ValMinHits=Meta.ValMinHits
MinHits=Meta.MinHits
SpatialOptBound=Meta.SpatialOptBound
JobSets=Meta.JobSets
AngularOptBound=Meta.AngularOptBound
Cycle=Meta.Cycles
plates=Meta.plates
Min_x=Meta.FiducialVolume[0]
Max_x=Meta.FiducialVolume[1]
Min_y=Meta.FiducialVolume[2]
Max_y=Meta.FiducialVolume[3]
#The function bellow helps to monitor the HTCondor jobs and keep the submission flow

UI.Msg('vanilla','Analysing the current script status...')
Status=Meta.Status[-1]

UI.Msg('status','Current status is',Status)
################ Set the execution sequence for the script
Program=[]
if type(JobSets) is int:
            TotJobs=JobSets
elif type(JobSets[0]) is int:
            TotJobs=np.sum(JobSets)
elif type(JobSets[0][0]) is int:
            TotJobs=0
            for lp in JobSets:
                TotJobs+=np.sum(lp)
for c in range(Cycle):
    prog_entry=[]
    prog_entry.append(' Sending tracks to the HTCondor, so track segment combinations can be formed...')
    prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/'+RecBatchID+'/','SpatialAlignmentResult_'+str(c),'Ra','.csv',RecBatchID,JobSets,'Ra_SpatiallyAlignBrick_Sub.py'])
    prog_entry.append([ " --MinHits ", " --ValMinHits "," --Size ", " --OptBound ", " --Plate "])
    prog_entry.append([MinHits, ValMinHits, Size, SpatialOptBound, '"'+str(plates)+'"'])
    prog_entry.append(TotJobs)
    prog_entry.append(LocalSub)
    prog_entry.append(["",""])
    prog_entry.append(False)
    prog_entry.append(False)

    if Mode=='RESET' and c==0:
            print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Delete'))
        #Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
    print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry))
    Program.append(prog_entry)
    Program.append('Custom: Spatial Cycle '+str(c))
for c in range(Cycle):
    prog_entry=[]
    prog_entry.append(' Sending tracks to the HTCondor, so track segment combinations can be formed...')
    prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/'+RecBatchID+'/','AngularAlignmentResult_'+str(c),'Rb','.csv',RecBatchID,JobSets,'Rb_AngularAlignBrick_Sub.py'])
    prog_entry.append([ " --MinHits ", " --ValMinHits "," --Size ", " --OptBound ", " --Plate "])
    prog_entry.append([MinHits, ValMinHits, Size, AngularOptBound, '"'+str(plates)+'"'])
    prog_entry.append(TotJobs)
    prog_entry.append(LocalSub)
    prog_entry.append(["",""])
    prog_entry.append(False)
    prog_entry.append(False)
    if Mode=='RESET' and c==0:
            print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Delete'))
        #Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
    print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry,'Create'))
    Program.append(prog_entry)
    Program.append('Custom: Angular Cycle '+str(c))
Program.append('Custom: Final')

while Status<len(Program):
    if Program[Status][:6]!='Custom':
        #Standard process here
        Result=UI.StandardProcess(Program,Status,SubGap,SubPause,RequestExtCPU,JobFlavour,ReqMemory,time_int,Patience,Meta,RecOutputMeta)
        if Result[0]:
             FreshStart=Result[1]
        else:
             Status=20
             break
    elif Program[Status][:21]=='Custom: Spatial Cycle':
        UI.Msg('result','Stage',Status,':Collecting results from the previous step')
        result=[]
        for i in range(0,len(JobSets)): #//Temporarily measure to save space || Update 13/08/23 - I have commented it out as it creates more problems than solves it
            for j in range(len(JobSets[i])):
                for k in range(JobSets[i][j]):
                  result_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_Ra'+'_'+RecBatchID+'_'+str(i)+'/Ra_'+RecBatchID+'_SpatialAlignmentResult_'+Program[Status][22:]+'_'+str(i)+'_'+str(j)+'_'+str(k)+'.csv'
                  res=UI.LogOperations(result_file_location,'r','N/A')
                  if len(res)>0:
                    result.append(res[0])
        result=pd.DataFrame(result,columns=['Type','Plate_ID','j','k','dx','FitX','ValFitX','dy','FitY','ValFitY'])
        log_result=result
        log_result['Cycle']=Program[Status][22:]
        if Program[Status][22:]=='0':
            log_result.to_csv(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/'+RecBatchID+'_REC_LOG.csv',mode="w",index=False)
        else:
            log_result.to_csv(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/'+RecBatchID+'_REC_LOG.csv',mode="a",index=False,header=False)
        result=result[['Type','Plate_ID','j','k','dx','dy']]
        required_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/R_'+RecBatchID+'_HITS.csv'
        data=pd.read_csv(required_file_location,header=0)

        data['j']=(data['x']-Min_x)/Size
        data['k']=(data['y']-Min_y)/Size
        data['j']=data['j'].apply(np.floor)
        data['k']=data['k'].apply(np.floor)
        result['Plate_ID'] = result['Plate_ID'].astype(int)
        result['dx'] = result['dx'].astype(float)
        result['dy'] = result['dy'].astype(float)
        result['j'] = result['j'].astype(int)
        result['k'] = result['k'].astype(int)
        data['j'] = data['j'].astype(int)
        data['k'] = data['k'].astype(int)
        data=pd.merge(data,result,on=['Plate_ID','j','k'],how='left')

        data['dx'] = data['dx'].fillna(0.0)
        data['dy'] = data['dy'].fillna(0.0)
        data['x']=data['x']+data['dx']
        data['y']=data['y']+data['dy']
        data.drop(['Type','dx','dy','k','j'],axis=1, inplace=True)
        validation_data = data[data.Track_No >= ValMinHits]
        validation_data = validation_data[validation_data.Track_No < MinHits]
        UI.Msg('result','Cycle '+Program[Status][22:]+' validation spatial residual value after spatial alignment is',round(UA.FitPlate(plates[0][0],0,0,validation_data,'Rec_Seg_ID',True),2),'microns')
        UI.Msg('result','Cycle '+Program[Status][22:]+' validation angular residual value after spatial alignment is',round(UA.FitPlateAngle(plates[0][0],0,0,validation_data,'Rec_Seg_ID',True)*1000,1),'milliradians')
        x_no=int(math.ceil((Max_x-Min_x)/Size))
        y_no=int(math.ceil((Max_y-Min_y)/Size))
        for j in range(x_no):
            for k in range(y_no):
                required_temp_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/R_'+RecBatchID+'_HITS_'+str(j)+'_'+str(k)+'.csv'
                x_min_cut=Min_x+(Size*j)
                x_max_cut=Min_x+(Size*(j+1))
                y_min_cut=Min_y+(Size*k)
                y_max_cut=Min_y+(Size*(k+1))
                temp_data=data[data.x >= x_min_cut]
                temp_data=temp_data[temp_data.x < x_max_cut]
                temp_data=temp_data[temp_data.y >= y_min_cut]
                temp_data=temp_data[temp_data.y < y_max_cut]
                temp_data.to_csv(required_temp_file_location,index=False)
                UI.Msg('location',"The granular hit data has been created successfully and written to",required_temp_file_location)
        data.to_csv(required_file_location,index=False)
        UI.Msg('location',"The hit data has been created successfully and written to",required_file_location)
        UI.UpdateStatus(Status+1,Meta,RecOutputMeta)


    elif Program[Status][:21]=='Custom: Angular Cycle':
        UI.Msg('result','Stage',Status,':Collecting results from the previous step')
        result=[]
        for i in range(0,len(JobSets)): #//Temporarily measure to save space || Update 13/08/23 - I have commented it out as it creates more problems than solves it
            for j in range(len(JobSets[i])):
                for k in range(JobSets[i][j]):
                  result_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_Rb'+'_'+RecBatchID+'_'+str(i)+'/Rb_'+RecBatchID+'_AngularAlignmentResult_'+Program[Status][22:]+'_'+str(i)+'_'+str(j)+'_'+str(k)+'.csv'
                  res=UI.LogOperations(result_file_location,'r','N/A')
                  if len(res)>0:
                    result.append(res[0])
        result=pd.DataFrame(result,columns=['Type','Plate_ID','j','k','dx','FitX','ValFitX','dy','FitY','ValFitY'])
        log_result=result
        log_result['Cycle']=Program[Status][22:]
        log_result.to_csv(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/'+RecBatchID+'_REC_LOG.csv',mode="a",index=False,header=False)
        result=result[['Type','Plate_ID','j','k','dx','dy']]
        required_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/R_'+RecBatchID+'_HITS.csv'
        data=pd.read_csv(required_file_location,header=0)
        data['j']=(data['x']-Min_x)/Size
        data['k']=(data['y']-Min_y)/Size
        data['j']=data['j'].apply(np.floor)
        data['k']=data['k'].apply(np.floor)
        result['Plate_ID'] = result['Plate_ID'].astype(int)
        result['dx'] = result['dx'].astype(float)
        result['dy'] = result['dy'].astype(float)
        result['j'] = result['j'].astype(int)
        result['k'] = result['k'].astype(int)
        data['j'] = data['j'].astype(int)
        data['k'] = data['k'].astype(int)
        data=pd.merge(data,result,on=['Plate_ID','j','k'],how='left')

        data['dx'] = data['dx'].fillna(0.0)
        data['dy'] = data['dy'].fillna(0.0)
        data['tx']=data['tx']+data['dx']
        data['ty']=data['ty']+data['dy']
        data.drop(['Type','dx','dy','k','j'],axis=1, inplace=True)
        validation_data = data[data.Track_No >= ValMinHits]
        validation_data = validation_data[validation_data.Track_No < MinHits]
        UI.Msg('result','Cycle '+Program[Status][22:]+' validation spatial residual value after spatial alignment is',round(UA.FitPlate(plates[0][0],0,0,validation_data,'Rec_Seg_ID',True),2),'microns')
        UI.Msg('result','Cycle '+Program[Status][22:]+' validation angular residual value after spatial alignment is',round(UA.FitPlateAngle(plates[0][0],0,0,validation_data,'Rec_Seg_ID',True)*1000,1),'milliradians')
        x_no=int(math.ceil((Max_x-Min_x)/Size))
        y_no=int(math.ceil((Max_y-Min_y)/Size))
        for j in range(x_no):
            for k in range(y_no):
                required_temp_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/R_'+RecBatchID+'_HITS_'+str(j)+'_'+str(k)+'.csv'
                x_min_cut=Min_x+(Size*j)
                x_max_cut=Min_x+(Size*(j+1))
                y_min_cut=Min_y+(Size*k)
                y_max_cut=Min_y+(Size*(k+1))
                temp_data=data[data.x >= x_min_cut]
                temp_data=temp_data[temp_data.x < x_max_cut]
                temp_data=temp_data[temp_data.y >= y_min_cut]
                temp_data=temp_data[temp_data.y < y_max_cut]
                temp_data.to_csv(required_temp_file_location,index=False)
                UI.Msg('location',"The granular hit data has been created successfully and written to",required_temp_file_location)
        data.to_csv(required_file_location,index=False)
        UI.Msg('location',"The hit data has been created successfully and written to",required_file_location)
        UI.UpdateStatus(Status+1,Meta,RecOutputMeta)

    elif Program[Status]=='Custom: Final':
        UI.Msg('location','Mapping the alignment transportation map to input data',initial_input_file_location)
        alignment_data_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/'+RecBatchID+'_REC_LOG.csv'
        UI.Msg('location','Loading alignment data from',alignment_data_location)

        ColUse=['Type','Plate_ID','j','k','dx','dy','Cycle']
        alignment_data=pd.read_csv(alignment_data_location,
                    header=0,
                    usecols=ColUse)
        alignment_data.drop_duplicates(subset=['Type','Plate_ID','j','k','Cycle'],keep='first',inplace=True)
        alignment_data.drop(['Cycle'],axis=1, inplace=True)
        alignment_data=alignment_data.groupby(['Type','Plate_ID','j','k']).agg({'dx': 'sum', 'dy': 'sum'}).reset_index()
        spatial_alignment_map=alignment_data[alignment_data.Type=='Spatial'].drop(['Type'],axis=1)
        angular_alignment_map=alignment_data[alignment_data.Type=='Angular'].drop(['Type'],axis=1)
        UI.Msg('location','Loading raw data from',initial_input_file_location)
        ini_data=pd.read_csv(initial_input_file_location,
                    header=0)
        ini_data['Plate_ID']=ini_data['z'].astype(int)

        ######    Measuring initial-alignment    ########
        data=ini_data
        if BrickID=='':
            ColUse=[TrackID,PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
        else:
            ColUse=[TrackID,BrickID,PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
        if BrickID=='':
            data[BrickID]='D'
        data=data[ColUse]
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

        print(UI.TimeStamp(),'Removing tracks which have less than',ValMinHits,'hits...')
        track_no_data=data.groupby(['Rec_Seg_ID'],as_index=False).count()
        track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty,PM.Hit_ID],axis=1)
        track_no_data=track_no_data.rename(columns={PM.x: "Track_No"})
        track_no_data['Random_Factor']=np.random.random(size=len(track_no_data))
        
        new_combined_data=pd.merge(data, track_no_data, how="left", on=["Rec_Seg_ID"])
        new_combined_data = new_combined_data[new_combined_data.Random_Factor <= SampleSize]
        grand_final_rows=len(new_combined_data.axes[0])
        print(UI.TimeStamp(),'The resampled data has ',grand_final_rows,' hits')
        new_combined_data=new_combined_data.drop(['Random_Factor'],axis=1)       
        new_combined_data = new_combined_data[new_combined_data.Track_No >= ValMinHits]
        new_combined_data['Plate_ID']=new_combined_data['z'].astype(int)
        new_combined_data=new_combined_data.sort_values(['Rec_Seg_ID',PM.x],ascending=[1,1])
        grand_final_rows=len(new_combined_data.axes[0])
        print(UI.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
        new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
        new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
        new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
        new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
        new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
        new_combined_data=new_combined_data.rename(columns={PM.Hit_ID: "Hit_ID"})
        validation_data = new_combined_data[new_combined_data.Track_No >= ValMinHits]
        validation_data = validation_data[validation_data.Track_No < MinHits]
        UI.Msg('result','There are ',len(plates),' plates')
        UI.Msg('result','Initial validation spatial residual value is',round(UA.FitPlate(plates[0][0],0,0,validation_data,'Rec_Seg_ID',True),2),'microns')
        UI.Msg('result','Initial validation angular residual value is',round(UA.FitPlateAngle(plates[0][0],0,0,validation_data,'Rec_Seg_ID',True)*1000,1),'milliradians')


        #### Saving the aligned file
        print(UI.TimeStamp(),'Preparing initial ini_data for the join...')
        ini_data['j']=(ini_data['x']-Min_x)/Size
        ini_data['k']=(ini_data['y']-Min_y)/Size
        ini_data['j']=ini_data['j'].apply(np.floor)
        ini_data['k']=ini_data['k'].apply(np.floor)
        print(UI.TimeStamp(),'Aligning spatial coordinates...')
        ini_data=pd.merge(ini_data,spatial_alignment_map,on=['Plate_ID','j','k'],how='left')
        ini_data['dx'] = ini_data['dx'].fillna(0.0)
        ini_data['dy'] = ini_data['dy'].fillna(0.0)
        ini_data[PM.x]=ini_data[PM.x]+ini_data['dx']
        ini_data[PM.y]=ini_data[PM.y]+ini_data['dy']
        ini_data.drop(['dx','dy'],axis=1, inplace=True)
        print(UI.TimeStamp(),'Aligning angular coordinates...')
        ini_data=pd.merge(ini_data,angular_alignment_map,on=['Plate_ID','j','k'],how='left')
        ini_data['dx'] = ini_data['dx'].fillna(0.0)
        ini_data['dy'] = ini_data['dy'].fillna(0.0)
        ini_data[PM.tx]=ini_data[PM.tx]+ini_data['dx']
        ini_data[PM.ty]=ini_data[PM.ty]+ini_data['dy']
        ini_data.drop(['Plate_ID','dx','dy','k','j'],axis=1, inplace=True)
        output_file_location=initial_input_file_location[:-4]+'_'+RecBatchID+'.csv'
        ini_data.to_csv(output_file_location,index=False)
        UI.Msg('location','Data has been realigned and saved in ',output_file_location)


        ######    Measuring post-realignment    ########
        print(UI.TimeStamp(),'Measuring the validation alignment...')

        data=ini_data
        if BrickID=='':
            ColUse=[TrackID,PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
        else:
            ColUse=[TrackID,BrickID,PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
        if BrickID=='':
            data[BrickID]='D'
        data=data[ColUse]
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

        print(UI.TimeStamp(),'Removing tracks which have less than',ValMinHits,'hits...')
        track_no_data=data.groupby(['Rec_Seg_ID'],as_index=False).count()
        track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty,PM.Hit_ID],axis=1)
        track_no_data=track_no_data.rename(columns={PM.x: "Track_No"})

        track_no_data['Random_Factor']=np.random.random(size=len(track_no_data))
        new_combined_data=pd.merge(data, track_no_data, how="left", on=["Rec_Seg_ID"])
        new_combined_data = new_combined_data[new_combined_data.Random_Factor <= SampleSize]
        grand_final_rows=len(new_combined_data.axes[0])
        UI.Msg('result','The resampled data has',grand_final_rows,'hits')
        new_combined_data=new_combined_data.drop(['Random_Factor'],axis=1)

        new_combined_data = new_combined_data[new_combined_data.Track_No >= ValMinHits]
        new_combined_data['Plate_ID']=new_combined_data['z'].astype(int)
        new_combined_data=new_combined_data.sort_values(['Rec_Seg_ID',PM.x],ascending=[1,1])
        grand_final_rows=len(new_combined_data.axes[0])
        print(UI.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
        new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
        new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
        new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
        new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
        new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
        new_combined_data=new_combined_data.rename(columns={PM.Hit_ID: "Hit_ID"})
        validation_data = new_combined_data[new_combined_data.Track_No >= ValMinHits]
        validation_data = validation_data[validation_data.Track_No < MinHits]
        UI.Msg('result','There are ',len(plates),' plates')
        UI.Msg('result','Final validation spatial residual value is',round(UA.FitPlate(plates[0][0],0,0,validation_data,'Rec_Seg_ID',True),2),'microns')
        UI.Msg('result','Final validation angular residual value is',round(UA.FitPlateAngle(plates[0][0],0,0,validation_data,'Rec_Seg_ID',True)*1000,1),'milliradians')
        UI.Msg('completed','Stage '+str(Status)+' has successfully completed')
        UI.UpdateStatus(Status+1,Meta,RecOutputMeta)
    UI.Msg('location','Loading previously saved data from',RecOutputMeta)
    MetaInput=UI.PickleOperations(RecOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
    Status=Meta.Status[-1]

if Status<20:
    #Removing the temp files that were generated by the process
    #print(UI.TimeStamp(),'Performing the cleanup... ')
    #print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'d',['Ra','Rb']))
    UI.Msg('completed',"Data alignment procedure has been completed")
else:
    UI.Msg('failed',"Segment merging has not been completed as one of the processes has timed out. Please run the script again (without Reset Mode).")
    exit()



