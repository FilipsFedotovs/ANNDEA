# Part of  ANNDEA  package.
# The purpose of this script to create samples for the tracking model training
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
import csv
import argparse
import pandas as pd #We use Panda for a routine data processing
import math #We use it for data manipulation
import os
import random
import time
import ast
import U_UI as UI #This is where we keep routine utility functions
import Parameters as PM #This is where we keep framework global parameters
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

UI.WelcomeMsg('Initialising ANNDEA Tracking training sample generation module...','Filips Fedotovs (PhD student at UCL)','Please reach out to filips.fedotovs@cern.ch for any queries')

parser = argparse.ArgumentParser(description='This script prepares training data for training the tracking model')
parser.add_argument('--Mode', help='Script will continue from the last checkpoint, unless you want to start from the scratch, then type "Reset"',default='')
parser.add_argument('--TrainSampleID',help="Give name to this train sample batch", default='MH_SND_Raw_Train_Data_6_6_12_All')
parser.add_argument('--Sampling',help="If the data is large, sampling helps to keep it manageable. If this script fails consider reducing this parameter.", default='1.0')
parser.add_argument('--Patience',help="How many HTCondor checks to perform before resubmitting the job?", default='15')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/eos/experiment/ship/ANNDEA/Data/SND_Raw_Input/SND_B11_FEDRARaw.csv')
parser.add_argument('--Xmin',help="This option restricts data to only those events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option restricts data to only those events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option restricts data to only those events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option restricts data to only those events that have tracks with hits y-coordinates that are below this value", default='0')
parser.add_argument('--ExcludeClassNames',help="What class headers to use?", default="['Flag','ProcID']")
parser.add_argument('--ExcludeClassValues',help="What class values to use?", default="[['11','-11'],['8']]")
parser.add_argument('--SubPause',help="How long to wait in minutes after submitting 10000 jobs?", default='60')
parser.add_argument('--SubGap',help="How long to wait in minutes after submitting 10000 jobs?", default='10000')
#The bellow are not important for the training smaples but if you want to augment the training data set above 1
parser.add_argument('--Z_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along z-axis.", default='1')
parser.add_argument('--Y_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along y-axis.", default='1')
parser.add_argument('--X_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along x-axis.", default='1')
parser.add_argument('--ReqMemory',help="How uch memory to request?", default='2 GB')
parser.add_argument('--RequestExtCPU',help="Would you like to request extra CPUs?", default=1)
parser.add_argument('--JobFlavour',help="Specifying the length of the HTCondor job walltime. Currently at 'workday' which is 8 hours.", default='workday')
parser.add_argument('--ForceStatus',help="Would you like the program run from specific status number? (Only for advance users)", default='0')
parser.add_argument('--HTCondorLog',help="Local submission?", default=False,type=bool)
parser.add_argument('--LocalSub',help="Local submission?", default='N')
######################################## Set variables  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
Sampling=float(args.Sampling)
TrainSampleID=args.TrainSampleID
Patience=int(args.Patience)
ReqMemory=args.ReqMemory
RequestExtCPU=int(args.RequestExtCPU)
ForceStatus=args.ForceStatus
JobFlavour=args.JobFlavour
HTCondorLog=args.HTCondorLog
LocalSub=(args.LocalSub=='Y')
input_file_location=args.f
SubPause=int(args.SubPause)*60
SubGap=int(args.SubGap)
Xmin,Xmax,Ymin,Ymax=float(args.Xmin),float(args.Xmax),float(args.Ymin),float(args.Ymax)
Z_overlap,Y_overlap,X_overlap=int(args.Z_overlap),int(args.Y_overlap),int(args.X_overlap)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0
if LocalSub:
   time_int=0
else:
    time_int=10

ExcludeClassNames=ast.literal_eval(args.ExcludeClassNames)
ExcludeClassValues=ast.literal_eval(args.ExcludeClassValues)
ColumnsToImport=[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.MC_Event_ID,PM.MC_Track_ID]
ExtraColumns=[]
BanDF=['-']
BanDF=pd.DataFrame(BanDF, columns=['Exclude'])
for i in range(len(ExcludeClassNames)):
        df=pd.DataFrame(ExcludeClassValues[i], columns=[ExcludeClassNames[i]])
        df['Exclude']='-'
        BanDF=pd.merge(BanDF,df,how='inner',on=['Exclude'])

        if (ExcludeClassNames[i] in ExtraColumns)==False:
                ExtraColumns.append(ExcludeClassNames[i])



stepX=PM.stepX #Size of the individual reconstruction volumes along the x-axis
stepY=PM.stepY #Size of the individual reconstruction volumes along the y-axis
stepZ=PM.stepZ #Size of the individual reconstruction volumes along the z-axis
cut_dt=PM.cut_dt #This cust help to discard hit pairs that are likely do not have a common mother track
cut_dr=PM.cut_dr
cut_dz=PM.cut_dz
testRatio=PM.testRatio #Usually about 5%
valRatio=PM.valRatio #Usually about 10%
TrainSampleOutputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/'+TrainSampleID+'_info.pkl' #For each training sample batch we create an individual meta file.
destination_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/'+TrainSampleID+'_TTr_OUTPUT_1.pkl' #The desired output
if Mode=='RESET':
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, TrainSampleID,'d',['MTr1a']))
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, TrainSampleID,'c'))
elif Mode=='CLEANUP':
     print(UI.ManageFolders(AFS_DIR, EOS_DIR, TrainSampleID,'d',['MTr1a']))
     exit()
else:
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, TrainSampleID,'c'))

########################################     Phase 1 - Create compact source file    #########################################
print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UI.TimeStamp(),bcolors.BOLD+'Stage 0:'+bcolors.ENDC+' Taking the file that has been supplied and creating the compact copies for the training set generation...')
output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/MTr1_'+TrainSampleID+'_hits.csv' #This is the compact data file that contains only relevant columns and rows
if os.path.isfile(output_file_location)==False:
        print(UI.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
        data=pd.read_csv(input_file_location,
                    header=0,
                    usecols=[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty])[[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty]]
        total_rows=len(data.axes[0])
        data[PM.Hit_ID] = data[PM.Hit_ID].astype(str) #We try to keep HIT ids as strings
        print(UI.TimeStamp(),'The raw data has ',total_rows,' hits')
        print(UI.TimeStamp(),'Removing unreconstructed hits...')
        data=data.dropna() #Removing nulls (not really applicable for this module but I put it just in case)
        final_rows=len(data.axes[0])
        print(UI.TimeStamp(),'The cleaned data has ',final_rows,' hits')
        data[PM.Hit_ID] = data[PM.Hit_ID].astype(int)
        data[PM.Hit_ID] = data[PM.Hit_ID].astype(str) #Why I am doing this twice? Need to investigate
        if SliceData: #Keeping only the relevant slice. Only works if we set at least one parameter (Xmin for instance) to non-zero
             print(UI.TimeStamp(),'Slicing the data...')
             data=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
             final_rows=len(data.axes[0])
             print(UI.TimeStamp(),'The sliced data has ',final_rows,' hits')
        data=data.rename(columns={PM.x: "x"})
        data=data.rename(columns={PM.y: "y"})
        data=data.rename(columns={PM.z: "z"})
        data=data.rename(columns={PM.tx: "tx"})
        data=data.rename(columns={PM.ty: "ty"})
        data=data.rename(columns={PM.Hit_ID: "Hit_ID"})
        data.to_csv(output_file_location,index=False)
        print(UI.TimeStamp(), bcolors.OKGREEN+"The segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)


###################### Phase 2 - Eval Data ######################################################
output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/ETr1_'+TrainSampleID+'_hits.csv' #This is similar to one above but also contains MC data
if os.path.isfile(output_file_location)==False:
    print(UI.TimeStamp(),'Creating Evaluation file...')
    print(UI.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
    data=pd.read_csv(input_file_location,
                header=0,
                usecols=ColumnsToImport+ExtraColumns)[ColumnsToImport+ExtraColumns]

    if len(ExtraColumns)>0:
            for c in ExtraColumns:
                data[c] = data[c].astype(str)
            data=pd.merge(data,BanDF,how='left',on=ExtraColumns)
            data=data.fillna('')
    else:
            data['Exclude']=''
    total_rows=len(data.axes[0])
    print(UI.TimeStamp(),'The raw data has ',total_rows,' hits')
    print(UI.TimeStamp(),'Removing unreconstructed hits...')
    data=data.dropna() #Unlikely to have in the hit data but keeping here just in case to prevent potential problems downstream
    final_rows=len(data.axes[0])
    print(UI.TimeStamp(),'The cleaned data has ',final_rows,' hits')
    data[PM.MC_Event_ID] = data[PM.MC_Event_ID].astype(str)
    data[PM.MC_Track_ID] = data[PM.MC_Track_ID].astype(str)
    data[PM.Hit_ID] = data[PM.Hit_ID].astype(str)
    data['MC_Mother_Track_ID'] = data[PM.MC_Event_ID] + '-'+ data['Exclude'] + data[PM.MC_Track_ID] #Track IDs are not unique and repeat for each event: crea
    data=data.drop([PM.MC_Event_ID],axis=1)
    data=data.drop([PM.MC_Track_ID],axis=1)
    data=data.drop(['Exclude'],axis=1)
    for c in ExtraColumns:
        data=data.drop([c],axis=1)
    if SliceData:
         print(UI.TimeStamp(),'Slicing the data...')
         data=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
         final_rows=len(data.axes[0])
         print(UI.TimeStamp(),'The sliced data has ',final_rows,' hits')
    #Even if particle leaves one hit it is still assigned MC Track ID - we cannot reconstruct these so we discard them so performance metrics are not skewed
    print(UI.TimeStamp(),'Removing tracks which have less than',PM.MinHitsTrack,'hits...')
    track_no_data=data.groupby(['MC_Mother_Track_ID'],as_index=False).count()
    track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty,PM.Hit_ID],axis=1)
    track_no_data=track_no_data.rename(columns={PM.x: "MC_Track_No"})
    new_combined_data=pd.merge(data, track_no_data, how="left", on=['MC_Mother_Track_ID'])
    new_combined_data = new_combined_data[new_combined_data.MC_Track_No >= PM.MinHitsTrack]  #We are only interested in MC Tracks that have a certain number of hits
    new_combined_data = new_combined_data.drop(["MC_Track_No"],axis=1)
    new_combined_data=new_combined_data.sort_values(['MC_Mother_Track_ID',PM.z],ascending=[1,1])
    grand_final_rows=len(new_combined_data.axes[0])
    print(UI.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
    new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
    new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
    new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
    new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
    new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
    new_combined_data=new_combined_data.rename(columns={PM.Hit_ID: "Hit_ID"})
    new_combined_data.to_csv(output_file_location,index=False)
    print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
    print(UI.TimeStamp(), bcolors.OKGREEN+"The track segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
print(UI.TimeStamp(),bcolors.OKGREEN+'Stage 0 has successfully completed'+bcolors.ENDC)
########################################     Preset framework parameters    #########################################



print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UI.TimeStamp(),bcolors.BOLD+'Stage 1:'+bcolors.ENDC+' Creating training sample meta data...')
if os.path.isfile(TrainSampleOutputMeta)==False: #A case of generating samples from scratch
    input_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/MTr1_'+TrainSampleID+'_hits.csv'
    print(UI.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
    data=pd.read_csv(input_file_location,header=0,usecols=['z','x','y'])
    print(UI.TimeStamp(),'Analysing data... ',bcolors.ENDC)
    z_offset=data['z'].min()
    data['z']=data['z']-z_offset #Reseting the coordinate origin to zero for this data set
    z_max=data['z'].max()
    y_offset=data['y'].min()
    x_offset=data['x'].min()
    data['x']=data['x']-x_offset #Reseting the coordinate origin to zero for this data set
    x_max=data['x'].max() #We need it to calculate how many clusters to create
    if Z_overlap==1:
            Zsteps=math.ceil((z_max)/stepZ)
    else:
            Zsteps=(math.ceil((z_max)/stepZ)*(Z_overlap))-1
    if X_overlap==1:
            Xsteps=math.ceil((x_max)/stepX)
    else:
            Xsteps=(math.ceil((x_max)/stepX)*(X_overlap))-1
    print(UI.TimeStamp(),'Distributing hit files...')
    print(UI.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
    data=pd.read_csv(input_file_location,header=0)
    with alive_bar(Xsteps,force_tty=True, title='Distributing hit files...') as bar:
             for i in range(Xsteps):
                     required_tfile_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/MTr1_'+TrainSampleID+'_'+str(i)+'_hits.csv'
                     if os.path.isfile(required_tfile_location)==False:
                         X_ID=int(i)/X_overlap
                         tdata=data.drop(data.index[data['x'] >= ((X_ID+1)*stepX)])  #Keeping the relevant z slice
                         tdata.drop(tdata.index[tdata['x'] < (X_ID*stepX)], inplace = True)  #Keeping the relevant z slice
                         tdata.to_csv(required_tfile_location,index=False)
                         #print(UI.TimeStamp(), bcolors.OKGREEN+"The segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+required_tfile_location+bcolors.ENDC)
                     bar()
    print(UI.TimeStamp(),'Distributing eval files...')
    input_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/ETr1_'+TrainSampleID+'_hits.csv' #This is similar to one above but also contains MC data
    data=pd.read_csv(input_file_location,header=0)
    with alive_bar(Xsteps,force_tty=True, title='Distributing hit files...') as bar:
             for i in range(Xsteps):
                     required_tfile_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/ETr1_'+TrainSampleID+'_'+str(i)+'_hits.csv'
                     if os.path.isfile(required_tfile_location)==False:
                         X_ID=int(i)/X_overlap
                         tdata=data.drop(data.index[data['x'] >= ((X_ID+1)*stepX)])  #Keeping the relevant z slice
                         tdata.drop(tdata.index[tdata['x'] < (X_ID*stepX)], inplace = True)  #Keeping the relevant z slice
                         tdata.to_csv(required_tfile_location,index=False)
                         #print(UI.TimeStamp(), bcolors.OKGREEN+"The segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+required_tfile_location+bcolors.ENDC)
                     bar()
    TrainDataMeta=UI.TrainingSampleMeta(TrainSampleID)
    TrainDataMeta.IniHitClusterMetaData(stepX, stepY, stepZ, cut_dt, cut_dr, cut_dz, testRatio, valRatio, z_offset, y_offset, x_offset, Xsteps, Zsteps,X_overlap, Y_overlap, Z_overlap)
    TrainDataMeta.UpdateStatus(1)
    print(UI.PickleOperations(TrainSampleOutputMeta,'w', TrainDataMeta)[1])

elif os.path.isfile(TrainSampleOutputMeta)==True:
    print(UI.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+TrainSampleOutputMeta+bcolors.ENDC)
    #Loading parameters from the Meta file if exists.
    MetaInput=UI.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
    stepX=Meta.stepX
    stepY=Meta.stepY
    stepZ=Meta.stepZ
    cut_dt=Meta.cut_dt
    cut_dr=Meta.cut_dr
    cut_dz=Meta.cut_dz
    testRatio=Meta.testRatio
    valRatio=Meta.valRatio
    stepX=stepX
    z_offset=Meta.z_offset
    y_offset=Meta.y_offset
    x_offset=Meta.x_offset
    Xsteps=Meta.Xsteps
    Zsteps=Meta.Zsteps
    Y_overlap=Meta.Y_overlap
    X_overlap=Meta.X_overlap
    Z_overlap=Meta.Z_overlap
print(UI.TimeStamp(),bcolors.OKGREEN+'Stage 1 has successfully completed'+bcolors.ENDC)

# ########################################     Preset framework parameters    #########################################

UI.Msg('vanilla','Analysing the current script status...')
Status=Meta.Status[-1]
if ForceStatus!='N':
    Status=int(ForceStatus)
UI.Msg('vanilla','Current stage is '+str(Status)+'...')

################ Set the execution sequence for the script
Program=[]
###### Stage 0
prog_entry=[]
job_sets=[]
for i in range(0,Xsteps):
                job_sets.append(Zsteps)
prog_entry.append(' Sending hit cluster to the HTCondor, so the model assigns weights between hits')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/','SelectedTrainClusters','MTr1','.csv',TrainSampleID,job_sets,'MTr1_GenerateTrainClusters_Sub.py'])
prog_entry.append([' --stepZ ', ' --stepY ', ' --stepX ', ' --cut_dt ', ' --cut_dr ', ' ----cut_dz ',' --Z_overlap ',' --Y_overlap ',' --X_overlap ', ' --Z_ID_Max '])
prog_entry.append([stepZ,stepY,stepX, cut_dt,cut_dr,cut_dz, Z_overlap,Y_overlap,X_overlap, Zsteps])
prog_entry.append(Xsteps*Zsteps)
prog_entry.append(LocalSub)
prog_entry.append('N/A')
prog_entry.append(HTCondorLog)
prog_entry.append(False)
Program.append(prog_entry)
print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry))

###### Stage 3
Program.append('Custom')


print(UI.TimeStamp(),'There are '+str(len(Program)+1)+' stages (0-'+str(len(Program)+1)+') of this script',bcolors.ENDC)
print(UI.TimeStamp(),'Current stage has a code',Status,bcolors.ENDC)
while Status<len(Program):
    if Program[Status]!='Custom':
        #Standard process here
       Result=UI.StandardProcess(Program,Status,SubGap,SubPause,RequestExtCPU,JobFlavour,ReqMemory,time_int,Patience)
       if Result[0]:
            UI.UpdateStatus(Status+1,Meta,TrainSampleOutputMeta)
            print(Status)
       else:
             Status=20
             break

    elif Status==2:
      #Non standard processes (that don't follow the general pattern) have been coded here
      print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
      print(UI.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' Using the results from previous steps to map merged trackIDs to the original reconstruction file')
      print('Wip')
      exit()
      # try:
      #   #Read the output with hit- ANN Track map
      #   FirstFile=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_RTr1c_'+RecBatchID+'_0'+'/RTr1c_'+RecBatchID+'_hit_cluster_rec_x_set_0.csv'
      #   print(UI.TimeStamp(),'Loading the file ',bcolors.OKBLUE+FirstFile+bcolors.ENDC)
      #   TrackMap=pd.read_csv(FirstFile,header=0)
      #   input_file_location=args.f
      #   print(UI.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
      #   #Reading the original file with Raw hits
      #   Data=pd.read_csv(input_file_location,header=0)
      #   Data[PM.Hit_ID] = Data[PM.Hit_ID].astype(str)
      #
      #   if SliceData: #If we want to perform reconstruction on the fraction of the Brick
      #      CutData=Data.drop(Data.index[(Data[PM.x] > Xmax) | (Data[PM.x] < Xmin) | (Data[PM.y] > Ymax) | (Data[PM.y] < Ymin)]) #The focus area where we reconstruct
      #   else:
      #      CutData=Data #If we reconstruct the whole brick we jsut take the whole data. No need to separate.
      #
      #   CutData.drop([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID'],axis=1,inplace=True,errors='ignore') #Removing old ANNDEA reconstruction results so we can overwrite with the new ones
      #   #Map reconstructed ANN tracks to hits in the Raw file - this is in essesential for the final output of the tracking
      #   TrackMap['HitID'] = TrackMap['HitID'].astype(str)
      #   CutData[PM.Hit_ID] = CutData[PM.Hit_ID].astype(str)
      #   CutData=pd.merge(CutData,TrackMap,how='left', left_on=[PM.Hit_ID], right_on=['HitID'])
      #
      #   CutData.drop(['HitID'],axis=1,inplace=True) #Make sure that HitID is not the Hit ID name in the raw data.
      #   Data=CutData
      #
      #
      #   #It was discovered that the output is not perfect: while the hit fidelity is achieved we don't have a full plate hit fidelity for a given track. It is still possible for a track to have multiple hits at one plate.
      #   #In order to fix it we need to apply some additional logic to those problematic tracks.
      #   print(UI.TimeStamp(),'Identifying problematic tracks where there is more than one hit per plate...')
      #   Hit_Map=Data[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.Hit_ID]] #Separating the hit map
      #   Data.drop([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID'],axis=1,inplace=True) #Remove the ANNDEA tracking info from the main data
      #
      #
      #
      #
      #
      #   Hit_Map=Hit_Map.dropna() #Remove unreconstructing hits - we are not interested in them atm
      #   Hit_Map_Stats=Hit_Map[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z,PM.Hit_ID]] #Calculating the stats
      #
      #   Hit_Map_Stats=Hit_Map_Stats.groupby([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID']).agg({PM.z:pd.Series.nunique,PM.Hit_ID: pd.Series.nunique}).reset_index() #Calculate the number fo unique plates and hits
      #   Ini_No_Tracks=len(Hit_Map_Stats)
      #   print(UI.TimeStamp(),bcolors.WARNING+'The initial number of tracks is '+ str(Ini_No_Tracks)+bcolors.ENDC)
      #   Hit_Map_Stats=Hit_Map_Stats.rename(columns={PM.z: "No_Plates",PM.Hit_ID:"No_Hits"}) #Renaming the columns so they don't interfere once we join it back to the hit map
      #   Hit_Map_Stats=Hit_Map_Stats[Hit_Map_Stats.No_Plates >= PM.MinHitsTrack]
      #   Prop_No_Tracks=len(Hit_Map_Stats)
      #   print(UI.TimeStamp(),bcolors.WARNING+'After dropping single hit tracks, left '+ str(Prop_No_Tracks)+' tracks...'+bcolors.ENDC)
      #   Hit_Map=pd.merge(Hit_Map,Hit_Map_Stats,how='inner',on = [RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID']) #Join back to the hit map
      #   Good_Tracks=Hit_Map[Hit_Map.No_Plates == Hit_Map.No_Hits] #For all good tracks the number of hits matches the number of plates, we won't touch them
      #   Good_Tracks=Good_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.Hit_ID]] #Just strip off the information that we don't need anymore
      #
      #   Bad_Tracks=Hit_Map[Hit_Map.No_Plates < Hit_Map.No_Hits] #These are the bad guys. We need to remove this extra hits
      #   Bad_Tracks=Bad_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.Hit_ID]]
      #
      #   #Id the problematic plates
      #   Bad_Tracks_Stats=Bad_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z,PM.Hit_ID]]
      #   Bad_Tracks_Stats=Bad_Tracks_Stats.groupby([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z])[PM.Hit_ID].nunique().reset_index() #Which plates have double hits?
      #   Bad_Tracks_Stats=Bad_Tracks_Stats.rename(columns={PM.Hit_ID: "Problem"}) #Renaming the columns so they don't interfere once we join it back to the hit map
      #   Bad_Tracks_Stats[RecBatchID+'_Brick_ID'] = Bad_Tracks_Stats[RecBatchID+'_Brick_ID'].astype(str)
      #   Bad_Tracks_Stats[RecBatchID+'_Track_ID'] = Bad_Tracks_Stats[RecBatchID+'_Track_ID'].astype(str)
      #   Bad_Tracks[RecBatchID+'_Brick_ID'] = Bad_Tracks[RecBatchID+'_Brick_ID'].astype(str)
      #   Bad_Tracks[RecBatchID+'_Track_ID'] = Bad_Tracks[RecBatchID+'_Track_ID'].astype(str)
      #   Bad_Tracks=pd.merge(Bad_Tracks,Bad_Tracks_Stats,how='inner',on = [RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z])
      #
      #
      #
      #   Bad_Tracks.sort_values([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z],ascending=[0,0,1],inplace=True)
      #
      #   Bad_Tracks_CP_File=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_RTr1c_'+RecBatchID+'_0'+'/RTr1c_'+RecBatchID+'_Bad_Tracks_CP.csv'
      #   if os.path.isfile(Bad_Tracks_CP_File)==False or Mode=='RESET':
      #
      #       Bad_Tracks_Head=Bad_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID']]
      #       Bad_Tracks_Head.drop_duplicates(inplace=True)
      #       Bad_Tracks_List=Bad_Tracks.values.tolist() #I find it is much easier to deal with tracks in list format when it comes to fitting
      #       Bad_Tracks_Head=Bad_Tracks_Head.values.tolist()
      #       Bad_Track_Pool=[]
      #
      #       #Bellow we build the track representatation that we can use to fit slopes
      #       with alive_bar(len(Bad_Tracks_Head),force_tty=True, title='Building track representations...') as bar:
      #                   for bth in Bad_Tracks_Head:
      #                      bar()
      #                      bth.append([])
      #                      bt=0
      #                      trigger=False
      #                      while bt<(len(Bad_Tracks_List)):
      #                          if (bth[0]==Bad_Tracks_List[bt][0] and bth[1]==Bad_Tracks_List[bt][1]):
      #                             if Bad_Tracks_List[bt][8]==1: #We only build polynomials for hits in a track that do not have duplicates - these are 'trusted hits'
      #                                bth[2].append(Bad_Tracks_List[bt][2:-2])
      #                             del Bad_Tracks_List[bt]
      #                             bt-=1
      #                             trigger=True
      #                          elif trigger:
      #                              break
      #                          else:
      #                              continue
      #                          bt+=1
      #
      #
      #       with alive_bar(len(Bad_Tracks_Head),force_tty=True, title='Fitting the tracks...') as bar:
      #        for bth in Bad_Tracks_Head:
      #          bar()
      #          if len(bth[2])==1: #Only one trusted hit - In these cases whe we take only tx and ty slopes of the single base track. Polynomial of the first degree and the equations of the line are x=ax+tx*z and y=ay+ty*z
      #              x=bth[2][0][0]
      #              z=bth[2][0][2]
      #              tx=bth[2][0][3]
      #              ax=x-tx*z
      #              bth.append(ax) #Append x intercept
      #              bth.append(tx) #Append x slope
      #              bth.append(0) #Append a placeholder slope (for polynomial case)
      #              y=bth[2][0][1]
      #              ty=bth[2][0][4]
      #              ay=y-ty*z
      #              bth.append(ay) #Append x intercept
      #              bth.append(ty) #Append x slope
      #              bth.append(0) #Append a placeholder slope (for polynomial case)
      #              del(bth[2])
      #          elif len(bth[2])==2: #Two trusted hits - In these cases whe we fit a polynomial of the first degree and the equations of the line are x=ax+tx*z and y=ay+ty*z
      #              x,y,z=[],[],[]
      #              x=[bth[2][0][0],bth[2][1][0]]
      #              y=[bth[2][0][1],bth[2][1][1]]
      #              z=[bth[2][0][2],bth[2][1][2]]
      #              tx=np.polyfit(z,x,1)[0]
      #              ax=np.polyfit(z,x,1)[1]
      #              ty=np.polyfit(z,y,1)[0]
      #              ay=np.polyfit(z,y,1)[1]
      #              bth.append(ax) #Append x intercept
      #              bth.append(tx) #Append x slope
      #              bth.append(0) #Append a placeholder slope (for polynomial case)
      #              bth.append(ay) #Append x intercept
      #              bth.append(ty) #Append x slope
      #              bth.append(0) #Append a placeholder slope (for polynomial case)
      #              del(bth[2])
      #          elif len(bth[2])==0:
      #              del(bth)
      #              continue
      #          else: #Three pr more trusted hits - In these cases whe we fit a polynomial of the second degree and the equations of the line are x=ax+(t1x*z)+(t2x*z*z) and y=ay+(t1y*z)+(t2y*z*z)
      #              x,y,z=[],[],[]
      #              for i in bth[2]:
      #                  x.append(i[0])
      #              for j in bth[2]:
      #                  y.append(j[1])
      #              for k in bth[2]:
      #                  z.append(k[2])
      #              t2x=np.polyfit(z,x,2)[0]
      #              t1x=np.polyfit(z,x,2)[1]
      #              ax=np.polyfit(z,x,2)[2]
      #
      #              t2y=np.polyfit(z,y,2)[0]
      #              t1y=np.polyfit(z,y,2)[1]
      #              ay=np.polyfit(z,y,2)[2]
      #
      #              bth.append(ax) #Append x intercept
      #              bth.append(t1x) #Append x slope
      #              bth.append(t2x) #Append a placeholder slope (for polynomial case)
      #              bth.append(ay) #Append x intercept
      #              bth.append(t1y) #Append x slope
      #              bth.append(t2y) #Append a placeholder slope (for polynomial case)
      #              del(bth[2])
      #
      #
      #       #Once we get coefficients for all tracks we convert them back to Pandas dataframe and join back to the data
      #       Bad_Tracks_Head=pd.DataFrame(Bad_Tracks_Head, columns = [RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID','ax','t1x','t2x','ay','t1y','t2y'])
      #
      #       print(UI.TimeStamp(),'Saving the checkpoint file ',bcolors.OKBLUE+Bad_Tracks_CP_File+bcolors.ENDC)
      #       Bad_Tracks_Head.to_csv(Bad_Tracks_CP_File,index=False)
      #   else:
      #      print(UI.TimeStamp(),'Loading the checkpoint file ',bcolors.OKBLUE+Bad_Tracks_CP_File+bcolors.ENDC)
      #      Bad_Tracks_Head=pd.read_csv(Bad_Tracks_CP_File,header=0)
      #      Bad_Tracks_Head=Bad_Tracks_Head[Bad_Tracks_Head.ax != '[]']
      #      Bad_Tracks_Head['ax'] = Bad_Tracks_Head['ax'].astype(float)
      #      Bad_Tracks_Head['ay'] = Bad_Tracks_Head['ay'].astype(float)
      #      Bad_Tracks_Head['t1x'] = Bad_Tracks_Head['t1x'].astype(float)
      #      Bad_Tracks_Head['t2x'] = Bad_Tracks_Head['t2x'].astype(float)
      #      Bad_Tracks_Head['t1y'] = Bad_Tracks_Head['t1y'].astype(float)
      #      Bad_Tracks_Head['t2y'] = Bad_Tracks_Head['t2y'].astype(float)
      #
      #   print(UI.TimeStamp(),'Removing problematic hits...')
      #   Bad_Tracks_Head[str(RecBatchID)+'_Brick_ID'] = Bad_Tracks_Head[str(RecBatchID)+'_Brick_ID'].astype(str)
      #   Bad_Tracks_Head[str(RecBatchID)+'_Track_ID'] = Bad_Tracks_Head[str(RecBatchID)+'_Track_ID'].astype(str)
      #   Bad_Tracks[str(RecBatchID)+'_Brick_ID'] = Bad_Tracks[str(RecBatchID)+'_Brick_ID'].astype(str)
      #   Bad_Tracks[str(RecBatchID)+'_Track_ID'] = Bad_Tracks[str(RecBatchID)+'_Track_ID'].astype(str)
      #
      #
      #   Bad_Tracks=pd.merge(Bad_Tracks,Bad_Tracks_Head,how='inner',on = [str(RecBatchID)+'_Brick_ID',str(RecBatchID)+'_Track_ID'])
      #
      #   print(UI.TimeStamp(),'Calculating x and y coordinates of the fitted line for all plates in the track...')
      #   #Calculating x and y coordinates of the fitted line for all plates in the track
      #   Bad_Tracks['new_x']=Bad_Tracks['ax']+(Bad_Tracks[PM.z]*Bad_Tracks['t1x'])+((Bad_Tracks[PM.z]**2)*Bad_Tracks['t2x'])
      #   Bad_Tracks['new_y']=Bad_Tracks['ay']+(Bad_Tracks[PM.z]*Bad_Tracks['t1y'])+((Bad_Tracks[PM.z]**2)*Bad_Tracks['t2y'])
      #
      #   #Calculating how far hits deviate from the fit polynomial
      #   print(UI.TimeStamp(),'Calculating how far hits deviate from the fit polynomial...')
      #   Bad_Tracks['d_x']=Bad_Tracks[PM.x]-Bad_Tracks['new_x']
      #   Bad_Tracks['d_y']=Bad_Tracks[PM.y]-Bad_Tracks['new_y']
      #
      #   Bad_Tracks['d_r']=Bad_Tracks['d_x']**2+Bad_Tracks['d_y']**2
      #   Bad_Tracks['d_r'] = Bad_Tracks['d_r'].astype(float)
      #   Bad_Tracks['d_r']=np.sqrt(Bad_Tracks['d_r']) #Absolute distance
      #   Bad_Tracks=Bad_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z,PM.Hit_ID,'d_r']]
      #
      #   #Sort the tracks and their hits by Track ID, Plate and distance to the perfect line
      #   print(UI.TimeStamp(),'Sorting the tracks and their hits by Track ID, Plate and distance to the perfect line...')
      #   Bad_Tracks.sort_values([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z,'d_r'],ascending=[0,0,1,1],inplace=True)
      #
      #   #If there are two hits per plate we will keep the one which is closer to the line
      #   Bad_Tracks.drop_duplicates(subset=[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z],keep='first',inplace=True)
      #   Bad_Tracks=Bad_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.Hit_ID]]
      #   Good_Tracks=pd.concat([Good_Tracks,Bad_Tracks]) #Combine all ANNDEA tracks together
      #   Data=pd.merge(Data,Good_Tracks,how='left', on=[PM.Hit_ID]) #Re-map corrected ANNDEA Tracks back to the main data
      #   output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'.csv' #Final output. We can use this file for further operations
      #   Data.to_csv(output_file_location,index=False)
      #   print(UI.TimeStamp(), bcolors.OKGREEN+"The tracked data has been written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
      #   print(UI.TimeStamp(),bcolors.OKGREEN+'Stage 4 has successfully completed'+bcolors.ENDC)
      #   UI.UpdateStatus(Status+1,Meta,RecOutputMeta)
      # except Exception as e:
      #     print(UI.TimeStamp(),bcolors.FAIL+'Stage 4 is uncompleted due to: '+str(e)+bcolors.ENDC)
      #     Status=21
      #     break
    MetaInput=UI.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
    Status=Meta.Status[-1]

if Status<20:
    #Removing the temp files that were generated by the process
    #print(UI.TimeStamp(),'Performing the cleanup... ')
    #print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'d',['RTr1a','RTr1b','RTr1c','RTr1d']))
    UI.Msg('success',"Segment merging has been completed")
else:
    UI.Msg('failed',"Segment merging has not been completed as one of the processes has timed out. Please run the script again (without Reset Mode).")
    exit()

# #The function bellow manages HTCondor jobs - so we don't have to do manually
# def AutoPilot(wait_min, interval_min, max_interval_tolerance):
#     print(UF.TimeStamp(),'Going on an autopilot mode for ',wait_min, 'minutes while checking HTCondor every',interval_min,'min',bcolors.ENDC)
#     wait_sec=wait_min*60
#     interval_sec=interval_min*60 #Converting min to sec as this time.sleep takes as an argument
#     intervals=int(math.ceil(wait_sec/interval_sec))
#     for interval in range(1,intervals+1):
#         time.sleep(interval_sec)
#         bad_pop=[]
#         print(UF.TimeStamp(),"Scheduled job checkup...") #Progress display
#         for k in range(0,Zsteps):
#          for i in range(0,Xsteps):
#               #Preparing HTCondor submission
#               OptionHeader = [' --Z_ID ', ' --stepX ',' --stepY ',' --stepZ ', ' --EOS ', " --AFS ", " --zOffset ", " --xOffset ", " --yOffset ", ' --cut_dt ', ' --cut_dr ', ' --testRatio ', ' --valRatio ', ' --X_ID ',' --TrainSampleID ',' --Y_overlap ',' --X_overlap ',' --Z_overlap ', ' --PY ']
#               OptionLine = [k, stepX,stepY,stepZ, EOS_DIR, AFS_DIR, z_offset, x_offset, y_offset, cut_dt,cut_dr,testRatio,valRatio, i,TrainSampleID,Y_overlap,X_overlap,Z_overlap,PY_DIR]
#               required_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/MTr1a_'+TrainSampleID+'_SelectedTrainClusters_'+str(k)+'_'+str(i)+'.pkl'
#               SHName = AFS_DIR + '/HTCondor/SH/SH_MTr1_'+ TrainSampleID+'_' + str(k) + '_' + str(i) + '.sh'
#               SUBName = AFS_DIR + '/HTCondor/SUB/SUB_MTr1_'+ TrainSampleID+'_'+ str(k) + '_' + str(i) + '.sub'
#               MSGName = AFS_DIR + '/HTCondor/MSG/MSG_MTr1_' + TrainSampleID+'_' + str(k) + '_' + str(i)
#               #The actual training sample generation is done by the script bellow
#               ScriptName = AFS_DIR + '/Code/Utilities/MTr1_GenerateTrainClusters_Sub.py '
#               if os.path.isfile(required_output_file_location)!=True: #Calculating the number of unfinished jobs
#                  bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-MTr1-'+TrainSampleID, False,RequestExtCPU,JobFlavour,ReqMemory])
#         if len(bad_pop)>0:
#               print(UF.TimeStamp(),bcolors.WARNING+'Autopilot status update: There are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
#               if interval%max_interval_tolerance==0: #If jobs are not received after fixed number of check-ups we resubmit. Jobs sometimes fail on HTCondor for various reasons.
#                  for bp in bad_pop:
#                      UF.SubmitJobs2Condor(bp) #Sumbitting all missing jobs
#                  print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
#         else:
#              return True
#     return False
#
# def Success(Finished):
#         if Finished:
#             print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
#             print(UF.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' All HTCondor jobs have completed.')
#             count=0
#             SampleCount=0
#             NodeFeatures=PM.num_node_features
#             EdgeFeatures=PM.num_edge_features
#             for k in range(0,Zsteps):
#                 progress=round((float(k)/float(Zsteps))*100,2)
#                 print(UF.TimeStamp(),"Collating results, progress is ",progress,' %') #Progress display
#                 for i in range(0,Xsteps):
#                         count+=1
#                         source_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/MTr1a_'+TrainSampleID+'_SelectedTrainClusters_'+str(k)+'_'+str(i)+'.pkl'
#                         destination_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/'+TrainSampleID+'_TTr_OUTPUT_'+str(count)+'.pkl'
#                         os.rename(source_output_file_location, destination_output_file_location)
#                         TrainingSample=UF.PickleOperations(destination_output_file_location,'r', 'N/A')[0]
#                         SampleCount+=len(TrainingSample)
#             print(UF.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+TrainSampleOutputMeta+bcolors.ENDC)
#             MetaInput=UF.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')
#             MetaInput[0].UpdateHitClusterMetaData(SampleCount,NodeFeatures,EdgeFeatures,count)
#             print(UF.PickleOperations(TrainSampleOutputMeta,'w', MetaInput[0])[1])
#             HTCondorTag="SoftUsed == \"ANNDEA-MTr-"+TrainSampleID+"\""
#             UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MTr1_'+TrainSampleID, ['MTr1a_'+TrainSampleID,'ETr1_'+TrainSampleID,'MTr1_'+TrainSampleID], HTCondorTag) #If successful we delete all temp files created by the process
#             print(UF.TimeStamp(),bcolors.OKGREEN+'Training samples are ready for the model creation/training'+bcolors.ENDC)
#             TrainSampleInputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/'+TrainSampleID+'_info.pkl'
#             print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+TrainSampleInputMeta+bcolors.ENDC)
#             MetaInput=UF.PickleOperations(TrainSampleInputMeta,'r', 'N/A')
#             print(MetaInput[1])
#             Meta=MetaInput[0]
#             TrainSamples=[]
#             ValSamples=[]
#             TestSamples=[]
#             for i in range(1,Meta.no_sets+1):
#                 flocation=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/'+TrainSampleID+'_TTr_OUTPUT_'+str(i)+'.pkl'
#                 print(UF.TimeStamp(),'Loading data from ',bcolors.OKBLUE+flocation+bcolors.ENDC)
#                 TrainClusters=UF.PickleOperations(flocation,'r', 'N/A')
#                 TrainClusters=TrainClusters[0]
#                 TrainFraction=int(math.floor(len(TrainClusters)*(1.0-(Meta.testRatio+Meta.valRatio))))
#                 ValFraction=int(math.ceil(len(TrainClusters)*Meta.valRatio))
#                 for smpl in range(0,TrainFraction):
#                            if TrainClusters[smpl].ClusterGraph.num_edges>0 and Sampling>=random.random(): #Only graph clusters with edges are kept + sampled
#                              TrainSamples.append(TrainClusters[smpl].ClusterGraph)
#                 for smpl in range(TrainFraction,TrainFraction+ValFraction):
#                            if TrainClusters[smpl].ClusterGraph.num_edges>0 and Sampling>=random.random():
#                              ValSamples.append(TrainClusters[smpl].ClusterGraph)
#                 for smpl in range(TrainFraction+ValFraction,len(TrainClusters)):
#                            if TrainClusters[smpl].ClusterGraph.num_edges>0 and Sampling>=random.random():
#                              TestSamples.append(TrainClusters[smpl].ClusterGraph)
#             output_train_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/'+TrainSampleID+'_TRAIN_SAMPLES'+'.pkl'
#             print(UF.PickleOperations(output_train_file_location,'w', TrainSamples)[1])
#             output_val_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/'+TrainSampleID+'_VAL_SAMPLES'+'.pkl'
#             print(UF.PickleOperations(output_val_file_location,'w', ValSamples)[1])
#             output_test_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/'+TrainSampleID+'_TEST_SAMPLES'+'.pkl'
#             print(UF.PickleOperations(output_test_file_location,'w', TestSamples)[1])
#             print(UF.TimeStamp(), bcolors.OKGREEN+"Train data has been re-generated successfully..."+bcolors.ENDC)
#             print(UF.TimeStamp(),bcolors.OKGREEN+'Please run MTr2_TrainModel.py after this to create/train a model'+bcolors.ENDC)
#             print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
#         else:
#             print(UF.TimeStamp(),bcolors.FAIL+'Unfortunately no results have been yield. Please rerun the script, and if the problem persists, check that the HTCondor jobs run adequately.'+bcolors.ENDC)
#             exit()
#
# if Mode=='RESET':
#    HTCondorTag="SoftUsed == \"ANNDEA-MTr1-"+TrainSampleID+"\""
#    UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MTr1_'+TrainSampleID, ['MTr1a_'+TrainSampleID,TrainSampleID+'_TTr_OUTPUT_'], HTCondorTag) #If we do total reset we want to delete all temp files left by a possible previous attempt
#    print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
# bad_pop=[]
# Status=False
# print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
# print(UF.TimeStamp(),bcolors.BOLD+'Stage 2:'+bcolors.ENDC+' Checking HTCondor jobs and preparing the job submission')
#
#
# for k in range(0,Zsteps):
#         progress=round((float(k)/float(Zsteps))*100,2)
#         print(UF.TimeStamp(),"progress is ",progress,' %') #Progress display
#         for i in range(0,Xsteps):
#              OptionHeader = [' --Z_ID ', ' --stepX ',' --stepY ',' --stepZ ', ' --EOS ', " --AFS ", " --zOffset ", " --xOffset ", " --yOffset ", ' --cut_dt ', ' --cut_dr ', ' --testRatio ', ' --valRatio ', ' --X_ID ',' --TrainSampleID ',' --Y_overlap ',' --X_overlap ',' --Z_overlap ', ' --PY ']
#              OptionLine = [k, stepX,stepY,stepZ, EOS_DIR, AFS_DIR, z_offset, x_offset, y_offset, cut_dt,cut_dr,testRatio,valRatio, i,TrainSampleID,Y_overlap,X_overlap,Z_overlap,PY_DIR]
#              required_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MTr1a_'+TrainSampleID+'_SelectedTrainClusters_'+str(k)+'_'+str(i)+'.pkl'
#              SHName = AFS_DIR + '/HTCondor/SH/SH_MTr1_'+ TrainSampleID+'_' + str(k) + '_' + str(i) + '.sh'
#              SUBName = AFS_DIR + '/HTCondor/SUB/SUB_MTr1_'+ TrainSampleID+'_'+ str(k) + '_' + str(i) + '.sub'
#              MSGName = AFS_DIR + '/HTCondor/MSG/MSG_MTr1_' + TrainSampleID+'_' + str(k) + '_' + str(i)
#              ScriptName = AFS_DIR + '/Code/Utilities/MTr1_GenerateTrainClusters_Sub.py '
#              if os.path.isfile(required_output_file_location)!=True:
#                 bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-MTr1-'+TrainSampleID, False,False])
# if len(bad_pop)==0:
#     Success(True)
#
#
# if (Zsteps*Xsteps)==len(bad_pop): #Scenario where all jobs are missing - doing a batch submission to save time
#     print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
#     print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
#     print(bcolors.BOLD+'If you would like to wait please enter enter the maximum wait time in minutes'+bcolors.ENDC)
#     print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
#     UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
#     if UserAnswer=='E':
#           print(UF.TimeStamp(),'OK, exiting now then')
#           exit()
#     if UserAnswer=='R':
#         print(UF.TimeStamp(),'Submitting jobs to HTCondor... ',bcolors.ENDC)
#         for k in range(0,Zsteps):
#                   OptionHeader = [' --Z_ID ', ' --stepX ',' --stepY ',' --stepZ ', ' --EOS ', " --AFS ", " --zOffset ", " --xOffset ", " --yOffset ", ' --cut_dt ', ' --cut_dr ', ' --testRatio ', ' --valRatio ', ' --X_ID ',' --TrainSampleID ',' --Y_overlap ',' --X_overlap ',' --Z_overlap ',' --PY ']
#                   OptionLine = [k, stepX,stepY,stepZ, EOS_DIR, AFS_DIR, z_offset, x_offset, y_offset, cut_dt,cut_dr,testRatio, valRatio,'$1',TrainSampleID,Y_overlap,X_overlap,Z_overlap,PY_DIR]
#                   SHName = AFS_DIR + '/HTCondor/SH/SH_MTr1_'+ TrainSampleID+'_' + str(k) + '.sh'
#                   SUBName = AFS_DIR + '/HTCondor/SUB/SUB_MTr1_'+ TrainSampleID+'_'+ str(k) + '.sub'
#                   MSGName = AFS_DIR + '/HTCondor/MSG/MSG_MTr1_' + TrainSampleID+'_' + str(k)
#                   ScriptName = AFS_DIR + '/Code/Utilities/MTr1_GenerateTrainClusters_Sub.py '
#                   UF.SubmitJobs2Condor([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, Xsteps, 'ANNDEA-MTr1-'+TrainSampleID, False,RequestExtCPU,JobFlavour,ReqMemory])
#                   print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
#         Success(AutoPilot(120,10,Patience))
#     else:
#         Success(AutoPilot(120,10,Patience))
#
# elif len(bad_pop)>0: #If some jobs have been complited, the missing jobs are submitted individually to avoidf overloading HTCondor for nothing
#       print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
#       print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
#       print(bcolors.BOLD+'If you would like to wait please enter enter the maximum wait time in minutes'+bcolors.ENDC)
#       print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
#       UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
#       if UserAnswer=='E':
#           print(UF.TimeStamp(),'OK, exiting now then')
#           exit()
#       if UserAnswer=='R':
#          for bp in bad_pop:
#               UF.SubmitJobs2Condor(bp)
#          print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
#          Success(AutoPilot(120,10,Patience))
#       else:
#          Success(AutoPilot(int(UserAnswer),10,Patience))




