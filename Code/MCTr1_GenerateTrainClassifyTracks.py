#This simple connects hits in the data to produce tracks
#Tracking Module of the ANNDEA package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
import argparse
import pandas as pd #We use Panda for a routine data processing
import math #We use it for data manipulation
import os
import time
import ast
import random
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
    if c[0]=='PY_DIR':
        PY_DIR=c[1]
csv_reader.close()
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import U_UI as UI #This is where we keep routine utility functions
import Parameters as PM #This is where we keep framework global parameters
UI.WelcomeMsg('Initialising ANNDEA track classification module training sample generator...','Filips Fedotovs (PhD student at UCL), Wenqing Xie (MSc student at UCL)','Please reach out to filips.fedotovs@cern.ch for any queries')
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
parser.add_argument('--ForceStatus',help="Would you like the program run from specific status number? (Only for advance users)", default='N')
parser.add_argument('--JobFlavour',help="Specifying the length of the HTCondor job walltime. Currently at 'workday' which is 8 hours.", default='workday')
parser.add_argument('--MinHitsTrack',help="What is the minimum number of hits per track?", default=PM.MinHitsTrack)
parser.add_argument('--SubPause',help="How long to wait in minutes after submitting 10000 jobs?", default='60')
parser.add_argument('--SubGap',help="How long to wait in minutes after submitting 10000 jobs?", default='10000')
parser.add_argument('--PY',help="Python libraries directory location", default='.')
parser.add_argument('--HTCondorLog',help="Local submission?", default=False,type=bool)
######################################## Parsing argument values  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
ClassHeaders=ast.literal_eval(args.ClassHeaders)
ClassNames=ast.literal_eval(args.ClassNames)
ClassValues=ast.literal_eval(args.ClassValues)
TrackID=args.TrackID
BrickID=args.BrickID
TrainSampleID=args.TrainSampleID
ForceStatus=args.ForceStatus
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
MinHitsTrack=int(args.MinHitsTrack)
LocalSub=(args.LocalSub=='Y')
HTCondorLog=args.HTCondorLog
if LocalSub:
   time_int=0
else:
    time_int=10
#Establishing paths
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
TrainSampleOutputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
required_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/MCTr1_'+TrainSampleID+'_TRACKS.csv'
ColumnsToImport=[TrackID,BrickID,PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.MC_Track_ID,PM.MC_Event_ID]
ExtraColumns=[]
for i in ClassNames:
    for j in i:
        if (j in ColumnsToImport)==False:
            ColumnsToImport.append(j)
        if (j in ExtraColumns)==False:
                ExtraColumns.append(j)

Regression=ClassValues[0][0]=='Reg'

########################################     Phase 1 - Create compact source file    #########################################
print(UI.TimeStamp(),bcolors.BOLD+'Stage 0:'+bcolors.ENDC+' Preparing the source data...')
if Mode=='RESET':
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, TrainSampleID,'d',['MCTr1a']))
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, TrainSampleID,'c'))
elif Mode=='CLEANUP':
     print(UI.ManageFolders(AFS_DIR, EOS_DIR, TrainSampleID,'d',['MCTr1a']))
     exit()
else:
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, TrainSampleID,'c'))



if os.path.isfile(required_file_location)==False:
        print(UI.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
        data=pd.read_csv(input_file_location,
                    header=0,
                    usecols=ColumnsToImport)
        total_rows=len(data.axes[0])

        print(UI.TimeStamp(),'The raw data has ',total_rows,' hits')
        print(UI.TimeStamp(),'Removing unreconstructed hits...')
        data=data.dropna()
        final_rows=len(data.axes[0])
        print(UI.TimeStamp(),'The cleaned data has ',final_rows,' hits')

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
             ValidEvents.drop([PM.x,PM.y,PM.z,PM.tx,PM.ty,'MC_Mother_Track_ID']+ExtraColumns,axis=1,inplace=True)
             ValidEvents.drop_duplicates(subset='Rec_Seg_ID',keep='first',inplace=True)
             data=pd.merge(data, ValidEvents, how="inner", on=['Rec_Seg_ID'])
             final_rows=len(data.axes[0])
             print(UI.TimeStamp(),'The sliced data has ',final_rows,' hits')


        output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/MCTr1_'+TrainSampleID+'_TRACKS.csv'
        if Regression:
            print(UI.TimeStamp(),'Normalising regression value',ExtraColumns[0])
            data_agg=data.groupby(['Rec_Seg_ID','MC_Mother_Track_ID']).agg(subject_reg_val=pd.NamedAgg(column=ClassNames[0][0], aggfunc=ClassValues[0][1])).reset_index()
            data_agg=data_agg.rename(columns={'subject_reg_val': ClassNames[0][0]})
            data_agg[ClassNames[0][0]]=data_agg[ClassNames[0][0]].astype(float)-(float(ClassValues[0][2])/2)
            data_agg[ClassNames[0][0]]=data_agg[ClassNames[0][0]].astype(float).div(float(ClassValues[0][2])/2)
            data.drop([ClassNames[0][0]],axis=1,inplace=True)
            data=pd.merge(data,data_agg, how="inner", on=['Rec_Seg_ID','MC_Mother_Track_ID'])
        print(UI.TimeStamp(),'Removing tracks which have less than',MinHitsTrack,'hits...')
        track_no_data=data.groupby(['MC_Mother_Track_ID','Rec_Seg_ID']+ExtraColumns,as_index=False).count()
        track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty],axis=1)
        track_no_data=track_no_data.rename(columns={PM.x: "Rec_Seg_No"})
        new_combined_data=pd.merge(data, track_no_data, how="left", on=['Rec_Seg_ID','MC_Mother_Track_ID']+ExtraColumns)
        new_combined_data = new_combined_data[new_combined_data.Rec_Seg_No >= MinHitsTrack]
        new_combined_data = new_combined_data.drop(["Rec_Seg_No"],axis=1)
        new_combined_data=new_combined_data.sort_values(['Rec_Seg_ID',PM.x],ascending=[1,1])
        grand_final_rows=len(new_combined_data.axes[0])
        print(UI.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
        new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
        new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
        new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
        new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
        new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
        new_combined_data.drop(['MC_Mother_Track_ID'],axis=1,inplace=True)
        new_combined_data.to_csv(output_file_location,index=False)
        data=new_combined_data[['Rec_Seg_ID']]
        print(UI.TimeStamp(),'Analysing the data sample in order to understand how many jobs to submit to HTCondor... ',bcolors.ENDC)
        data.drop_duplicates(subset='Rec_Seg_ID',keep='first',inplace=True)
        data = data.values.tolist()
        no_submissions=math.ceil(len(data)/PM.MaxSegments)
        print(UI.TimeStamp(), bcolors.OKGREEN+"The track segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
        Meta=UI.JobMeta(TrainSampleID)
        Meta.UpdateJobMeta(['ClassHeaders','ClassNames','ClassValues','MaxSegments','no_submissions','MinHitsTrack'], [ClassHeaders,ClassNames,ClassValues,PM.MaxSegments,JobSets,MinHitsTrack])
        Meta.UpdateStatus(0)
        print(UI.PickleOperations(TrainSampleOutputMeta,'w', Meta)[1])
        print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
        print(UI.TimeStamp(),bcolors.OKGREEN+'Stage 0 has successfully completed'+bcolors.ENDC)
elif os.path.isfile(TrainSampleOutputMeta)==True:
    print(UI.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+TrainSampleOutputMeta+bcolors.ENDC)
    MetaInput=UI.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]

ClassHeaders=Meta.ClassHeaders
ClassNames=Meta.ClassNames
ClassValues=Meta.ClassValues
JobSets=Meta.JobSets
MaxSegments=Meta.MaxSegments
TotJobs=JobSets

########################################     Preset framework parameters    #########################################
Program=[]


#The function bellow helps to automate the submission process
UI.Msg('vanilla','Analysing the current script status...')
Status=Meta.Status[-1]
if ForceStatus!='N':
    Status=int(ForceStatus)
UI.Msg('vanilla','Current stage is '+str(Status)+'...')
###### Stage 0
prog_entry=[]
prog_entry.append('Sending tracks to HTCondor for conversion int training/validation samples...')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/','IDseeds','MCTr1a','.pkl',TrainSampleID,JobSets,'MCTr1a_GenerateRawTrackSamples_Sub.py'])
prog_entry.append([ " --MaxSegments ", " --ClassNames "," --ClassValues "])
prog_entry.append([MaxSegments,'"'+str(ClassNames)+'"','"'+str(ClassValues)+'"'])
prog_entry.append(JobSets)
prog_entry.append(LocalSub)
prog_entry.append('N/A')
prog_entry.append(HTCondorLog)
prog_entry.append(False)
Program.append(prog_entry)
print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry))
###### Stage 1
Program.append('Custom')
###### Stage 2
Program.append('Custom')

print(UI.TimeStamp(),'There are ',len(Program),' stages of this script', bcolors.ENDC)
print(UI.TimeStamp(),'Current status has a stage',Status+1,bcolors.ENDC)

while Status<len(Program):
      if Program[Status]!='Custom':
        #Standard process here
        Result=UI.StandardProcess(Program,Status,SubGap,SubPause,RequestExtCPU,JobFlavour,ReqMemory,time_int,Patience)
        if Result[0]:
            UI.UpdateStatus(Status+1,Meta,TrainSampleOutputMeta)
        else:
            Status=20
            break
      if Program[Status]=='Custom':
          if Status==1:
            print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
            print(UI.TimeStamp(),bcolors.BOLD+'Stage 2:'+bcolors.ENDC+' Collecting and de-duplicating the results from stage 1')
            for i in range(JobSets):
                    req_file=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/Temp_MCTr1a_'+TrainSampleID+'_0/MCTr1a_'+TrainSampleID+'_IDseeds_'+str(i)+'.pkl'
                    output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/Temp_MCTr1a_'+TrainSampleID+'_0/MCTr1b_'+TrainSampleID+'_SelectedTrackSamples_'+str(i)+'.pkl'
                    base_data=UI.PickleOperations(req_file,'r', 'N/A')[0]
                    ExtractedData=[]
                    if Regression==False:
                        min_len=len([j for j in base_data if j.Label==0])
                        for j in range(len(ClassHeaders)+1):
                            if len([k for k in base_data if k.Label==j])!=0:
                               ExtractedData.append([k for k in base_data if k.Label==j])
                               min_len=min(len([k for k in base_data if k.Label==j]),min_len)
                        TotalData=[]
                        for s in range(len(ExtractedData)):
                            TotalData+=random.sample(ExtractedData[s],min_len)
                        print(UI.PickleOperations(output_file_location,'w', TotalData)[1])
                    else: print(UI.PickleOperations(output_file_location,'w', base_data)[1])
            print(UI.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
            UI.UpdateStatus(Status+1,Meta,TrainSampleOutputMeta)
            Status+=1
            continue

          if Status==2:
              print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
              print(UI.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' Taking the list of seeds previously generated by Stage 2, converting them into Emulsion Objects and doing more rigorous selection')
              TotalData=[]
              for i in range(JobSets):
                    req_file=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/Temp_MCTr1a_'+TrainSampleID+'_0/MCTr1b_'+TrainSampleID+'_SelectedTrackSamples_'+str(i)+'.pkl'
                    base_data=UI.PickleOperations(req_file,'r', 'N/A')[0]
                    TotalData+=base_data
              ValidationSampleSize=int(round(min((len(TotalData)*float(PM.valRatio)),PM.MaxValSampleSize),0))
              random.shuffle(TotalData)
              output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_OUTPUT.pkl'
              print(UI.PickleOperations(output_file_location,'w', TotalData[:ValidationSampleSize])[1])
              TotalData=TotalData[ValidationSampleSize:]
              No_Train_Files=int(math.ceil(len(TotalData)/TrainSampleSize))
              for i in range(0,No_Train_Files):
                  output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_OUTPUT_'+str(i+1)+'.pkl'
                  print(UI.PickleOperations(output_file_location,'w', TotalData[(i*TrainSampleSize):min(len(TotalData),((i+1)*TrainSampleSize))])[1])
              print(UI.TimeStamp(),bcolors.OKGREEN+'Stage 2 has successfully completed'+bcolors.ENDC)
              UI.UpdateStatus(Status+1,Meta,TrainSampleOutputMeta)
              Status+=1
              continue
      MetaInput=UI.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')
      Meta=MetaInput[0]
      Status=Meta.Status[-1]
if Status==3:
    #Removing the temp files that were generated by the process
    print(UI.TimeStamp(),'Performing the cleanup... ')
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, TrainSampleID,'d',['MCTr1a']))
    UI.Msg('success',"Train sample generation has been completed")
else:
      UI.Msg('failed',"Reconstruction has not been completed as one of the processes has timed out. Please run the script again (without Reset Mode).")
      exit()



