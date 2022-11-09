#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
import argparse
import pandas as pd #We use Panda for a routine data processing
import math #We use it for data manipulation
import os
import random
import time

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
print(bcolors.HEADER+"######################     Initialising ANNDEA Train Cluster Generation module     #####################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script prepares training data for training the tracking model')
parser.add_argument('--Mode', help='Script will continue from the last checkpoint, unless you want to start from the scratch, then type "Reset"',default='')
parser.add_argument('--TrainSampleID',help="Give name to this train sample", default='SHIP_TrainSample_v1')
parser.add_argument('--Sampling',help="How much sampling?", default='1.0')
parser.add_argument('--Patience',help="How many checks before resubmitting the job?", default='15')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/eos/user/f/ffedship/ANNDEA/Data/REC_SET/SND_Emulsion_FEDRA_Raw_B11_B21_B51_B12B52_B13B53_B14B54.csv')
parser.add_argument('--Xmin',help="This option restricts data to only those events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option restricts data to only those events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option restricts data to only those events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option restricts data to only those events that have tracks with hits y-coordinates that are below this value", default='0')
parser.add_argument('--Z_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along z-axis.", default='1')
parser.add_argument('--Y_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along y-axis.", default='1')
parser.add_argument('--X_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along x-axis.", default='1')
######################################## Set variables  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
Sampling=float(args.Sampling)
TrainSampleID=args.TrainSampleID
Patience=int(args.Patience)
input_file_location=args.f
Xmin,Xmax,Ymin,Ymax=float(args.Xmin),float(args.Xmax),float(args.Ymin),float(args.Ymax)
Z_overlap,Y_overlap,X_overlap=int(args.Z_overlap),int(args.Y_overlap),int(args.X_overlap)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0
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
stepX=PM.stepX
stepY=PM.stepY
stepZ=PM.stepZ
cut_dt=PM.cut_dt
cut_dr=PM.cut_dr
testRatio=PM.testRatio
valRatio=PM.valRatio
TrainSampleOutputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
destination_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TTr_OUTPUT_1.pkl'
if os.path.isfile(destination_output_file_location) and Mode!='RESET':
    TrainSampleInputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
    print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+TrainSampleInputMeta+bcolors.ENDC)
    MetaInput=UF.PickleOperations(TrainSampleInputMeta,'r', 'N/A')
    print(MetaInput[1])
    Meta=MetaInput[0]
    TrainSamples=[]
    ValSamples=[]
    TestSamples=[]
    for i in range(1,Meta.no_sets+1):
        flocation=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TTr_OUTPUT_'+str(i)+'.pkl'
        print(UF.TimeStamp(),'Loading data from ',bcolors.OKBLUE+flocation+bcolors.ENDC)
        TrainClusters=UF.PickleOperations(flocation,'r', 'N/A')
        TrainClusters=TrainClusters[0]
        TrainFraction=int(math.floor(len(TrainClusters)*(1.0-(Meta.testRatio+Meta.valRatio))))
        ValFraction=int(math.ceil(len(TrainClusters)*Meta.valRatio))
        for smpl in range(0,TrainFraction):
                   if TrainClusters[smpl].ClusterGraph.num_edges>0 and Sampling>=random.random():
                     TrainSamples.append(TrainClusters[smpl].ClusterGraph)
        for smpl in range(TrainFraction,TrainFraction+ValFraction):
                   if TrainClusters[smpl].ClusterGraph.num_edges>0 and Sampling>=random.random():
                     ValSamples.append(TrainClusters[smpl].ClusterGraph)
        for smpl in range(TrainFraction+ValFraction,len(TrainClusters)):
                   if TrainClusters[smpl].ClusterGraph.num_edges>0 and Sampling>=random.random():
                     TestSamples.append(TrainClusters[smpl].ClusterGraph)
    output_train_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_SAMPLES'+'.pkl'
    print(UF.PickleOperations(output_train_file_location,'w', TrainSamples))[1]
    output_val_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_SAMPLES'+'.pkl'
    print(UF.PickleOperations(output_val_file_location,'w', ValSamples))[1]
    output_test_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TEST_SAMPLES'+'.pkl'
    print(UF.PickleOperations(output_test_file_location,'w', ValSamples))[1]
    print(UF.TimeStamp(), bcolors.OKGREEN+"Train data has been re-generated successfully..."+bcolors.ENDC)
    exit()


########################################     Phase 1 - Create compact source file    #########################################
print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(),bcolors.BOLD+'Stage 0:'+bcolors.ENDC+' Taking the file that has been supplied and creating the compact copies for the training set generation...')
output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MTr1_'+TrainSampleID+'_hits.csv'
if os.path.isfile(output_file_location)==False or Mode=='RESET':
        print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
        data=pd.read_csv(input_file_location,
                    header=0,
                    usecols=[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty])[[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty]]
        total_rows=len(data.axes[0])
        data[PM.Hit_ID] = data[PM.Hit_ID].astype(str)
        print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
        print(UF.TimeStamp(),'Removing unreconstructed hits...')
        data=data.dropna()
        final_rows=len(data.axes[0])
        print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
        data[PM.Hit_ID] = data[PM.Hit_ID].astype(int)
        data[PM.Hit_ID] = data[PM.Hit_ID].astype(str)
        if SliceData:
             print(UF.TimeStamp(),'Slicing the data...')
             data=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
             final_rows=len(data.axes[0])
             print(UF.TimeStamp(),'The sliced data has ',final_rows,' hits')
        data=data.rename(columns={PM.x: "x"})
        data=data.rename(columns={PM.y: "y"})
        data=data.rename(columns={PM.z: "z"})
        data=data.rename(columns={PM.tx: "tx"})
        data=data.rename(columns={PM.ty: "ty"})
        data=data.rename(columns={PM.Hit_ID: "Hit_ID"})
        data.to_csv(output_file_location,index=False)
        print(UF.TimeStamp(), bcolors.OKGREEN+"The segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)


###################### Phase 2 - Eval Data ######################################################
output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/ETr1_'+TrainSampleID+'_hits.csv'
if os.path.isfile(output_file_location)==False or Mode=='RESET':
    print(UF.TimeStamp(),'Creating Evaluation file...')
    print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
    data=pd.read_csv(input_file_location,
                header=0,
                usecols=[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.MC_Track_ID,PM.MC_Event_ID])[[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.MC_Track_ID,PM.MC_Event_ID]]

    total_rows=len(data.axes[0])
    print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
    print(UF.TimeStamp(),'Removing unreconstructed hits...')
    data=data.dropna()
    final_rows=len(data.axes[0])
    print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
    data[PM.MC_Event_ID] = data[PM.MC_Event_ID].astype(str)
    data[PM.MC_Track_ID] = data[PM.MC_Track_ID].astype(str)
    data[PM.Hit_ID] = data[PM.Hit_ID].astype(str)
    data['MC_Mother_Track_ID'] = data[PM.MC_Event_ID] + '-' + data[PM.MC_Track_ID]
    data=data.drop([PM.MC_Event_ID],axis=1)
    data=data.drop([PM.MC_Track_ID],axis=1)
    if SliceData:
         print(UF.TimeStamp(),'Slicing the data...')
         data=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
         final_rows=len(data.axes[0])
         print(UF.TimeStamp(),'The sliced data has ',final_rows,' hits')
    print(UF.TimeStamp(),'Removing tracks which have less than',PM.MinHitsTrack,'hits...')
    track_no_data=data.groupby(['MC_Mother_Track_ID'],as_index=False).count()
    track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty,PM.Hit_ID],axis=1)
    track_no_data=track_no_data.rename(columns={PM.x: "MC_Track_No"})
    new_combined_data=pd.merge(data, track_no_data, how="left", on=['MC_Mother_Track_ID'])
    new_combined_data = new_combined_data[new_combined_data.MC_Track_No >= PM.MinHitsTrack]
    new_combined_data = new_combined_data.drop(["MC_Track_No"],axis=1)
    new_combined_data=new_combined_data.sort_values(['MC_Mother_Track_ID',PM.z],ascending=[1,1])
    grand_final_rows=len(new_combined_data.axes[0])
    print(UF.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
    new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
    new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
    new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
    new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
    new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
    new_combined_data=new_combined_data.rename(columns={PM.Hit_ID: "Hit_ID"})
    new_combined_data.to_csv(output_file_location,index=False)
    print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
    print(UF.TimeStamp(), bcolors.OKGREEN+"The track segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 0 has successfully completed'+bcolors.ENDC)
########################################     Preset framework parameters    #########################################

print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(),bcolors.BOLD+'Stage 1:'+bcolors.ENDC+' Creating training sample meta data...')
if os.path.isfile(TrainSampleOutputMeta)==False or Mode=='RESET':
    input_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MTr1_'+TrainSampleID+'_hits.csv'
    print(UF.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
    data=pd.read_csv(input_file_location,header=0,usecols=['z','x','y'])
    print(UF.TimeStamp(),'Analysing data... ',bcolors.ENDC)
    z_offset=data['z'].min()
    data['z']=data['z']-z_offset
    z_max=data['z'].max()
    y_offset=data['y'].min()
    x_offset=data['x'].min()
    data['x']=data['x']-x_offset
    x_max=data['x'].max()
    if X_overlap==1:
       Xsteps=math.ceil((x_max)/stepX)
    else:
       Xsteps=(math.ceil((x_max)/stepX)*(X_overlap))-1

    if Z_overlap==1:
       Zsteps=math.ceil((z_max)/stepZ)
    else:
       Zsteps=(math.ceil((z_max)/stepZ)*(Z_overlap))-1
    TrainDataMeta=UF.TrainingSampleMeta(TrainSampleID)
    TrainDataMeta.IniHitClusterMetaData(stepX,stepY,stepZ,cut_dt,cut_dr,testRatio,valRatio,z_offset,y_offset,x_offset, Xsteps, Zsteps,X_overlap, Y_overlap, Z_overlap)
    print(UF.PickleOperations(TrainSampleOutputMeta,'w', TrainDataMeta)[1])

elif os.path.isfile(TrainSampleOutputMeta)==True:
    print(UF.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+TrainSampleOutputMeta+bcolors.ENDC)
    MetaInput=UF.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
    stepX=Meta.stepX
    stepY=Meta.stepY
    stepZ=Meta.stepZ
    cut_dt=Meta.cut_dt
    cut_dr=Meta.cut_dr
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
print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 1 has successfully completed'+bcolors.ENDC)
def AutoPilot(wait_min, interval_min, max_interval_tolerance):
    print(UF.TimeStamp(),'Going on an autopilot mode for ',wait_min, 'minutes while checking HTCondor every',interval_min,'min',bcolors.ENDC)
    wait_sec=wait_min*60
    interval_sec=interval_min*60
    intervals=int(math.ceil(wait_sec/interval_sec))
    for interval in range(1,intervals+1):
        time.sleep(interval_sec)
        bad_pop=[]
        print(UF.TimeStamp(),"Scheduled job checkup...") #Progress display
        for k in range(0,Zsteps):
         for i in range(0,Xsteps):
              OptionHeader = [' --Z_ID ', ' --stepX ',' --stepY ',' --stepZ ', ' --EOS ', " --AFS ", " --zOffset ", " --xOffset ", " --yOffset ", ' --cut_dt ', ' --cut_dr ', ' --testRatio ', ' --valRatio ', ' --X_ID ',' --TrainSampleID ',' --Y_overlap ',' --X_overlap ',' --Z_overlap ']
              OptionLine = [k, stepX,stepY,stepZ, EOS_DIR, AFS_DIR, z_offset, x_offset, y_offset, cut_dt,cut_dr,testRatio,valRatio, i,TrainSampleID,Y_overlap,X_overlap,Z_overlap]
              required_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MTr1a_'+TrainSampleID+'_SelectedTrainClusters_'+str(k)+'_'+str(i)+'.pkl'
              SHName = AFS_DIR + '/HTCondor/SH/SH_MTr1_'+ TrainSampleID+'_' + str(k) + '_' + str(i) + '.sh'
              SUBName = AFS_DIR + '/HTCondor/SUB/SUB_MTr1_'+ TrainSampleID+'_'+ str(k) + '_' + str(i) + '.sub'
              MSGName = AFS_DIR + '/HTCondor/MSG/MSG_MTr1_' + TrainSampleID+'_' + str(k) + '_' + str(i)
              ScriptName = AFS_DIR + '/Code/Utilities/MTr1_GenerateTrainClusters_Sub.py '
              if os.path.isfile(required_output_file_location)!=True:
                 bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-MTr1-'+TrainSampleID, False,False])
        if len(bad_pop)>0:
              print(UF.TimeStamp(),bcolors.WARNING+'Autopilot status update: There are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
              if interval%max_interval_tolerance==0:
                 for bp in bad_pop:
                     UF.SubmitJobs2Condor(bp)
                 print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
        else:
             return True
    return False

def Success(Finished):
        if Finished:
            print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
            print(UF.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' All HTCondor jobs have completed.')
            count=0
            SampleCount=0
            NodeFeatures=PM.num_node_features
            EdgeFeatures=PM.num_edge_features
            for k in range(0,Zsteps):
                progress=round((float(k)/float(Zsteps))*100,2)
                print(UF.TimeStamp(),"Collating results, progress is ",progress,' %') #Progress display
                for i in range(0,Xsteps):
                        count+=1
                        source_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MTr1a_'+TrainSampleID+'_SelectedTrainClusters_'+str(k)+'_'+str(i)+'.pkl'
                        destination_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TTr_OUTPUT_'+str(count)+'.pkl'
                        os.rename(source_output_file_location, destination_output_file_location)
                        TrainingSample=UF.PickleOperations(destination_output_file_location,'r', 'N/A')[0]
                        SampleCount+=len(TrainingSample)
            print(UF.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+TrainSampleOutputMeta+bcolors.ENDC)
            MetaInput=UF.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')
            MetaInput[0].UpdateHitClusterMetaData(SampleCount,NodeFeatures,EdgeFeatures,count)
            print(UF.PickleOperations(TrainSampleOutputMeta,'w', MetaInput[0])[1])
            HTCondorTag="SoftUsed == \"ANNDEA-MTr-"+TrainSampleID+"\""
            UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MTr1_'+TrainSampleID, ['MTr1a_'+TrainSampleID,'ETr1_'+TrainSampleID,'MTr1_'+TrainSampleID], HTCondorTag)
            print(UF.TimeStamp(),bcolors.OKGREEN+'Training samples are ready for the model creation/training'+bcolors.ENDC)
            TrainSampleInputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
            print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+TrainSampleInputMeta+bcolors.ENDC)
            MetaInput=UF.PickleOperations(TrainSampleInputMeta,'r', 'N/A')
            print(MetaInput[1])
            Meta=MetaInput[0]
            TrainSamples=[]
            ValSamples=[]
            TestSamples=[]
#            for i in range(1,Meta.no_sets+1):
            for i in range(1,10):
                flocation=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TTr_OUTPUT_'+str(i)+'.pkl'
                print(UF.TimeStamp(),'Loading data from ',bcolors.OKBLUE+flocation+bcolors.ENDC)
                TrainClusters=UF.PickleOperations(flocation,'r', 'N/A')
                TrainClusters=TrainClusters[0]
                TrainFraction=int(math.floor(len(TrainClusters)*(1.0-(Meta.testRatio+Meta.valRatio))))
                ValFraction=int(math.ceil(len(TrainClusters)*Meta.valRatio))
                for smpl in range(0,TrainFraction):
                           if TrainClusters[smpl].ClusterGraph.num_edges>0 and Sampling>=random.random():
                             TrainSamples.append(TrainClusters[smpl].ClusterGraph)
                for smpl in range(TrainFraction,TrainFraction+ValFraction):
                           if TrainClusters[smpl].ClusterGraph.num_edges>0 and Sampling>=random.random():
                             ValSamples.append(TrainClusters[smpl].ClusterGraph)
                for smpl in range(TrainFraction+ValFraction,len(TrainClusters)):
                           if TrainClusters[smpl].ClusterGraph.num_edges>0 and Sampling>=random.random():
                             TestSamples.append(TrainClusters[smpl].ClusterGraph)
            output_train_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_SAMPLES'+'.pkl'
            print(UF.PickleOperations(output_train_file_location,'w', TrainSamples)[1])
            output_val_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_SAMPLES'+'.pkl'
            print(UF.PickleOperations(output_val_file_location,'w', ValSamples)[1])
            output_test_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TEST_SAMPLES'+'.pkl'
            print(UF.PickleOperations(output_test_file_location,'w', ValSamples)[1])
            print(UF.TimeStamp(), bcolors.OKGREEN+"Train data has been re-generated successfully..."+bcolors.ENDC)
            print(UF.TimeStamp(),bcolors.OKGREEN+'Please run MTr2_TrainModel.py after this to create/train a model'+bcolors.ENDC)
            print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
        else:
            print(UF.TimeStamp(),bcolors.FAIL+'Unfortunately no results have been yield. Please rerun the script, and if the problem persists, check that the HTCondor jobs run adequately.'+bcolors.ENDC)
            exit()

if Mode=='RESET':
   HTCondorTag="SoftUsed == \"ANNDEA-MTr1-"+TrainSampleID+"\""
   UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MTr1_'+TrainSampleID, ['MTr1a_'+TrainSampleID,TrainSampleID+'_TTr_OUTPUT_'], HTCondorTag)
   print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
bad_pop=[]
Status=False
print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(),bcolors.BOLD+'Stage 2:'+bcolors.ENDC+' Checking HTCondor jobs and preparing the job submission')


for k in range(0,Zsteps):
        progress=round((float(k)/float(Zsteps))*100,2)
        print(UF.TimeStamp(),"progress is ",progress,' %') #Progress display
        for i in range(0,Xsteps):
             OptionHeader = [' --Z_ID ', ' --stepX ',' --stepY ',' --stepZ ', ' --EOS ', " --AFS ", " --zOffset ", " --xOffset ", " --yOffset ", ' --cut_dt ', ' --cut_dr ', ' --testRatio ', ' --valRatio ', ' --X_ID ',' --TrainSampleID ',' --Y_overlap ',' --X_overlap ',' --Z_overlap ']
             OptionLine = [k, stepX,stepY,stepZ, EOS_DIR, AFS_DIR, z_offset, x_offset, y_offset, cut_dt,cut_dr,testRatio,valRatio, i,TrainSampleID,Y_overlap,X_overlap,Z_overlap]
             required_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MTr1a_'+TrainSampleID+'_SelectedTrainClusters_'+str(k)+'_'+str(i)+'.pkl'
             SHName = AFS_DIR + '/HTCondor/SH/SH_MTr1_'+ TrainSampleID+'_' + str(k) + '_' + str(i) + '.sh'
             SUBName = AFS_DIR + '/HTCondor/SUB/SUB_MTr1_'+ TrainSampleID+'_'+ str(k) + '_' + str(i) + '.sub'
             MSGName = AFS_DIR + '/HTCondor/MSG/MSG_MTr1_' + TrainSampleID+'_' + str(k) + '_' + str(i)
             ScriptName = AFS_DIR + '/Code/Utilities/MTr1_GenerateTrainClusters_Sub.py '
             if os.path.isfile(required_output_file_location)!=True:
                bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-MTr1-'+TrainSampleID, False,False])
if len(bad_pop)==0:
    Success(True)


if (Zsteps*Xsteps)==len(bad_pop):
    print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
    print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
    print(bcolors.BOLD+'If you would like to wait please enter enter the maximum wait time in minutes'+bcolors.ENDC)
    print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
    UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
    if UserAnswer=='E':
          print(UF.TimeStamp(),'OK, exiting now then')
          exit()
    if UserAnswer=='R':
        print(UF.TimeStamp(),'Submitting jobs to HTCondor... ',bcolors.ENDC)
        for k in range(0,Zsteps):
                  OptionHeader = [' --Z_ID ', ' --stepX ',' --stepY ',' --stepZ ', ' --EOS ', " --AFS ", " --zOffset ", " --xOffset ", " --yOffset ", ' --cut_dt ', ' --cut_dr ', ' --testRatio ', ' --valRatio ', ' --X_ID ',' --TrainSampleID ',' --Y_overlap ',' --X_overlap ',' --Z_overlap ']
                  OptionLine = [k, stepX,stepY,stepZ, EOS_DIR, AFS_DIR, z_offset, x_offset, y_offset, cut_dt,cut_dr,testRatio, valRatio,'$1',TrainSampleID,Y_overlap,X_overlap,Z_overlap]
                  SHName = AFS_DIR + '/HTCondor/SH/SH_MTr1_'+ TrainSampleID+'_' + str(k) + '.sh'
                  SUBName = AFS_DIR + '/HTCondor/SUB/SUB_MTr1_'+ TrainSampleID+'_'+ str(k) + '.sub'
                  MSGName = AFS_DIR + '/HTCondor/MSG/MSG_MTr1_' + TrainSampleID+'_' + str(k)
                  ScriptName = AFS_DIR + '/Code/Utilities/MTr1_GenerateTrainClusters_Sub.py '
                  UF.SubmitJobs2Condor([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, Xsteps, 'ANNDEA-MTr1-'+TrainSampleID, False,False])
                  print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
        Success(AutoPilot(120,10,Patience))
    else:
        Success(AutoPilot(120,10,Patience))

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
         Success(AutoPilot(120,10,Patience))
      else:
         Success(AutoPilot(int(UserAnswer),10,Patience))




