# Part of  ANNDEA  package.
# The purpose of this script to create samples for the tracking model training
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
import argparse
import pandas as pd #We use Panda for a routine data processing
import math #We use it for data manipulation
import os
import random
import time

# import sys
# sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
from Utilities import UtilityFunctions as UF #This is where we keep routine utility functions
from Utilities import Parameters as PM #This is where we keep framework global parameters
from Utilities import PrintingUtility 
from Utilities import PandasUtility
from Utilities import PickleUtility


class bcolors:   #We use it for the interface
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

PrintingUtility.print_head()

parser = argparse.ArgumentParser(description='This script prepares training data for training the tracking model')
parser.add_argument('--Mode', help='Script will continue from the last checkpoint, unless you want to start from the scratch, then type "Reset"',default='')
parser.add_argument('--TrainSampleID',help="Give name to this train sample batch", default='MH_SND_Raw_Train_Data_6_6_12_All')
parser.add_argument('--Sampling',help="If the data is large, sampling helps to keep it manageable. If this script fails consider reducing this parameter.", default='1.0')
parser.add_argument('--Patience',help="How many HTCondor checks to perform before resubmitting the job?", default='15')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/eos/user/f/ffedship/ANNDEA/Data/REC_SET/SND_Emulsion_FEDRA_Raw_B11_B21_B51_B12B52_B13B53_B14B54.csv')
parser.add_argument('--Xmin',help="This option restricts data to only those events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option restricts data to only those events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option restricts data to only those events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option restricts data to only those events that have tracks with hits y-coordinates that are below this value", default='0')
#The bellow are not important for the training smaples but if you want to augment the training data set above 1
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


stepX=PM.stepX #Size of the individual reconstruction volumes along the x-axis
stepY=PM.stepY #Size of the individual reconstruction volumes along the y-axis
stepZ=PM.stepZ #Size of the individual reconstruction volumes along the z-axis
cut_dt=PM.cut_dt #This cust help to discard hit pairs that are likely do not have a common mother track
cut_dr=PM.cut_dr
testRatio=PM.testRatio #Usually about 5%
valRatio=PM.valRatio #Usually about 10%

######## Utility Functions ###########################################
def generate_train_val_test_samples(TrainSampleID):
    TrainSampleInputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
    Meta= PickleUtility.load(TrainSampleInputMeta) #Reading the meta file
    TrainSamples=[]
    ValSamples=[]
    TestSamples=[]
    TrainFraction = 1.0-(Meta.testRatio+Meta.valRatio)
    ValFraction = Meta.valRatio
    for i in range(1,Meta.no_sets+1):
        flocation=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TTr_OUTPUT_'+str(i)+'.pkl' #Looking for constituent input file
        TrainClusters = PickleUtility.load(flocation) #Reading the train sample file
        dataSize = len(TrainClusters)    
        TrainSize = math.floor(dataSize*TrainFraction)  #Calculating the size of the training sample pool
        ValSize = math.ceil(dataSize*ValFraction) #Calculating the size of the validation sample pool
        for i in range(0, dataSize):
            # apply random sampling 
            if Sampling>=random.random():
                continue 
            sample = TrainClusters[i] 
            #discard graphs do not contain edges
            if sample.ClusterGraph.num_edges<=0 :
                continue
            # separate the samples into training set, validation set and test set 
            if i < TrainSize:
                TrainSamples.append(sample.ClusterGraph)
            else if i < TrainSize + ValSize:
                ValSamples.append(sample.ClusterGraph)
            else :
                TestSamples.append(sample.ClusterGraph)
    #Write the final output
    output_train_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_SAMPLES'+'.pkl'
    output_val_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_SAMPLES'+'.pkl'
    output_test_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TEST_SAMPLES'+'.pkl'
    PickleUtility.write(output_train_file_location,TrainSamples)
    PickleUtility.write(output_val_file_location,ValSamples)
    PickleUtility.write(output_test_file_location,TestSamples)
    print(UF.TimeStamp(), bcolors.OKGREEN+"Train data has been re-generated successfully..."+bcolors.ENDC)
######## End of Utility Functions ###########################################


TrainSampleOutputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl' #For each training sample batch we create an individual meta file.
destination_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TTr_OUTPUT_1.pkl' #The desired output
if os.path.isfile(destination_output_file_location) and Mode!='RESET': #If we have it, we don't have to start from the scratch.
    generate_train_val_test_samples(TrainSampleID)
    exit()


########################################     Phase 1 - Create compact source file    #########################################
print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(),bcolors.BOLD+'Stage 0:'+bcolors.ENDC+' Taking the file that has been supplied and creating the compact copies for the training set generation...')
output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MTr1_'+TrainSampleID+'_hits.csv' #This is the compact data file that contains only relevant columns and rows
if os.path.isfile(output_file_location)==False or Mode=='RESET':
    data = PandasUtility.load_data(input_file_location)
    usecols=[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
    data = PandasUtility.select_columns(data, usecols)
    data = PandasUtility.remove_unreconstructed_hits(data)
    data = PandasUtility.rename_hit_columns(data)
    data = PandasUtility.convert_ID_to_string(data)
    if SliceData:
        data = PandasUtility.slice_hit_data(data, Xmin, Xmax, Ymin, Ymax)
    PrintingUtility.print_message("Saving segment data...")
    PandasUtility.save_data(data, output_file_location)


###################### Phase 2 - Eval Data ######################################################
#This is similar to one above but also contains MC data
output_file_location=EOS_DIR+'/ANNADEA/Data/TRAIN_SET/ETr1_'+TrainSampleID+'_hits.csv' 
if os.path.isfile(output_file_location)==False or Mode=='RESET':
    print(UF.TimeStamp(),'Creating Evaluation file...')
    data = PandasUtility.load_data(input_file_location)
    usecols=[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.MC_Track_ID,PM.MC_Event_ID]
    data = PandasUtility.select_columns(data, usecols)
    data = PandasUtility.remove_unreconstructed_hits(data)
    data = PandasUtility.rename_hit_columns(data)
    data = PandasUtility.convert_ID_to_string(data)
    if SliceData:
        data = PandasUtility.slice_hit_data(data, Xmin, Xmax, Ymin, Ymax)
    #Even if particle leaves one hit it is still assigned MC Track ID - we cannot reconstruct these so we discard them so performance metrics are not skewed
    #We are only interested in MC Tracks that have a certain number of hits
    data = PandasUtility.remove_ill_mc_tracks(data)  
    PandasUtility.save_data(data, output_file_location)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 0 has successfully completed'+bcolors.ENDC)

########################################     Preset framework parameters    #########################################

print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(),bcolors.BOLD+'Stage 1:'+bcolors.ENDC+' Creating training sample meta data...')

#A case of generating samples from scratch
if os.path.isfile(TrainSampleOutputMeta)==False or Mode=='RESET': 
    input_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MTr1_'+TrainSampleID+'_hits.csv'
    data = PandasUtility.load_data(input_file_location)
    usecols=['z','x','y']
    data = PandasUtility.select_columns(data, usecols)
    data = PandasUtility.reset_origin(data)
    
    #We need it to calculate how many clusters to create
    x_max=data['x'].max()
    z_max=data['z'].max()
    Xsteps=(math.ceil((x_max)/stepX)*(X_overlap))
    Zsteps=(math.ceil((z_max)/stepZ)*(Z_overlap))
    # don't know why but keep the convention
    if X_overlap > 1:
       Xsteps -= 1 
    if Z_overlap > 1:
        Zsteps -= 1
    TrainDataMeta=UF.TrainingSampleMeta(TrainSampleID)
    TrainDataMeta.IniHitClusterMetaData(
        stepX,stepY,stepZ,
        cut_dt,cut_dr,
        testRatio,valRatio,
        z_offset,y_offset,x_offset,
        Xsteps, Zsteps,
        X_overlap, Y_overlap, Z_overlap
    )
    PickleUtility.write(TrainSampleOutputMeta,TrainDataMeta)


elif os.path.isfile(TrainSampleOutputMeta)==True:
    print(UF.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+TrainSampleOutputMeta+bcolors.ENDC)
    #Loading parameters from the Meta file if exists.
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

if Mode=='RESET':
   HTCondorTag="SoftUsed == \"ANNDEA-MTr1-"+TrainSampleID+"\""
   UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MTr1_'+TrainSampleID, ['MTr1a_'+TrainSampleID,TrainSampleID+'_TTr_OUTPUT_'], HTCondorTag) #If we do total reset we want to delete all temp files left by a possible previous attempt
   print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
bad_pop=[]
Status=False

print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(),bcolors.BOLD+'Stage 2:'+bcolors.ENDC+' Checking HTCondor jobs and preparing the job submission')


############## Belows are some utility functions #########################################

# Check condor jobs that are unfinished 
def check_unfinished_jobs(Zsteps,Xsteps,bad_pop):
    bad_pop=[]
    print(UF.TimeStamp(),"Scheduled job checkup...") 
    for k in range(0,Zsteps):
        #Progress display
        progress=round((float(k)/float(Zsteps))*100,2)
        print(UF.TimeStamp(),"progress is ",progress,' %') 
        for i in range(0,Xsteps):
            #Preparing HTCondor submission
            OptionHeader = [' --Z_ID ', ' --stepX ',' --stepY ',' --stepZ ', ' --EOS ', " --AFS ", " --zOffset ", " --xOffset ", " --yOffset ", ' --cut_dt ', ' --cut_dr ', ' --testRatio ', ' --valRatio ', ' --X_ID ',' --TrainSampleID ',' --Y_overlap ',' --X_overlap ',' --Z_overlap ']
            OptionLine = [k, stepX,stepY,stepZ, EOS_DIR, AFS_DIR, z_offset, x_offset, y_offset, cut_dt,cut_dr,testRatio,valRatio, i,TrainSampleID,Y_overlap,X_overlap,Z_overlap]
            required_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MTr1a_'+TrainSampleID+'_SelectedTrainClusters_'+str(k)+'_'+str(i)+'.pkl'
            SHName = AFS_DIR + '/HTCondor/SH/SH_MTr1_'+ TrainSampleID+'_' + str(k) + '_' + str(i) + '.sh'
            SUBName = AFS_DIR + '/HTCondor/SUB/SUB_MTr1_'+ TrainSampleID+'_'+ str(k) + '_' + str(i) + '.sub'
            MSGName = AFS_DIR + '/HTCondor/MSG/MSG_MTr1_' + TrainSampleID+'_' + str(k) + '_' + str(i)
            #The actual training sample generation is done by the script bellow
            ScriptName = AFS_DIR + '/Code/Utilities/MTr1_GenerateTrainClusters_Sub.py '
            if os.path.isfile(required_output_file_location)!=True: #Calculating the number of unfinished jobs
                bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-MTr1-'+TrainSampleID, False,False])
    return bad_pop

#The function bellow manages HTCondor jobs - so we don't have to do manually
def submit_job_to_condor_auto(wait_min, interval_min, max_interval_tolerance):
    print(
        UF.TimeStamp(),
        'Going on an autopilot mode for ', wait_min, 'min',
        'minutes while checking HTCondor every', interval_min, 'min',
        bcolors.ENDC
    )
    #Converting min to sec as this time.sleep takes as an argument
    interval_sec=interval_min*60
    nIntervals=int(math.ceil(wait_min/interval_min))
    for i in range(0,nIntervals):
        bad_pop=check_unfinished_jobs(Zsteps,Xsteps,bad_pop)
        if len(bad_pop) ==0:
            return True
        # else case: there are still jobs in condor
        print(
            UF.TimeStamp(), bcolors.WARNING+
            'Autopilot status update: There are still', len(bad_pop), 'HTCondor jobs remaining'
            +bcolors.ENDC
        )
        #If jobs are not received after fixed number of check-ups we resubmit. 
        #Jobs sometimes fail on HTCondor for various reasons.
        if i+1 % max_interval_tolerance == 0: 
            for bp in bad_pop:
                UF.SubmitJobs2Condor(bp) #Sumbitting all missing jobs
        time.sleep(interval_sec)
    return False

# Let users to choose whether resubmit jobs when condor jobs are unfinished
def deal_remain_jobs(bad_pop):
    print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
    print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
    print(bcolors.BOLD+'If you would like to wait please enter enter the maximum wait time in minutes'+bcolors.ENDC)
    print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
    UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
    if UserAnswer=='E':
        print(UF.TimeStamp(),'OK, exiting now then')
        exit()
    # parameters for auto submission to condor
    wait_min = 120 
    interval_min = 10
    max_interval_tolerance = Patience
    if UserAnswer=='R':
        #Scenario where all jobs are missing - doing a batch submission to save time
        if len(bad_pop) == (Zsteps*Xsteps): 
            for k in range(0,Zsteps):
                OptionHeader = [' --Z_ID ', ' --stepX ',' --stepY ',' --stepZ ', ' --EOS ', " --AFS ", " --zOffset ", " --xOffset ", " --yOffset ", ' --cut_dt ', ' --cut_dr ', ' --testRatio ', ' --valRatio ', ' --X_ID ',' --TrainSampleID ',' --Y_overlap ',' --X_overlap ',' --Z_overlap ']
                OptionLine = [k, stepX,stepY,stepZ, EOS_DIR, AFS_DIR, z_offset, x_offset, y_offset, cut_dt,cut_dr,testRatio, valRatio,'$1',TrainSampleID,Y_overlap,X_overlap,Z_overlap]
                SHName = AFS_DIR + '/HTCondor/SH/SH_MTr1_'+ TrainSampleID+'_' + str(k) + '.sh'
                SUBName = AFS_DIR + '/HTCondor/SUB/SUB_MTr1_'+ TrainSampleID+'_'+ str(k) + '.sub'
                MSGName = AFS_DIR + '/HTCondor/MSG/MSG_MTr1_' + TrainSampleID+'_' + str(k)
                ScriptName = AFS_DIR + '/Code/Utilities/MTr1_GenerateTrainClusters_Sub.py '
                UF.SubmitJobs2Condor([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, Xsteps, 'ANNDEA-MTr1-'+TrainSampleID, False,False])         
        else:
            for bp in bad_pop:
                UF.SubmitJobs2Condor(bp)
        print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
    else:
        wait_min = UserAnswer
    
    finished = submit_job_to_condor_auto(wait_min,interval_min,max_interval_tolerance)
    return finished
####################### End of utility functions #############################################
####################### Below are main logic again ########################################### 

bad_pop = check_unfinished_jobs(Zsteps,Xsteps,bad_pop)
# all jobs are finished if no item in bad_pop
job_finished = (len(bad_pop)==0)
if job_finished == False: 
    job_finished = deal_remain_jobs(bad_pop)
# if jobs still fail after auto submit
if job_finished == False:
    print(
        UF.TimeStamp(),bcolors.FAIL+
        'Unfortunately no results have been yield.'+
        'Please rerun the script, and if the problem persists, check that the HTCondor jobs run adequately.'+
        bcolors.ENDC
    )
    exit()

# After all jobs are finished
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
UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MTr1_'+TrainSampleID, ['MTr1a_'+TrainSampleID,'ETr1_'+TrainSampleID,'MTr1_'+TrainSampleID], HTCondorTag) #If successful we delete all temp files created by the process
print(UF.TimeStamp(),bcolors.OKGREEN+'Training samples are ready for the model creation/training'+bcolors.ENDC)



TrainSampleInputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
Meta= PickleUtility.load(TrainSampleInputMeta) #Reading the meta file
TrainSamples=[]
ValSamples=[]
TestSamples=[]
TrainFraction = 1.0-(Meta.testRatio+Meta.valRatio)
ValFraction = dataSize*Meta.valRatio
for i in range(1,Meta.no_sets+1):
    flocation=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TTr_OUTPUT_'+str(i)+'.pkl' #Looking for constituent input file
    TrainClusters = PickleUtility.load(flocation) #Reading the train sample file
    dataSize = len(TrainClusters)    
    TrainSize = math.floor(dataSize*TrainFraction)  #Calculating the size of the training sample pool
    ValSize = math.ceil(dataSize*ValFraction) #Calculating the size of the validation sample pool
    for i in range(0, dataSize):
        # apply random sampling 
        if Sampling>=random.random():
            continue 
        sample = TrainClusters[i] 
        #discard graphs do not contain edges
        if sample.ClusterGraph.num_edges<=0 :
            continue 
        if i < TrainSize:
            TrainSamples.append(sample.ClusterGraph)
        else if i < TrainSize + ValSize:
            ValSamples.append(sample.ClusterGraph)
        else :
            TestSamples.append(sample.ClusterGraph)

#Write the final output
output_train_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_SAMPLES'+'.pkl'
output_val_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_SAMPLES'+'.pkl'
output_test_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TEST_SAMPLES'+'.pkl'
PickleUtility.write(output_train_file_location,TrainSamples)
PickleUtility.write(output_val_file_location,ValSamples)
PickleUtility.write(output_test_file_location,TestSamples)

print(UF.TimeStamp(), bcolors.OKGREEN+"Train data has been re-generated successfully..."+bcolors.ENDC)
print(UF.TimeStamp(),bcolors.OKGREEN+'Please run MTr2_TrainModel.py after this to create/train a model'+bcolors.ENDC)
print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)










