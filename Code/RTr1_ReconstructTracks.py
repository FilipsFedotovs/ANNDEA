#This script connects hits in the data to produce tracks
#Tracking Module of the ANNDEA package
#Made by Filips Fedotovs

########################################    Import libraries    #############################################
import csv
import ast
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
import pandas as pd #We use Panda for a routine data processing
pd.options.mode.chained_assignment = None #Silence annoying warnings
import math #We use it for data manipulation
import numpy as np
import os
import time
from alive_progress import alive_bar
import argparse
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
print(bcolors.HEADER+"######################        Initialising ANNDEA Hit Tracking module              #####################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script prepares training data for training the tracking model')
parser.add_argument('--Mode', help='Script will continue from the last checkpoint, unless you want to start from the scratch, then type "Reset"',default='')
parser.add_argument('--ModelName',help="WHat GNN model would you like to use?", default='MH_SND_Tracking_5_80_5_80')
parser.add_argument('--Patience',help="How many checks to do before resubmitting the job?", default='30')
parser.add_argument('--SubPause',help="How long to wait in minutes after submitting 10000 jobs?", default='60')
parser.add_argument('--SubGap',help="How long to wait in minutes after submitting 10000 jobs?", default='10000')
parser.add_argument('--RecBatchID',help="Give this reconstruction batch an ID", default='Test_Batch')
parser.add_argument('--LocalSub',help="Local submission?", default='N')
parser.add_argument('--TrackFitCut',help="Track Fit cut Residual", default="['1000','10','200']")
parser.add_argument('--ForceStatus',help="Would you like the program run from specific status number? (Only for advance users)", default='0')
parser.add_argument('--RequestExtCPU',help="Would you like to request extra CPUs?", default=1)
parser.add_argument('--JobFlavour',help="Specifying the length of the HTCondor job walltime. Currently at 'workday' which is 8 hours.", default='workday')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/eos/experiment/ship/ANNDEA/Data/SND_Emulsion_FEDRA_Raw_B31.csv')
parser.add_argument('--Xmin',help="This option restricts data to only those events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option restricts data to only those events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option restricts data to only those events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option restricts data to only those events that have tracks with hits y-coordinates that are below this value", default='0')
parser.add_argument('--Z_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along z-axis. (In order to avoid segmentation this value should be more than 1)", default='3')
parser.add_argument('--Y_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along y-axis. (In order to avoid segmentation this value should be more than 1)", default='2')
parser.add_argument('--X_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along x-axis. (In order to avoid segmentation this value should be more than 1)", default='2')
parser.add_argument('--CheckPoint',help="Save cluster sets during individual cluster tracking.", default='N')
parser.add_argument('--ReqMemory',help="How much memory?", default='2 GB')

######################################## Parsing argument values  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
ModelName=args.ModelName
RecBatchID=args.RecBatchID
Patience=int(args.Patience)
SubPause=int(args.SubPause)*60
TrackFitCut=ast.literal_eval(args.TrackFitCut)
SubGap=int(args.SubGap)
LocalSub=(args.LocalSub=='Y')
if LocalSub:
   time_int=0
else:
    time_int=10
JobFlavour=args.JobFlavour
RequestExtCPU=int(args.RequestExtCPU)
ReqMemory=args.ReqMemory
input_file_location=args.f
Xmin,Xmax,Ymin,Ymax=float(args.Xmin),float(args.Xmax),float(args.Ymin),float(args.Ymax)
Z_overlap,Y_overlap,X_overlap=int(args.Z_overlap),int(args.Y_overlap),int(args.X_overlap)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0 #We don't slice data if all values are set to zero simultaneousy (which is the default setting)

#Loading Directory locations
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
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import UtilityFunctions as UF #This is where we keep routine utility functions
import Parameters as PM #This is where we keep framework global parameters

#Establishing paths
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
print(UF.TimeStamp(),bcolors.BOLD+'Preparation 1/3:'+bcolors.ENDC+' Setting up metafiles...')
#Loading the model meta file
print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+Model_Meta_Path+bcolors.ENDC)

if args.ModelName=='blank':
   print(UF.TimeStamp(),bcolors.WARNING+'You have specified the model name as "blank": This means that no GNN model will be used as part of the tracking process which can degrade the tracking performance.'+bcolors.ENDC)
   UserAnswer=input(bcolors.BOLD+"Do you want to continue? (y/n)\n"+bcolors.ENDC)
   if UserAnswer.upper()=='N':
       exit()
   stepX=PM.stepX
   stepY=PM.stepY
   stepZ=PM.stepZ
   cut_dt=PM.cut_dt
   cut_dr=PM.cut_dr
elif os.path.isfile(Model_Meta_Path):
       Model_Meta_Raw=UF.PickleOperations(Model_Meta_Path, 'r', 'N/A')
       print(Model_Meta_Raw[1])
       Model_Meta=Model_Meta_Raw[0]
       stepX=Model_Meta.stepX
       stepY=Model_Meta.stepY
       stepZ=Model_Meta.stepZ
       cut_dt=Model_Meta.cut_dt
       cut_dr=Model_Meta.cut_dr
else:
       print(UF.TimeStamp(),bcolors.FAIL+'Fail! No existing model meta files have been found, exiting now'+bcolors.ENDC)
       exit()

########################################     Phase 1 - Create compact source file    #########################################
print(UF.TimeStamp(),bcolors.BOLD+'Preparation 2/3:'+bcolors.ENDC+' Preparing the source data...')
required_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1_'+RecBatchID+'_hits.csv'
if os.path.isfile(required_file_location)==False or Mode=='RESET':
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
         try:
             data[PM.Hit_ID] = data[PM.Hit_ID].astype(int)
         except:
             print(UF.TimeStamp(), bcolors.WARNING+"Hit ID is already in the string format, skipping the reformatting step..."+bcolors.ENDC)
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
         data.to_csv(required_file_location,index=False)
         print(UF.TimeStamp(), bcolors.OKGREEN+"The segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+required_file_location+bcolors.ENDC)

# ########################################     Preset framework parameters    #########################################
input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1_'+RecBatchID+'_hits.csv'
print(UF.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
data=pd.read_csv(input_file_location,header=0,usecols=['z','x','y'])
print(UF.TimeStamp(),'Analysing data... ',bcolors.ENDC)
z_offset=data['z'].min()
data['z']=data['z']-z_offset
z_max=data['z'].max()
if Z_overlap==1:
    Zsteps=math.ceil((z_max)/stepZ)
else:
    Zsteps=(math.ceil((z_max)/stepZ)*(Z_overlap))-1
y_offset=data['y'].min()
x_offset=data['x'].min()
data['x']=data['x']-x_offset
data['y']=data['y']-y_offset
x_max=data['x'].max()
y_max=data['y'].max()
FreshStart=True
Program=[]
#Calculating the number of volumes that will be sent to HTCondor for reconstruction. Account for overlap if specified.
if X_overlap==1:
    Xsteps=math.ceil((x_max)/stepX)
else:
    Xsteps=(math.ceil((x_max)/stepX)*(X_overlap))-1

if Y_overlap==1:
    Ysteps=math.ceil((y_max)/stepY)
else:
    Ysteps=(math.ceil((y_max)/stepY)*(Y_overlap))-1

#Defining handy functions to make the code little cleaner

#The function bellow helps to monitor the HTCondor jobs and keep the submission flow
def AutoPilot(wait_min, interval_min, max_interval_tolerance,program):
     print(UF.TimeStamp(),'Going on an autopilot mode for ',wait_min, 'minutes while checking HTCondor every',interval_min,'min',bcolors.ENDC)
     wait_sec=wait_min*60
     interval_sec=interval_min*60
     intervals=int(math.ceil(wait_sec/interval_sec))
     for interval in range(1,intervals+1):
         time.sleep(interval_sec)
         print(UF.TimeStamp(),"Scheduled job checkup...") #Progress display
         bad_pop=UF.CreateCondorJobs(program[1][0],
                                    program[1][1],
                                    program[1][2],
                                    program[1][3],
                                    program[1][4],
                                    program[1][5],
                                    program[1][6],
                                    program[1][7],
                                    program[1][8],
                                    program[2],
                                    program[3],
                                    program[1][9],
                                    False)
         if len(bad_pop)>0:
               print(UF.TimeStamp(),bcolors.WARNING+'Autopilot status update: There are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
               if interval%max_interval_tolerance==0:
                  for bp in bad_pop:
                      UF.SubmitJobs2Condor(bp,program[5],RequestExtCPU,JobFlavour,ReqMemory)
                  print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
         else:
              return True,False
     return False,False
#The function bellow helps to automate the submission process
def StandardProcess(program,status,freshstart):
        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.BOLD+'Stage '+str(status)+':'+bcolors.ENDC+str(program[status][0]))
        batch_sub=program[status][4]>1
        bad_pop=UF.CreateCondorJobs(program[status][1][0],
                                    program[status][1][1],
                                    program[status][1][2],
                                    program[status][1][3],
                                    program[status][1][4],
                                    program[status][1][5],
                                    program[status][1][6],
                                    program[status][1][7],
                                    program[status][1][8],
                                    program[status][2],
                                    program[status][3],
                                    program[status][1][9],
                                    False)
        if len(bad_pop)==0:
             print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+' has successfully completed'+bcolors.ENDC)
             return True,False


        elif (program[status][4])==len(bad_pop):
                 bad_pop=UF.CreateCondorJobs(program[status][1][0],
                                    program[status][1][1],
                                    program[status][1][2],
                                    program[status][1][3],
                                    program[status][1][4],
                                    program[status][1][5],
                                    program[status][1][6],
                                    program[status][1][7],
                                    program[status][1][8],
                                    program[status][2],
                                    program[status][3],
                                    program[status][1][9],
                                    batch_sub)
                 print(UF.TimeStamp(),'Submitting jobs to HTCondor... ',bcolors.ENDC)
                 _cnt=0
                 for bp in bad_pop:
                          if _cnt>SubGap:
                              print(UF.TimeStamp(),'Pausing submissions for  ',str(int(SubPause/60)), 'minutes to relieve congestion...',bcolors.ENDC)
                              time.sleep(SubPause)
                              _cnt=0
                          UF.SubmitJobs2Condor(bp,program[status][5],RequestExtCPU,JobFlavour,ReqMemory)
                          _cnt+=bp[6]
                 if program[status][5]:
                    print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+' has successfully completed'+bcolors.ENDC)
                    return True,False
                 elif AutoPilot(600,time_int,Patience,program[status]):
                        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+' has successfully completed'+bcolors.ENDC)
                        return True,False
                 else:
                        print(UF.TimeStamp(),bcolors.FAIL+'Stage '+str(status)+' is uncompleted...'+bcolors.ENDC)
                        return False,False


        elif len(bad_pop)>0:
            if freshstart:
                   print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to wait please enter enter the maximum wait time in minutes'+bcolors.ENDC)
                   print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
                   UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
                   if UserAnswer=='E':
                       print(UF.TimeStamp(),'OK, exiting now then')
                       exit()
                   if UserAnswer=='R':
                      _cnt=0
                      for bp in bad_pop:
                           if _cnt>SubGap:
                              print(UF.TimeStamp(),'Pausing submissions for  ',str(int(SubPause/60)), 'minutes to relieve congestion...',bcolors.ENDC)
                              time.sleep(SubPause)
                              _cnt=0
                           UF.SubmitJobs2Condor(bp,program[status][5],RequestExtCPU,JobFlavour,ReqMemory)
                           _cnt+=bp[6]
                      print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
                      if program[status][5]:
                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+' has successfully completed'+bcolors.ENDC)
                          return True,False
                      elif AutoPilot(600,time_int,Patience,program[status]):
                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+ 'has successfully completed'+bcolors.ENDC)
                          return True,False
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage '+str(status)+' is uncompleted...'+bcolors.ENDC)
                          return False,False
                   else:
                      if program[status][5]:
                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+' has successfully completed'+bcolors.ENDC)
                          return True,False
                      elif AutoPilot(int(UserAnswer),time_int,Patience,program[status]):
                          print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+ 'has successfully completed'+bcolors.ENDC)
                          return True,False
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage '+str(status)+' is uncompleted...'+bcolors.ENDC)
                          return False,False
            else:
                      _cnt=0
                      for bp in bad_pop:
                           if _cnt>SubGap:
                              print(UF.TimeStamp(),'Pausing submissions for  ',str(int(SubPause/60)), 'minutes to relieve congestion...',bcolors.ENDC)
                              time.sleep(SubPause)
                              _cnt=0
                           UF.SubmitJobs2Condor(bp,program[status][5],RequestExtCPU,JobFlavour,ReqMemory)
                           _cnt+=bp[6]
                      if program[status][5]:
                           print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+' has successfully completed'+bcolors.ENDC)
                           return True,False
                      elif AutoPilot(600,time_int,Patience,program[status]):
                           print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+ 'has successfully completed'+bcolors.ENDC)
                           return True,False
                      else:
                          print(UF.TimeStamp(),bcolors.FAIL+'Stage '+str(status)+' is uncompleted...'+bcolors.ENDC)
                          return False,False

#If we chose reset mode we do a full cleanup.
# #Reconstructing a single brick can cause in gereation of 100s of thousands of files - need to make sure that we remove them.
if Mode=='RESET':
    print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
    HTCondorTag="SoftUsed == \"ANNDEA-RTr1a-"+RecBatchID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'RTr1_'+RecBatchID, ['RTr1a','RTr1b','RTr1c','RTr1d',RecBatchID+'_RTr_OUTPUT_CLEANED.csv'], HTCondorTag)
    FreshStart=False
if Mode=='CLEANUP':
    Status=5
else:
    Status=int(args.ForceStatus)
################ Set the execution sequence for the script
###### Stage 0
prog_entry=[]
job_sets=[]
for i in range(0,Xsteps):
                job_sets.append(Ysteps)
prog_entry.append(' Sending hit cluster to the HTCondor, so the model assigns weights between hits')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','hit_cluster_rec_set','RTr1a','.csv',RecBatchID,job_sets,'RTr1a_ReconstructTracks_Sub.py'])
prog_entry.append([' --stepZ ', ' --stepY ', ' --stepX ', " --zOffset ", " --yOffset ", " --xOffset ", ' --cut_dt ', ' --cut_dr ', ' --ModelName ',' --Z_overlap ',' --Y_overlap ',' --X_overlap ', ' --Z_ID_Max ', ' --CheckPoint ', ' --TrackFitCutRes ',' --TrackFitCutSTD ',' --TrackFitCutMRes '])
prog_entry.append([stepZ,stepY,stepX,z_offset, y_offset, x_offset, cut_dt,cut_dr, ModelName,Z_overlap,Y_overlap,X_overlap, Zsteps, args.CheckPoint]+TrackFitCut)
prog_entry.append(Xsteps*Ysteps)
prog_entry.append(LocalSub)
Program.append(prog_entry)

if Mode=='RESET':
   print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
#Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))

###### Stage 1
prog_entry=[]
job_sets=Xsteps
prog_entry.append(' Sending hit cluster to the HTCondor, so the reconstructed clusters can be merged along y-axis')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','hit_cluster_rec_y_set','RTr1b','.csv',RecBatchID,job_sets,'RTr1b_LinkSegmentsY_Sub.py'])
prog_entry.append([' --Y_ID_Max ', ' --i '])
prog_entry.append([Ysteps,Xsteps])
prog_entry.append(Xsteps)
prog_entry.append(LocalSub)
Program.append(prog_entry)
if Mode=='RESET':
   print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
#Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))

###### Stage 2
prog_entry=[]
job_sets=1
prog_entry.append(' Sending hit cluster to the HTCondor, so the reconstructed clusters can be merged along x-axis')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','hit_cluster_rec_x_set','RTr1c','.csv',RecBatchID,job_sets,'RTr1c_LinkSegmentsX_Sub.py'])
prog_entry.append([' --X_ID_Max '])
prog_entry.append([Xsteps])
prog_entry.append(1)
prog_entry.append(True) #This part we can execute locally, no need for HTCondor
Program.append(prog_entry)
if Mode=='RESET':
   print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
#Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))

###### Stage 3
Program.append('Custom')


print(UF.TimeStamp(),'There are '+str(len(Program)+1)+' stages (0-'+str(len(Program)+1)+') of this script',bcolors.ENDC)
print(UF.TimeStamp(),'Current stage has a code',Status,bcolors.ENDC)
while Status<len(Program):
    if Program[Status]!='Custom':
        #Standard process here
        Result=StandardProcess(Program,Status,FreshStart)
        if Result[0]:
            FreshStart=Result[1]
            if int(args.ForceStatus)==0:
                Status+=1
                continue
            else:
                exit()
        else:
            Status=len(Program)+1
            break

    elif Status==3:
      #Non standard processes (that don't follow the general pattern) have been coded here
      print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
      print(UF.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' Using the results from previous steps to map merged trackIDs to the original reconstruction file')
      try:
            #Read the output with hit- ANN Track map
            FirstFile=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RTr1c_'+RecBatchID+'_0'+'/RTr1c_'+RecBatchID+'_hit_cluster_rec_x_set_0.csv'
            print(UF.TimeStamp(),'Loading the file ',bcolors.OKBLUE+FirstFile+bcolors.ENDC)
            TrackMap=pd.read_csv(FirstFile,header=0)
            input_file_location=args.f
            print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
            #Reading the original file with Raw hits
            Data=pd.read_csv(input_file_location,header=0)
            Data[PM.Hit_ID] = Data[PM.Hit_ID].astype(str)

            if SliceData: #If we want to perform reconstruction on the fraction of the Brick
               CutData=Data.drop(Data.index[(Data[PM.x] > Xmax) | (Data[PM.x] < Xmin) | (Data[PM.y] > Ymax) | (Data[PM.y] < Ymin)]) #The focus area where we reconstruct
            else:
               CutData=Data #If we reconstruct the whole brick we jsut take the whole data. No need to separate.

            CutData.drop([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID'],axis=1,inplace=True,errors='ignore') #Removing old ANNDEA reconstruction results so we can overwrite with the new ones
            #Map reconstructed ANN tracks to hits in the Raw file - this is in essesential for the final output of the tracking
            TrackMap['HitID'] = TrackMap['HitID'].astype(str)
            CutData[PM.Hit_ID] = CutData[PM.Hit_ID].astype(str)
            CutData=pd.merge(CutData,TrackMap,how='left', left_on=[PM.Hit_ID], right_on=['HitID'])

            CutData.drop(['HitID'],axis=1,inplace=True) #Make sure that HitID is not the Hit ID name in the raw data.
            Data=CutData


            #It was discovered that the output is not perfect: while the hit fidelity is achieved we don't have a full plate hit fidelity for a given track. It is still possible for a track to have multiple hits at one plate.
            #In order to fix it we need to apply some additional logic to those problematic tracks.
            print(UF.TimeStamp(),'Identifying problematic tracks where there is more than one hit per plate...')
            Hit_Map=Data[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.Hit_ID]] #Separating the hit map
            Data.drop([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID'],axis=1,inplace=True) #Remove the ANNDEA tracking info from the main data





            Hit_Map=Hit_Map.dropna() #Remove unreconstructing hits - we are not interested in them atm
            Hit_Map_Stats=Hit_Map[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z,PM.Hit_ID]] #Calculating the stats

            Hit_Map_Stats=Hit_Map_Stats.groupby([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID']).agg({PM.z:pd.Series.nunique,PM.Hit_ID: pd.Series.nunique}).reset_index() #Calculate the number fo unique plates and hits
            Ini_No_Tracks=len(Hit_Map_Stats)
            print(UF.TimeStamp(),bcolors.WARNING+'The initial number of tracks is '+ str(Ini_No_Tracks)+bcolors.ENDC)
            Hit_Map_Stats=Hit_Map_Stats.rename(columns={PM.z: "No_Plates",PM.Hit_ID:"No_Hits"}) #Renaming the columns so they don't interfere once we join it back to the hit map
            Hit_Map_Stats=Hit_Map_Stats[Hit_Map_Stats.No_Plates >= PM.MinHitsTrack]
            Prop_No_Tracks=len(Hit_Map_Stats)
            print(UF.TimeStamp(),bcolors.WARNING+'After dropping single hit tracks, left '+ str(Prop_No_Tracks)+' tracks...'+bcolors.ENDC)
            Hit_Map=pd.merge(Hit_Map,Hit_Map_Stats,how='inner',on = [RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID']) #Join back to the hit map
            Good_Tracks=Hit_Map[Hit_Map.No_Plates == Hit_Map.No_Hits] #For all good tracks the number of hits matches the number of plates, we won't touch them
            Good_Tracks=Good_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.Hit_ID]] #Just strip off the information that we don't need anymore

            Bad_Tracks=Hit_Map[Hit_Map.No_Plates < Hit_Map.No_Hits] #These are the bad guys. We need to remove this extra hits
            Bad_Tracks=Bad_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.Hit_ID]]

            #Id the problematic plates
            Bad_Tracks_Stats=Bad_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z,PM.Hit_ID]]
            Bad_Tracks_Stats=Bad_Tracks_Stats.groupby([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z])[PM.Hit_ID].nunique().reset_index() #Which plates have double hits?
            Bad_Tracks_Stats=Bad_Tracks_Stats.rename(columns={PM.Hit_ID: "Problem"}) #Renaming the columns so they don't interfere once we join it back to the hit map
            Bad_Tracks_Stats[RecBatchID+'_Brick_ID'] = Bad_Tracks_Stats[RecBatchID+'_Brick_ID'].astype(str)
            Bad_Tracks_Stats[RecBatchID+'_Track_ID'] = Bad_Tracks_Stats[RecBatchID+'_Track_ID'].astype(str)
            Bad_Tracks[RecBatchID+'_Brick_ID'] = Bad_Tracks[RecBatchID+'_Brick_ID'].astype(str)
            Bad_Tracks[RecBatchID+'_Track_ID'] = Bad_Tracks[RecBatchID+'_Track_ID'].astype(str)
            Bad_Tracks=pd.merge(Bad_Tracks,Bad_Tracks_Stats,how='inner',on = [RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z])



            Bad_Tracks.sort_values([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z],ascending=[0,0,1],inplace=True)

            Bad_Tracks_CP_File=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_RTr1c_'+RecBatchID+'_0'+'/RTr1c_'+RecBatchID+'_Bad_Tracks_CP.csv'
            if os.path.isfile(Bad_Tracks_CP_File)==False or Mode=='RESET':

                Bad_Tracks_Head=Bad_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID']]
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
                Bad_Tracks_Head=pd.DataFrame(Bad_Tracks_Head, columns = [RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID','ax','t1x','t2x','ay','t1y','t2y'])

                print(UF.TimeStamp(),'Saving the checkpoint file ',bcolors.OKBLUE+Bad_Tracks_CP_File+bcolors.ENDC)
                Bad_Tracks_Head.to_csv(Bad_Tracks_CP_File,index=False)
            else:
               print(UF.TimeStamp(),'Loading the checkpoint file ',bcolors.OKBLUE+Bad_Tracks_CP_File+bcolors.ENDC)
               Bad_Tracks_Head=pd.read_csv(Bad_Tracks_CP_File,header=0)
               Bad_Tracks_Head=Bad_Tracks_Head[Bad_Tracks_Head.ax != '[]']
               Bad_Tracks_Head['ax'] = Bad_Tracks_Head['ax'].astype(float)
               Bad_Tracks_Head['ay'] = Bad_Tracks_Head['ay'].astype(float)
               Bad_Tracks_Head['t1x'] = Bad_Tracks_Head['t1x'].astype(float)
               Bad_Tracks_Head['t2x'] = Bad_Tracks_Head['t2x'].astype(float)
               Bad_Tracks_Head['t1y'] = Bad_Tracks_Head['t1y'].astype(float)
               Bad_Tracks_Head['t2y'] = Bad_Tracks_Head['t2y'].astype(float)

            print(UF.TimeStamp(),'Removing problematic hits...')
            Bad_Tracks=pd.merge(Bad_Tracks,Bad_Tracks_Head,how='inner',on = [RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID'])



            print(UF.TimeStamp(),'Calculating x and y coordinates of the fitted line for all plates in the track...')
            #Calculating x and y coordinates of the fitted line for all plates in the track
            Bad_Tracks['new_x']=Bad_Tracks['ax']+(Bad_Tracks[PM.z]*Bad_Tracks['t1x'])+((Bad_Tracks[PM.z]**2)*Bad_Tracks['t2x'])
            Bad_Tracks['new_y']=Bad_Tracks['ay']+(Bad_Tracks[PM.z]*Bad_Tracks['t1y'])+((Bad_Tracks[PM.z]**2)*Bad_Tracks['t2y'])

            #Calculating how far hits deviate from the fit polynomial
            print(UF.TimeStamp(),'Calculating how far hits deviate from the fit polynomial...')
            Bad_Tracks['d_x']=Bad_Tracks[PM.x]-Bad_Tracks['new_x']
            Bad_Tracks['d_y']=Bad_Tracks[PM.y]-Bad_Tracks['new_y']

            Bad_Tracks['d_r']=Bad_Tracks['d_x']**2+Bad_Tracks['d_y']**2
            Bad_Tracks['d_r'] = Bad_Tracks['d_r'].astype(float)
            Bad_Tracks['d_r']=np.sqrt(Bad_Tracks['d_r']) #Absolute distance
            Bad_Tracks=Bad_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z,PM.Hit_ID,'d_r']]

            #Sort the tracks and their hits by Track ID, Plate and distance to the perfect line
            print(UF.TimeStamp(),'Sorting the tracks and their hits by Track ID, Plate and distance to the perfect line...')
            Bad_Tracks.sort_values([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z,'d_r'],ascending=[0,0,1,1],inplace=True)

            #If there are two hits per plate we will keep the one which is closer to the line
            Bad_Tracks.drop_duplicates(subset=[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z],keep='first',inplace=True)
            Bad_Tracks=Bad_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.Hit_ID]]
            Good_Tracks=pd.concat([Good_Tracks,Bad_Tracks]) #Combine all ANNDEA tracks together
            Data=pd.merge(Data,Good_Tracks,how='left', on=[PM.Hit_ID]) #Re-map corrected ANNDEA Tracks back to the main data
            output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_RTr_OUTPUT_CLEANED.csv' #Final output. We can use this file for further operations
            Data.to_csv(output_file_location,index=False)
            print(UF.TimeStamp(), bcolors.OKGREEN+"The tracked data has been written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
            print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 4 has successfully completed'+bcolors.ENDC)
            Status=4
       except Exception as e:
          print(UF.TimeStamp(),bcolors.FAIL+'Stage 4 is uncompleted due to: '+str(e)+bcolors.ENDC)
          Status=5
          break
if Status==4:
    # Removing the temp files that were generated by the process
    print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
    HTCondorTag="SoftUsed == \"ANNDEA-RTr1a-"+RecBatchID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'RTr1a_'+RecBatchID, [], HTCondorTag)
    HTCondorTag="SoftUsed == \"ANNDEA-RTr1b-"+RecBatchID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'RTr1b_'+RecBatchID, [], HTCondorTag)
    for p in Program:
        if p!='Custom':
           print(UF.TimeStamp(),UF.ManageTempFolders(p,'Delete'))
    print(UF.TimeStamp(), bcolors.OKGREEN+"Reconstruction has been completed"+bcolors.ENDC)
    exit()
else:
    print(UF.TimeStamp(), bcolors.FAIL+"Reconstruction has not been completed as one of the processes has timed out or --ForceStatus!=0 option was chosen. Please run the script again (without Reset Mode)."+bcolors.ENDC)
    exit()



