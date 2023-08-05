#This is the training module for the tracker. This script is the master script that just manages HTCondor submissions
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
if PY_DIR!='': #Temp solution - the decision was made to move all libraries to EOS drive as AFS get locked during heavy HTCondor submission loads
    sys.path=['',PY_DIR]
    sys.path.append('/usr/lib64/python36.zip')
    sys.path.append('/usr/lib64/python3.6')
    sys.path.append('/usr/lib64/python3.6/lib-dynload')
    sys.path.append('/usr/lib64/python3.6/site-packages')
    sys.path.append('/usr/lib/python3.6/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')
#import libraries
import argparse
import ast
import os
import time
import math
import htcondor
schedd=htcondor.Schedd()
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#Set the parsing module
parser = argparse.ArgumentParser(description='Enter training job parameters')
parser.add_argument('--ModelName',help="Please enter the name of the model that you want to train", default='MH_SND_Tracking_1_20_1_20_Meta')
parser.add_argument('--ModelParams',help="Please enter the model params: '[<Number of MLP layers>, <'MLP hidden size'>, <Number of IN layers>, <'IN hidden size'>]'", default='[1,20,1,20]')
parser.add_argument('--TrainParams',help="Please enter the train params: '[<Learning Rate>, <Batch size>, <Epochs>, <Epoch fractions>]'", default='[0.0001, 4, 10, 1]')
parser.add_argument('--TrainSampleID',help="What training sample would you like to use?", default='MH_SND_Raw_Train_Data_6_6_12_All')
parser.add_argument('--Mode',help="Please enter 'Reset' if you want to overwrite the existing model", default='')
parser.add_argument('--JobFlavour',help="Specifying the length of the HTCondor job walltime. Currently at 'workday' which is 8 hours.", default='workday')
parser.add_argument('--ReqMemory',help="How much memory to request?", default='2 GB')
parser.add_argument('--RequestExtCPU',help="Would you like to request extra CPUs? How Many?", default=1)
args = parser.parse_args()

#setting main learning parameters
Mode=args.Mode.upper()
ModelName=args.ModelName
ModelParamsStr='"'+args.ModelParams+'"' #Add apostrophies so command is passed correctly to the submission script
TrainParamsStr='"'+args.TrainParams+'"' #Add apostrophies so command is passed correctly to the submission script
ModelParams=ast.literal_eval(args.ModelParams) #Here we converting parsed list string to list type so we can work with it
TrainParams=ast.literal_eval(args.TrainParams) #Here we converting parsed list string to list type so we can work with it
TrainSampleID=args.TrainSampleID
JobFlavour=args.JobFlavour
ReqMemory=args.ReqMemory
RequestExtCPU=int(args.RequestExtCPU)

#Loading Data configurations
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubDataDIR=EOSsubDIR+'/'+'Data'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
import UtilityFunctions as UF
import Parameters as PM
import datetime

prog_entry=[]
job_sets=[1]
prog_entry.append('Training the model')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','training_model','MTr2','.pkl',ModelName,job_sets,''])
prog_entry.append([''])
prog_entry.append([1])
prog_entry.append(1)
prog_entry.append('NA')
print(prog_entry)
if Mode=='RESET':
   print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))

print(bcolors.HEADER+"####################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################  Initialising ANNDEA model training module       #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################            Written by Filips Fedotovs            #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################               PhD Student at UCL                 #########################"+bcolors.ENDC)
print(bcolors.HEADER+"###################### For troubleshooting please contact filips.fedotovs@cern.ch ##################"+bcolors.ENDC)
print(bcolors.HEADER+"####################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
#This code fragment covers the Algorithm logic on the first run

#The function bellow measures the slope of the loss/accuracy line. If it gets too flat (determined by PM.TST parameter) we stop the training. Useful for automation
def ModelTrainingSaturation(Meta):
    LossDataForChecking=[]
    AccDataForChecking=[]
    for i in range(1,len(Meta)):
               LossDataForChecking.append(Meta[i][6])
               AccDataForChecking.append(Meta[i][7])
    LossGrad=UF.GetEquationOfLine(LossDataForChecking)[0]
    AccGrad=UF.GetEquationOfLine(AccDataForChecking)[0]
    return LossGrad>=-PM.TST and AccGrad<=PM.TST
#The function bellow monitors and manages the HTCondor submission
def AutoPilot(wait_min, interval_min):
    print(UF.TimeStamp(),'Going on an autopilot mode for ',wait_min, 'minutes while checking HTCondor every',interval_min,'min',bcolors.ENDC)
    wait_sec=wait_min*60
    interval_sec=interval_min*60
    intervals=int(math.ceil(wait_sec/interval_sec))
    Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
    for interval in range(1,intervals+1):
       time.sleep(interval_sec)
       print(UF.TimeStamp(),"Scheduled job checkup...") #Progress display
       print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+Model_Meta_Path+bcolors.ENDC)
       Model_Meta_Raw=UF.PickleOperations(Model_Meta_Path, 'r', 'N/A')
       print(Model_Meta_Raw[1])
       Model_Meta=Model_Meta_Raw[0]
       Model_Status=Model_Meta.ModelTrainStatus(PM.TST)
       if Model_Status==1:
              print(UF.TimeStamp(),bcolors.WARNING+'Warning, the model seems to be over saturated'+bcolors.ENDC)
              print(UF.TimeStamp(),'Aborting the training...')
              exit()
       elif Model_Status==2:
                 print(UF.TimeStamp(), bcolors.OKGREEN+"Training session has been completed, starting another session..."+bcolors.ENDC)
                 HTCondorTag="SoftUsed == \"ANNDEA-MTr2-"+ModelName+"\""
                 UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MTr2_'+ModelName, ['N/A'], HTCondorTag)
                 OptionHeader = [' --TrainParams ', " --TrainSampleID "]
                 OptionLine = [TrainParamsStr, TrainSampleID]
                 Job=UF.CreateCondorJobs(AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','N/A','MTr2','N/A',ModelName,1,OptionHeader,OptionLine,'MTr2_TrainModel_Sub.py',False,"['','']", True, False)[0]
                 TrainSampleInputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
                 print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+TrainSampleInputMeta+bcolors.ENDC)
                 Model_Meta.IniTrainingSession(TrainSampleID, datetime.datetime.now(), TrainParams)
                 print(UF.PickleOperations(Model_Meta_Path, 'w', Model_Meta)[1])
                 UF.SubmitJobs2Condor(Job,False,RequestExtCPU,JobFlavour,ReqMemory)
                 print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
                 print(bcolors.OKGREEN+"......................................................................."+bcolors.ENDC)
                 interval_count=1
       else:
                HTCondorTag="SoftUsed == \"ANNDEA-MUTr2-"+ModelName+"\""
                q=schedd.query(constraint=HTCondorTag,projection=['CLusterID','JobStatus'])
                if len(q)==0:
                    print(UF.TimeStamp(),bcolors.FAIL+'The HTCondor job has failed, resubmitting...'+bcolors.ENDC)
                    OptionHeader = [' --TrainParams ', " --TrainSampleID "]
                    OptionLine = [TrainParamsStr, TrainSampleID]
                    Job=UF.CreateCondorJobs(AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','N/A','MTr2','N/A',ModelName,1,OptionHeader,OptionLine,'MTr2_TrainModel_Sub.py',False,"['','']", True, False)[0]
                    UF.SubmitJobs2Condor(Job,False,RequestExtCPU,JobFlavour,ReqMemory)
                    print(bcolors.BOLD+"The job has been submitted..."+bcolors.ENDC)
                else:
                    print(UF.TimeStamp(),bcolors.WARNING+'Warning, the training is still running on HTCondor, the job parameters are listed bellow:'+bcolors.ENDC)
                    print(q)
                    continue
    return True

if Mode=='RESET':
 HTCondorTag="SoftUsed == \"ANNDEA-MTr2-"+ModelName+"\""
 UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MTr2_'+ModelName, ['N/A'], HTCondorTag)
 OptionHeader = [' --ModelParams ', ' --TrainParams ', " --TrainSampleID "]
 OptionLine = [ModelParamsStr, TrainParamsStr, TrainSampleID]
 Job=UF.CreateCondorJobs(AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','N/A','MTr2','N/A',ModelName,1,OptionHeader,OptionLine,'MTr2_TrainModel_Sub.py',False,"['','']", True, False)[0]
 TrainSampleInputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
 print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+TrainSampleInputMeta+bcolors.ENDC)
 MetaInput=UF.PickleOperations(TrainSampleInputMeta,'r', 'N/A')
 print(MetaInput[1])
 Meta=MetaInput[0]
 Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
 ModelMeta=UF.ModelMeta(ModelName)
 ModelMeta.IniModelMeta(ModelParams, 'PyTorch', Meta, 'TCN', 'GNN')
 ModelMeta.IniTrainingSession(TrainSampleID, datetime.datetime.now(), TrainParams)
 print(UF.PickleOperations(Model_Meta_Path, 'w', ModelMeta)[1])
 UF.SubmitJobs2Condor(Job,False,RequestExtCPU,JobFlavour,ReqMemory)
 print(bcolors.BOLD+"The job has been submitted..."+bcolors.ENDC)
 print(bcolors.BOLD+'If you would like to stop training and exit please enter E'+bcolors.ENDC)
 print(bcolors.BOLD+'If you would like to continue training on autopilot please type waiting time in minutes'+bcolors.ENDC)
 UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
 if UserAnswer=='E':
                  print(UF.TimeStamp(),'OK, exiting now then')
                  exit()
 else:
    AutoPilot(int(UserAnswer),30)
else:
     Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
     print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+Model_Meta_Path+bcolors.ENDC)
     if os.path.isfile(Model_Meta_Path):
       Model_Meta_Raw=UF.PickleOperations(Model_Meta_Path, 'r', 'N/A')
       print(Model_Meta_Raw[1])
       Model_Meta=Model_Meta_Raw[0]
       Models_Status=Model_Meta.ModelTrainStatus(PM.TST)
       if Models_Status==1:
              print(UF.TimeStamp(),bcolors.WARNING+'Warning, the model seems to be over saturated'+bcolors.ENDC)
              Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
              print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+Model_Meta_Path+bcolors.ENDC)
              if os.path.isfile(Model_Meta_Path):
                   Model_Meta_Raw=UF.PickleOperations(Model_Meta_Path, 'r', 'N/A')
                   print(Model_Meta_Raw[1])
                   Model_Meta=Model_Meta_Raw[0]
                   Header=Model_Meta.TrainSessionsData[0][0]+['Model Parameters','Train Sample ID','LR','Batch Size','Normalised Epochs']
                   New_Data=[Header]
                   Print_New_Data=[] 
                   counter=0
                   print(UF.TimeStamp(),bcolors.OKGREEN+'The model training profile is printed bellow: '+bcolors.ENDC)
                   for TSD in range(len(Model_Meta.TrainSessionsData)):
                       for Record in Model_Meta.TrainSessionsData[TSD][1:]:
                           counter+=1
                           New_Data.append(Record+[Model_Meta.ModelParameters,Model_Meta.TrainSessionsDataID[TSD],Model_Meta.TrainSessionsParameters[TSD][0],Model_Meta.TrainSessionsParameters[TSD][1],counter])
                           Print_New_Data.append(Record)
                   print(pd.DataFrame(Print_New_Data, columns=Model_Meta.TrainSessionsData[0][0]))
                   Model_Meta_csv=EOSsubModelDIR+'/'+args.ModelName+'_Out.csv'
                   UF.LogOperations(Model_Meta_csv,'w', New_Data)
                   print(UF.TimeStamp(),bcolors.OKGREEN+'Csv output has been saved as '+bcolors.ENDC+bcolors.OKBLUE+Model_Meta_csv+bcolors.ENDC)  
              print(bcolors.BOLD+'If you would like to stop training and exit please enter E'+bcolors.ENDC)
              print(bcolors.BOLD+'If you would like to resubmit your script enter R'+bcolors.ENDC)
              print(bcolors.BOLD+'If you would like to continue training on autopilot please type waiting time in minutes'+bcolors.ENDC)
              UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
              if UserAnswer=='E':
                  print(UF.TimeStamp(),'OK, exiting now then')
                  exit()
              if UserAnswer=='R':
                 HTCondorTag="SoftUsed == \"ANNDEA-MTr2-"+ModelName+"\""
                 UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MTr2_'+ModelName, ['N/A'], HTCondorTag)
                 OptionHeader = [' --TrainParams ', " --TrainSampleID "]
                 OptionLine = [TrainParamsStr,TrainSampleID]
                 Job=UF.CreateCondorJobs(AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','N/A','MTr2','N/A',ModelName,1,OptionHeader,OptionLine,'MTr2_TrainModel_Sub.py',False,"['','']", True, False)[0]
                 Model_Meta.IniTrainingSession(TrainSampleID, datetime.datetime.now(), TrainParams)
                 UF.SubmitJobs2Condor(Job,False,RequestExtCPU,JobFlavour,ReqMemory)
                 print(bcolors.BOLD+"The job has been submitted..."+bcolors.ENDC)
                 print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
                 exit()
              else:
                 OptionHeader = [' --TrainParams ', " --TrainSampleID "]
                 OptionLine = [TrainParamsStr,TrainSampleID]
                 Job=UF.CreateCondorJobs(AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','N/A','MTr2','N/A',ModelName,1,OptionHeader,OptionLine,'MTr2_TrainModel_Sub.py',False,"['','']", True, False)[0]
                 Model_Meta.IniTrainingSession(TrainSampleID, datetime.datetime.now(), TrainParams)
                 print(UF.PickleOperations(Model_Meta_Path, 'w', Model_Meta)[1])
                 UF.SubmitJobs2Condor(Job,False,RequestExtCPU,JobFlavour,ReqMemory)
                 print(bcolors.BOLD+"The job has been submitted..."+bcolors.ENDC)
                 print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
                 AutoPilot(int(UserAnswer),30)
       elif Models_Status==2:
                 print(UF.TimeStamp(),bcolors.OKGREEN+'The training session has been completed'+bcolors.ENDC)
                 Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
                 print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+Model_Meta_Path+bcolors.ENDC)
                 if os.path.isfile(Model_Meta_Path):
                   Model_Meta_Raw=UF.PickleOperations(Model_Meta_Path, 'r', 'N/A')
                   print(Model_Meta_Raw[1])
                   Model_Meta=Model_Meta_Raw[0]
                   Header=Model_Meta.TrainSessionsData[0][0]+['Model Parameters','Train Sample ID','LR','Batch Size','Normalised Epochs']
                   New_Data=[Header]
                   Print_New_Data=[] 
                   counter=0
                   print(UF.TimeStamp(),bcolors.OKGREEN+'The model training profile is printed bellow: '+bcolors.ENDC)
                   for TSD in range(len(Model_Meta.TrainSessionsData)):
                       for Record in Model_Meta.TrainSessionsData[TSD][1:]:
                           counter+=1
                           New_Data.append(Record+[Model_Meta.ModelParameters,Model_Meta.TrainSessionsDataID[TSD],Model_Meta.TrainSessionsParameters[TSD][0],Model_Meta.TrainSessionsParameters[TSD][1],counter])
                           Print_New_Data.append(Record)
                   print(pd.DataFrame(Print_New_Data, columns=Model_Meta.TrainSessionsData[0][0]))
                   Model_Meta_csv=EOSsubModelDIR+'/'+args.ModelName+'_Out.csv'
                   UF.LogOperations(Model_Meta_csv,'w', New_Data)
                   print(UF.TimeStamp(),bcolors.OKGREEN+'Csv output has been saved as '+bcolors.ENDC+bcolors.OKBLUE+Model_Meta_csv+bcolors.ENDC) 
                 print(bcolors.BOLD+'If you would like to stop training and exit please enter E'+bcolors.ENDC)
                 print(bcolors.BOLD+'If you would like to submit another one and exit, enter S'+bcolors.ENDC)
                 print(bcolors.BOLD+'If you would like to submit another one and continue training on the autopilot please type waiting time in minutes'+bcolors.ENDC)
                 UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
                 if UserAnswer=='E':
                  print(UF.TimeStamp(),'OK, exiting now then')
                  exit()
                 elif UserAnswer=='S':
                     OptionHeader = [' --TrainParams ', " --TrainSampleID "]
                     OptionLine = [TrainParamsStr, TrainSampleID]
                     Job=UF.CreateCondorJobs(AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','N/A','MTr2','N/A',ModelName,1,OptionHeader,OptionLine,'MTr2_TrainModel_Sub.py',False,"['','']", True, False)[0]
                     Model_Meta.IniTrainingSession(TrainSampleID, datetime.datetime.now(), TrainParams)
                     print(UF.PickleOperations(Model_Meta_Path, 'w', Model_Meta)[1])
                     UF.SubmitJobs2Condor(Job,False,RequestExtCPU,JobFlavour,ReqMemory)
                     print(bcolors.BOLD+"The job has been submitted..."+bcolors.ENDC)
                 else:
                     OptionHeader = [' --TrainParams ', " --TrainSampleID "]
                     OptionLine = [TrainParamsStr, TrainSampleID]
                     Job=UF.CreateCondorJobs(AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','N/A','MTr2','N/A',ModelName,1,OptionHeader,OptionLine,'MTr2_TrainModel_Sub.py',False,"['','']", True, False)[0]
                     Model_Meta.IniTrainingSession(TrainSampleID, datetime.datetime.now(), TrainParams)
                     print(UF.PickleOperations(Model_Meta_Path, 'w', Model_Meta)[1])
                     UF.SubmitJobs2Condor(Job,False,RequestExtCPU,JobFlavour,ReqMemory)
                     print(bcolors.BOLD+"The job has been submitted..."+bcolors.ENDC)
                     AutoPilot(int(UserAnswer),30)

       else:
           HTCondorTag="SoftUsed == \"ANNDEA-MTr2-"+ModelName+"\""
           q=schedd.query(constraint=HTCondorTag,projection=['CLusterID','JobStatus'])
           if len(q)==0:
                print(UF.TimeStamp(),bcolors.FAIL+'HTCondor has failed...'+bcolors.ENDC)
           else:
                print(UF.TimeStamp(),bcolors.WARNING+'Warning, the training is still running on HTCondor, the job parameters are listed bellow:'+bcolors.ENDC)
                print(q)
           print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
           print(bcolors.BOLD+'If you would like to resubmit your script enter R'+bcolors.ENDC)
           print(bcolors.BOLD+'If you would like to put the script on the autopilot type number of minutes to wait'+bcolors.ENDC)
           UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
           if UserAnswer=='E':
                  print(UF.TimeStamp(),'OK, exiting now then')
                  exit()
           elif UserAnswer=='R':
                 HTCondorTag="SoftUsed == \"ANNDEA-MTr2-"+ModelName+"\""
                 UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MTr2_'+ModelName, ['N/A'], HTCondorTag)
                 OptionHeader = [' --TrainParams ', " --TrainSampleID "]
                 OptionLine = [TrainParamsStr, TrainSampleID]
                 Job=UF.CreateCondorJobs(AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','N/A','MTr2','N/A',ModelName,1,OptionHeader,OptionLine,'MTr2_TrainModel_Sub.py',False,"['','']", True,False)[0]
                 UF.SubmitJobs2Condor(Job,False,RequestExtCPU,JobFlavour,ReqMemory)
                 print(bcolors.BOLD+"The job has been submitted..."+bcolors.ENDC)
                 print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
                 exit()
           else:
              AutoPilot(int(UserAnswer),30)
     else:
                 print(UF.TimeStamp(),bcolors.WARNING+'Warning! No existing meta files have been found, starting everything from the scratch.'+bcolors.ENDC)
                 HTCondorTag="SoftUsed == \"ANNDEA-MTr2-"+ModelName+"\""
                 UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MTr2_'+ModelName, ['N/A'], HTCondorTag)
                 OptionHeader = [' --ModelParams ', ' --TrainParams ', " --TrainSampleID "]
                 OptionLine = [ModelParamsStr, TrainParamsStr, TrainSampleID]
                 Job=UF.CreateCondorJobs(AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','N/A','MTr2','N/A',ModelName,1,OptionHeader,OptionLine,'MTr2_TrainModel_Sub.py',False,"['','']", True, False)[0]
                 TrainSampleInputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
                 print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+TrainSampleInputMeta+bcolors.ENDC)
                 MetaInput=UF.PickleOperations(TrainSampleInputMeta,'r', 'N/A')
                 print(MetaInput[1])
                 Meta=MetaInput[0]
                 Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
                 ModelMeta=UF.ModelMeta(ModelName)
                 ModelMeta.IniModelMeta(ModelParams, 'PyTorch', Meta, 'TCN', 'GNN') #For tracking, only this type of GNN is used.
                 ModelMeta.IniTrainingSession(TrainSampleID, datetime.datetime.now(), TrainParams)
                 print(UF.PickleOperations(Model_Meta_Path, 'w', ModelMeta)[1])
                 UF.SubmitJobs2Condor(Job,False,RequestExtCPU,JobFlavour,ReqMemory)
                 print(bcolors.BOLD+"The job has been submitted..."+bcolors.ENDC)
                 AutoPilot(600,30)
print(UF.TimeStamp(),bcolors.OKGREEN+'Exiting the program...'+bcolors.ENDC)
print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
exit()


