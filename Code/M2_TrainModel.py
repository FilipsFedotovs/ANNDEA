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
    sys.path.append('/usr/lib64/python39.zip')
    sys.path.append('/usr/lib64/python3.9')
    sys.path.append('/usr/lib64/python3.9/lib-dynload')
    sys.path.append('/usr/lib64/python3.9/site-packages')
    sys.path.append('/usr/lib/python3.9/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')

#import libraries
import argparse
import pandas as pd
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
parser.add_argument('--ModelName',help="Which model would you like to use as a base for training (please enter N if you want to train a new model from scratch)", default='Default')
parser.add_argument('--ModelType',help="What Neural Network type would you like to use: CNN/GNN?", default='GNN')
parser.add_argument('--ModelArchitecture',help="What Type of Image/Graph: CNN, CNN-E", default='GMM-6N-IC')
parser.add_argument('--ModelParams',help="Please enter the model params", default="[[150,3],[150,3],[150,3],[150,3],[],[],[],[],[],[],[7,2],[3000,3000,20000,50]]")
parser.add_argument('--TrainParams',help="Please enter the train params: '[<Learning Rate>, <Batch size>, <Epochs>]'", default='[0.0001, 4, 10]')
parser.add_argument('--TrainSampleID',help="Give name to this train sample", default='SHIP_TrainSample_v1')
parser.add_argument('--Mode',help="Please enter 'Reset' if you want to overwrite the existing model", default='')
parser.add_argument('--JobFlavour',help="Specifying the length of the HTCondor job wall-time. Currently at 'workday' which is 8 hours.", default='workday')
parser.add_argument('--ReqMemory',help="How much memory to request?", default='4 GB')
parser.add_argument('--Wait',help="How many minutes to wait for a job", default='30')
parser.add_argument('--RequestExtCPU',help="Would you like to request extra CPUs? How Many?", default=2)
parser.add_argument('--LocalSub',help="Local submission?", default='N')
parser.add_argument('--Memory',help="How uch memory to request?", default='2 GB')
parser.add_argument('--CPU',help="How many CPUs?", default=1)
args = parser.parse_args()

#setting main learning parameters
Mode=args.Mode.upper()
ModelName=args.ModelName
ModelType=args.ModelType
ModelArchitecture=args.ModelArchitecture
ModelParamsStr='"'+args.ModelParams+'"'
TrainParamsStr='"'+args.TrainParams+'"'
ModelParams=ast.literal_eval(args.ModelParams)
TrainParams=ast.literal_eval(args.TrainParams)
TrainSampleID=args.TrainSampleID
JobFlavour=args.JobFlavour
Memory=args.Memory
LocalSub=(args.LocalSub=='Y')
CPU=int(args.CPU)
Wait=int(args.Wait)


#Loading Data configurations
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubDataDIR=EOSsubDIR+'/'+'Data'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import U_UI as UI #This is where we keep routine utility functions
import U_ML as ML #This is where we keep routine utility functions
import Parameters as PM
import datetime

if Mode=='RESET':
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, ModelName,'d',['M2']))
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, ModelName,'c'))
    import shutil
    print(EOSsubModelDIR+'/'+args.ModelName)
    print(EOSsubModelDIR+'/'+args.ModelName+'.keras')
    shutil.rmtree(EOSsubModelDIR+'/'+args.ModelName,True)
    shutil.rmtree(EOSsubModelDIR+'/'+args.ModelName+'.keras',True)
else:
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, ModelName,'c'))

OptionHeader = [' --TrainParams ', " --TrainSampleID "]
OptionLine = [TrainParamsStr, TrainSampleID]
prog_entry=[]
prog_entry.append(' Sending hit cluster to the HTCondor, so the reconstructed clusters can be merged along z-axis')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/'+ModelName+'/','hit_cluster_rec_z_set','M2','.pkl',ModelName,1,''])
prog_entry.append([''])
prog_entry.append([1])
prog_entry.append(1)
prog_entry.append('N/A')
print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry))
if ModelType=='CNN':
                    Job=UI.CreateCondorJobs(AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','N/A','M2','N/A',ModelName,1,OptionHeader,OptionLine,'M2_TrainModel_Sub.py',False,"['','']", True, True)[0]
elif ModelType=='GNN' and ModelArchitecture=='TCN':
                    Job=UI.CreateCondorJobs(AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','N/A','M2','N/A',ModelName,1,OptionHeader,OptionLine,'MTr4_TrainModel_Sub.py',False,"['','']", True, False)[0]
elif ModelType=='MLP':
                    Job=UI.CreateCondorJobs(AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','N/A','M2','N/A',ModelName,1,OptionHeader,OptionLine,'MTr2_TrainSeedModel_Sub.py',False,"['','']", True, False)[0]
else:
                    Job=UI.CreateCondorJobs(AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/','N/A','M2','N/A',ModelName,1,OptionHeader,OptionLine,'M2_TrainModel_Sub.py',False,"['','']", True, False)[0]

UI.WelcomeMsg('Initialising ANNDEA Model Training module...','Filips Fedotovs (PhD student at UCL), Wenqing Xie (MSc student at UCL), Leah Wolf (MSc student at UCL), Henry Wilson (MSc student at UCL)','Please reach out to filips.fedotovs@cern.ch for any queries')
print(UI.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
#This code fragment covers the Algorithm logic on the first run
def AutoPilot(wait_min, interval_min):
    print(UI.TimeStamp(),'Going on an autopilot mode for ',wait_min, 'minutes while checking HTCondor every',interval_min,'min',bcolors.ENDC)
    wait_sec=wait_min*60
    interval_sec=interval_min*60
    intervals=int(math.ceil(wait_sec/interval_sec))
    Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
    for interval in range(1,intervals+1):
       time.sleep(interval_sec)
       print(UI.TimeStamp(),"Scheduled job checkup...") #Progress display
       print(UI.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+Model_Meta_Path+bcolors.ENDC)
       Model_Meta_Raw=UI.PickleOperations(Model_Meta_Path, 'r', 'N/A')
       print(Model_Meta_Raw[1])
       Model_Meta=Model_Meta_Raw[0]
       Model_Status=Model_Meta.ModelTrainStatus(PM.TST)
       if Model_Status==1:
              print(UI.TimeStamp(),bcolors.WARNING+'Warning, the model seems to be over saturated'+bcolors.ENDC)
              print(UI.TimeStamp(),'Aborting the training...')
              exit()
       elif Model_Status==2:
                 print(UI.TimeStamp(), bcolors.OKGREEN+"Training is finished, starting another session..."+bcolors.ENDC)
                 Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
                 print(UI.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+Model_Meta_Path+bcolors.ENDC)
                 if os.path.isfile(Model_Meta_Path):
                       Model_Meta_Raw=UI.PickleOperations(Model_Meta_Path, 'r', 'N/A')
                       print(Model_Meta_Raw[1])
                       Model_Meta=Model_Meta_Raw[0]
                       Header=Model_Meta.TrainSessionsData[0][0]+['Model Parameters','Train Sample ID','LR','Batch Size','Normalised Epochs']
                       New_Data=[Header]
                       Print_New_Data=[] 
                       counter=0
                       print(UI.TimeStamp(),bcolors.OKGREEN+'The model training profile is printed bellow: '+bcolors.ENDC)
                       for TSD in range(len(Model_Meta.TrainSessionsData)):
                           for Record in Model_Meta.TrainSessionsData[TSD][1:]:
                               counter+=1
                               New_Data.append(Record+[Model_Meta.ModelParameters,Model_Meta.TrainSessionsDataID[TSD],Model_Meta.TrainSessionsParameters[TSD][0],Model_Meta.TrainSessionsParameters[TSD][1],counter])
                               Print_New_Data.append(Record)
                       print(pd.DataFrame(Print_New_Data, columns=Model_Meta.TrainSessionsData[0][0]))
                       Model_Meta_csv=EOSsubModelDIR+'/'+args.ModelName+'_Out.csv'
                       UI.LogOperations(Model_Meta_csv,'w', New_Data)
                       print(UI.TimeStamp(),bcolors.OKGREEN+'Csv output has been saved as '+bcolors.ENDC+bcolors.OKBLUE+Model_Meta_csv+bcolors.ENDC)
                 TrainSampleInputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
                 print(UI.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+TrainSampleInputMeta+bcolors.ENDC)
                 Model_Meta.IniTrainingSession(TrainSampleID, datetime.datetime.now(), TrainParams)
                 print(UI.PickleOperations(Model_Meta_Path, 'w', Model_Meta)[1])
                 UI.SubmitJobs2Condor(Job,LocalSub,CPU,JobFlavour,Memory)
                 print(UI.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
                 print(bcolors.OKGREEN+"......................................................................."+bcolors.ENDC)
       else:
                try:
                    HTCondorTag="SoftUsed == \"ANNDEA-M2-"+ModelName+"\""
                    q=schedd.query(constraint=HTCondorTag,projection=['CLusterID','JobStatus'])
                    if len(q)==0:
                        print(UI.TimeStamp(),bcolors.FAIL+'The HTCondor job has failed, resubmitting...'+bcolors.ENDC)
                        UI.SubmitJobs2Condor(Job,LocalSub,CPU,JobFlavour,Memory)
                        print(bcolors.BOLD+"The job has been submitted..."+bcolors.ENDC)
                    else:
                        print(UI.TimeStamp(),bcolors.WARNING+'Warning, the training is still running on HTCondor, the job parameters are listed bellow:'+bcolors.ENDC)
                        print(q)
                        continue
                except:
                    print(UI.TimeStamp(),bcolors.FAIL+'No response from HTCondor, will check again in a while...:'+bcolors.ENDC)
                    continue

    return True

if Mode=='RESET':
 TrainSampleInputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
 print(UI.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+TrainSampleInputMeta+bcolors.ENDC)
 MetaInput=UI.PickleOperations(TrainSampleInputMeta,'r', 'N/A')
 print(MetaInput[1])
 Meta=MetaInput[0]
 Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
 ModelMeta=ML.ModelMeta(ModelName)
 if ModelType=='CNN':
    ModelMeta.IniModelMeta(ModelParams, 'Tensorflow', Meta, ModelArchitecture, 'CNN')
 elif ModelType=='GNN':
    ModelMeta.IniModelMeta(ModelParams, 'PyTorch', Meta, ModelArchitecture, 'GNN')
 elif ModelType=='MLP':
    ModelMeta.IniModelMeta(ModelParams, 'PyTorch', Meta, ModelArchitecture, 'GNN')
 ModelMeta.IniTrainingSession(TrainSampleID, datetime.datetime.now(), TrainParams)
 print(UI.PickleOperations(Model_Meta_Path, 'w', ModelMeta)[1])
 UI.SubmitJobs2Condor(Job,LocalSub,CPU,JobFlavour,Memory)
 print(bcolors.BOLD+'If you would like to stop training and exit please enter E'+bcolors.ENDC)
 print(bcolors.BOLD+'If you would like to continue training on autopilot please type waiting time in minutes'+bcolors.ENDC)
 UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
 if UserAnswer=='E':
                  print(UI.TimeStamp(),'OK, exiting now then')
                  exit()
 else:
    AutoPilot(int(UserAnswer),Wait)
else:
     Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
     print(UI.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+Model_Meta_Path+bcolors.ENDC)
     if os.path.isfile(Model_Meta_Path):
       Model_Meta_Raw=UI.PickleOperations(Model_Meta_Path, 'r', 'N/A')
       print(Model_Meta_Raw[1])
       Model_Meta=Model_Meta_Raw[0]
       Models_Status=Model_Meta.ModelTrainStatus(PM.TST)
       if Models_Status==1:
              print(UI.TimeStamp(),bcolors.WARNING+'Warning, the model seems to be over saturated'+bcolors.ENDC)
              Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
              print(UI.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+Model_Meta_Path+bcolors.ENDC)
              if os.path.isfile(Model_Meta_Path):
                   Model_Meta_Raw=UI.PickleOperations(Model_Meta_Path, 'r', 'N/A')
                   print(Model_Meta_Raw[1])
                   Model_Meta=Model_Meta_Raw[0]
                   Header=Model_Meta.TrainSessionsData[0][0]+['Model Parameters','Train Sample ID','LR','Batch Size','Normalised Epochs']
                   New_Data=[Header]
                   Print_New_Data=[] 
                   counter=0
                   print(UI.TimeStamp(),bcolors.OKGREEN+'The model training profile is printed bellow: '+bcolors.ENDC)
                   for TSD in range(len(Model_Meta.TrainSessionsData)):
                       for Record in Model_Meta.TrainSessionsData[TSD][1:]:
                           counter+=1
                           New_Data.append(Record+[Model_Meta.ModelParameters,Model_Meta.TrainSessionsDataID[TSD],Model_Meta.TrainSessionsParameters[TSD][0],Model_Meta.TrainSessionsParameters[TSD][1],counter])
                           Print_New_Data.append(Record)
                   print(pd.DataFrame(Print_New_Data, columns=Model_Meta.TrainSessionsData[0][0]))
                   Model_Meta_csv=EOSsubModelDIR+'/'+args.ModelName+'_Out.csv'
                   UI.LogOperations(Model_Meta_csv,'w', New_Data)
                   print(UI.TimeStamp(),bcolors.OKGREEN+'Csv output has been saved as '+bcolors.ENDC+bcolors.OKBLUE+Model_Meta_csv+bcolors.ENDC)
              print(bcolors.BOLD+'If you would like to stop training and exit please enter E'+bcolors.ENDC)
              print(bcolors.BOLD+'If you would like to resubmit your script and exit enter R'+bcolors.ENDC)
              print(bcolors.BOLD+'If you would like to continue training on autopilot please type waiting time in minutes'+bcolors.ENDC)
              UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
              if UserAnswer=='E':
                  print(UI.TimeStamp(),'OK, exiting now then')
                  exit()
              if UserAnswer=='R':
                 Model_Meta.IniTrainingSession(TrainSampleID, datetime.datetime.now(), TrainParams)
                 print(UI.PickleOperations(Model_Meta_Path, 'w', Model_Meta)[1])
                 UI.SubmitJobs2Condor(Job,LocalSub,CPU,JobFlavour,Memory)
                 print(bcolors.BOLD+"The job has been submitted..."+bcolors.ENDC)
                 print(UI.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
                 exit()
              else:
                 Model_Meta.IniTrainingSession(TrainSampleID, datetime.datetime.now(), TrainParams)
                 print(UI.PickleOperations(Model_Meta_Path, 'w', Model_Meta)[1])
                 UI.SubmitJobs2Condor(Job,LocalSub,CPU,JobFlavour,Memory)
                 print(bcolors.BOLD+"The job has been submitted..."+bcolors.ENDC)
                 print(UI.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
                 AutoPilot(int(UserAnswer),Wait)
       elif Models_Status==2:
                 print(UI.TimeStamp(),bcolors.OKGREEN+'The training session has been completed'+bcolors.ENDC)
                 Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
                 print(UI.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+Model_Meta_Path+bcolors.ENDC)
                 if os.path.isfile(Model_Meta_Path):
                   Model_Meta_Raw=UI.PickleOperations(Model_Meta_Path, 'r', 'N/A')
                   print(Model_Meta_Raw[1])
                   Model_Meta=Model_Meta_Raw[0]
                   Header=Model_Meta.TrainSessionsData[0][0]+['Model Parameters','Train Sample ID','LR','Batch Size','Normalised Epochs']
                   New_Data=[Header]
                   Print_New_Data=[] 
                   counter=0
                   print(UI.TimeStamp(),bcolors.OKGREEN+'The model training profile is printed bellow: '+bcolors.ENDC)
                   for TSD in range(len(Model_Meta.TrainSessionsData)):
                       for Record in Model_Meta.TrainSessionsData[TSD][1:]:
                           counter+=1
                           New_Data.append(Record+[Model_Meta.ModelParameters,Model_Meta.TrainSessionsDataID[TSD],Model_Meta.TrainSessionsParameters[TSD][0],Model_Meta.TrainSessionsParameters[TSD][1],counter])
                           Print_New_Data.append(Record)
                   print(pd.DataFrame(Print_New_Data, columns=Model_Meta.TrainSessionsData[0][0]))
                   Model_Meta_csv=EOSsubModelDIR+'/'+args.ModelName+'_Out.csv'
                   UI.LogOperations(Model_Meta_csv,'w', New_Data)
                   print(UI.TimeStamp(),bcolors.OKGREEN+'Csv output has been saved as '+bcolors.ENDC+bcolors.OKBLUE+Model_Meta_csv+bcolors.ENDC)
                 print(bcolors.BOLD+'If you would like to stop training and exit please enter E'+bcolors.ENDC)
                 print(bcolors.BOLD+'If you would like to submit another one and exit enter S'+bcolors.ENDC)
                 print(bcolors.BOLD+'If you would like to continue training on autopilot please type waiting time in minutes'+bcolors.ENDC)
                 UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
                 if UserAnswer=='E':
                  print(UI.TimeStamp(),'OK, exiting now then')
                  exit()
                 elif UserAnswer=='S':
                    HTCondorTag="SoftUsed == \"ANNDEA-M2-"+ModelName+"\""
                    Model_Meta.IniTrainingSession(TrainSampleID, datetime.datetime.now(), TrainParams)
                    print(UI.PickleOperations(Model_Meta_Path, 'w', Model_Meta)[1])
                    UI.SubmitJobs2Condor(Job,LocalSub,CPU,JobFlavour,Memory)
                    print(bcolors.BOLD+"The job has been submitted..."+bcolors.ENDC)
                 else:
                     HTCondorTag="SoftUsed == \"ANNDEA-M2-"+ModelName+"\""
                     Model_Meta.IniTrainingSession(TrainSampleID, datetime.datetime.now(), TrainParams)
                     print(UI.PickleOperations(Model_Meta_Path, 'w', Model_Meta)[1])
                     UI.SubmitJobs2Condor(Job,LocalSub,CPU,JobFlavour,Memory)
                     print(bcolors.BOLD+"The job has been submitted..."+bcolors.ENDC)
                     AutoPilot(int(UserAnswer),Wait)

       else:
           HTCondorTag="SoftUsed == \"ANNDEA-M2-"+ModelName+"\""
           q=schedd.query(constraint=HTCondorTag,projection=['CLusterID','JobStatus'])
           if len(q)==0:
                print(UI.TimeStamp(),bcolors.FAIL+'HTCondor has failed...'+bcolors.ENDC)
           else:
                print(UI.TimeStamp(),bcolors.WARNING+'Warning, the training is still running on HTCondor, the job parameters are listed bellow:'+bcolors.ENDC)
                print(q)
           print(bcolors.BOLD+'If you would like to wait and exit please enter E'+bcolors.ENDC)
           print(bcolors.BOLD+'If you would like to resubmit your script enter R'+bcolors.ENDC)
           print(bcolors.BOLD+'If you would like to put the script on the autopilot type number of minutes to wait'+bcolors.ENDC)
           UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
           if UserAnswer=='E':
                  print(UI.TimeStamp(),'OK, exiting now then')
                  exit()
           elif UserAnswer=='R':
                 UI.SubmitJobs2Condor(Job,LocalSub,CPU,JobFlavour,Memory)
                 print(bcolors.BOLD+"The job has been submitted..."+bcolors.ENDC)
                 print(UI.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
                 exit()
           else:
              AutoPilot(int(UserAnswer),Wait)
     else:
                 print(UI.TimeStamp(),bcolors.WARNING+'Warning! No existing meta files have been found, starting everything from the scratch.'+bcolors.ENDC)
                 HTCondorTag="SoftUsed == \"ANNDEA-M2-"+ModelName+"\""
                 TrainSampleInputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
                 print(UI.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+TrainSampleInputMeta+bcolors.ENDC)
                 MetaInput=UI.PickleOperations(TrainSampleInputMeta,'r', 'N/A')
                 print(MetaInput[1])
                 Meta=MetaInput[0]
                 Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
                 ModelMeta=ML.ModelMeta(ModelName)
                 if ModelType=='CNN':
                    ModelMeta.IniModelMeta(ModelParams, 'Tensorflow', Meta, ModelArchitecture, 'CNN')
                 elif ModelType=='GNN':
                    ModelMeta.IniModelMeta(ModelParams, 'PyTorch', Meta, ModelArchitecture, 'GNN')
                 ModelMeta.IniTrainingSession(TrainSampleID, datetime.datetime.now(), TrainParams)
                 print(UI.PickleOperations(Model_Meta_Path, 'w', ModelMeta)[1])
                 UI.SubmitJobs2Condor(Job,LocalSub,CPU,JobFlavour,Memory)
                 print(bcolors.BOLD+"The job has been submitted..."+bcolors.ENDC)
                 AutoPilot(600,Wait)
print(UI.TimeStamp(),bcolors.OKGREEN+'Training is finished then, thank you and goodbye'+bcolors.ENDC)
print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
exit()


