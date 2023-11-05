###This file contains the standard UI utility functions that are commonly used in ANNDEA packages

#Libraries used
import csv
import os
import subprocess
import datetime
import numpy as np
import shutil
import time

#Graphic section

#Use to give colour to the messages
class bc:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#This utility provides Timestamps for print messages
def TimeStamp():
 return "["+datetime.datetime.now().strftime("%D")+' '+datetime.datetime.now().strftime("%H:%M:%S")+"]"

#This utility provides automates welcome messages
def WelcomeMsg(Title, Help):
    ANNDEA_logo='Welcome to ANNDEA'
    line=''
    title=''
    separator='-------------------'
    for l in ANNDEA_logo:
        line+=l
        print(bc.HEADER+line+bc.ENDC, end="\r",flush=True)
        print(bc.HEADER+separator+bc.ENDC, end="\r",flush=True)
        time.sleep(0.1)
    for t in Title:
        title+=t
        print(bc.HEADER+title+bc.ENDC, end="\r",flush=True)
        time.sleep(0.1)



def CleanFolder(folder,key):
    if key=='':
      for the_file in os.listdir(folder):
                file_path=os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
    else:
      for the_file in os.listdir(folder):
                file_path=os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path) and (key in the_file):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
#This function automates csv read/write operations

def LogOperations(flocation,mode, message):
    if mode=='a':
        csv_writer_log=open(flocation,"a")
        log_writer = csv.writer(csv_writer_log)
        if len(message)>0:
         for m in message:
          log_writer.writerow(m)
        csv_writer_log.close()
    if mode=='w':
        csv_writer_log=open(flocation,"w")
        log_writer = csv.writer(csv_writer_log)
        if len(message)>0:
         for m in message:
           log_writer.writerow(m)
        csv_writer_log.close()
    if mode=='r':
        csv_reader_log=open(flocation,"r")
        log_reader = csv.reader(csv_reader_log)
        return list(log_reader)

def PickleOperations(flocation,mode, message):
    import pickle
    if mode=='w':
        pickle_writer_log=open(flocation,"wb")
        pickle.dump(message, pickle_writer_log)
        pickle_writer_log.close()
        return ('',"UF.PickleOperations Message: Data has been written successfully into "+flocation)
    if mode=='r':
        pickle_writer_log=open(flocation,'rb')
        result=pickle.load(pickle_writer_log)
        pickle_writer_log.close()
        return (result,"UF.PickleOperations Message: Data has been loaded successfully from "+flocation)

def RecCleanUp(AFS_DIR, EOS_DIR, Process, FileNames, ProcessId):
      subprocess.call(['condor_rm', '-constraint', ProcessId])
      EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
      EOSsubModelDIR=EOSsubDIR+'/'+'Data/REC_SET'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      folder =  AFS_DIR+'/HTCondor/SH'
      CleanFolder(folder,'SH_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/SUB'
      CleanFolder(folder,'SUB_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/MSG'
      CleanFolder(folder,'MSG_'+Process+'_')

def EvalCleanUp(AFS_DIR, EOS_DIR, Process, FileNames, ProcessId):
      subprocess.call(['condor_rm', '-constraint', ProcessId])
      EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
      EOSsubModelDIR=EOSsubDIR+'/'+'Data/TEST_SET'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      folder =  AFS_DIR+'/HTCondor/SH'
      CleanFolder(folder,'SH_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/SUB'
      CleanFolder(folder,'SUB_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/MSG'
      CleanFolder(folder,'MSG_'+Process+'_')

def TrainCleanUp(AFS_DIR, EOS_DIR, Process, FileNames, ProcessId):
      subprocess.call(['condor_rm', '-constraint', ProcessId])
      EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
      EOSsubModelDIR=EOSsubDIR+'/'+'Data/TRAIN_SET'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      EOSsubModelDIR=EOSsubDIR+'/'+'Models'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      folder =  AFS_DIR+'/HTCondor/SH'
      CleanFolder(folder,'SH_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/SUB'
      CleanFolder(folder,'SUB_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/MSG'
      CleanFolder(folder,'MSG_'+Process+'_')

def CreateCondorJobs(AFS,EOS,PY,path,o,pfx,sfx,ID,loop_params,OptionHeader,OptionLine,Sub_File,batch_sub=False,Exception=['',''], Log=False, GPU=False):
   if Exception[0]==" --PlateZ ":
    if batch_sub==False:
        from alive_progress import alive_bar
        bad_pop=[]
        TotJobs=0
        if type(loop_params) is int:
            nest_lvl=1
            TotJobs=loop_params
        elif type(loop_params[0]) is int:
            nest_lvl=2
            TotJobs=np.sum(loop_params)
        elif type(loop_params[0][0]) is int:
            nest_lvl=3
            for lp in loop_params:
                TotJobs+=np.sum(lp)
        OH=OptionHeader+[' --EOS '," --AFS ", " --PY ", " --BatchID "]
        OL=OptionLine+[EOS, AFS, PY, ID]
        TotJobs=int(TotJobs)

        with alive_bar(TotJobs,force_tty=True, title='Checking the results from HTCondor') as bar:
             if nest_lvl==2:
                 for i in range(len(loop_params)):
                     for j in range(int(loop_params[i])):

                               required_output_file_location=EOS+'/'+path+'/Temp_'+pfx+'_'+ID+'_'+str(i)+'/'+pfx+'_'+ID+'_'+o+'_'+str(i)+'_'+str(j)+sfx
                               bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               if os.path.isfile(required_output_file_location)!=True:
                                  bad_pop.append([OH+[' --i ', ' --j ', ' --p ', ' --o ',' --pfx ', ' --sfx ', Exception[0]], OL+[i, j, path,o, pfx, sfx, Exception[1][i][0]], SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-'+pfx+'-'+ID, Log, GPU])

             if nest_lvl==3:
                 for i in range(len(loop_params)):
                     for j in range(len(loop_params[i])):
                         for k in range(loop_params[i][j]):
                               required_output_file_location=EOS+'/'+path+'/Temp_'+pfx+'_'+ID+'_'+str(i)+'/'+pfx+'_'+ID+'_'+o+'_'+str(i)+'_'+str(j)+sfx
                               bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '_' + str(k) +'.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '_' + str(k) +'.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '_' + str(k)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               if os.path.isfile(required_output_file_location)!=True:
                                  bad_pop.append([OH+[' --i ', ' --j ',' --k ', ' --p ', ' --o ',' --pfx ', ' --sfx ', Exception[0]], OL+[i, j,k, path,o, pfx, sfx, Exception[1][i][0]], SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
        return(bad_pop)
    else:
        from alive_progress import alive_bar
        bad_pop=[]
        TotJobs=0
        if type(loop_params) is int:
            nest_lvl=1
            TotJobs=loop_params
        elif type(loop_params[0]) is int:
            nest_lvl=2
            TotJobs=np.sum(loop_params)
        elif type(loop_params[0][0]) is int:
            nest_lvl=3
            for lp in loop_params:
                TotJobs+=np.sum(lp)
        OH=OptionHeader+[' --EOS '," --AFS ", " --PY ", " --BatchID "]
        OL=OptionLine+[EOS, AFS, PY, ID]
        TotJobs=int(TotJobs)
        with alive_bar(TotJobs,force_tty=True, title='Checking the results from HTCondor') as bar:
             if nest_lvl==2:
                 for i in range(len(loop_params)):
                               bar.text = f'-> Preparing batch submission...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) + '.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               bad_pop.append([OH+[' --i ', ' --j ', ' --p ', ' --o ',' --pfx ', ' --sfx ', Exception[0]], OL+[i, '$1', path,o, pfx, sfx, Exception[1][i][0]], SHName, SUBName, MSGName, ScriptName, loop_params[i], 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
             if nest_lvl==3:
                 for i in range(len(loop_params)):
                     for j in range(loop_params[i]):
                               bar.text = f'-> Preparing batch submission...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j)+'.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) +'_' + str(j)+'.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i) +'_' + str(j)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               bad_pop.append([OH+[' --i ', ' --j ',' --k ', ' --p ', ' --o ',' --pfx ', ' --sfx ', Exception[0]], OL+[i, j, '$1', path,o, pfx, sfx, Exception[1][i][0]], SHName, SUBName, MSGName, ScriptName, loop_params[i], 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
        return(bad_pop)
   else:
    if batch_sub==False:
        from alive_progress import alive_bar
        bad_pop=[]
        TotJobs=0
        if type(loop_params) is int:
            nest_lvl=1
            TotJobs=loop_params
        elif type(loop_params[0]) is int:
            nest_lvl=2
            TotJobs=np.sum(loop_params)
        elif type(loop_params[0][0]) is int:
            nest_lvl=3
            for lp in loop_params:
                TotJobs+=np.sum(lp)
        OH=OptionHeader+[' --EOS '," --AFS ", " --PY ", " --BatchID "]
        OL=OptionLine+[EOS, AFS, PY, ID]
        TotJobs=int(TotJobs)
        with alive_bar(TotJobs,force_tty=True, title='Checking the results from HTCondor') as bar:
             if nest_lvl==1:
                 for i in range(loop_params):

                               required_output_file_location=EOS+'/'+path+'/Temp_'+pfx+'_'+ID+'_'+str(0)+'/'+pfx+'_'+ID+'_'+o+'_'+str(i)+sfx
                               bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(0)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(0)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) + '.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(0)+'/MSG_'+pfx+'_'+ ID+'_' + str(i)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               if os.path.isfile(required_output_file_location)!=True:
                                  bad_pop.append([OH+[' --i ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OL+[i, path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
             if nest_lvl==2:
                 for i in range(len(loop_params)):
                     for j in range(loop_params[i]):
                               required_output_file_location=EOS+'/'+path+'/Temp_'+pfx+'_'+ID+'_'+str(i)+'/'+pfx+'_'+ID+'_'+o+'_'+str(i)+'_'+str(j)+sfx
                               bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               if os.path.isfile(required_output_file_location)!=True:
                                  bad_pop.append([OH+[' --i ', ' --j ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OL+[i, j, path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])

             if nest_lvl==3:
                 for i in range(len(loop_params)):
                     for j in range(len(loop_params[i])):
                         for k in range(loop_params[i][j]):
                               required_output_file_location=EOS+'/'+path+'/Temp_'+pfx+'_'+ID+'_'+str(i)+'/'+pfx+'_'+ID+'_'+o+'_'+str(i)+'_'+str(j) + '_' + str(k)+sfx
                               bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '_' + str(k) +'.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '_' + str(k) +'.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '_' + str(k)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               if os.path.isfile(required_output_file_location)!=True:
                                  bad_pop.append([OH+[' --i ', ' --j ',' --k ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OL+[i, j,k, path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
        return(bad_pop)
    else:
        from alive_progress import alive_bar
        bad_pop=[]
        TotJobs=0
        if type(loop_params) is int:
            nest_lvl=1
            TotJobs=loop_params
        elif type(loop_params[0]) is int:
            nest_lvl=2
            TotJobs=np.sum(loop_params)
        elif type(loop_params[0][0]) is int:
            nest_lvl=3
            for lp in loop_params:
                TotJobs+=np.sum(lp)
        OH=OptionHeader+[' --EOS '," --AFS ", " --PY ", " --BatchID "]
        OL=OptionLine+[EOS, AFS, PY, ID]
        TotJobs=int(TotJobs)
        with alive_bar(TotJobs,force_tty=True, title='Checking the results from HTCondor') as bar:
             if nest_lvl==2:
                 for i in range(len(loop_params)):
                               bar.text = f'-> Preparing batch submission...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) + '.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               bad_pop.append([OH+[' --i ', ' --j ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OL+[i, '$1', path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, loop_params[i], 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
             if nest_lvl==3:
                 for i in range(len(loop_params)):
                     for j in range(len(loop_params[i])):
                               bar.text = f'-> Preparing batch submission...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+ '_' + ID+ '_' + str(i) + '_' + str(j)+'.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_' +pfx+ '_' + ID+ '_' + str(i) + '_' + str(j)+'.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_' +pfx+'_' + ID+ '_' + str(i) + '_' + str(j)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               bad_pop.append([OH+[' --i ', ' --j ',' --k ', ' --p ', ' --o ',' --pfx ', ' --sfx '],OL+[i, j, '$1', path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, loop_params[i][j], 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
             if nest_lvl==1:
                               bar.text = f'-> Preparing batch submission...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(0)+'/SH_'+pfx+'_'+ ID+'.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_' +pfx+ '_' +ID+ '_' +str(0)+'/SUB_' +pfx+ '_' + ID+'.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(0)+'/MSG_'+pfx+'_' + ID
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               bad_pop.append([OH+[' --i ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OL+['$1', path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, loop_params, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
        return(bad_pop)
   return []

def SubmitJobs2Condor(job,local=False,ExtCPU=1,JobFlavour='workday', ExtMemory='2 GB'):
    if local:
       OptionLine = job[0][0]+str(job[1][0])
       for line in range(1,len(job[0])):
                OptionLine+=job[0][line]
                OptionLine+=str(job[1][line])
                TotalLine = 'python3 ' + job[5] + OptionLine
       submission_line='python3 '+job[5]+OptionLine
       for j in range(0,job[6]):
         act_submission_line=submission_line.replace('$1',str(j))
         subprocess.call([act_submission_line],shell=True)
         print(bcolors.OKGREEN+act_submission_line+" has been successfully executed"+bcolors.ENDC)
    else:
        SHName = job[2]
        SUBName = job[3]
        if job[8]:
            MSGName=job[4]
        OptionLine = job[0][0]+str(job[1][0])
        for line in range(1,len(job[0])):
            OptionLine+=job[0][line]
            OptionLine+=str(job[1][line])
        f = open(SUBName, "w")
        f.write("executable = " + SHName)
        f.write("\n")
        if job[8]:
            f.write("output ="+MSGName+".out")
            f.write("\n")
            f.write("error ="+MSGName+".err")
            f.write("\n")
            f.write("log ="+MSGName+".log")
            f.write("\n")
        f.write('requirements = (CERNEnvironment =!= "qa")')
        f.write("\n")
        if job[9]:
            f.write('request_gpus = 1')
            f.write("\n")
        if ExtCPU>1 and job[9]==False:
            f.write('request_cpus = '+str(ExtCPU))
            f.write("\n")
        f.write('request_memory = '+str(ExtMemory))
        f.write("\n")
        f.write('arguments = $(Process)')
        f.write("\n")
        f.write('+SoftUsed = '+'"'+job[7]+'"')
        f.write("\n")
        f.write('transfer_output_files = ""')
        f.write("\n")
        f.write('+JobFlavour = "'+JobFlavour+'"')
        f.write("\n")
        f.write('queue ' + str(job[6]))
        f.write("\n")
        f.close()
        TotalLine = 'python3 ' + job[5] + OptionLine
        f = open(SHName, "w")
        f.write("#!/bin/bash")
        f.write("\n")
        f.write("set -ux")
        f.write("\n")
        f.write(TotalLine)
        f.write("\n")
        f.close()
        subprocess.call(['condor_submit', SUBName])
        print(TotalLine, " has been successfully submitted")

def ManageTempFolders(spi,op_type):
    if type(spi[1][8]) is int:
       _tot=spi[1][8]
    else:
       _tot=len(spi[1][8])
    if op_type=='Create':
       if type(spi[1][8]) is int:
           try:
              os.mkdir(spi[1][1]+spi[1][3]+'Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0))
           except OSError as error:
               print(bcolors.WARNING+spi[1][1]+spi[1][3]+'Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0)+" already exists"+bcolors.ENDC)
           try:
              os.mkdir(spi[1][0]+'/HTCondor/SUB/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0))
           except OSError as error:
               print(bcolors.WARNING+spi[1][0]+'/HTCondor/SUB/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0)+" already exists"+bcolors.ENDC)
           try:
              os.mkdir(spi[1][0]+'/HTCondor/SH/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0))
           except OSError as error:
               print(bcolors.WARNING+spi[1][0]+'/HTCondor/SH/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0)+" already exists"+bcolors.ENDC)
           try:
              os.mkdir(spi[1][0]+'/HTCondor/MSG/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0))

           except OSError as error:
               print(bcolors.WARNING+spi[1][0]+'/HTCondor/MSG/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0)+" already exists"+bcolors.ENDC)
       else:

           for i in range(_tot):
               try:
                  os.mkdir(spi[1][1]+spi[1][3]+'Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i))
               except OSError as error:
                  continue
               try:
                  os.mkdir(spi[1][0]+'/HTCondor/SUB/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i))
               except OSError as error:
                  continue
               try:
                  os.mkdir(spi[1][0]+'/HTCondor/SH/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i))
               except OSError as error:
                  continue
               try:
                  os.mkdir(spi[1][0]+'/HTCondor/MSG/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i))
               except OSError as error:
                  continue
       return 'Temporary folders have been created'
    if op_type=='Delete':
       if (type(spi[1][8]) is int):
           shutil.rmtree(spi[1][1]+spi[1][3]+'Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0),True)
           shutil.rmtree(spi[1][0]+'/HTCondor/SUB/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0),True)
           shutil.rmtree(spi[1][0]+'/HTCondor/SH/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0),True)
           shutil.rmtree(spi[1][0]+'/HTCondor/MSG/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0),True)
       else:
           for i in range(_tot):
               shutil.rmtree(spi[1][1]+spi[1][3]+'Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i),True)
               shutil.rmtree(spi[1][0]+'/HTCondor/SUB/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i),True)
               shutil.rmtree(spi[1][0]+'/HTCondor/SH/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i),True)
               shutil.rmtree(spi[1][0]+'/HTCondor/MSG/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i),True)
       return 'Temporary folders have been deleted'

