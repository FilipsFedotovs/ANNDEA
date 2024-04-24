###This file contains the standard UI utility functions that are commonly used in ANNDEA packages

#Libraries used
import csv
import os
import subprocess
import datetime
import numpy as np
import shutil
import time
import math
from alive_progress import alive_bar
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
    GREY= '\033[90m'

#This utility provides Timestamps for print messages
def TimeStamp():
 return "["+datetime.datetime.now().strftime("%D")+' '+datetime.datetime.now().strftime("%H:%M:%S")+"]"
def PrintLine(Content):
    line='   '
    for l in range(len(Content)):
        line+=Content[l]
        if l<len(Content)-1:
            print(bc.HEADER+line+bc.ENDC, end="\r",flush=True)
        else:
            print(bc.HEADER+line+bc.ENDC)
        time.sleep(0.02)
    print('                                                                                                                                    ')
#This utility provides automates welcome messages
def WelcomeMsg(Title, Author, Contact):
    ANNDEA_logo='Welcome to ANNDEA'
    print('                                                                                                                                    ')
    print('                                                                                                                                    ')
    print(bc.HEADER+"########################################################################################################"+bc.ENDC)
    print('                                                                                                                                    ')
    PrintLine(ANNDEA_logo)
    PrintLine('Developed by: '+Author)
    PrintLine(Contact)
    PrintLine(Title)
    print('                                                                                                                                    ')
    print(bc.HEADER+"########################################################################################################"+bc.ENDC)
    print('                                                                                                                                    ')
    print('                                                                                                                                    ')
class TrainingSampleMeta:
      def __init__(self,TrainDataID):
          self.TrainDataID=TrainDataID
      def __eq__(self, other):
        return (self.TrainDataID) == (other.TrainDataID)
      def __hash__(self):
        return hash(self.TrainDataID)
      def IniHitClusterMetaData(self,stepX,stepY,stepZ,cut_dt,cut_dr,testRatio,valRatio,z_offset,y_offset,x_offset, Xsteps, Zsteps,X_overlap,Y_overlap,Z_overlap):
          self.stepX=stepX
          self.stepY=stepY
          self.stepZ=stepZ
          self.cut_dt=cut_dt
          self.cut_dr=cut_dr
          self.testRatio=testRatio
          self.valRatio=valRatio
          self.z_offset=z_offset
          self.y_offset=y_offset
          self.x_offset=x_offset
          self.Xsteps=Xsteps
          self.Zsteps=Zsteps
          self.X_overlap=X_overlap
          self.Y_overlap=Y_overlap
          self.Z_overlap=Z_overlap
      def IniTrackMetaData(self,ClassHeaders,ClassNames,ClassValues,MaxSegments,JobSets,MinHitsTrack):
          self.ClassHeaders=ClassHeaders
          self.ClassNames=ClassNames
          self.ClassValues=ClassValues
          self.MaxSegments=MaxSegments
          self.JobSets=JobSets
          self.MinHitsTrack=MinHitsTrack
      def IniTrackSeedMetaData(self,MaxSLG,MaxSTG,MaxDOCA,MaxAngle,JobSets,MaxSegments,VetoMotherTrack,MaxSeeds,MinHitsTrack
                               ):
          self.MaxSLG=MaxSLG
          self.MaxSTG=MaxSTG
          self.MaxDOCA=MaxDOCA
          self.MaxAngle=MaxAngle
          self.JobSets=JobSets
          self.MaxSegments=MaxSegments
          self.MaxSeeds=MaxSeeds
          self.VetoMotherTrack=VetoMotherTrack
          self.MinHitsTrack=MinHitsTrack
      def IniBrickAlignMetaData(self,Size,ValMinHits,MinHits,SpatialOptBound,AngularOptBound,JobSets,Cycles,plates,FiducialVolume
                               ):
          self.Size=Size
          self.MinHits=MinHits
          self.ValMinHits=ValMinHits
          self.SpatialOptBound=SpatialOptBound
          self.AngularOptBound=AngularOptBound
          self.Cycles=Cycles
          self.JobSets=JobSets
          self.plates=plates
          self.FiducialVolume=FiducialVolume
      def IniVertexSeedMetaData(self,MaxDST,MaxVXT,MaxDOCA,MaxAngle,JobSets,MaxSegments,MaxSeeds,MinHitsTrack,FiducialVolumeCut,ExcludeClassNames,ExcludeClassValues
                               ):
          self.MaxDST=MaxDST
          self.MaxVXT=MaxVXT
          self.MaxDOCA=MaxDOCA
          self.MaxAngle=MaxAngle
          self.JobSets=JobSets
          self.MaxSegments=MaxSegments
          self.MaxSeeds=MaxSeeds
          self.ClassNames=ExcludeClassNames
          self.ClassValues=ExcludeClassValues
          self.MinHitsTrack=MinHitsTrack
          self.FiducialVolumeCut = FiducialVolumeCut
      def UpdateHitClusterMetaData(self,NoS,NoNF,NoEF,NoSets):
          self.num_node_features=NoNF
          self.num_edge_features=NoEF
          self.tot_sample_size=NoS
          self.no_sets=NoSets
      def UpdateStatus(self, status):
          if hasattr(self,'Status'):
            self.Status.append(status)
          else:
            self.Status=[status]
def Msg(type,content,content2='',content3=''):
      if type=='status':
          print(TimeStamp(),bc.BOLD+str(content)+' '+bc.ENDC+str(content2))
      if type=='location':
          print(TimeStamp(),content,bc.OKBLUE+content2+bc.ENDC)
      if type=='success':
          print(TimeStamp(),content,bc.OKBLUE+content2+bc.ENDC)
      if type=='result':
          print(TimeStamp(),content,bc.BOLD+str(content2)+bc.ENDC,content3)
      if type=='vanilla':
          print(TimeStamp(),content)
      if type=='comleted':
         print(TimeStamp(),bc.OKGREEN+content+bc.ENDC)
      if type=='failed':
         print(TimeStamp(),bc.FAIL+content+bc.ENDC)
def UpdateStatus(status,meta,output):
    meta.UpdateStatus(status)
    print(PickleOperations(output,'w', meta)[1])

def CalculateNJobs(JobSet):
            if type(JobSet) is int:
                        return 1,JobSet
            elif type(JobSet[0]) is int:
                        return 2,np.sum(JobSet)
            return 0,0
def CreateCondorJobs(AFS,EOS,PY,path,o,pfx,sfx,ID,loop_params,OptionHeader,OptionLine,Sub_File,batch_sub=False,PlaceHolder='N/A', Log=False, GPU=False):
    if batch_sub==False:
        from alive_progress import alive_bar
        bad_pop=[]
        TotJobs=CalculateNJobs(loop_params)[1]
        nest_lvl=CalculateNJobs(loop_params)[0]
        OH=OptionHeader+[' --EOS '," --AFS ", " --PY ", " --BatchID "]
        OL=OptionLine+[EOS, AFS, PY, ID]
        TotJobs=int(TotJobs)
        with alive_bar(TotJobs,force_tty=True, title='Checking the results from HTCondor') as bar:
             if nest_lvl==1:
                 for i in range(loop_params):

                               required_output_file_location=EOS+'/'+path+'/Temp_'+pfx+'_'+ID+'_'+str(0)+'/'+pfx+'_'+ID+'_'+o+'_'+str(i)+sfx
                               bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/'+ID+'/Temp_'+pfx+'_'+ID+'_'+str(0)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '.sh'
                               SUBName = AFS + '/HTCondor/SUB/'+ID+'/Temp_'+pfx+'_'+ID+'_'+str(0)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) + '.sub'
                               MSGName = AFS + '/HTCondor/MSG/'+ID+'/Temp_'+pfx+'_'+ID+'_'+str(0)+'/MSG_'+pfx+'_'+ ID+'_' + str(i)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               if os.path.isfile(required_output_file_location)!=True:
                                  bad_pop.append([OH+[' --i ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OL+[i, path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
             if nest_lvl==2:
                 for i in range(len(loop_params)):
                     for j in range(loop_params[i]):
                               required_output_file_location=EOS+'/'+path+'/Temp_'+pfx+'_'+ID+'_'+str(i)+'/'+pfx+'_'+ID+'_'+o+'_'+str(i)+'_'+str(j)+sfx
                               bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/'+ID+'/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '.sh'
                               SUBName = AFS + '/HTCondor/SUB/'+ID+'/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '.sub'
                               MSGName = AFS + '/HTCondor/MSG/'+ID+'/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               if os.path.isfile(required_output_file_location)!=True:
                                  bad_pop.append([OH+[' --i ', ' --j ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OL+[i, j, path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
        return(bad_pop)
    else:
        from alive_progress import alive_bar
        bad_pop=[]
        TotJobs=0
        TotJobs=CalculateNJobs(loop_params)[1]
        nest_lvl=CalculateNJobs(loop_params)[0]
        OH=OptionHeader+[' --EOS '," --AFS ", " --PY ", " --BatchID "]
        OL=OptionLine+[EOS, AFS, PY, ID]
        TotJobs=int(TotJobs)
        with alive_bar(TotJobs,force_tty=True, title='Checking the results from HTCondor') as bar:
             if nest_lvl==2:
                 for i in range(len(loop_params)):
                               bar.text = f'-> Preparing batch submission...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/'+ID+'/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '.sh'
                               SUBName = AFS + '/HTCondor/SUB/'+ID+'/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) + '.sub'
                               MSGName = AFS + '/HTCondor/MSG/'+ID+'/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               bad_pop.append([OH+[' --i ', ' --j ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OL+[i, '$1', path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, loop_params[i], 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
             if nest_lvl==1:
                               bar.text = f'-> Preparing batch submission...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/'+ID+'/Temp_'+pfx+'_'+ID+'_'+str(0)+'/SH_'+pfx+'_'+ ID+'.sh'
                               SUBName = AFS + '/HTCondor/SUB/'+ID+'/Temp_' +pfx+ '_' +ID+ '_' +str(0)+'/SUB_' +pfx+ '_' + ID+'.sub'
                               MSGName = AFS + '/HTCondor/MSG/'+ID+'/Temp_'+pfx+'_'+ID+'_'+str(0)+'/MSG_'+pfx+'_' + ID
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
         print(bc.OKGREEN+act_submission_line+" has been successfully executed"+bc.ENDC)
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
def AutoPilot(wait_min, interval_min, max_interval_tolerance,program,RequestExtCPU,JobFlavour,ReqMemory):
     print(TimeStamp(),'Going on an autopilot mode for ',wait_min, 'minutes while checking HTCondor every',interval_min,'min',bc.ENDC)
     wait_sec=wait_min*60
     interval_sec=interval_min*60
     intervals=int(math.ceil(wait_sec/interval_sec))
     for interval in range(1,intervals+1):
         time.sleep(interval_sec)
         print(TimeStamp(),"Scheduled job checkup...") #Progress display
         bad_pop=CreateCondorJobs(program[1][0],
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
                                    False,
                                    program[6],
                                    program[7],
                                    program[8])
         if len(bad_pop)>0:
               print(TimeStamp(),bc.WARNING+'Autopilot status update: There are still', len(bad_pop), 'HTCondor jobs remaining'+bc.ENDC)
               if interval%max_interval_tolerance==0:
                  for bp in bad_pop:
                      SubmitJobs2Condor(bp,program[5],RequestExtCPU,JobFlavour,ReqMemory)
                  print(TimeStamp(), bc.OKGREEN+"All jobs have been resubmitted"+bc.ENDC)
         else:
              return True,False
     return False,False
def StandardProcess(program,status,SubGap,SubPause,RequestExtCPU,JobFlavour,ReqMemory,time_int,Patience):
        print(bc.HEADER+"#############################################################################################"+bc.ENDC)
        print(TimeStamp(),bc.BOLD+'Stage '+str(status)+':'+bc.ENDC+str(program[status][0]))
        batch_sub=program[status][4]>1
        bad_pop=CreateCondorJobs(program[status][1][0],
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
                                    False,
                                    program[status][6],
                                    program[status][7],
                                    program[status][8])
        if len(bad_pop)==0:
             print(TimeStamp(),bc.OKGREEN+'Stage '+str(status)+' has successfully completed'+bc.ENDC)
             return True,False
        elif (program[status][4])==len(bad_pop):
                 bad_pop=CreateCondorJobs(program[status][1][0],
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
                                    batch_sub,
                                    program[status][6],
                                    program[status][7],
                                    program[status][8])
                 print(TimeStamp(),'Submitting jobs to HTCondor... ',bc.ENDC)
                 _cnt=0
                 for bp in bad_pop:
                          if _cnt>SubGap:
                              print(TimeStamp(),'Pausing submissions for  ',str(int(SubPause/60)), 'minutes to relieve congestion...',bc.ENDC)
                              time.sleep(SubPause)
                              _cnt=0
                          SubmitJobs2Condor(bp,program[status][5],RequestExtCPU,JobFlavour,ReqMemory)
                          _cnt+=bp[6]
                 if program[status][5]:
                    print(TimeStamp(),bc.OKGREEN+'Stage '+str(status)+' has successfully completed'+bc.ENDC)
                    return True,False
                 elif AutoPilot(600,time_int,Patience,program[status],RequestExtCPU, JobFlavour,ReqMemory):
                        print(TimeStamp(),bc.OKGREEN+'Stage '+str(status)+' has successfully completed'+bc.ENDC)
                        return True,False
                 else:
                        print(TimeStamp(),bc.FAIL+'Stage '+str(status)+' is uncompleted...'+bc.ENDC)
                        return False,False
        elif len(bad_pop)>0:
            # if freshstart:
                   print(TimeStamp(),bc.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bc.ENDC)
                   print(bc.BOLD+'If you would like to wait and exit please enter E'+bc.ENDC)
                   print(bc.BOLD+'If you would like to wait please enter enter the maximum wait time in minutes'+bc.ENDC)
                   print(bc.BOLD+'If you would like to resubmit please enter R'+bc.ENDC)
                   UserAnswer=input(bc.BOLD+"Please, enter your option\n"+bc.ENDC)
                   if UserAnswer=='E':
                       print(TimeStamp(),'OK, exiting now then')
                       exit()
                   if UserAnswer=='R':
                      _cnt=0
                      for bp in bad_pop:
                           if _cnt>SubGap:
                              print(TimeStamp(),'Pausing submissions for  ',str(int(SubPause/60)), 'minutes to relieve congestion...',bc.ENDC)
                              time.sleep(SubPause)
                              _cnt=0
                           SubmitJobs2Condor(bp,program[status][5],RequestExtCPU,JobFlavour,ReqMemory)
                           _cnt+=bp[6]
                      print(TimeStamp(), bc.OKGREEN+"All jobs have been resubmitted"+bc.ENDC)
                      if program[status][5]:
                          print(TimeStamp(),bc.OKGREEN+'Stage '+str(status)+' has successfully completed'+bc.ENDC)
                          return True,False
                      elif AutoPilot(600,time_int,Patience,program[status],RequestExtCPU, JobFlavour,ReqMemory):
                          print(TimeStamp(),bc.OKGREEN+'Stage '+str(status)+ 'has successfully completed'+bc.ENDC)
                          return True,False
                      else:
                          print(TimeStamp(),bc.FAIL+'Stage '+str(status)+' is uncompleted...'+bc.ENDC)
                          return False,False
                   else:
                      if program[status][5]:
                          print(TimeStamp(),bc.OKGREEN+'Stage '+str(status)+' has successfully completed'+bc.ENDC)
                          return True,False
                      elif AutoPilot(int(UserAnswer),time_int,Patience,program[status],RequestExtCPU, JobFlavour,ReqMemory):
                          print(TimeStamp(),bc.OKGREEN+'Stage '+str(status)+ 'has successfully completed'+bc.ENDC)
                          return True,False
                      else:
                          print(TimeStamp(),bc.FAIL+'Stage '+str(status)+' is uncompleted...'+bc.ENDC)
                          return False,False

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
        if type(message) is list:
            return ('',"PickleOperations Message: "+str(len(message))+" records have(s) been written successfully into "+flocation)
        else:
            return ('',"PickleOperations Message: 1 item has been written successfully into "+flocation)
    if mode=='r':
        pickle_writer_log=open(flocation,'rb')
        result=pickle.load(pickle_writer_log)
        pickle_writer_log.close()
        if type(result) is list:
            return (result,"PickleOperations Message: "+str(len(result))+" records have(s) been loaded successfully from "+flocation)
        else:
            return (result,"PickleOperations Message: 1 item has been loaded successfully from "+flocation)

def ManageFolders(AFS_DIR, EOS_DIR, BatchID,op_type,HTCondorJobs=[]):
    if op_type=='c':
       EOSsubDIR=EOS_DIR+'/'+'ANNDEA/Data/'
       try:
          os.mkdir(EOSsubDIR+'REC_SET/'+BatchID)
       except OSError as error:
          print(bc.WARNING+EOSsubDIR+'REC_SET/'+BatchID+" already exists"+bc.ENDC)
       try:
          os.mkdir(EOSsubDIR+'TEST_SET/'+BatchID)
       except OSError as error:
          print(bc.WARNING+EOSsubDIR+'TEST_SET/'+BatchID+" already exists"+bc.ENDC)
       try:
          os.mkdir(EOSsubDIR+'TRAIN_SET/'+BatchID)
       except OSError as error:
          print(bc.WARNING+EOSsubDIR+'TRAIN_SET/'+BatchID+" already exists"+bc.ENDC)
       HTCondorDir =  AFS_DIR+'/HTCondor/'
       try:
          os.mkdir(HTCondorDir+'SH/'+BatchID)
       except OSError as error:
          print(bc.WARNING+HTCondorDir+'SH/'+BatchID+" already exists"+bc.ENDC)
       try:
          os.mkdir(HTCondorDir+'SUB/'+BatchID)
       except OSError as error:
          print(bc.WARNING+HTCondorDir+'SUB/'+BatchID+" already exists"+bc.ENDC)
       try:
          os.mkdir(HTCondorDir+'MSG/'+BatchID)
       except OSError as error:
          print(bc.WARNING+HTCondorDir+'MSG/'+BatchID+" already exists"+bc.ENDC)
       return 'Main folders for the reconstruction job '+BatchID+' have been created'

    if op_type=='d':
       Msg('vanilla','Performing the cleanup... ')
       EOSsubDIR=EOS_DIR+'/'+'ANNDEA/Data/'
       shutil.rmtree(EOSsubDIR+'REC_SET/'+BatchID,True)
       shutil.rmtree(EOSsubDIR+'TEST_SET/'+BatchID,True)
       shutil.rmtree(EOSsubDIR+'TRAIN_SET/'+BatchID,True)
       HTCondorDir =  AFS_DIR+'/HTCondor/'
       shutil.rmtree(HTCondorDir+'SH/'+BatchID,True)
       shutil.rmtree(HTCondorDir+'SUB/'+BatchID,True)
       shutil.rmtree(HTCondorDir+'MSG/'+BatchID,True)
       for HTJ in HTCondorJobs:
           HTCondorTag="SoftUsed == \"ANNDEA-"+HTJ+"-"+BatchID+"\""
           subprocess.call(['condor_rm', '-constraint', HTCondorTag])
       return 'Main folders for the reconstruction job '+BatchID+' have been deleted'

def ManageTempFolders(spi):
   if type(spi[1][8]) is int:
       _tot=spi[1][8]
   else:
       _tot=len(spi[1][8])

   if type(spi[1][8]) is int:
       try:
          os.mkdir(spi[1][1]+spi[1][3]+'Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0))
       except OSError as error:
           print(bc.WARNING+spi[1][1]+spi[1][3]+'Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0)+" already exists"+bc.ENDC)
       try:
          os.mkdir(spi[1][0]+'/HTCondor/SUB/'+spi[1][7]+'/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0))
       except OSError as error:
           print(bc.WARNING+spi[1][0]+'/HTCondor/SUB/'+spi[1][7]+'/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0)+" already exists"+bc.ENDC)
       try:
          os.mkdir(spi[1][0]+'/HTCondor/SH/'+spi[1][7]+'/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0))
       except OSError as error:
           print(bc.WARNING+spi[1][0]+'/HTCondor/SH/'+spi[1][7]+'/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0)+" already exists"+bc.ENDC)
       try:
          os.mkdir(spi[1][0]+'/HTCondor/MSG/'+spi[1][7]+'/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0))

       except OSError as error:
           print(bc.WARNING+spi[1][0]+'/HTCondor/MSG/'+spi[1][7]+'/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(0)+" already exists"+bc.ENDC)
   else:
       for i in range(_tot):
           try:
              os.mkdir(spi[1][1]+spi[1][3]+'Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i))
           except OSError as error:
              continue
           try:
              os.mkdir(spi[1][0]+'/HTCondor/SUB/'+spi[1][7]+'/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i))
           except OSError as error:
              continue
           try:
              os.mkdir(spi[1][0]+'/HTCondor/SH/'+spi[1][7]+'/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i))
           except OSError as error:
              continue
           try:
              os.mkdir(spi[1][0]+'/HTCondor/MSG/'+spi[1][7]+'/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i))
           except OSError as error:
              continue
   return 'Temporary folders have been created'







