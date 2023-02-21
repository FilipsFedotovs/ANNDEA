import PrintingUtility
from PrintingUtility import bcolors
# temporarily
import UtilityFunctions as UF


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
        OptionHeader+=[' --EOS '," --AFS ", " --PY ", " --BatchID "]
        OptionLine+=[EOS, AFS, PY, ID]
        TotJobs=int(TotJobs)
        with alive_bar(TotJobs,force_tty=True, title='Checking the results from HTCondor') as bar:
             if nest_lvl==2:
                 for i in range(len(loop_params)):
                     for j in range(loop_params[i]):
                               required_output_file_location=EOS+'/'+path+'/'+pfx+'_'+ID+'_'+o+'_'+str(i)+'_'+str(j)+sfx
                               bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/SH_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '.sh'
                               SUBName = AFS + '/HTCondor/SUB/SUB_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '.sub'
                               MSGName = AFS + '/HTCondor/MSG/MSG_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               if os.path.isfile(required_output_file_location)!=True:
                                  bad_pop.append([OptionHeader+[' --i ', ' --j ', ' --p ', ' --o ',' --pfx ', ' --sfx ', Exception[0]], OptionLine+[i, j, path,o, pfx, sfx, Exception[1][j][0]], SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])

             if nest_lvl==3:
                 for i in range(len(loop_params)):
                     for j in range(len(loop_params[i])):
                         for k in range(loop_params[i][j]):
                               required_output_file_location=EOS+'/'+path+'/'+pfx+'_'+ID+'_'+o+'_'+str(i)+'_'+str(j)+sfx
                               bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/SH_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '_' + str(k) +'.sh'
                               SUBName = AFS + '/HTCondor/SUB/SUB_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '_' + str(k) +'.sub'
                               MSGName = AFS + '/HTCondor/MSG/MSG_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '_' + str(k)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               if os.path.isfile(required_output_file_location)!=True:
                                  bad_pop.append([OptionHeader+[' --i ', ' --j ',' --k ', ' --p ', ' --o ',' --pfx ', ' --sfx ', Exception[0]], OptionLine+[i, j,k, path,o, pfx, sfx, Exception[1][j][0]], SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
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
        OptionHeader+=[' --EOS '," --AFS ", " --PY ", " --BatchID "]
        OptionLine+=[EOS, AFS, PY, ID]
        TotJobs=int(TotJobs)
        with alive_bar(TotJobs,force_tty=True, title='Checking the results from HTCondor') as bar:
             if nest_lvl==2:
                 for i in range(len(loop_params)):
                               bar.text = f'-> Preparing batch submission...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/SH_'+pfx+'_'+ ID+'_' + str(i) + '.sh'
                               SUBName = AFS + '/HTCondor/SUB/SUB_'+pfx+'_'+ ID+'_' + str(i) + '.sub'
                               MSGName = AFS + '/HTCondor/MSG/MSG_'+pfx+'_'+ ID+'_' + str(i)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               bad_pop.append([OptionHeader+[' --i ', ' --j ', ' --p ', ' --o ',' --pfx ', ' --sfx ', Exception[0]], OptionLine+[i, '$1', path,o, pfx, sfx, Exception[1][i][0]], SHName, SUBName, MSGName, ScriptName, loop_params[i], 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
             if nest_lvl==3:
                 for i in range(len(loop_params)):
                     for j in range(loop_params[i]):
                               bar.text = f'-> Preparing batch submission...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/SH_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j)+'.sh'
                               SUBName = AFS + '/HTCondor/SUB/SUB_'+pfx+'_'+ ID+'_' + str(i) +'_' + str(j)+'.sub'
                               MSGName = AFS + '/HTCondor/MSG/MSG_'+pfx+'_'+ ID+'_' + str(i) +'_' + str(j)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               bad_pop.append([OptionHeader+[' --i ', ' --j ',' --k ', ' --p ', ' --o ',' --pfx ', ' --sfx ', Exception[0]], OptionLine+[i, j, '$1', path,o, pfx, sfx, Exception[1][i][0]], SHName, SUBName, MSGName, ScriptName, loop_params[i], 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
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
        OptionHeader+=[' --EOS '," --AFS ", " --PY ", " --BatchID "]
        OptionLine+=[EOS, AFS, PY, ID]
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
                                  bad_pop.append([OptionHeader+[' --i ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OptionLine+[i, path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
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
                                  bad_pop.append([OptionHeader+[' --i ', ' --j ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OptionLine+[i, j, path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])

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
                                  bad_pop.append([OptionHeader+[' --i ', ' --j ',' --k ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OptionLine+[i, j,k, path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
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
        OptionHeader+=[' --EOS '," --AFS ", " --PY ", " --BatchID "]
        OptionLine+=[EOS, AFS, PY, ID]
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
                               bad_pop.append([OptionHeader+[' --i ', ' --j ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OptionLine+[i, '$1', path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, loop_params[i], 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
             if nest_lvl==3:
                 for i in range(len(loop_params)):
                     for j in range(len(loop_params[i])):
                               bar.text = f'-> Preparing batch submission...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j)+'.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) +'_' + str(j)+'.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i) +'_' + str(j)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               bad_pop.append([OptionHeader+[' --i ', ' --j ',' --k ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OptionLine+[i, j, '$1', path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, loop_params[i][j], 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
             if nest_lvl==1:
                               bar.text = f'-> Preparing batch submission...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(0)+'/SH_'+pfx+'_'+ ID+'.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(0)+'/SUB_'+pfx+'_'+ ID+'.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(0)+'/MSG_'+pfx+'_'+ ID
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               bad_pop.append([OptionHeader+[' --i ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OptionLine+['$1', path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, loop_params, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
        return(bad_pop)
   return []


def SubmitJobs2Condor(job,local=False):
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
        f.write('arguments = $(Process)')
        f.write("\n")
        f.write('+SoftUsed = '+'"'+job[7]+'"')
        f.write("\n")
        f.write('transfer_output_files = ""')
        f.write("\n")
        f.write('+JobFlavour = "workday"')
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

# convert job array to a dictionary so that it is easier for process
def job_to_dict(job_array, required_output_file_location=None):
    dict = {
        "OptionHeader": job_array[0], 
        "OptionLine": job_array[1], 
        "SHName": job_array[2], 
        "SUBName": job_array[3], 
        "MSGName": job_array[4], 
        "ScriptName": job_array[5], 
        "Queue": job_array[6],
        "SoftUsed",:job_array[7],
        "EnableMSG": job_array[8],
        "RequestGPU": job_array[9],
        "required_output_file_location": required_output_file_location
    }
    return dict

# the input is a job dictionary. 
# You can always convert job to dictionary with "job_to_dict" function
def SubmitJob2Condor_wx(job):
    text=[
        "executable = " + job["SHName"],
        'requirements = (CERNEnvironment =!= "qa")' ,
        'arguments = $(Process)',
        '+SoftUsed = ' + '"' + job["SoftUsed"] + '"',
        'transfer_output_files = ""',
        '+JobFlavour = "workday"',
        'queue ' + str(job["Queue"]),
    ]
    if job["EnableMSG"] == True:
        msg_text = [
            "output =" + job["MSGName"] + ".out",
            "error =" + job["MSGName"] + ".err",
            "log ="+ job["MSGName"] +".log",      
        ]
        text += msg_text
    if job["RequestGPU"] == True:
        gpu_text = [
            'request_gpus = 1',
        ]
        text += gpu_text
    with open(job["SUBName"], 'w') as f:
        for line in text:
            f.write(line)
            f.write('\n')

    optionLine = ""
    for i in range(len(job["OptionLine"])):
        OptionLine+= job["OptionHeader"][i]
        OptionLine+= str(job["OptionLine"][i])
    TotalLine = 'python3 ' + job["ScriptName"] + OptionLine
    sh_text = [
        "#!/bin/bash",
        "set -ux",
        TotalLine,
    ]
    with open(job["SHName"], 'w') as f:
        for line in sh_text:
            f.write(line)
            f.write('\n')
    subprocess.call(['condor_submit', job["SUBName"]])
    print(TotalLine, " has been successfully submitted")


def SubmitJobs2Condor_wx(job_list):
    print(bcolors.OKGREEN,"There are", len(bad_pop), "jobs in the list",bcolors.ENDC)
    for job in job_list:
        #Sumbitting all jobs    
        SubmitJob2Condor_wx(job)
    print(bcolors.OKGREEN+"All jobs have been submitted"+bcolors.ENDC)



def collect_unfinished_jobs(job_list):
    print("Scheduled job checkup...") 
    unfinished_job_list = []
    # job is a dictionary here
    for job in job_list:
        if os.path.isfile(job["required_output_file_location"])==False: 
            unfinished_job_list.append(job)
    print("There are", len(unfinished_job_list), "jobs unfinished...")
    return unfinished_job_list

# the new version of auto pilot, can be used directly to submit jobs
# no need to do other submission first 
def AutoSubmitJobs(
    job_list, 
    wait_min, 
    interval_min, 
    max_interval_tolerance
):
    print(
        UF.TimeStamp(),
        'Going on an autopilot mode for ', wait_min, 'min',
        'minutes while checking HTCondor every', interval_min, 'min',
        bcolors.ENDC
    )
    #Converting min to sec as this time.sleep takes as an argument
    interval_sec=interval_min*60
    nIntervals=int(math.ceil(wait_min/interval_min))

    unfinished_jobs = job_list
    for i in range(0,nIntervals):
        unfinished_jobs=collect_unfinished_jobs(unfinished_jobs)
        if len(unfinished_jobs) ==0:
            print("All jobs are finished!")
            return True
        # else case: there are still jobs in condor
        print(
            UF.TimeStamp(), bcolors.WARNING+
            'Autopilot status update: There are still', len(unfinished_jobs), 'HTCondor jobs remaining'
            +bcolors.ENDC
        )
        #If jobs are not received after fixed number of check-ups we resubmit. 
        #Jobs sometimes fail on HTCondor for various reasons.
        if i % max_interval_tolerance == 0: 
            SubmitJobs2Condor_wx(unfinished_jobs) #Sumbitting all missing jobs
        time.sleep(interval_sec)
    return False