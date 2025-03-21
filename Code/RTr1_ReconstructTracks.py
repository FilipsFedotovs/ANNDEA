#This script connects hits in the data to produce tracks
#Tracking Module of the ANNDEA package
#Made by Filips Fedotovs

########################################    Import libraries    #############################################
import csv
#import ast
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
import pandas as pd #We use Panda for a routine data processing
pd.options.mode.chained_assignment = None #Silence annoying warnings
import math #We use it for data manipulation
import numpy as np
import os
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
import U_UI as UI #This is where we keep routine utility functions
import Parameters as PM #This is where we keep framework global parameters
import U_HC as HC_l
UI.WelcomeMsg('Initialising ANNDEA Tracking module...','Filips Fedotovs (PhD student at UCL), Dinis Beleza (MSci student at UCL)','Please reach out to filips.fedotovs@cern.ch for any queries')
#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script prepares training data for training the tracking model')
parser.add_argument('--Mode', help='Script will continue from the last checkpoint, unless you want to start from the scratch, then type "Reset"',default='')
parser.add_argument('--SeedModel',help="WHat GNN model would you like to use?", default='N')
parser.add_argument('--GraphModel',help="WHat GNN model would you like to use?", default='MH_SND_Tracking_5_80_5_80')
parser.add_argument('--Patience',help="How many checks to do before resubmitting the job?", default='30')
parser.add_argument('--SubPause',help="How long to wait in minutes after submitting 10000 jobs?", default='60')
parser.add_argument('--SubGap',help="How long to wait in minutes after submitting 10000 jobs?", default='10000')
parser.add_argument('--RecBatchID',help="Give this reconstruction batch an ID", default='Test_Batch')
parser.add_argument('--LocalSub',help="Local submission?", default='N')
parser.add_argument('--CheckPoint',help="Save cluster sets during individual cluster tracking.", default='N')
parser.add_argument('--CalibrateSeedBatch',help="Optimise the maximum edge per job parameter", default='N')
parser.add_argument('--ForceStatus',help="Would you like the program run from specific status number? (Only for advance users)", default='N')
parser.add_argument('--CPU',help="Would you like to request extra CPUs?", default=1)
parser.add_argument('--JobFlavour',help="Specifying the length of the HTCondor job walltime. Currently at 'workday' which is 8 hours.", default='workday')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/eos/experiment/ship/ANNDEA/Data/SND_Emulsion_FEDRA_Raw_B31.csv')
parser.add_argument('--Xmin',help="This option restricts data to only those events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option restricts data to only those events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option restricts data to only those events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option restricts data to only those events that have tracks with hits y-coordinates that are below this value", default='0')
parser.add_argument('--Zoverlap',help="Enter the level of overlap in integer number between reconstruction blocks along z-axis. (In order to avoid segmentation this value should be more than 1)", default='3')
parser.add_argument('--Yoverlap',help="Enter the level of overlap in integer number between reconstruction blocks along y-axis. (In order to avoid segmentation this value should be more than 1)", default='2')
parser.add_argument('--Xoverlap',help="Enter the level of overlap in integer number between reconstruction blocks along x-axis. (In order to avoid segmentation this value should be more than 1)", default='2')
parser.add_argument('--Memory',help="How much memory?", default='2 GB')
# parser.add_argument('--FixedPosition',help="Use to temporary reconstruct only specific sections of the emulsion data at a time (based on x-coordinate domain)", default=-1,type=int)
parser.add_argument('--HTCondorLog',help="Local submission?", default=False,type=bool)
parser.add_argument('--SeedFlowLog',help="Enable tracking of the seed cutflow?", default='N')

######################################## Parsing argument values  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
SeedModel=args.SeedModel
GraphModel=args.GraphModel
RecBatchID=args.RecBatchID
Patience=int(args.Patience)
SubPause=int(args.SubPause)*60
#TrackFitCut=ast.literal_eval(args.TrackFitCut)
SubGap=int(args.SubGap)
HTCondorLog=args.HTCondorLog
ForceStatus=args.ForceStatus
LocalSub=(args.LocalSub=='Y')
# FixedPosition=args.FixedPosition
CalibrateSeedBatch=(args.CalibrateSeedBatch=='Y')
SeedFlowLog=args.SeedFlowLog
JobFlavour=args.JobFlavour
CPU=int(args.CPU)
Memory=args.Memory
input_file_location=args.f
Zoverlap,Yoverlap,Xoverlap=int(args.Zoverlap),int(args.Yoverlap),int(args.Xoverlap)
Xmin,Xmax,Ymin,Ymax=float(args.Xmin),float(args.Xmax),float(args.Ymin),float(args.Ymax)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0 #We don't slice data if all values are set to zero simultaneousy (which is the default setting)

if LocalSub: time_int=0
else: time_int=10

if Mode=='RESET':
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'d',['RTr1a','RTr1b','RTr1c','RTr1d','RTr1e']))
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'c'))
elif Mode=='CLEANUP':
     print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'d',['RTr1a','RTr1b','RTr1c','RTr1d','RTr1e']))
     exit()
else:
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'c'))


#Establishing paths
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
Model_Meta_Path=EOSsubModelDIR+'/'+args.GraphModel+'_Meta'
print(UI.TimeStamp(),bcolors.BOLD+'Preparation 1/3:'+bcolors.ENDC+' Setting up metafiles...')

#Loading the model meta file
print(UI.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+Model_Meta_Path+bcolors.ENDC)

if args.GraphModel=='blank':
   print(UI.TimeStamp(),bcolors.WARNING+'You have specified the model name as "blank": This means that no GNN model will be used as part of the tracking process which can degrade the tracking performance.'+bcolors.ENDC)
   UserAnswer=input(bcolors.BOLD+"Do you want to continue? (y/n)\n"+bcolors.ENDC)
   if UserAnswer.upper()=='N':
       exit()
   stepX,stepY, stepZ = PM.stepX, PM.stepY, PM.stepZ
   cut_dz, cut_dt, cut_dr =PM.cut_dz, PM.cut_dt, PM.cut_dr

elif os.path.isfile(Model_Meta_Path):
       Model_Meta_Raw=UI.PickleOperations(Model_Meta_Path, 'r', 'N/A')
       print(Model_Meta_Raw[1])
       Model_Meta=Model_Meta_Raw[0]
       stepX, stepY, stepZ = Model_Meta.stepX, Model_Meta.stepY, Model_Meta.stepZ
       cut_dz, cut_dt, cut_dr = Model_Meta.cut_dz, Model_Meta.cut_dt, Model_Meta.cut_dr
else:
       print(UI.TimeStamp(),bcolors.FAIL+'Fail! No existing model meta files have been found, exiting now'+bcolors.ENDC)
       exit()
#
# def CP_CleanUp(prog,status):
#     jobs=prog[status][1][8]
#     eos=prog[status][1][1]
#     p=prog[status][1][3]
#     o=prog[status][1][4]
#     pfx=prog[status][1][5]
#     sfx=prog[status][1][6]
#     rec_batch_id=prog[status][1][7]
#     tot_jobs=UI.CalculateNJobs(jobs)[1]
#     jobs_del=0
#     with alive_bar(int(tot_jobs),force_tty=True, title='Deleting the unnecessary temp files...') as bar:
#         for i in range(len(jobs)):
#             for j in range(len(jobs[i])):
#                 for k in range(jobs[i][j]):
#                    bar()
#                    output_file_location=eos+p+'Temp_'+pfx+'_'+rec_batch_id+'_'+str(i)+'_'+str(j)+'/'+pfx+'_'+rec_batch_id+'_'+o+'_'+str(i)+'_'+str(j)+'_'+str(k)+sfx
#                    if os.path.isfile(output_file_location):
#                         CheckPointFile_Ini=eos+p+'/Temp_'+pfx+'_'+rec_batch_id+'_'+str(i)+'_'+str(j)+'/'+pfx+'_'+rec_batch_id+'_'+o+'_'+str(i)+'_'+str(j)+'_'+str(k) +'_CP_Ini.pkl'
#                         CheckPointFile_Edge=eos+p+'/Temp_'+pfx+'_'+rec_batch_id+'_'+str(i)+'_'+str(j)+'/'+pfx+'_'+rec_batch_id+'_'+o+'_'+str(i)+'_'+str(j) +'_'+str(k) +'_CP_Edge.pkl'
#                         CheckPointFile_ML=eos+p+'/Temp_'+pfx+'_'+rec_batch_id+'_'+str(i)+'_'+str(j)+'/'+pfx+'_'+rec_batch_id+'_'+o+'_'+str(i)+'_'+str(j)+'_'+str(k) + '_CP_ML.csv'
#                         CheckPointFile_Prep_1=eos+p+'/Temp_'+pfx+'_'+rec_batch_id+'_'+str(i)+'_'+str(j)+'/'+pfx+'_'+rec_batch_id+'_'+o+'_'+str(i)+'_'+str(j) +'_'+str(k) +'_CP_Prep_1.csv'
#                         CheckPointFile_Prep_2=eos+p+'/Temp_'+pfx+'_'+rec_batch_id+'_'+str(i)+'_'+str(j)+'/'+pfx+'_'+rec_batch_id+'_'+o+'_'+str(i)+'_'+str(j) +'_'+str(k) +'_CP_Prep_2.csv'
#                         CheckPointFile_Tracking_TH=eos+p+'/Temp_'+pfx+'_'+rec_batch_id+'_'+str(i)+'_'+str(j)+'/'+pfx+'_'+rec_batch_id+'_'+o+'_'+str(i)+'_'+str(j)+'_'+str(k) +'_CP_Tracking_TH.csv'
#                         CheckPointFile_Tracking_RP=eos+p+'/Temp_'+pfx+'_'+rec_batch_id+'_'+str(i)+'_'+str(j)+'/'+pfx+'_'+rec_batch_id+'_'+o+'_'+str(i)+'_'+str(j)+'_'+str(k) +'_CP_Tracking_RP.csv'
#                         files_to_delete=[CheckPointFile_Ini,CheckPointFile_Edge,CheckPointFile_ML,CheckPointFile_Prep_1,CheckPointFile_Prep_2,CheckPointFile_Tracking_TH,CheckPointFile_Tracking_RP]
#                         for f in files_to_delete:
#                             if os.path.isfile(f):
#                                 UI.Msg('location','Deleting:',f)
#                                 os.remove(f)
#                                 jobs_del+=1
#     return jobs_del

########################################     Phase 1 - Create compact source file    #########################################

print(UI.TimeStamp(),bcolors.BOLD+'Preparation 2/3:'+bcolors.ENDC+' Preparing the source data...')
required_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/RTr1_'+RecBatchID+'_hits.csv'
RecOutputMeta=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/'+RecBatchID+'_info.pkl'
if os.path.isfile(required_file_location)==False:
         print(UI.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
         have_mc=False
         if SeedFlowLog:
             try:
                 data=pd.read_csv(input_file_location,
                             header=0,
                             usecols=[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.MC_Event_ID,PM.MC_Track_ID])[[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.MC_Event_ID,PM.MC_Track_ID]]
                 have_mc=True
             except Exception as e:
                 UI.Msg('failed', str(e))
                 data=pd.read_csv(input_file_location,
                             header=0,
                             usecols=[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty])[[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty]]
         total_rows=len(data.axes[0])
         data[PM.Hit_ID] = data[PM.Hit_ID].astype(str)
         print(UI.TimeStamp(),'The raw data has ',total_rows,' hits')
         print(UI.TimeStamp(),'Removing unreconstructed hits...')
         data=data.dropna()
         final_rows=len(data.axes[0])
         print(UI.TimeStamp(),'The cleaned data has ',final_rows,' hits')
         try:
             data[PM.Hit_ID] = data[PM.Hit_ID].astype(int)
         except:
             print(UI.TimeStamp(), bcolors.WARNING+"Hit ID is already in the string format, skipping the reformatting step..."+bcolors.ENDC)
         data[PM.Hit_ID] = data[PM.Hit_ID].astype(str)

         if have_mc:
             data[PM.MC_Event_ID] = data[PM.MC_Event_ID].astype(str)
             data[PM.MC_Track_ID] = data[PM.MC_Track_ID].astype(str)
             data['MC_Super_Track_ID'] = data[PM.MC_Event_ID] + '-' + data[PM.MC_Track_ID] #Track IDs are not unique and repeat for each event: crea
             data=data.drop([PM.MC_Event_ID],axis=1)
             data=data.drop([PM.MC_Track_ID],axis=1)
             data=data.rename(columns={"MC_Super_Track_ID": "MC_Track_ID"})
         if SliceData:
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
         print(UI.TimeStamp(),'Analysing data... ',bcolors.ENDC)
         z_offset=data['z'].min()
         data['z']=data['z']-z_offset
         z_max=data['z'].max()
         if Zoverlap==1:
            Zsteps=math.ceil((z_max)/stepZ)
         else:
            Zsteps=(math.ceil((z_max)/stepZ)*(Zoverlap))-1
         y_offset=data['y'].min()
         x_offset=data['x'].min()
         data['x']=data['x']-x_offset
         data['y']=data['y']-y_offset
         x_max=data['x'].max()
         y_max=data['y'].max()
        #Calculating the number of volumes that will be sent to HTCondor for reconstruction. Account for overlap if specified.
         if Xoverlap==1:
            Xsteps=math.ceil((x_max)/stepX)
         else:
            Xsteps=(math.ceil((x_max)/stepX)*(Xoverlap))-1
        
         if Yoverlap==1:
            Ysteps=math.ceil((y_max)/stepY)
         else:
            Ysteps=(math.ceil((y_max)/stepY)*(Yoverlap))-1
         print(UI.TimeStamp(),'Distributing input files...')
         with alive_bar(Xsteps*Ysteps*Zsteps,force_tty=True, title='Distributing input files...') as bar:
             for i in range(Xsteps):
                 for j in range(Ysteps):
                     for k in range(Zsteps):
                         required_tfile_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/RTr1_'+RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_clusters.pkl'
                         if os.path.isfile(required_tfile_location)==False:
                             Y_ID=int(j)/Yoverlap
                             X_ID=int(i)/Xoverlap
                             Z_ID=int(k)/Zoverlap
                             tdata=data.drop(data.index[data['x'] >= ((X_ID+1)*stepX)])  #Keeping the relevant z slice
                             tdata.drop(tdata.index[tdata['x'] < (X_ID*stepX)], inplace = True)  #Keeping the relevant z slice
                             tdata.drop(tdata.index[tdata['y'] >= ((Y_ID+1)*stepY)], inplace = True)  #Keeping the relevant z slice
                             tdata.drop(tdata.index[tdata['y'] < (Y_ID*stepY)], inplace = True)  #Keeping the relevant z slice
                             tdata.drop(tdata.index[tdata['z'] >= ((Z_ID+1)*stepZ)], inplace = True)  #Keeping the relevant z slice
                             tdata.drop(tdata.index[tdata['z'] < (Z_ID*stepZ)], inplace = True)  #Keeping the relevant z slice
                             tdata_list=tdata.values.tolist()
                             print(UI.TimeStamp(),'Creating the cluster', X_ID,Y_ID,Z_ID)
                             HC=HC_l.HitCluster([X_ID,Y_ID,Z_ID],[stepX,stepY,stepZ]) #Initializing the cluster
                             print(UI.TimeStamp(),'Decorating the cluster')
                             HC.LoadClusterHits(tdata_list) #Decorating the Clusters with Hit information
                             UI.PickleOperations(required_tfile_location,'w',HC)
                         bar()
         data.to_csv(required_file_location,index=False)

         Meta=UI.JobMeta(RecBatchID)
         Meta.UpdateJobMeta(['stepX', 'stepY', 'stepZ', 'cut_dt', 'cut_dr', 'cut_dz', 'y_offset', 'x_offset', 'Xoverlap', 'Yoverlap', 'Zoverlap', 'Xsteps', 'Ysteps', 'Zsteps'],[stepX, stepY, stepZ, cut_dt, cut_dr, cut_dz, y_offset, x_offset, Xoverlap, Yoverlap, Zoverlap, Xsteps, Ysteps, Zsteps])
         Meta.UpdateStatus(0)
         print(UI.PickleOperations(RecOutputMeta,'w', Meta)[1])
         UI.Msg('completed','Stage 0 has successfully completed')
         print(UI.TimeStamp(), bcolors.OKGREEN+"The segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+required_file_location+bcolors.ENDC)
elif os.path.isfile(RecOutputMeta)==True:
    print(UI.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+RecOutputMeta+bcolors.ENDC)
    MetaInput=UI.PickleOperations(RecOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
Zsteps=Meta.Zsteps
Ysteps=Meta.Ysteps
Xsteps=Meta.Xsteps
stepZ=Meta.stepZ
stepY=Meta.stepY
stepX=Meta.stepX
cut_dt=Meta.cut_dt
cut_dr=Meta.cut_dr
cut_dz=Meta.cut_dz
if hasattr(Meta, 'job_sets') and hasattr(Meta, 'n_graph_hobs'):
    job_sets=Meta.job_sets
    n_graph_hobs=Meta.n_graph_hobs

# ########################################     Preset framework parameters    #########################################

UI.Msg('vanilla','Analysing the current script status...')
Status=Meta.Status[-1]
if ForceStatus!='N':
    Status=int(ForceStatus)
UI.Msg('vanilla','Current stage is '+str(Status)+'...')

################ Set the execution sequence for the script
Program=[]
###### Stage 0
if hasattr(Meta, 'job_sets')==hasattr(Meta, 'n_graph_jobs')==False:
    graph_job_set=[]
    n_graph_jobs=0
    with alive_bar(Xsteps*Ysteps*Zsteps,force_tty=True, title='Estimating number of jobs...') as bar:
        for i in range(0,Xsteps):
            graph_job_set.append([])
            for j in range(0,Ysteps):
                graph_job_set[i].append([])
                for k in range(0,Zsteps):
                    bar()
                    tfile_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/RTr1_'+RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_clusters.pkl'
                    HC=UI.PickleOperations(tfile_location,'r','N/A')[0]
                    n_edg=len(HC.RawClusterGraph)
                    job_iter=0
                    acc_edg=0
                    for n_e in range(1,n_edg+1):
                            acc_edg+=n_edg-n_e
                            if acc_edg>=PM.MaxEdgesPerJob:
                                job_iter+=1
                                acc_edg=0
                    if acc_edg>0:
                         job_iter+=1
                    n_graph_jobs+=job_iter
                    graph_job_set[i][j].append(job_iter)
    Meta.graph_job_set=graph_job_set
    Meta.n_graph_jobs=n_graph_jobs
    UI.UpdateStatus(Status,Meta,RecOutputMeta)
else:
    graph_job_set=Meta.graph_job_set
    n_graph_jobs=Meta.n_graph_jobs


if CalibrateSeedBatch:
    print(job_sets)
    x=input('Continue(y/n)?')
    if x!='y':
        exit()

prog_entry=[]
prog_entry.append(' Sending hit cluster to the HTCondor, so the graph seed can be generated')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/'+RecBatchID+'/','hit_cluster_edges','RTr1a','.pkl',RecBatchID,graph_job_set,'RTr1a_GenerateEdges_Sub.py'])
prog_entry.append([' --cut_dt ', ' --cut_dr ',' --cut_dz ',' --MaxEdgesPerJob '])
prog_entry.append([cut_dt,cut_dr,cut_dz,str(PM.MaxEdgesPerJob)])
prog_entry.append(n_graph_jobs)
prog_entry.append(LocalSub)
prog_entry.append('N/A')
prog_entry.append(HTCondorLog)
prog_entry.append(False)
Program.append(prog_entry)
print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry))

prog_entry=[]
job_sets=[]
for i in range(0,Xsteps):
                job_set=[]
                for j in range(0,Ysteps):
                    job_set.append(Zsteps)
                job_sets.append(job_set)
prog_entry.append(' Sending hit cluster to the HTCondor, so the graph seed can be consolidated')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/'+RecBatchID+'/','hit_cluster_edges_consolidated','RTr1b','.pkl',RecBatchID,job_sets,'RTr1b_ConsolidateEdges_Sub.py'])
prog_entry.append([' --GraphProgram '])
prog_entry.append(['"'+str(Program[0][1][8])+'"'])
prog_entry.append(Xsteps*Ysteps*Zsteps)
prog_entry.append(LocalSub)
prog_entry.append('N/A')
prog_entry.append(HTCondorLog)
prog_entry.append(False)
Program.append(prog_entry)
print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry))

###### Stage 2
prog_entry=[]
job_sets=[]
for i in range(0,Xsteps):
                job_set=[]
                for j in range(0,Ysteps):
                    job_set.append(Zsteps)
                job_sets.append(job_set)
prog_entry.append(' Sending hit cluster to the HTCondor, so the model assigns weights between hits')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/'+RecBatchID+'/','hit_cluster_rec_set','RTr1c','.csv',RecBatchID,job_sets,'RTr1c_ReconstructTracks_Sub.py'])
#prog_entry.append([' --stepZ ', ' --stepY ', ' --stepX ', ' --ModelName ', ' --CheckPoint ', ' --TrackFitCutRes ',' --TrackFitCutSTD ',' --TrackFitCutMRes '])
prog_entry.append([' --ModelName ', ' --CheckPoint '])
#prog_entry.append([stepZ,stepY,stepX,ModelName,args.CheckPoint]+TrackFitCut)
prog_entry.append([ModelName,args.CheckPoint])
prog_entry.append(Xsteps*Ysteps*Zsteps)
prog_entry.append(LocalSub)
prog_entry.append('N/A')
prog_entry.append(HTCondorLog)
prog_entry.append(False)
Program.append(prog_entry)
print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry))



if FixedPosition<0:
    ###### Stage 3
    prog_entry=[]
    job_sets=[]
    for i in range(0,Xsteps):
                    job_sets.append(Ysteps)
    prog_entry.append(' Sending hit cluster to the HTCondor, so the reconstructed clusters can be merged along z-axis')
    prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/'+RecBatchID+'/','hit_cluster_rec_z_set','RTr1d','.csv',RecBatchID,job_sets,'RTr1d_LinkSegmentsZ_Sub.py'])
    prog_entry.append([' --Z_ID_Max ', ' --i ',' --j '])
    prog_entry.append([Zsteps,Xsteps,Ysteps])
    prog_entry.append(Xsteps*Ysteps)
    prog_entry.append(LocalSub)
    prog_entry.append('N/A')
    prog_entry.append(HTCondorLog)
    prog_entry.append(False)
    Program.append(prog_entry)
    print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry))

    ###### Stage 4
    prog_entry=[]
    job_sets=Xsteps
    prog_entry.append(' Sending hit cluster to the HTCondor, so the reconstructed clusters can be merged along y-axis')
    prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/'+RecBatchID+'/','hit_cluster_rec_y_set','RTr1e','.csv',RecBatchID,job_sets,'RTr1e_LinkSegmentsY_Sub.py'])
    prog_entry.append([' --Y_ID_Max ', ' --i '])
    prog_entry.append([Ysteps,Xsteps])
    prog_entry.append(Xsteps)
    prog_entry.append(LocalSub)
    prog_entry.append('N/A')
    prog_entry.append(HTCondorLog)
    prog_entry.append(False)
    Program.append(prog_entry)
    print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry))

    ###### Stage 5
    prog_entry=[]
    job_sets=1
    prog_entry.append(' Sending hit cluster to the HTCondor, so the reconstructed clusters can be merged along x-axis')
    prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/'+RecBatchID+'/','hit_cluster_rec_x_set','RTr1f','.csv',RecBatchID,job_sets,'RTr1f_LinkSegmentsX_Sub.py'])
    prog_entry.append([' --X_ID_Max '])
    prog_entry.append([Xsteps])
    prog_entry.append(1)
    prog_entry.append(True) #This part we can execute locally, no need for HTCondor
    prog_entry.append('N/A')
    prog_entry.append(HTCondorLog)
    prog_entry.append(False)
    Program.append(prog_entry)
    print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry))

    ###### Stage 6
    Program.append('Custom')
else:
    ###### Temp cleanup stage 3
    Program.append('Custom')


print(UI.TimeStamp(),'There are '+str(len(Program)+1)+' stages (0-'+str(len(Program)+1)+') of this script',bcolors.ENDC)
print(UI.TimeStamp(),'Current stage has a code',Status,bcolors.ENDC)
while Status<len(Program):
    if Program[Status]!='Custom':
       Result=UI.StandardProcess(Program,Status,SubGap,SubPause,RequestExtCPU,JobFlavour,ReqMemory,time_int,Patience,FixedPosition)
       if Result[0]:
            UI.UpdateStatus(Status+1,Meta,RecOutputMeta)
       else:
             Status=20
             break

    # elif Status==1:
    #     print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
    #     print(UI.TimeStamp(),bcolors.BOLD+'Stage 1:'+bcolors.ENDC+' Consolidating the edge generation files from the previous step...')
    #     with alive_bar(Program[0][4],force_tty=True, title='Consolidation progress...') as bar:
    #         if FixedPosition<0:
    #             for i in range(len(Program[0][1][8])):
    #                 for j in range(len(Program[0][1][8][i])):
    #                     for k in range(len(Program[0][1][8][i][j])):
    #                         if Program[0][1][8][i][j][k]>0:
    #                             master_file=EOS_DIR+Program[0][1][3]+'/Temp_'+Program[0][1][5]+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'/'+Program[0][1][5]+'_'+RecBatchID+'_'+Program[0][1][4]+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_0.pkl'
    #                             master_data=UI.PickleOperations(master_file,'r','')[0]
    #                             bar()
    #                             for l in range(1,Program[0][1][8][i][j][k]):
    #                                 slave_file=EOS_DIR+Program[0][1][3]+'/Temp_'+Program[0][1][5]+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'/'+Program[0][1][5]+'_'+RecBatchID+'_'+Program[0][1][4]+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)+'.pkl'
    #                                 slave_data=UI.PickleOperations(slave_file,'r','')[0]
    #                                 master_data.RawEdgeGraph+=slave_data.RawEdgeGraph
    #                                 master_data.HitPairs+=slave_data.HitPairs
    #                                 bar()
    #                             output_file=EOS_DIR+Program[2][1][3]+'/Temp_'+Program[2][1][5]+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'/'+Program[0][1][5]+'_'+RecBatchID+'_'+Program[0][1][4]+'_'+str(i)+'_'+str(j)+'_'+str(k)+'.pkl'
    #                             print(UI.PickleOperations(output_file,'w',master_data)[1])
    #                         else:
    #                             bar()
    #                             continue
    #         else:
    #             i=FixedPosition
    #             for j in range(len(Program[0][1][8][i])):
    #                 for k in range(len(Program[0][1][8][i][j])):
    #                     if Program[0][1][8][i][j][k]>0:
    #                         master_file=EOS_DIR+Program[0][1][3]+'/Temp_'+Program[0][1][5]+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'/'+Program[0][1][5]+'_'+RecBatchID+'_'+Program[0][1][4]+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_0.pkl'
    #                         master_data=UI.PickleOperations(master_file,'r','')[0]
    #                         bar()
    #                         for l in range(1,Program[0][1][8][i][j][k]):
    #                             slave_file=EOS_DIR+Program[0][1][3]+'/Temp_'+Program[0][1][5]+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'/'+Program[0][1][5]+'_'+RecBatchID+'_'+Program[0][1][4]+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)+'.pkl'
    #                             slave_data=UI.PickleOperations(slave_file,'r','')[0]
    #                             master_data.RawEdgeGraph+=slave_data.RawEdgeGraph
    #                             master_data.HitPairs+=slave_data.HitPairs
    #                             bar()
    #                         output_file=EOS_DIR+Program[2][1][3]+'/Temp_'+Program[2][1][5]+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'/'+Program[0][1][5]+'_'+RecBatchID+'_'+Program[0][1][4]+'_'+str(i)+'_'+str(j)+'_'+str(k)+'.pkl'
    #                         print(UI.PickleOperations(output_file,'w',master_data)[1])
    #                     else:
    #                         bar()
    #                         continue
    #     UI.Msg('success',"The hit cluster files were successfully consolidated.")
    #     UI.UpdateStatus(Status+1,Meta,RecOutputMeta)
    elif Status==3:
        print('We are here')
        exit()
        i=FixedPosition
        with alive_bar(Ysteps*Zsteps*Program[0][1][8][i][j][k],force_tty=True, title='Deleting the files that are not needed anymore...') as bar:
                    for j in range(len(Program[0][1][8][i])):
                        for k in range(len(Program[0][1][8][i][j])):
                            for l in range(Program[0][1][8][i][j][k]):
                                    del_file_location_1=EOS_DIR+Program[0][1][3]+'/Temp_RTr1a_'+RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'/RTr1a_'+RecBatchID+'_hit_cluster_edges_'+str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)+'.pkl'
                                    del_file_location_2=EOS_DIR+Program[0][1][3]+'/Temp_RTr1b_'+RecBatchID+'_'+str(i)+'_'+str(j)+'/RTr1a_'+RecBatchID+'_hit_cluster_edges_'+str(i)+'_'+str(j)+'_'+str(k)+'.pkl'
                                    if os.path.isfile(del_file_location_1):
                                         os.remove(del_file_location_1)
                                    if os.path.isfile(del_file_location_2):
                                         os.remove(del_file_location_2)
                                    bar()
        exit()
    elif Status==6:
      #Non standard processes (that don't follow the general pattern) have been coded here
      print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
      print(UI.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' Using the results from previous steps to map merged trackIDs to the original reconstruction file')
      try:
        #Read the output with hit- ANN Track map
        FirstFile=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_RTr1e_'+RecBatchID+'_0'+'/RTr1e_'+RecBatchID+'_hit_cluster_rec_x_set_0.csv'
        print(UI.TimeStamp(),'Loading the file ',bcolors.OKBLUE+FirstFile+bcolors.ENDC)
        TrackMap=pd.read_csv(FirstFile,header=0)
        input_file_location=args.f
        print(UI.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
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
        print(UI.TimeStamp(),'Identifying problematic tracks where there is more than one hit per plate...')
        Hit_Map=Data[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.Hit_ID]] #Separating the hit map
        Data.drop([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID'],axis=1,inplace=True) #Remove the ANNDEA tracking info from the main data





        Hit_Map=Hit_Map.dropna() #Remove unreconstructing hits - we are not interested in them atm
        Hit_Map_Stats=Hit_Map[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z,PM.Hit_ID]] #Calculating the stats

        Hit_Map_Stats=Hit_Map_Stats.groupby([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID']).agg({PM.z:pd.Series.nunique,PM.Hit_ID: pd.Series.nunique}).reset_index() #Calculate the number fo unique plates and hits
        Ini_No_Tracks=len(Hit_Map_Stats)
        print(UI.TimeStamp(),bcolors.WARNING+'The initial number of tracks is '+ str(Ini_No_Tracks)+bcolors.ENDC)
        Hit_Map_Stats=Hit_Map_Stats.rename(columns={PM.z: "No_Plates",PM.Hit_ID:"No_Hits"}) #Renaming the columns so they don't interfere once we join it back to the hit map
        Hit_Map_Stats=Hit_Map_Stats[Hit_Map_Stats.No_Plates >= PM.MinHitsTrack]
        Prop_No_Tracks=len(Hit_Map_Stats)
        print(UI.TimeStamp(),bcolors.WARNING+'After dropping single hit tracks, left '+ str(Prop_No_Tracks)+' tracks...'+bcolors.ENDC)
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

        Bad_Tracks_CP_File=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_RTr1c_'+RecBatchID+'_0'+'/RTr1c_'+RecBatchID+'_Bad_Tracks_CP.csv'
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

            print(UI.TimeStamp(),'Saving the checkpoint file ',bcolors.OKBLUE+Bad_Tracks_CP_File+bcolors.ENDC)
            Bad_Tracks_Head.to_csv(Bad_Tracks_CP_File,index=False)
        else:
           print(UI.TimeStamp(),'Loading the checkpoint file ',bcolors.OKBLUE+Bad_Tracks_CP_File+bcolors.ENDC)
           Bad_Tracks_Head=pd.read_csv(Bad_Tracks_CP_File,header=0)
           Bad_Tracks_Head=Bad_Tracks_Head[Bad_Tracks_Head.ax != '[]']
           Bad_Tracks_Head['ax'] = Bad_Tracks_Head['ax'].astype(float)
           Bad_Tracks_Head['ay'] = Bad_Tracks_Head['ay'].astype(float)
           Bad_Tracks_Head['t1x'] = Bad_Tracks_Head['t1x'].astype(float)
           Bad_Tracks_Head['t2x'] = Bad_Tracks_Head['t2x'].astype(float)
           Bad_Tracks_Head['t1y'] = Bad_Tracks_Head['t1y'].astype(float)
           Bad_Tracks_Head['t2y'] = Bad_Tracks_Head['t2y'].astype(float)

        print(UI.TimeStamp(),'Removing problematic hits...')
        Bad_Tracks_Head[str(RecBatchID)+'_Brick_ID'] = Bad_Tracks_Head[str(RecBatchID)+'_Brick_ID'].astype(str)
        Bad_Tracks_Head[str(RecBatchID)+'_Track_ID'] = Bad_Tracks_Head[str(RecBatchID)+'_Track_ID'].astype(str)
        Bad_Tracks[str(RecBatchID)+'_Brick_ID'] = Bad_Tracks[str(RecBatchID)+'_Brick_ID'].astype(str)
        Bad_Tracks[str(RecBatchID)+'_Track_ID'] = Bad_Tracks[str(RecBatchID)+'_Track_ID'].astype(str)


        Bad_Tracks=pd.merge(Bad_Tracks,Bad_Tracks_Head,how='inner',on = [str(RecBatchID)+'_Brick_ID',str(RecBatchID)+'_Track_ID'])

        print(UI.TimeStamp(),'Calculating x and y coordinates of the fitted line for all plates in the track...')
        #Calculating x and y coordinates of the fitted line for all plates in the track
        Bad_Tracks['new_x']=Bad_Tracks['ax']+(Bad_Tracks[PM.z]*Bad_Tracks['t1x'])+((Bad_Tracks[PM.z]**2)*Bad_Tracks['t2x'])
        Bad_Tracks['new_y']=Bad_Tracks['ay']+(Bad_Tracks[PM.z]*Bad_Tracks['t1y'])+((Bad_Tracks[PM.z]**2)*Bad_Tracks['t2y'])

        #Calculating how far hits deviate from the fit polynomial
        print(UI.TimeStamp(),'Calculating how far hits deviate from the fit polynomial...')
        Bad_Tracks['d_x']=Bad_Tracks[PM.x]-Bad_Tracks['new_x']
        Bad_Tracks['d_y']=Bad_Tracks[PM.y]-Bad_Tracks['new_y']

        Bad_Tracks['d_r']=Bad_Tracks['d_x']**2+Bad_Tracks['d_y']**2
        Bad_Tracks['d_r'] = Bad_Tracks['d_r'].astype(float)
        Bad_Tracks['d_r']=np.sqrt(Bad_Tracks['d_r']) #Absolute distance
        Bad_Tracks=Bad_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z,PM.Hit_ID,'d_r']]

        #Sort the tracks and their hits by Track ID, Plate and distance to the perfect line
        print(UI.TimeStamp(),'Sorting the tracks and their hits by Track ID, Plate and distance to the perfect line...')
        Bad_Tracks.sort_values([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z,'d_r'],ascending=[0,0,1,1],inplace=True)

        #If there are two hits per plate we will keep the one which is closer to the line
        Bad_Tracks.drop_duplicates(subset=[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z],keep='first',inplace=True)
        Bad_Tracks=Bad_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.Hit_ID]]
        Good_Tracks=pd.concat([Good_Tracks,Bad_Tracks]) #Combine all ANNDEA tracks together
        Data=pd.merge(Data,Good_Tracks,how='left', on=[PM.Hit_ID]) #Re-map corrected ANNDEA Tracks back to the main data
        output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'.csv' #Final output. We can use this file for further operations
        Data.to_csv(output_file_location,index=False)
        print(UI.TimeStamp(), bcolors.OKGREEN+"The tracked data has been written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
        print(UI.TimeStamp(),bcolors.OKGREEN+'Stage 4 has successfully completed'+bcolors.ENDC)
        UI.UpdateStatus(Status+1,Meta,RecOutputMeta)
      except Exception as e:
          print(UI.TimeStamp(),bcolors.FAIL+'Stage 4 is uncompleted due to: '+str(e)+bcolors.ENDC)
          Status=21
          break
    MetaInput=UI.PickleOperations(RecOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
    Status=Meta.Status[-1]

if Status<20:
    x=input("Would you like to remove temp files (y/n)?")
    if x=='y':
        #Removing the temp files that were generated by the process
        print(UI.TimeStamp(),'Performing the cleanup... ')
        print(UI.ManageFolders(AFS_DIR, EOS_DIR, RecBatchID,'d',['RTr1a','RTr1b','RTr1c','RTr1d','RTr1e']))
        UI.Msg('success',"Segment merging has been completed")
else:
    UI.Msg('failed',"Segment merging has not been completed as one of the processes has timed out. Please run the script again (without Reset Mode).")
    exit()




