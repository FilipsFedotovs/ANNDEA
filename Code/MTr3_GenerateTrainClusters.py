# Part of  ANNDEA  package.
# The purpose of this script to create samples for the tracking model training
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
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
import argparse
import pandas as pd #We use Panda for a routine data processing
import math #We use it for data manipulation
import os
import random
import time
import ast
import U_UI as UI #This is where we keep routine utility functions
import Parameters as PM #This is where we keep framework global parameters
from alive_progress import alive_bar
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
print(bcolors.HEADER+"######################     Initialising ANNDEA Tracking Train Sample Generation module     #############"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
parser = argparse.ArgumentParser(description='This script prepares training data for training the tracking model')
parser.add_argument('--Mode', help='Script will continue from the last checkpoint, unless you want to start from the scratch, then type "Reset"',default='')
parser.add_argument('--TrainSampleID',help="Give name to this train sample batch", default='MH_SND_Raw_Train_Data_6_6_12_All')
parser.add_argument('--Sampling',help="If the data is large, sampling helps to keep it manageable. If this script fails consider reducing this parameter.", default='1.0')
parser.add_argument('--Patience',help="How many HTCondor checks to perform before resubmitting the job?", default='15')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/eos/experiment/ship/ANNDEA/Data/SND_Raw_Input/SND_B11_FEDRARaw.csv')
parser.add_argument('--Xmin',help="This option restricts data to only those events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option restricts data to only those events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option restricts data to only those events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option restricts data to only those events that have tracks with hits y-coordinates that are below this value", default='0')
parser.add_argument('--ExcludeClassNames',help="What class headers to use?", default="[]")
parser.add_argument('--ExcludeClassValues',help="What class values to use?", default="[]")
parser.add_argument('--SubPause',help="How long to wait in minutes after submitting 10000 jobs?", default='60')
parser.add_argument('--SubGap',help="How long to wait in minutes after submitting 10000 jobs?", default='10000')
#The bellow are not important for the training smaples but if you want to augment the training data set above 1
parser.add_argument('--Zoverlap',help="Enter the level of overlap in integer number between reconstruction blocks along z-axis.", default='1')
parser.add_argument('--Yoverlap',help="Enter the level of overlap in integer number between reconstruction blocks along y-axis.", default='1')
parser.add_argument('--Xoverlap',help="Enter the level of overlap in integer number between reconstruction blocks along x-axis.", default='1')
parser.add_argument('--Memory',help="How uch memory to request?", default='2 GB')
parser.add_argument('--CPU',help="How many CPUs?", default=1)
parser.add_argument('--JobFlavour',help="Specifying the length of the HTCondor job walltime. Currently at 'workday' which is 8 hours.", default='workday')
parser.add_argument('--ForceStatus',help="Would you like the program run from specific status number? (Only for advance users)", default='0')
parser.add_argument('--HTCondorLog',help="Local submission?", default=False,type=bool)
parser.add_argument('--LocalSub',help="Local submission?", default='N')
parser.add_argument('--SeedFlowLog',help="Enable tracking of the seed cutflow?", default='N')
parser.add_argument('--ModelName',help="Model used to refine seeds", default='N')
######################################## Set variables  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
Sampling=float(args.Sampling)
TrainSampleID=args.TrainSampleID
Patience=int(args.Patience)
Memory=args.Memory
ModelName=args.ModelName
CPU=int(args.CPU)
ForceStatus=args.ForceStatus
JobFlavour=args.JobFlavour
HTCondorLog=args.HTCondorLog
SeedFlowLog=args.SeedFlowLog
LocalSub=(args.LocalSub=='Y')
input_file_location=args.f
SubPause=int(args.SubPause)*60
SubGap=int(args.SubGap)
Xmin,Xmax,Ymin,Ymax=float(args.Xmin),float(args.Xmax),float(args.Ymin),float(args.Ymax)
Zoverlap,Yoverlap,Xoverlap=int(args.Zoverlap),int(args.Yoverlap),int(args.Xoverlap)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0

if LocalSub:
   time_int=0
else:
   time_int=10

ExcludeClassNames=ast.literal_eval(args.ExcludeClassNames)
ExcludeClassValues=ast.literal_eval(args.ExcludeClassValues)
ColumnsToImport=[PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.MC_Event_ID,PM.MC_Track_ID]
ExtraColumns=[]
BanDF=['-']
BanDF=pd.DataFrame(BanDF, columns=['Exclude'])
for i in range(len(ExcludeClassNames)):
        df=pd.DataFrame(ExcludeClassValues[i], columns=[ExcludeClassNames[i]])
        df['Exclude']='-'
        BanDF=pd.merge(BanDF,df,how='inner',on=['Exclude'])

        if (ExcludeClassNames[i] in ExtraColumns)==False:
                ExtraColumns.append(ExcludeClassNames[i])


stepX=PM.stepX #Size of the individual reconstruction volumes along the x-axis
stepY=PM.stepY #Size of the individual reconstruction volumes along the y-axis
stepZ=PM.stepZ #Size of the individual reconstruction volumes along the z-axis
cut_dt=PM.cut_dt #This cust help to discard hit pairs that are likely do not have a common mother track
cut_dr=PM.cut_dr
cut_dz=PM.cut_dz
testRatio=PM.testRatio #Usually about 5%
valRatio=PM.valRatio #Usually about 10%

TrainSampleOutputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl' #For each training sample batch we create an individual meta file.
destination_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TTr_OUTPUT_1.pkl' #The desired output


if Mode=='RESET':
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, TrainSampleID,'d',['MTr3']))
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, TrainSampleID,'c'))
    if os.path.isfile(TrainSampleOutputMeta):
        os.remove(TrainSampleOutputMeta)
elif Mode=='CLEANUP':
     print(UI.ManageFolders(AFS_DIR, EOS_DIR, TrainSampleID,'d',['MTr3']))
     if os.path.isfile(TrainSampleOutputMeta):
        os.remove(TrainSampleOutputMeta)
     exit()
else:
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, TrainSampleID,'c'))


########################################     Phase 1 - Create compact source file    #########################################
print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UI.TimeStamp(),bcolors.BOLD+'Stage 0:'+bcolors.ENDC+' Taking the file that has been supplied and creating the compact copies for the training set generation...')
output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/MTr3_'+TrainSampleID+'_hits.csv' #This is the compact data file that contains only relevant columns and rows
if os.path.isfile(output_file_location)==False:
        print(UI.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
        data=pd.read_csv(input_file_location,
                    header=0,
                    usecols=ColumnsToImport+ExtraColumns)[ColumnsToImport+ExtraColumns]
        if len(ExtraColumns)>0:
            for c in ExtraColumns:
                data[c] = data[c].astype(str)
            data=pd.merge(data,BanDF,how='left',on=ExtraColumns)
            data=data.fillna('')
        else:
            data['Exclude']=''
        total_rows=len(data.axes[0])
        data[PM.Hit_ID] = data[PM.Hit_ID].astype(str) #We try to keep HIT ids as strings
        print(UI.TimeStamp(),'The raw data has ',total_rows,' hits')
        print(UI.TimeStamp(),'Removing unreconstructed hits...')
        data=data.dropna() #Removing nulls (not really applicable for this module but I put it just in case)
        final_rows=len(data.axes[0])
        print(UI.TimeStamp(),'The cleaned data has ',final_rows,' hits')
        data[PM.MC_Event_ID] = data[PM.MC_Event_ID].astype(str)
        data[PM.MC_Track_ID] = data[PM.MC_Track_ID].astype(str)
        data['MC_Super_Track_ID'] = data[PM.MC_Event_ID] + '-'+ data['Exclude'] + data[PM.MC_Track_ID] #Track IDs are not unique and repeat for each event: crea
        data=data.drop([PM.MC_Event_ID],axis=1)
        data=data.drop([PM.MC_Track_ID],axis=1)
        data=data.drop(['Exclude'],axis=1)
        for c in ExtraColumns:
            data=data.drop([c],axis=1)
        if SliceData: #Keeping only the relevant slice. Only works if we set at least one parameter (Xmin for instance) to non-zero
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
        data=data.rename(columns={"MC_Super_Track_ID": "MC_Track_ID"})
        data.to_csv(output_file_location,index=False)
        print(UI.TimeStamp(), bcolors.OKGREEN+"The segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)

print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UI.TimeStamp(),bcolors.BOLD+'Stage 1:'+bcolors.ENDC+' Creating training sample meta data...')
if os.path.isfile(TrainSampleOutputMeta)==False: #A case of generating samples from scratch
    input_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/MTr3_'+TrainSampleID+'_hits.csv'
    print(UI.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
    data=pd.read_csv(input_file_location,header=0,usecols=['x','y','z'])
    print(UI.TimeStamp(),'Analysing data... ',bcolors.ENDC)
    y_offset=data['y'].min()
    x_offset=data['x'].min()
    z_offset=data['z'].min()
    data['x']=data['x']-x_offset #Reseting the coordinate origin to zero for this data set
    data['y']=data['y']-y_offset #Reseting the coordinate origin to zero for this data set
    data['z']=data['z']-z_offset #Reseting the coordinate origin to zero for this data set
    x_max=data['x'].max() #We need it to calculate how many clusters to create
    y_max=data['y'].max()
    z_max=data['z'].max()
    if Xoverlap==1:
            Xsteps=math.ceil((x_max)/stepX)
    else:
            Xsteps=(math.ceil((x_max)/stepX)*(Xoverlap))-1
    if Yoverlap==1:
            Ysteps=math.ceil((y_max)/stepY)
    else:
            Ysteps=(math.ceil((y_max)/stepY)*(Yoverlap))-1
    if Zoverlap==1:
       Zsteps=math.ceil((z_max)/stepZ)
    else:
       Zsteps=(math.ceil((z_max)/stepZ)*(Zoverlap))-1

    print(UI.TimeStamp(),'Distributing hit files...')
    print(UI.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
    data=pd.read_csv(input_file_location,header=0)
    data['x']=data['x']-x_offset #Reseting the coordinate origin to zero for this data set
    data['y']=data['y']-y_offset #Reseting the coordinate origin to zero for this data set
    data['z']=data['z']-z_offset #Reseting the coordinate origin to zero for this data set
    n_jobs=0
    jobs=[]
    with alive_bar(Xsteps*Ysteps*Zsteps,force_tty=True, title='Sampling hit files...') as bar:
        for i in range(Xsteps):
            for j in range(Ysteps):
                     for k in range(Zsteps):
                         required_tfile_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/MTr3_'+TrainSampleID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_hits.csv'
                         if os.path.isfile(required_tfile_location)==False:
                             X_ID=int(i)/Xoverlap
                             Y_ID=int(j)/Yoverlap
                             Z_ID=int(k)/Yoverlap
                             tdata=data.drop(data.index[data['x'] >= ((X_ID+1)*stepX)])  #Keeping the relevant z slice
                             tdata.drop(tdata.index[tdata['x'] < (X_ID*stepX)], inplace = True)  #Keeping the relevant z slice
                             tdata.drop(tdata.index[tdata['y'] >= ((Y_ID+1)*stepY)], inplace = True)  #Keeping the relevant z slice
                             tdata.drop(tdata.index[tdata['y'] < (Y_ID*stepY)], inplace = True)  #Keeping the relevant z slice
                             tdata.drop(tdata.index[tdata['z'] >= ((Z_ID+1)*stepZ)], inplace = True)  #Keeping the relevant z slice
                             tdata.drop(tdata.index[tdata['z'] < (Z_ID*stepZ)], inplace = True)  #Keeping the relevant z slice
                             if len(tdata)>1:
                                 if Sampling>=random.random():
                                     tdata.to_csv(required_tfile_location,index=False)
                                     job_comb=[i, j, k]
                                     jobs.append(job_comb)
                                     n_jobs+=1
                                     print(UI.TimeStamp(), bcolors.OKGREEN+"The hit data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+required_tfile_location+bcolors.ENDC)
                         bar()
    TrainDataMeta=UI.JobMeta(TrainSampleID)
    TrainDataMeta.UpdateJobMeta(['stepX', 'stepY', 'stepZ', 'cut_dt', 'cut_dr', 'cut_dz', 'testRatio', 'valRatio', 'y_offset', 'x_offset','Xoverlap', 'Yoverlap', 'Zoverlap'],[stepX, stepY, stepZ, cut_dt, cut_dr, cut_dz, testRatio, valRatio, y_offset, x_offset,Xoverlap, Yoverlap, Zoverlap])
    TrainDataMeta.UpdateJobMeta(['jobs', 'n_jobs'],[jobs, n_jobs])
    TrainDataMeta.UpdateStatus(1)
    Meta=TrainDataMeta
    print(UI.PickleOperations(TrainSampleOutputMeta,'w', TrainDataMeta)[1])
elif os.path.isfile(TrainSampleOutputMeta)==True:
    print(UI.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+TrainSampleOutputMeta+bcolors.ENDC)
    #Loading parameters from the Meta file if exists.
    MetaInput=UI.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
    stepX=Meta.stepX
    stepY=Meta.stepY
    stepZ=Meta.stepZ
    cut_dt=Meta.cut_dt
    cut_dr=Meta.cut_dr
    cut_dz=Meta.cut_dz
    testRatio=Meta.testRatio
    valRatio=Meta.valRatio
    y_offset=Meta.y_offset
    x_offset=Meta.x_offset
    Yoverlap=Meta.Yoverlap
    Xoverlap=Meta.Xoverlap
    Zoverlap=Meta.Zoverlap

    jobs=Meta.jobs
    n_jobs=Meta.n_jobs

# ########################################     Preset framework parameters    #########################################

UI.Msg('vanilla','Analysing the current script status...')
Status=Meta.Status[-1]
if ForceStatus!='N':
    Status=int(ForceStatus)
UI.Msg('vanilla','Current stage is '+str(Status)+'...')

################ Set the execution sequence for the script
Program=[]
###### Stage 0
prog_entry=[]
prog_entry.append(' Sending hit cluster to the HTCondor, so the model assigns weights between hits')
prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/','SelectedTrainClusters','MTr3','.pkl',TrainSampleID,n_jobs,'MTr3_GenerateTrainClusters_Sub.py'])
prog_entry.append([' --stepY ', ' --stepX ', ' --stepZ ', ' --cut_dt ', ' --cut_dr ', ' --cut_dz ',' --Yoverlap ',' --Xoverlap ',' --Zoverlap ', ' --jobs ', ' --SeedFlowLog ',  ' --ModelName '])
prog_entry.append([stepY, stepX, stepZ, cut_dt,cut_dr,cut_dz, Yoverlap, Xoverlap, Zoverlap, '"'+str(jobs)+'"', SeedFlowLog, ModelName])
prog_entry.append(n_jobs)
prog_entry.append(LocalSub)
prog_entry.append('N/A')
prog_entry.append(HTCondorLog)
prog_entry.append(False)
Program.append(prog_entry)
print(UI.TimeStamp(),UI.ManageTempFolders(prog_entry))

###### Stage 3
Program.append('Custom')
print(UI.TimeStamp(),'There are '+str(len(Program)+1)+' stages (0-'+str(len(Program)+1)+') of this script',bcolors.ENDC)
print(UI.TimeStamp(),'Current stage has a code',Status,bcolors.ENDC)
while Status<len(Program):
    if Program[Status]!='Custom':
        #Standard process here
       Result=UI.StandardProcess(Program,Status,SubGap,SubPause,CPU,JobFlavour,Memory,time_int,Patience)
       if Result[0]:
            UI.UpdateStatus(Status+1,Meta,TrainSampleOutputMeta)
       else:
             Status=20
             break

    elif Status==1:
        try:
            #Non standard processes (that don't follow the general pattern) have been coded here
            print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
            print(UI.TimeStamp(),bcolors.BOLD+'Stage 2:'+bcolors.ENDC+' Accumulating results from the previous step')
            SampleCount=0
            Samples=[]
            SeedFlowLabels=['All','Excluding self-permutations', 'Excluding duplicates','Excluding seeds on the same plate', 'Cut on dz', 'Cut on dtx', 'Cut on dty' , 'Cut on drx', 'Cut on dry', 'MLP filter', 'GNN filter', 'Tracking process' ]
            SeedFlowValuesAll=[0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            SeedFlowValuesTrue=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            with alive_bar(n_jobs,force_tty=True, title='Consolidating the output...') as bar:
                for i in range(n_jobs):
                        source_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/Temp_MTr3_'+TrainSampleID+'_0/MTr3_'+TrainSampleID+'_SelectedTrainSeedClusters_'+str(i)+'.pkl'
                        UI.Msg('location','Analysing file:',content2=source_output_file_location)
                        SampleCluster=UI.PickleOperations(source_output_file_location,'r', 'N/A')[0]
                        SeedFlowValuesAll = [a + b for a, b in zip(SeedFlowValuesAll, SampleCluster.SeedFlowValuesAll)]
                        SeedFlowValuesTrue = [a + b for a, b in zip(SeedFlowValuesTrue, SampleCluster.SeedFlowValuesTrue)]
                        Samples+=(SampleCluster)
                        bar()

            UI.Msg('vanilla','Printing the seed cutflow...')
            headers = SeedFlowLabels
            first_row = SeedFlowValuesAll
            second_row = SeedFlowValuesTrue

            # Create a DataFrame
            data = [first_row, second_row]
            df = pd.DataFrame(data, columns=headers)

            # Print the table with borders
            print(df.to_string(index=False))
            Meta.UpdateJobMeta(['SeedFlowLabels', 'SeedFlowValuesAll', 'SeedFlowValuesTrue'],[SeedFlowLabels, SeedFlowValuesAll, SeedFlowValuesTrue])
            random.shuffle(Samples)
            print(UI.PickleOperations(TrainSampleOutputMeta,'w', Meta)[1])
            TrainSamples=[]
            ValSamples=[]
            TestSamples=[]
            TrainFraction=int(math.floor(len(Samples)*(1.0-(Meta.testRatio+Meta.valRatio))))
            ValFraction=int(math.ceil(len(Samples)*Meta.valRatio))

            for s in range(0,TrainFraction):
                        TrainSamples.append(Samples[s])
            for s in range(TrainFraction,TrainFraction+ValFraction):
                         ValSamples.append(Samples[s])
            for s in range(TrainFraction+ValFraction,len(Samples)):
                         TestSamples.append(Samples[s])
            output_train_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_SEEDS'+'.pkl'
            print(UI.PickleOperations(output_train_file_location,'w', TrainSamples)[1])
            output_val_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_SEEDS'+'.pkl'
            print(UI.PickleOperations(output_val_file_location,'w', ValSamples)[1])
            output_test_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TEST_SEEDS'+'.pkl'
            print(UI.PickleOperations(output_test_file_location,'w', TestSamples)[1])
            print(UI.TimeStamp(), bcolors.OKGREEN+"Train data has been re-generated successfully..."+bcolors.ENDC)
            print(UI.TimeStamp(),bcolors.OKGREEN+'Please run MTr4_TrainModel.py after this to create/train a model'+bcolors.ENDC)
            UI.UpdateStatus(Status+1,Meta,TrainSampleOutputMeta)
            print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
        except Exception as e:
          print(UI.TimeStamp(),bcolors.FAIL+'Stage 2 is uncompleted due to: '+str(e)+bcolors.ENDC)
          Status=21
          break
    MetaInput=UI.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
    Status=Meta.Status[-1]

if Status<20:

    #Removing the temp files that were generated by the process
    print(UI.TimeStamp(),'Performing the cleanup... ')
    print(UI.ManageFolders(AFS_DIR, EOS_DIR, TrainSampleID,'d',['MTr1']))
    UI.Msg('success',"Segment merging has been completed")
else:
    UI.Msg('failed',"Segment merging has not been completed as one of the processes has timed out. Please run the script again (without Reset Mode).")
    exit()



# def Success(Finished):
#         if Finished:
#             print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
#             print(UF.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' All HTCondor jobs have completed.')
#             count=0
#             SampleCount=0
#             NodeFeatures=PM.num_node_features
#             EdgeFeatures=PM.num_edge_features
#             for k in range(0,Zsteps):
#                 progress=round((float(k)/float(Zsteps))*100,2)
#                 print(UF.TimeStamp(),"Collating results, progress is ",progress,' %') #Progress display
#                 for i in range(0,Xsteps):
#                         count+=1
#                         source_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MTr1a_'+TrainSampleID+'_SelectedTrainClusters_'+str(k)+'_'+str(i)+'.pkl'
#                         destination_output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TTr_OUTPUT_'+str(count)+'.pkl'
#                         os.rename(source_output_file_location, destination_output_file_location)
#                         TrainingSample=UF.PickleOperations(destination_output_file_location,'r', 'N/A')[0]
#                         SampleCount+=len(TrainingSample)
#             print(UF.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+TrainSampleOutputMeta+bcolors.ENDC)
#             MetaInput=UF.PickleOperations(TrainSampleOutputMeta,'r', 'N/A')
#             MetaInput[0].UpdateHitClusterMetaData(SampleCount,NodeFeatures,EdgeFeatures,count)
#             print(UF.PickleOperations(TrainSampleOutputMeta,'w', MetaInput[0])[1])
#             HTCondorTag="SoftUsed == \"ANNDEA-MTr-"+TrainSampleID+"\""
#             UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'MTr1_'+TrainSampleID, ['MTr1a_'+TrainSampleID,'ETr1_'+TrainSampleID,'MTr1_'+TrainSampleID], HTCondorTag) #If successful we delete all temp files created by the process
#             print(UF.TimeStamp(),bcolors.OKGREEN+'Training samples are ready for the model creation/training'+bcolors.ENDC)
#             TrainSampleInputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
#             print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+TrainSampleInputMeta+bcolors.ENDC)
#             MetaInput=UF.PickleOperations(TrainSampleInputMeta,'r', 'N/A')
#             print(MetaInput[1])
#             Meta=MetaInput[0]
#             TrainSamples=[]
#             ValSamples=[]
#             TestSamples=[]
#             for i in range(1,Meta.no_sets+1):
#                 flocation=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TTr_OUTPUT_'+str(i)+'.pkl'
#                 print(UF.TimeStamp(),'Loading data from ',bcolors.OKBLUE+flocation+bcolors.ENDC)
#                 TrainClusters=UF.PickleOperations(flocation,'r', 'N/A')
#                 TrainClusters=TrainClusters[0]
#                 TrainFraction=int(math.floor(len(TrainClusters)*(1.0-(Meta.testRatio+Meta.valRatio))))
#                 ValFraction=int(math.ceil(len(TrainClusters)*Meta.valRatio))
#                 for smpl in range(0,TrainFraction):
#                            if TrainClusters[smpl].ClusterGraph.num_edges>0 and Sampling>=random.random(): #Only graph clusters with edges are kept + sampled
#                              TrainSamples.append(TrainClusters[smpl].ClusterGraph)
#                 for smpl in range(TrainFraction,TrainFraction+ValFraction):
#                            if TrainClusters[smpl].ClusterGraph.num_edges>0 and Sampling>=random.random():
#                              ValSamples.append(TrainClusters[smpl].ClusterGraph)
#                 for smpl in range(TrainFraction+ValFraction,len(TrainClusters)):
#                            if TrainClusters[smpl].ClusterGraph.num_edges>0 and Sampling>=random.random():
#                              TestSamples.append(TrainClusters[smpl].ClusterGraph)
#             output_train_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_SAMPLES'+'.pkl'
#             print(UF.PickleOperations(output_train_file_location,'w', TrainSamples)[1])
#             output_val_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_SAMPLES'+'.pkl'
#             print(UF.PickleOperations(output_val_file_location,'w', ValSamples)[1])
#             output_test_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TEST_SAMPLES'+'.pkl'
#             print(UF.PickleOperations(output_test_file_location,'w', TestSamples)[1])
#             print(UF.TimeStamp(), bcolors.OKGREEN+"Train data has been re-generated successfully..."+bcolors.ENDC)
#             print(UF.TimeStamp(),bcolors.OKGREEN+'Please run M2_TrainModel.py after this to create/train a model'+bcolors.ENDC)
#             print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
#         else:
#             print(UF.TimeStamp(),bcolors.FAIL+'Unfortunately no results have been yield. Please rerun the script, and if the problem persists, check that the HTCondor jobs run adequately.'+bcolors.ENDC)
#             exit()






