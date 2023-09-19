#This script aligns the brick based on the reconstructed volumene tracks
#Tracking Module of the ANNDEA package
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
    sys.path.append('/usr/lib64/python36.zip')
    sys.path.append('/usr/lib64/python3.6')
    sys.path.append('/usr/lib64/python3.6/lib-dynload')
    sys.path.append('/usr/lib64/python3.6/site-packages')
    sys.path.append('/usr/lib/python3.6/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')
import UtilityFunctions as UF #This is where we keep routine utility functions
import Parameters as PM #This is where we keep framework global parameters
import pandas as pd #We use Panda for a routine data processing
pd.options.mode.chained_assignment = None #Silence annoying warnings
import math #We use it for data manipulation
import numpy as np
import os
import time
from alive_progress import alive_bar
import argparse
import ast
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
print(bcolors.HEADER+"#########                 Initialising ANNDEA ECC Alignment module                       ###############"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)

#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script prepares training data for training the tracking model')
parser.add_argument('--SubPause',help="How long to wait in minutes after submitting 10000 jobs?", default='60')
parser.add_argument('--SubGap',help="How long to wait in minutes after submitting 10000 jobs?", default='10000')
parser.add_argument('--LocalSub',help="Local submission?", default='N')
parser.add_argument('--RequestExtCPU',help="Would you like to request extra CPUs? How Many?", default=1)
parser.add_argument('--JobFlavour',help="Specifying the length of the HTCondor job wall time. Currently at 'workday' which is 8 hours.", default='workday')
parser.add_argument('--TrackID',help="What track name is used?", default='ANN_Track_ID')
parser.add_argument('--BrickID',help="What brick ID name is used?", default='ANN_Brick_ID')
parser.add_argument('--Mode', help='Script will continue from the last checkpoint, unless you want to start from the scratch, then type "Reset"',default='')
parser.add_argument('--Patience',help="How many checks to do before resubmitting the job?", default='15')
parser.add_argument('--RecBatchID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_Rec_Raw_UR.csv')
parser.add_argument('--Size',help="Split the cross section of the brick in the squares with the size being a length of such a square.", default='0')
parser.add_argument('--ReqMemory',help="Specifying the length of the HTCondor job walltime. Currently at 'workday' which is 8 hours.", default='2 GB')
parser.add_argument('--MinHits',help="What is the minimum number of hits per track?", default=50,type=int)
parser.add_argument('--ValMinHits',help="What is the validation minimum number of hits per track?", default=45,type=int)
parser.add_argument('--Cycle',help="Number of cycles", default='1')
parser.add_argument('--SpatialOptBound',help="Size", default='200')
parser.add_argument('--AngularOptBound',help="Size", default='2')

######################################## Parsing argument values  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
RecBatchID=args.RecBatchID
Patience=int(args.Patience)
TrackID=args.TrackID
BrickID=args.BrickID

MinHits=int(args.MinHits)
ValMinHits=int(args.ValMinHits)
Size=int(args.Size)
SpatialOptBound=args.SpatialOptBound
AngularOptBound=args.AngularOptBound
Cycle=int(args.Cycle)

SubPause=int(args.SubPause)*60
SubGap=int(args.SubGap)
LocalSub=(args.LocalSub=='Y')
if LocalSub:
   time_int=0
else:
    time_int=10
JobFlavour=args.JobFlavour
RequestExtCPU=int(args.RequestExtCPU)
ReqMemory=args.ReqMemory
Patience=int(args.Patience)
initial_input_file_location=args.f
FreshStart=True
#Establishing paths
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
RecOutputMeta=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_info.pkl'
required_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/R_'+RecBatchID+'_HITS.csv'

#Defining some functions
def FitPlate(PlateZ,dx,dy,input_data,Track_ID):
    change_df = pd.DataFrame([[PlateZ,dx,dy]], columns = ['Plate_ID','dx','dy'])
    temp_data=input_data[[Track_ID,'x','y','z','Track_No','Plate_ID']]
    temp_data=pd.merge(temp_data,change_df,on='Plate_ID',how='left')
    temp_data['dx'] = temp_data['dx'].fillna(0.0)
    temp_data['dy'] = temp_data['dy'].fillna(0.0)
    temp_data['x']=temp_data['x']+temp_data['dx']
    temp_data['y']=temp_data['y']+temp_data['dy']
    temp_data=temp_data[[Track_ID,'x','y','z','Track_No']]
    Tracks_Head=temp_data[[Track_ID]]
    Tracks_Head.drop_duplicates(inplace=True)
    Tracks_List=temp_data.values.tolist() #I find it is much easier to deal with tracks in list format when it comes to fitting
    Tracks_Head=Tracks_Head.values.tolist()
    #Bellow we build the track representatation that we can use to fit slopes
    for bth in Tracks_Head:
                   bth.append([])
                   bt=0
                   trigger=False
                   while bt<(len(Tracks_List)):
                       if bth[0]==Tracks_List[bt][0]:

                           bth[1].append(Tracks_List[bt][1:4])
                           del Tracks_List[bt]
                           bt-=1
                           trigger=True
                       elif trigger:
                            break
                       else:
                            continue
                       bt+=1
    for bth in Tracks_Head:
           x,y,z=[],[],[]
           for b in bth[1]:
               x.append(b[0])
               y.append(b[1])
               z.append(b[2])
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
           del(bth[1])
    #Once we get coefficients for all tracks we convert them back to Pandas dataframe and join back to the data
    Tracks_Head=pd.DataFrame(Tracks_Head, columns = [Track_ID,'ax','t1x','t2x','ay','t1y','t2y'])

    temp_data=pd.merge(temp_data,Tracks_Head,how='inner',on = [Track_ID])
    #Calculating x and y coordinates of the fitted line for all plates in the track
    temp_data['new_x']=temp_data['ax']+(temp_data['z']*temp_data['t1x'])+((temp_data['z']**2)*temp_data['t2x'])
    temp_data['new_y']=temp_data['ay']+(temp_data['z']*temp_data['t1y'])+((temp_data['z']**2)*temp_data['t2y'])
    #Calculating how far hits deviate from the fit polynomial
    temp_data['d_x']=temp_data['x']-temp_data['new_x']
    temp_data['d_y']=temp_data['y']-temp_data['new_y']
    temp_data['d_r']=temp_data['d_x']**2+temp_data['d_y']**2
    temp_data['d_r'] = temp_data['d_r'].astype(float)
    temp_data['d_r']=np.sqrt(temp_data['d_r']) #Absolute distance
    temp_data=temp_data[[Track_ID,'Track_No','d_r']]
    temp_data=temp_data.groupby([Track_ID,'Track_No']).agg({'d_r':'sum'}).reset_index()

    temp_data=temp_data.agg({'d_r':'sum','Track_No':'sum'})
    temp_data=temp_data.values.tolist()
    fit=temp_data[0]/temp_data[1]
    return fit



def FitPlateAngle(PlateZ,dtx,dty,input_data,Track_ID):
    change_df = pd.DataFrame([[PlateZ,dtx,dty]], columns = ['Plate_ID','dtx','dty'])
    temp_data=input_data[[Track_ID,'x','y','z','tx','ty','Track_No','Plate_ID']]
    temp_data=pd.merge(temp_data,change_df,on='Plate_ID',how='left')
    temp_data['dtx'] = temp_data['dtx'].fillna(0.0)
    temp_data['dty'] = temp_data['dty'].fillna(0.0)
    temp_data['tx']=temp_data['tx']+temp_data['dtx']
    temp_data['ty']=temp_data['ty']+temp_data['dty']
    temp_data=temp_data[[Track_ID,'x','y','z','tx','ty','Track_No']]
    Tracks_Head=temp_data[[Track_ID]]
    Tracks_Head.drop_duplicates(inplace=True)
    Tracks_List=temp_data.values.tolist() #I find it is much easier to deal with tracks in list format when it comes to fitting
    Tracks_Head=Tracks_Head.values.tolist()
    #Bellow we build the track representatation that we can use to fit slopes
    for bth in Tracks_Head:
                   bth.append([])
                   bt=0
                   trigger=False
                   while bt<(len(Tracks_List)):
                       if bth[0]==Tracks_List[bt][0]:

                           bth[1].append(Tracks_List[bt][1:4])
                           del Tracks_List[bt]
                           bt-=1
                           trigger=True
                       elif trigger:
                            break
                       else:
                            continue
                       bt+=1
    for bth in Tracks_Head:
           x,y,z=[],[],[]
           for b in bth[1]:
               x.append(b[0])
               y.append(b[1])
               z.append(b[2])
           tx=np.polyfit(z,x,1)[0]
           ty=np.polyfit(z,y,1)[0]
           bth.append(tx) #Append x slope
           bth.append(ty) #Append x slope
           del(bth[1])
    #Once we get coefficients for all tracks we convert them back to Pandas dataframe and join back to the data
    Tracks_Head=pd.DataFrame(Tracks_Head, columns = [Track_ID,'ntx','nty'])

    temp_data=pd.merge(temp_data,Tracks_Head,how='inner',on = [Track_ID])

    #Calculating x and y coordinates of the fitted line for all plates in the track
    #Calculating how far hits deviate from the fit polynomial
    temp_data['d_tx']=temp_data['tx']-temp_data['ntx']
    temp_data['d_ty']=temp_data['ty']-temp_data['nty']

    temp_data['d_tr']=temp_data['d_tx']**2+temp_data['d_ty']**2
    temp_data['d_tr'] = temp_data['d_tr'].astype(float)
    temp_data['d_tr']=np.sqrt(temp_data['d_tr']) #Absolute distance

    temp_data=temp_data[[Track_ID,'Track_No','d_tr']]
    temp_data=temp_data.groupby([Track_ID,'Track_No']).agg({'d_tr':'sum'}).reset_index()
    temp_data=temp_data.agg({'d_tr':'sum','Track_No':'sum'})
    temp_data=temp_data.values.tolist()
    fit=temp_data[0]/temp_data[1]
    return fit
########################################     Phase 1 - Create compact source file    #########################################
print(UF.TimeStamp(),bcolors.BOLD+'Stage 0:'+bcolors.ENDC+' Preparing the source data...')

if os.path.isfile(required_file_location)==False or Mode=='RESET':
        print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+initial_input_file_location+bcolors.ENDC)
        if BrickID=='':
            ColUse=[TrackID,PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
        else:
            ColUse=[TrackID,BrickID,PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
        data=pd.read_csv(initial_input_file_location,
                    header=0,
                    usecols=ColUse)
        Min_x=data[PM.x].min()
        Max_x=data[PM.x].max()
        Min_y=data[PM.y].min()
        Max_y=data[PM.y].max()
        if BrickID=='':
            data[BrickID]='D'
        total_rows=len(data.axes[0])
        print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
        print(UF.TimeStamp(),'Removing unreconstructed hits...')
        data=data.dropna()

        final_rows=len(data.axes[0])
        print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
        data[BrickID] = data[BrickID].astype(str)
        data[TrackID] = data[TrackID].astype(str)

        data['Rec_Seg_ID'] = data[TrackID] + '-' + data[BrickID]
        data=data.drop([TrackID],axis=1)
        data=data.drop([BrickID],axis=1)

        print(UF.TimeStamp(),'Removing tracks which have less than',ValMinHits,'hits...')
        track_no_data=data.groupby(['Rec_Seg_ID'],as_index=False).count()
        track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty,PM.Hit_ID],axis=1)
        track_no_data=track_no_data.rename(columns={PM.x: "Track_No"})
        new_combined_data=pd.merge(data, track_no_data, how="left", on=["Rec_Seg_ID"])
        new_combined_data = new_combined_data[new_combined_data.Track_No >= ValMinHits]
        new_combined_data['Plate_ID']=new_combined_data['z'].astype(int)

        new_combined_data=new_combined_data.sort_values(['Rec_Seg_ID',PM.x],ascending=[1,1])
        grand_final_rows=len(new_combined_data.axes[0])
        print(UF.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
        new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
        new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
        new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
        new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
        new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
        new_combined_data=new_combined_data.rename(columns={PM.Hit_ID: "Hit_ID"})
        train_data = new_combined_data[new_combined_data.Track_No >= MinHits]
        validation_data = new_combined_data[new_combined_data.Track_No >= ValMinHits]
        validation_data = validation_data[validation_data.Track_No < MinHits]
        print(UF.TimeStamp(),'Analysing the data sample in order to understand how many jobs to submit to HTCondor... ',bcolors.ENDC)
        Sets=new_combined_data.z.unique().size

        x_no=int(math.ceil((Max_x-Min_x)/Size))

        print(UF.TimeStamp(),'Working out the number of plates to align')
        plates=train_data[['Plate_ID']].sort_values(['Plate_ID'],ascending=[1])
        plates.drop_duplicates(inplace=True)
        plates=plates.values.tolist() #I find it is much easier to deal with tracks in list format when it comes to fitting
        print(UF.TimeStamp(),'There are',len(plates),'plates')
        print(UF.TimeStamp(),'Initial validation spatial residual value is',bcolors.BOLD+str(round(FitPlate(plates[0][0],0,0,validation_data,'Rec_Seg_ID'),2))+bcolors.ENDC, 'microns')
        print(UF.TimeStamp(),'Initial validatio residual value is',bcolors.BOLD+str(round(FitPlateAngle(plates[0][0],0,0,validation_data,'Rec_Seg_ID')*1000,1))+bcolors.ENDC, 'milliradians')

        y_no=int(math.ceil((Max_y-Min_y)/Size))
        for j in range(x_no):
            for k in range(y_no):
                required_temp_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/R_'+RecBatchID+'_HITS_'+str(j)+'_'+str(k)+'.csv'
                x_min_cut=Min_x+(Size*j)
                x_max_cut=Min_x+(Size*(j+1))
                y_min_cut=Min_y+(Size*k)
                y_max_cut=Min_y+(Size*(k+1))
                temp_data=new_combined_data[new_combined_data.x >= x_min_cut]
                temp_data=temp_data[temp_data.x < x_max_cut]
                temp_data=temp_data[temp_data.y >= y_min_cut]
                temp_data=temp_data[temp_data.y < y_max_cut]
                temp_data.to_csv(required_temp_file_location,index=False)
                print(UF.TimeStamp(), bcolors.OKGREEN+"The granular hit data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+required_temp_file_location+bcolors.ENDC)
        JobSets=[]
        new_combined_data.to_csv(required_file_location,index=False)
        print(UF.TimeStamp(), bcolors.OKGREEN+"The hit data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+required_file_location+bcolors.ENDC)

        for i in range(Sets):
            JobSets.append([])
            for j in range(x_no):
                JobSets[i].append(y_no)
        Meta=UF.TrainingSampleMeta(RecBatchID)
        Meta.IniBrickAlignMetaData(Size,ValMinHits,MinHits,SpatialOptBound,AngularOptBound,JobSets,Cycle,plates,[Min_x,Max_x,Min_y,Max_y])
        Meta.UpdateStatus(0)
        print(UF.PickleOperations(RecOutputMeta,'w', Meta)[1])
        print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.OKGREEN+'Stage 0 has successfully completed'+bcolors.ENDC)
elif os.path.isfile(RecOutputMeta)==True:
    print(UF.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+RecOutputMeta+bcolors.ENDC)
    MetaInput=UF.PickleOperations(RecOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
Size=Meta.Size
ValMinHits=Meta.ValMinHits
MinHits=Meta.MinHits
SpatialOptBound=Meta.SpatialOptBound
JobSets=Meta.JobSets
AngularOptBound=Meta.AngularOptBound
Cycle=Meta.Cycles
plates=Meta.plates
Min_x=Meta.FiducialVolume[0]
Max_x=Meta.FiducialVolume[1]
Min_y=Meta.FiducialVolume[2]
Max_y=Meta.FiducialVolume[3]
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
                                    False,
                                    program[6])
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
                                    False,
                                    program[status][6])


        if len(bad_pop)==0:
             print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(status)+' has successfully completed'+bcolors.ENDC)
             UpdateStatus(status+1)
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
                                    batch_sub,
                                    program[status][6])
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
            # if freshstart:
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

def UpdateStatus(status):
    Meta.UpdateStatus(status)
    print(UF.PickleOperations(RecOutputMeta,'w', Meta)[1])

if Mode=='RESET':
    print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
    #HTCondorTag="SoftUsed == \"ANNDEA-R-"+RecBatchID+"\""
    #UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R_'+RecBatchID, ['R_'+RecBatchID], HTCondorTag)
    FreshStart=False
    UpdateStatus(0)
    Status=0
else:
    print(UF.TimeStamp(),'Analysing the current script status...',bcolors.ENDC)
    Status=Meta.Status[-1]

print(UF.TimeStamp(),'Current status is ',Status,bcolors.ENDC)
################ Set the execution sequence for the script
Program=[]

# UpdateStatus(0)
# Status=0

if Mode=='CLEANUP':
    UpdateStatus(19)
    Status=19

# ###### Stage 2

TotJobs=0

if type(JobSets) is int:
            TotJobs=JobSets
elif type(JobSets[0]) is int:
            TotJobs=np.sum(JobSets)
elif type(JobSets[0][0]) is int:
            for lp in JobSets:
                TotJobs+=np.sum(lp)
for c in range(Cycle):
    prog_entry=[]
    prog_entry.append(' Sending tracks to the HTCondor, so track segment combinations can be formed...')
    prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','SpatialAlignmentResult_'+str(c),'Ra','.csv',RecBatchID,JobSets,'Ra_SpatiallyAlignBrick_Sub.py'])
    prog_entry.append([ " --MinHits ", " --ValMinHits "," --Size ", " --OptBound ", " --Plate "])
    prog_entry.append([MinHits, ValMinHits, Size, SpatialOptBound, '"'+str(plates)+'"'])
    prog_entry.append(TotJobs)
    prog_entry.append(LocalSub)
    prog_entry.append(["",""])
    if Mode=='RESET' and c==0:
            print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
        #Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
    print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))
    Program.append(prog_entry)
    Program.append('Custom: Spatial Cycle '+str(c))
for c in range(Cycle):
    prog_entry=[]
    prog_entry.append(' Sending tracks to the HTCondor, so track segment combinations can be formed...')
    prog_entry.append([AFS_DIR,EOS_DIR,PY_DIR,'/ANNDEA/Data/REC_SET/','AngularAlignmentResult_'+str(c),'Rb','.csv',RecBatchID,JobSets,'Rb_AngularAlignBrick_Sub.py'])
    prog_entry.append([ " --MinHits ", " --ValMinHits "," --Size ", " --OptBound ", " --Plate "])
    prog_entry.append([MinHits, ValMinHits, Size, AngularOptBound, '"'+str(plates)+'"'])
    prog_entry.append(TotJobs)
    prog_entry.append(LocalSub)
    prog_entry.append(["",""])
    if Mode=='RESET' and c==0:
            print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Delete'))
        #Setting up folders for the output. The reconstruction of just one brick can easily generate >100k of files. Keeping all that blob in one directory can cause problems on lxplus.
    print(UF.TimeStamp(),UF.ManageTempFolders(prog_entry,'Create'))
    Program.append(prog_entry)
    Program.append('Custom: Angular Cycle '+str(c))
Program.append('Custom: Final')

# Program.append('Custom - TrackMapping')
while Status<len(Program):
    if Program[Status][:6]!='Custom':
        #Standard process here
        Result=StandardProcess(Program,Status,FreshStart)
        if Result[0]:
             FreshStart=Result[1]
        else:
             Status=20
             break
    elif Program[Status][:21]=='Custom: Spatial Cycle':
        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.BOLD+'Stage '+str(Status)+':'+bcolors.ENDC+' Collecting results from the previous step')
        result=[]
        for i in range(0,len(JobSets)): #//Temporarily measure to save space || Update 13/08/23 - I have commented it out as it creates more problems than solves it
            for j in range(len(JobSets[i])):
                for k in range(JobSets[i][j]):
                  result_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_Ra'+'_'+RecBatchID+'_'+str(i)+'/Ra_'+RecBatchID+'_SpatialAlignmentResult_'+Program[Status][22:]+'_'+str(i)+'_'+str(j)+'_'+str(k)+'.csv'
                  result.append(UF.LogOperations(result_file_location,'r','N/A')[0])
        result=pd.DataFrame(result,columns=['Type','Plate_ID','j','k','dx','FitX','ValFitX','dy','FitY','ValFitY'])
        log_result=result
        log_result['Cycle']=Program[Status][22:]
        if Program[Status][22:]=='0':
            log_result.to_csv(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv',mode="w",index=False)
        else:
            log_result.to_csv(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv',mode="a",index=False,header=False)
        result=result[['Type','Plate_ID','j','k','dx','dy']]
        required_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/R_'+RecBatchID+'_HITS.csv'
        data=pd.read_csv(required_file_location,header=0)

        data['j']=(data['x']-Min_x)/Size
        data['k']=(data['y']-Min_y)/Size
        data['j']=data['j'].apply(np.floor)
        data['k']=data['k'].apply(np.floor)
        result['Plate_ID'] = result['Plate_ID'].astype(int)
        result['dx'] = result['dx'].astype(float)
        result['dy'] = result['dy'].astype(float)
        result['j'] = result['j'].astype(int)
        result['k'] = result['k'].astype(int)
        data['j'] = data['j'].astype(int)
        data['k'] = data['k'].astype(int)
        data=pd.merge(data,result,on=['Plate_ID','j','k'],how='left')

        data['dx'] = data['dx'].fillna(0.0)
        data['dy'] = data['dy'].fillna(0.0)
        data['x']=data['x']+data['dx']
        data['y']=data['y']+data['dy']
        data.drop(['Type','dx','dy','k','j'],axis=1, inplace=True)
        validation_data = data[data.Track_No >= ValMinHits]
        validation_data = validation_data[validation_data.Track_No < MinHits]
        print(UF.TimeStamp(),'Cycle '+Program[Status][22:]+' validation spatial residual value after spatial alignment is',bcolors.BOLD+str(round(FitPlate(plates[0][0],0,0,validation_data,'Rec_Seg_ID'),2))+bcolors.ENDC, 'microns')
        print(UF.TimeStamp(),'Cycle '+Program[Status][22:]+' validation angular residual value after spatial alignment is',bcolors.BOLD+str(round(FitPlateAngle(plates[0][0],0,0,validation_data,'Rec_Seg_ID')*1000,1))+bcolors.ENDC, 'milliradians')
        x_no=int(math.ceil((Max_x-Min_x)/Size))
        y_no=int(math.ceil((Max_y-Min_y)/Size))
        for j in range(x_no):
            for k in range(y_no):
                required_temp_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/R_'+RecBatchID+'_HITS_'+str(j)+'_'+str(k)+'.csv'
                x_min_cut=Min_x+(Size*j)
                x_max_cut=Min_x+(Size*(j+1))
                y_min_cut=Min_y+(Size*k)
                y_max_cut=Min_y+(Size*(k+1))
                temp_data=data[data.x >= x_min_cut]
                temp_data=temp_data[temp_data.x < x_max_cut]
                temp_data=temp_data[temp_data.y >= y_min_cut]
                temp_data=temp_data[temp_data.y < y_max_cut]
                temp_data.to_csv(required_temp_file_location,index=False)
                print(UF.TimeStamp(), bcolors.OKGREEN+"The granular hit data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+required_temp_file_location+bcolors.ENDC)
        data.to_csv(required_file_location,index=False)
        print(UF.TimeStamp(), bcolors.OKGREEN+"The hit data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+required_file_location+bcolors.ENDC)
        UpdateStatus(Status+1)




        # print(UF.TimeStamp(),'Analysing the data sample in order to understand how many jobs to submit to HTCondor... ',bcolors.ENDC)

    elif Program[Status][:21]=='Custom: Angular Cycle':
        print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(),bcolors.BOLD+'Stage '+str(Status)+':'+bcolors.ENDC+' Collecting results from the previous step')
        result=[]
        for i in range(0,len(JobSets)): #//Temporarily measure to save space || Update 13/08/23 - I have commented it out as it creates more problems than solves it
            for j in range(len(JobSets[i])):
                for k in range(JobSets[i][j]):
                  result_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/Temp_Rb'+'_'+RecBatchID+'_'+str(i)+'/Rb_'+RecBatchID+'_AngularAlignmentResult_'+Program[Status][22:]+'_'+str(i)+'_'+str(j)+'_'+str(k)+'.csv'
                  result.append(UF.LogOperations(result_file_location,'r','N/A')[0])
        result=pd.DataFrame(result,columns=['Type','Plate_ID','j','k','dx','FitX','ValFitX','dy','FitY','ValFitY'])
        log_result=result
        log_result['Cycle']=Program[Status][22:]
        log_result.to_csv(EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv',mode="a",index=False,header=False)
        result=result[['Type','Plate_ID','j','k','dx','dy']]
        required_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/R_'+RecBatchID+'_HITS.csv'
        data=pd.read_csv(required_file_location,header=0)

        data['j']=(data['x']-Min_x)/Size
        data['k']=(data['y']-Min_y)/Size
        data['j']=data['j'].apply(np.floor)
        data['k']=data['k'].apply(np.floor)
        result['Plate_ID'] = result['Plate_ID'].astype(int)
        result['dx'] = result['dx'].astype(float)
        result['dy'] = result['dy'].astype(float)
        result['j'] = result['j'].astype(int)
        result['k'] = result['k'].astype(int)
        data['j'] = data['j'].astype(int)
        data['k'] = data['k'].astype(int)
        data=pd.merge(data,result,on=['Plate_ID','j','k'],how='left')

        data['dx'] = data['dx'].fillna(0.0)
        data['dy'] = data['dy'].fillna(0.0)
        data['tx']=data['tx']+data['dx']
        data['ty']=data['ty']+data['dy']
        data.drop(['Type','dx','dy','k','j'],axis=1, inplace=True)
        validation_data = data[data.Track_No >= ValMinHits]
        validation_data = validation_data[validation_data.Track_No < MinHits]
        print(UF.TimeStamp(),'Cycle '+Program[Status][22:]+' validation spatial residual value after angular alignment is',bcolors.BOLD+str(round(FitPlate(plates[0][0],0,0,validation_data,'Rec_Seg_ID'),2))+bcolors.ENDC, 'microns')
        print(UF.TimeStamp(),'Cycle '+Program[Status][22:]+' validation angular residual value after angular alignment is',bcolors.BOLD+str(round(FitPlateAngle(plates[0][0],0,0,validation_data,'Rec_Seg_ID')*1000,1))+bcolors.ENDC, 'milliradians')
        x_no=int(math.ceil((Max_x-Min_x)/Size))
        y_no=int(math.ceil((Max_y-Min_y)/Size))
        for j in range(x_no):
            for k in range(y_no):
                required_temp_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/R_'+RecBatchID+'_HITS_'+str(j)+'_'+str(k)+'.csv'
                x_min_cut=Min_x+(Size*j)
                x_max_cut=Min_x+(Size*(j+1))
                y_min_cut=Min_y+(Size*k)
                y_max_cut=Min_y+(Size*(k+1))
                temp_data=data[data.x >= x_min_cut]
                temp_data=temp_data[temp_data.x < x_max_cut]
                temp_data=temp_data[temp_data.y >= y_min_cut]
                temp_data=temp_data[temp_data.y < y_max_cut]
                temp_data.to_csv(required_temp_file_location,index=False)
                print(UF.TimeStamp(), bcolors.OKGREEN+"The granular hit data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+required_temp_file_location+bcolors.ENDC)
        data.to_csv(required_file_location,index=False)
        print(UF.TimeStamp(), bcolors.OKGREEN+"The hit data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+required_file_location+bcolors.ENDC)
        UpdateStatus(Status+1)
    elif Program[Status]=='Custom: Final':
        print(UF.TimeStamp(),'Mapping the alignment transportation map to input data',bcolors.OKBLUE+initial_input_file_location+bcolors.ENDC)
        alignment_data_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_REC_LOG.csv'
        print(UF.TimeStamp(),'Loading alignment data from',bcolors.OKBLUE+alignment_data_location+bcolors.ENDC)
        ColUse=['Type','Plate_ID','j','k','dx','dy','Cycle']
        alignment_data=pd.read_csv(alignment_data_location,
                    header=0,
                    usecols=ColUse)
        alignment_data.drop_duplicates(subset=['Type','Plate_ID','j','k','Cycle'],keep='first',inplace=True)
        alignment_data.drop(['Cycle'],axis=1, inplace=True)
        alignment_data=alignment_data.groupby(['Type','Plate_ID','j','k']).agg({'dx': 'sum', 'dy': 'sum'}).reset_index()
        spatial_alignment_map=alignment_data[alignment_data.Type=='Spatial'].drop(['Type'],axis=1)
        angular_alignment_map=alignment_data[alignment_data.Type=='Angular'].drop(['Type'],axis=1)

        print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+initial_input_file_location+bcolors.ENDC)
        data=pd.read_csv(initial_input_file_location,
                    header=0)
        data['Plate_ID']=data['z'].astype(int)
        print(UF.TimeStamp(),'Preparing initial data for the join...')
        data['j']=(data['x']-Min_x)/Size
        data['k']=(data['y']-Min_y)/Size
        data['j']=data['j'].apply(np.floor)
        data['k']=data['k'].apply(np.floor)
        print(UF.TimeStamp(),'Aligning spatial coordinates...')
        data=pd.merge(data,spatial_alignment_map,on=['Plate_ID','j','k'],how='left')
        data['dx'] = data['dx'].fillna(0.0)
        data['dy'] = data['dy'].fillna(0.0)
        data[PM.x]=data[PM.x]+data['dx']
        data[PM.y]=data[PM.y]+data['dy']
        data.drop(['dx','dy'],axis=1, inplace=True)
        print(UF.TimeStamp(),'Aligning angular coordinates...')
        data=pd.merge(data,angular_alignment_map,on=['Plate_ID','j','k'],how='left')
        data['dx'] = data['dx'].fillna(0.0)
        data['dy'] = data['dy'].fillna(0.0)
        data[PM.tx]=data[PM.tx]+data['dx']
        data[PM.ty]=data[PM.ty]+data['dy']
        data.drop(['Plate_ID','dx','dy','k','j'],axis=1, inplace=True)
        output_file_location=initial_input_file_location[:-4]+'_'+RecBatchID+'.csv'
        data.to_csv(output_file_location,index=False)
        print(UF.TimeStamp(),'Data has been realigned and saved in ',bcolors.OKBLUE+output_file_location+bcolors.ENDC)
        print(UF.TimeStamp(),'Measuring the validation alignment...')

        if BrickID=='':
            ColUse=[TrackID,PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
        else:
            ColUse=[TrackID,BrickID,PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty]
        if BrickID=='':
            data[BrickID]='D'
        data=data[ColUse]
        total_rows=len(data.axes[0])
        print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
        print(UF.TimeStamp(),'Removing unreconstructed hits...')
        data=data.dropna()

        final_rows=len(data.axes[0])
        print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
        data[BrickID] = data[BrickID].astype(str)
        data[TrackID] = data[TrackID].astype(str)

        data['Rec_Seg_ID'] = data[TrackID] + '-' + data[BrickID]
        data=data.drop([TrackID],axis=1)
        data=data.drop([BrickID],axis=1)

        print(UF.TimeStamp(),'Removing tracks which have less than',ValMinHits,'hits...')
        track_no_data=data.groupby(['Rec_Seg_ID'],as_index=False).count()
        track_no_data=track_no_data.drop([PM.y,PM.z,PM.tx,PM.ty,PM.Hit_ID],axis=1)
        track_no_data=track_no_data.rename(columns={PM.x: "Track_No"})
        new_combined_data=pd.merge(data, track_no_data, how="left", on=["Rec_Seg_ID"])
        new_combined_data = new_combined_data[new_combined_data.Track_No >= ValMinHits]
        new_combined_data['Plate_ID']=new_combined_data['z'].astype(int)
        new_combined_data=new_combined_data.sort_values(['Rec_Seg_ID',PM.x],ascending=[1,1])
        grand_final_rows=len(new_combined_data.axes[0])
        print(UF.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
        new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
        new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
        new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
        new_combined_data=new_combined_data.rename(columns={PM.tx: "tx"})
        new_combined_data=new_combined_data.rename(columns={PM.ty: "ty"})
        new_combined_data=new_combined_data.rename(columns={PM.Hit_ID: "Hit_ID"})
        validation_data = new_combined_data[new_combined_data.Track_No >= ValMinHits]
        validation_data = validation_data[validation_data.Track_No < MinHits]
        print(UF.TimeStamp(),'There are',len(plates),'plates')
        print(UF.TimeStamp(),'Final validation spatial residual value is',bcolors.BOLD+str(round(FitPlate(plates[0][0],0,0,validation_data,'Rec_Seg_ID'),2))+bcolors.ENDC, 'microns')
        print(UF.TimeStamp(),'Final validation residual value is',bcolors.BOLD+str(round(FitPlateAngle(plates[0][0],0,0,validation_data,'Rec_Seg_ID')*1000,1))+bcolors.ENDC, 'milliradians')



    #             # print(UF.TimeStamp(),bcolors.OKGREEN+'Stage '+str(Status)+' has successfully completed'+bcolors.ENDC)
    #             # UpdateStatus(Status+1)
    print(UF.TimeStamp(),'Loading previously saved data from ',bcolors.OKBLUE+RecOutputMeta+bcolors.ENDC)
    MetaInput=UF.PickleOperations(RecOutputMeta,'r', 'N/A')
    Meta=MetaInput[0]
    Status=Meta.Status[-1]

if Status<20:
    #Removing the temp files that were generated by the process
    print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
    HTCondorTag="SoftUsed == \"ANNDEA-R1-"+RecBatchID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R1_'+RecBatchID, ['R1'], HTCondorTag)
    HTCondorTag="SoftUsed == \"ANNDEA-R1a-"+RecBatchID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R1a_'+RecBatchID, [], HTCondorTag)
    HTCondorTag="SoftUsed == \"ANNDEA-R1b-"+RecBatchID+"\""
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R1b_'+RecBatchID, [], HTCondorTag)
    for p in Program:
        if p[:6]!='Custom':
           print(UF.TimeStamp(),UF.ManageTempFolders(p,'Delete'))
    print(UF.TimeStamp(), bcolors.OKGREEN+"Segment merging has been completed"+bcolors.ENDC)
else:
    print(UF.TimeStamp(), bcolors.FAIL+"Segment merging has not been completed as one of the processes has timed out. Please run the script again (without Reset Mode)."+bcolors.ENDC)
    exit()



