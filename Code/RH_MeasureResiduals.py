#Aligns the real data brick
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
import scipy
from scipy.optimize import minimize_scalar
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
print(bcolors.HEADER+"#########                  Initialising ANNDEA Spatial Brick Alignment module             ##############"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)

#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script prepares training data for training the tracking model')

parser.add_argument('--MinHits',help="What is the minimum number of hits per track?", default='50')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_Rec_Raw_UR.csv')
parser.add_argument('--Name',help="Name of log", default='Realigned')
parser.add_argument('--BrickID',help="Name of log", default='MC_Event_ID')
parser.add_argument('--TrackID',help="Name of log", default='MC_Track_ID')



######################################## Parsing argument values  #############################################################
args = parser.parse_args()
initial_input_file_location=args.f
MinHits=int(args.MinHits)
BrickID=args.BrickID
TrackID=args.TrackID
name=args.Name
output_file_location=initial_input_file_location[:-4]+'_'+name+'.csv'
#output_temp_location=initial_input_file_location[:-4]+'_Alignment-start_'+str(MinHits)+'.csv'

def MeasureResiduals(input_data):
    temp_data=input_data[['Track_ID','x','y','z','Track_Hit_No']]
    Tracks_Head=temp_data[['Track_ID']]
    Tracks_Head.drop_duplicates(inplace=True)
    Tracks_List=temp_data.values.tolist() #I find it is much easier to deal with tracks in list format when it comes to fitting
    Tracks_Head=Tracks_Head.values.tolist()
    #Bellow we build the track representatation that we can use to fit slopes
    with alive_bar(len(Tracks_Head),force_tty=True, title='Build the track representatation: Phase 1') as bar:
        for bth in Tracks_Head:
                   bar()
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
    with alive_bar(len(Tracks_Head),force_tty=True, title='Build the track representatation: Phase 2') as bar:
        for bth in Tracks_Head:
           bar()
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
    Tracks_Head=pd.DataFrame(Tracks_Head, columns = ['Track_ID','ax','t1x','t2x','ay','t1y','t2y'])

    temp_data=pd.merge(temp_data,Tracks_Head,how='inner',on = ['Track_ID'])
    #Calculating x and y coordinates of the fitted line for all plates in the track
    temp_data['new_x']=temp_data['ax']+(temp_data['z']*temp_data['t1x'])+((temp_data['z']**2)*temp_data['t2x'])
    temp_data['new_y']=temp_data['ay']+(temp_data['z']*temp_data['t1y'])+((temp_data['z']**2)*temp_data['t2y'])
    #Calculating how far hits deviate from the fit polynomial
    temp_data['d_x']=temp_data['x']-temp_data['new_x']
    temp_data['d_y']=temp_data['y']-temp_data['new_y']
    temp_data['Residual']=temp_data['d_x']**2+temp_data['d_y']**2
    temp_data['Residual'] = temp_data['Residual'].astype(float)
    temp_data['Residual']=np.sqrt(temp_data['Residual']) #Absolute distance
    temp_data['Max_Deviation']=temp_data['Residual'] #Absolute distance
    temp_data=temp_data[['Track_ID','Track_Hit_No','Residual','Max_Deviation']]
    temp_data=temp_data.groupby(['Track_ID','Track_Hit_No']).agg({'Residual':'sum','Max_Deviation':'max'}).reset_index()
    out_temp_data=temp_data.agg({'Residual':'sum','Max_Deviation':'sum','Track_Hit_No':'sum'})
    out_temp_data=out_temp_data.values.tolist()
    temp_data['Residual']=temp_data['Residual']/temp_data['Track_Hit_No']
    fit=out_temp_data[0]/out_temp_data[2]
    fit2=out_temp_data[1]/out_temp_data[2]
    return fit,fit2,temp_data

########################################     Phase 1 - Create compact source file    #########################################
print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+initial_input_file_location+bcolors.ENDC)
raw_data=pd.read_csv(initial_input_file_location,
                header=0,usecols=[BrickID,TrackID,'x','y','z'])

total_rows=len(raw_data)
print(UF.TimeStamp(),'The raw data has',total_rows,'hits')
print(UF.TimeStamp(),'Removing tracks which have less than',MinHits,'hits...')
raw_data[BrickID] = raw_data[BrickID].astype(str)
raw_data[TrackID] = raw_data[TrackID].astype(str)
raw_data['Track_ID'] = raw_data[BrickID] + '-' + raw_data[TrackID]
raw_data=raw_data.drop([TrackID],axis=1)
raw_data=raw_data.drop([BrickID],axis=1)

track_no_data=raw_data.groupby(['Track_ID'],as_index=False).count()
track_no_data=track_no_data.drop(['y','z'],axis=1)
track_no_data=track_no_data.rename(columns={'x': "Track_Hit_No"})
new_combined_data=pd.merge(raw_data, track_no_data, how="left", on=['Track_ID'])
new_combined_data=new_combined_data.sort_values(['Track_ID','z'],ascending=[1,1])
new_combined_data = new_combined_data[new_combined_data.Track_Hit_No >= MinHits]

result=MeasureResiduals(new_combined_data)
print(result[0])
print(result[1])
result[2].to_csv(output_file_location,index=False)
print('Result has been saved 2 file', output_file_location)
exit()

