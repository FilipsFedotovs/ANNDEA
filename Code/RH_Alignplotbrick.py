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
print(bcolors.HEADER+"#########                  Initialising ANNDEA Brick Alignment module                    ###############"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)

#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script prepares training data for training the tracking model')

parser.add_argument('--MinHits',help="What is the minimum number of hits per track?", default='2')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_Rec_Raw_UR.csv')

######################################## Parsing argument values  #############################################################
args = parser.parse_args()
initial_input_file_location=args.f
MinHits=int(args.MinHits)
output_file_location=initial_input_file_location[:-4]+'_Re-Aligned_'+str(MinHits)+'.csv'
output_log_location=initial_input_file_location[:-4]+'_Alignment-log_'+str(MinHits)+'.csv'
output_temp_location=initial_input_file_location[:-4]+'_Alignment-start_'+str(MinHits)+'.csv'
residuals = []     #WC create list to store residual
def FitPlate(PlateZ,input_data):
    change_df = pd.DataFrame([[PlateZ]], columns = ['Plate_ID'])
    temp_data=input_data[['FEDRA_Track_ID','x','y','z','Track_Hit_No','Plate_ID']]
    temp_data=pd.merge(temp_data,change_df,on='Plate_ID',how='left')
    temp_data=temp_data[['FEDRA_Track_ID','x','y','z','Track_Hit_No','Plate_ID']]
    Tracks_Head=temp_data[['FEDRA_Track_ID']]
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
    Tracks_Head=pd.DataFrame(Tracks_Head, columns = ['FEDRA_Track_ID','ax','t1x','t2x','ay','t1y','t2y'])

    temp_data=pd.merge(temp_data,Tracks_Head,how='inner',on = ['FEDRA_Track_ID'])
    #Calculating x and y coordinates of the fitted line for all plates in the track
    temp_data['new_x']=temp_data['ax']+(temp_data['z']*temp_data['t1x'])+((temp_data['z']**2)*temp_data['t2x'])
    temp_data['new_y']=temp_data['ay']+(temp_data['z']*temp_data['t1y'])+((temp_data['z']**2)*temp_data['t2y'])
    #Calculating how far hits deviate from the fit polynomial
    temp_data['d_x']=temp_data['x']-temp_data['new_x']
    temp_data['d_y']=temp_data['y']-temp_data['new_y']
    temp_data['d_r']=temp_data['d_x']**2+temp_data['d_y']**2
    temp_data['d_r'] = temp_data['d_r'].astype(float)
    temp_data['d_r']=np.sqrt(temp_data['d_r']) #Absolute distance
    temp_data=temp_data[['x','y','Plate_ID','d_r', 'd_x', 'd_y']]
    temp_data['angle']=np.arctan2(temp_data['d_y'],temp_data['d_x'])
    temp_data['angle']=np.degrees(temp_data['angle'])
    temp_data = temp_data[temp_data.Plate_ID == PlateZ]
    temp_data=temp_data.drop(['Plate_ID','d_x','d_y'],axis=1)
    for _, row in temp_data.iterrows(): #WC append residuals to list
        residuals.append({"x": row["x"], "y": row["y"], "dr": row["d_r"]})
    import seaborn as sns
    import matplotlib.pyplot as plt
    residuals_df = pd.DataFrame(residuals) #WC
    num_bins = 35
    residuals_df['x_bin'] = pd.cut(residuals_df['x'], bins=num_bins, labels=False)
    residuals_df['y_bin'] = pd.cut(residuals_df['y'], bins=num_bins, labels=False)
    heatmap_data = residuals_df.groupby(['x_bin', 'y_bin'])['dr'].mean().reset_index()
    heatmap_data = heatmap_data.pivot('y_bin', 'x_bin', 'dr')
    plt.figure(figsize=(10,10))
    plt.imshow(heatmap_data, cmap='hot', origin='lower')
    plt.colorbar(label='dr')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Residual Heatmap')
    plt.savefig('Heatmap.png')    #WC End of addition code

    sns.heatmap(temp_data.pivot(index='y', columns='x', values='d_r'))
    plt.show()
    print(temp_data)
    
    exit()

########################################     Phase 1 - Create compact source file    #########################################
print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+initial_input_file_location+bcolors.ENDC)
raw_data=pd.read_csv(initial_input_file_location,
                header=0)

total_rows=len(raw_data)
print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
print(UF.TimeStamp(),'Removing unreconstructed hits...')
data=raw_data.dropna()
final_rows=len(data)
print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
print(UF.TimeStamp(),'Removing tracks which have less than',MinHits,'hits...')
track_no_data=data.groupby(['FEDRA_Track_ID'],as_index=False).count()
track_no_data=track_no_data.drop(['Hit_ID','y','z','tx','ty'],axis=1)
track_no_data=track_no_data.rename(columns={'x': "Track_Hit_No"})
new_combined_data=pd.merge(data, track_no_data, how="left", on=['FEDRA_Track_ID'])
new_combined_data = new_combined_data[new_combined_data.Track_Hit_No >= MinHits]
new_combined_data=new_combined_data.drop(['Hit_ID','tx','ty'],axis=1)
new_combined_data=new_combined_data.sort_values(['FEDRA_Track_ID','z'],ascending=[1,1])
new_combined_data['FEDRA_Track_ID']=new_combined_data['FEDRA_Track_ID'].astype(int)
new_combined_data['Plate_ID']=new_combined_data['z'].astype(int)

print(UF.TimeStamp(),'Working out the number of plates to align')
plates=new_combined_data[['Plate_ID']].sort_values(['Plate_ID'],ascending=[1])
plates.drop_duplicates(inplace=True)
plates=plates.values.tolist() #I find it is much easier to deal with tracks in list format when it comes to fitting
print(UF.TimeStamp(),'There are ',len(plates),' plates')

global_logdata = []
iterator = 0

tot_jobs = len(plates)*2
alignment_map=[]
with alive_bar(tot_jobs,force_tty=True, title='Optimising the alignment configuration...') as bar:
    for p in plates:
       am=[p[0]]
       print(FitPlate(p[0],new_combined_data))



