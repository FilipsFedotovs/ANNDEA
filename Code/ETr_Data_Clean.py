#This script removes long tracks from the data
#Optional Module of the ANNDEA package
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
import U_UI as UI
import Parameters as PM #This is where we keep framework global parameters
import pandas as pd #We use Panda for a routine data processing
pd.options.mode.chained_assignment = None #Silence annoying warnings
import numpy as np
import argparse

UI.WelcomeMsg('Initialising emulsion data clean module...','Filips Fedotovs (PhD student at UCL)', 'Please reach out to filips.fedotovs@cern.ch for any queries')

#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script prepares training data for training the tracking model')
parser.add_argument('--TrackID',help="What track name is used?", default='FEDRA_Track_ID')
parser.add_argument('--Mode', help='Script will continue from the last checkpoint, unless you want to start from the scratch, then type "Reset"',default='')
parser.add_argument('--o',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_Rec_Raw_UR.csv')
parser.add_argument('--Size',help="Split the cross section of the brick in the squares with the size being a length of such a square.", default='0')
parser.add_argument('--MaxLen',help="What is the minimum number of hits per track?", default=50,type=int)
######################################## Parsing argument values  #############################################################
args = parser.parse_args()
Mode=args.Mode.upper()
TrackID=args.TrackID
MaxLen=int(args.MaxLen)
Size=int(args.Size)
initial_input_file_location=args.f

output_file_location_KLT=initial_input_file_location[:-4]+'_'+args.o+'_KLT.csv'
output_file_location_ELT=initial_input_file_location[:-4]+'_'+args.o+'_ELT.csv'


########################################     Phase 1 - Create compact source file    #########################################
UI.Msg('location','Loading raw data from',initial_input_file_location)
ColUse=[TrackID,PM.Hit_ID,PM.x,PM.y,PM.z,PM.tx,PM.ty]

data=pd.read_csv(initial_input_file_location,
            header=0,
            usecols=ColUse)
Volume=(PM.stepX/10000)*(PM.stepY/10000)*(PM.stepZ/10000)
Min_x=data[PM.x].min()
Max_x=data[PM.x].max()
Min_y=data[PM.y].min()
Max_y=data[PM.y].max()
Min_z=data[PM.z].min()
Max_z=data[PM.z].max()
UI.Msg('vanilla','The snapshot of the data is printed below:')
print(data)
def Density(data,msg):
    UI.Msg('vanilla',msg)
    data_agg=data[[PM.Hit_ID,PM.x,PM.y,PM.z]]
    data_agg[PM.x]=data_agg[PM.x]-Min_x
    data_agg[PM.y]=data_agg[PM.y]-Min_y
    data_agg[PM.z]=data_agg[PM.z]-Min_z
    data_agg[PM.x]=data_agg[PM.x]/PM.stepX
    data_agg[PM.y]=data_agg[PM.y]/PM.stepY
    data_agg[PM.z]=data_agg[PM.z]/PM.stepZ
    data_agg[PM.x]=data_agg[PM.x].apply(np.ceil)
    data_agg[PM.y]=data_agg[PM.y].apply(np.ceil)
    data_agg[PM.z]=data_agg[PM.z].apply(np.ceil)
    data_agg=data_agg.groupby([PM.x,PM.y,PM.z],as_index=False).nunique()
    UI.Msg('result','Minimum hits per cluster: ',int(data_agg[PM.Hit_ID].min()),' hits')
    print('-------------------------------')
    UI.Msg('result','Average hits per cluster: ',int(data_agg[PM.Hit_ID].mean()),' hits')
    print('-------------------------------')
    UI.Msg('result','Maximum hits per cluster: ',int(data_agg[PM.Hit_ID].max()),' hits')
    print('-------------------------------')

    return 1

print('-------------------------------------------------------------')
Density(data,'Calculating the initial density of the data')
print('-------------------------------------------------------------')

track_data=data.dropna()
rank_track_data=track_data[[PM.z]].drop_duplicates()
rank_track_data['Plate_ID']=rank_track_data[PM.z].rank().astype('int')
track_data=pd.merge(track_data,rank_track_data,how='left', on=[PM.z])
track_data['Min_Plate_ID']=track_data['Plate_ID']
track_data['Max_Plate_ID']=track_data['Plate_ID']
min_track_data=track_data.groupby([TrackID],as_index=False)['Min_Plate_ID'].min()
max_track_data=track_data.groupby([TrackID],as_index=False)['Max_Plate_ID'].max()
track_data=pd.merge(min_track_data, max_track_data,how='inner',on=[TrackID])
track_data['Plate_Length']=track_data['Max_Plate_ID']-track_data['Min_Plate_ID']+1
track_data=track_data[track_data.Plate_Length > MaxLen]
track_data=track_data.drop(['Min_Plate_ID','Max_Plate_ID','Plate_Length'],axis=1)
KLT_data=pd.merge(data,track_data,how='inner',on=[TrackID])
print(KLT_data)
print('-------------------------------------------------------------')
Density(KLT_data,'Calculating the density of the data with long tracks only...')
print('-------------------------------------------------------------')

track_data['ELT_Flag']=True
ELT_data=pd.merge(data,track_data,how='left',on=[TrackID])
ELT_data=ELT_data[ELT_data.ELT_Flag != True]

print('-------------------------------------------------------------')
Density(KLT_data,'Calculating the density of the data with long tracks only..')
print('-------------------------------------------------------------')
print(ELT_data)
print('-------------------------------------------------------------')
Density(ELT_data,'Calculating the density of the data without long tracks...')
print('-------------------------------------------------------------')
ELT_data=ELT_data.drop(['ELT_Flag'],axis=1)

x=input('Would you like to save the files? (Press y):')
if x=='y':
    KLT_data.to_csv(output_file_location_KLT)
    UI.Msg('location', 'The data with long tracks only is written to:',output_file_location_KLT)

    ELT_data.to_csv(output_file_location_ELT)
    UI.Msg('location', 'The data without long tracks is written to:',output_file_location_ELT)

exit()




