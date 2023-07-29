# Not an official part of ANNDEA  package but used to merge together SND data from different bricks and do some manipulations on the data + split Training/Testing data segments
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
import argparse
import ast
import pandas as pd #We use Panda for a routine data processing
pd.options.mode.chained_assignment = None #Disable annoying Pandas warnings
import os
import psutil #This script is very m
import gc
process = psutil.Process(os.getpid())
class bcolors:   #We use it for the interface (Colour fonts and so on)
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#Loading Directory locations
csv_reader=open('../config',"r")
config = list(csv.reader(csv_reader))
for c in config:
    if c[0]=='AFS_DIR':
        AFS_DIR=c[1]
    if c[0]=='EOS_DIR':
        EOS_DIR=c[1]
csv_reader.close()
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')


import UtilityFunctions as UF #This is where we keep routine utility functions

#Setting the parser
parser = argparse.ArgumentParser(description='Not an official part of ANNDEA  package but used to merge together SND data from different bricks and do some manipulations on the data + split Training/Testing data segments')
parser.add_argument('--f',help="Please enter the full path directory there the files are located", default='/eos/user/a/aiulian/sim_fedra/mergeneutrinos_260422_1e2nu_1e5mu/newtracking_110522/')
parser.add_argument('--TestBricks',help="Which ECC bricks are excluded from the training data for Testing?", default="['11']")
parser.add_argument('--Gap',help="Offset along z?", default="50000")
parser.add_argument('--Test',help="Test?", default="Y")

args = parser.parse_args()

print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(),bcolors.BOLD+'Stage 1:'+bcolors.ENDC+' Preparing the input data...')

#This function bellow sets Mother IDs of two tracks that come from the same mother but originate in the different ECC brick to -2 so we avoid considering it as a Truth vertex
#For reconstruction purposes we treat those tracks as incoming background in the given brick as the vertex for those tracks would be already reconstructed in the originating brick
def MotherIDNorm(row):
        if row['MC_Track_Start_Z']>=row['Fiducial_Cut_z_LB'] and row['MC_Track_Start_Z']<=row['Fiducial_Cut_z_UB']: #Track originates in the brick - ok. Might add x&y bounds in the future
          return row['MCMotherID']
        else:
          return -2

input_file_location=args.f
no_quadrants=4  #SND config consists of 4 quadrant
no_brick_layers=5 # Current SND Emulsion detector is made of 5 walls
req_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/SND_Raw_Data_Agg.csv' #This is a support file that we create, more info bellow.

if os.path.isfile(req_file_location)==False: #This is the part where we create the aggregated file that contains MC Track starting z-coordinates.
    #In order to calculate it the whole data for detector has to be considered - very memory demanding operation, thats why I have splitted it up
    columns_to_extract=['z','MCEvent','MCTrack']
    gap=int(args.Gap) #Gap between walls - keep it sufficiently large >20000 micrometers so ANNDEA does not try to join/vertex two tracks from the different bricks
    Test=args.Test=='Y' #If Yes we only produce Test files
    TestBricks=ast.literal_eval(args.TestBricks) #Extracting list from the parser
    data=None
    for q in range(1,no_quadrants+1):
        for bl in range(1,no_brick_layers+1):
            input_file=input_file_location+'b0000'+str(bl)+str(q)+'/brick'+str(bl)+str(q)+'.csv' #Creating a path to access the files
            print(UF.TimeStamp(), 'Loading ',bcolors.OKBLUE+input_file+bcolors.ENDC)
            new_data=pd.read_csv(input_file,header=0,usecols=columns_to_extract)
            new_data['Z']=(new_data['z']+(bl*(77585+gap)))-gap  #Resetting the local z-coordinate to the global one so we don't superimpose tracks in the same volume
            new_data.drop(['z'],axis=1,inplace=True)
            new_data['MC_Track']=new_data['MCEvent'].astype(str)+'-'+new_data['MCTrack'].astype(str) #Creating unique MC Track identifier

            new_data.drop(['MCEvent','MCTrack'],axis=1,inplace=True)
            data=pd.concat([data,new_data])
    del new_data
    gc.collect()
    data_agg=data.groupby(by=['MC_Track'])['Z'].min().reset_index()
    data_agg=data_agg.rename(columns={'Z': 'MC_Track_Start_Z'})  #MC Tracks are unique for the whole detector setup.
    data_agg.to_csv(req_file_location,index=False)
    print(UF.TimeStamp(), bcolors.OKGREEN+"The data was written to :"+bcolors.ENDC, bcolors.OKBLUE+req_file_location+bcolors.ENDC)
    print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
else: #Once we have the aggregated file we can use it to calculate the true starting positions of the MC Tracks
    columns_to_extract=['ID','x','y','z','TX','TY','MCEvent','MCTrack','MCMotherID','P','MotherPDG','PdgCode', 'ProcID', 'FEDRATrackID']
    gap=int(args.Gap)
    Test=args.Test=='Y'
    TestBricks=ast.literal_eval(args.TestBricks)
    data=None
    data_agg=pd.read_csv(req_file_location,header=0)
    for q in range(1,no_quadrants+1):
        for bl in range(1,no_brick_layers+1):
            if Test and (str(bl)+str(q) in TestBricks)==False:
                continue
            elif Test==False and (str(bl)+str(q) in TestBricks):
                continue
            else:
                input_file=input_file_location+'b0000'+str(bl)+str(q)+'/brick'+str(bl)+str(q)+'.csv'
                print(UF.TimeStamp(), 'Loading ',bcolors.OKBLUE+input_file+bcolors.ENDC)
                new_data=pd.read_csv(input_file,header=0,usecols=columns_to_extract)
                input_vx_file=input_file_location+'b0000'+str(bl)+str(q)+'/brick'+str(bl)+str(q)+'_vertices.csv'
                new_vx_data=pd.read_csv(input_vx_file,header=0)
                new_data=pd.merge(new_data,new_vx_data,how='left',on=['FEDRATrackID'])
                new_data['Brick_ID']=str(bl)+str(q)
                new_data['Z']=(new_data['z']+(bl*(77585+gap)))-gap
                new_data.drop(['z'],axis=1,inplace=True)
                new_data['MC_Event_ID']=str(bl)+str(q)+'-'+new_data['MCEvent'].astype(str)
                new_data['MC_Track']=new_data['MCEvent'].astype(str)+'-'+new_data['MCTrack'].astype(str)
                new_data['Fiducial_Cut_z_LB']=new_data['Z'].min() #Local z-bounds for tracks in a given brick
                new_data['Fiducial_Cut_z_UB']=new_data['Z'].max() #Local z-bounds for tracks in a given brick - might consider adding x&y bounds
                new_data.drop(['MCEvent'],axis=1,inplace=True)
                new_data=pd.merge(new_data,data_agg,how='inner',on=['MC_Track'])
                new_data['MC_Mother_ID']=new_data.apply(MotherIDNorm,axis=1)
                new_data.drop(['MCMotherID','MC_Track_Start_Z','MC_Track'],axis=1,inplace=True)
                data=pd.concat([data,new_data])
    #Just renaming the columns so they make more sense.
    data=data.rename(columns={'ID': 'Hit_ID'})
    data=data.rename(columns={'TX': 'tx'})
    data=data.rename(columns={'TY': 'ty'})
    data=data.rename(columns={'MCTrack': 'MC_Track_ID'})
    data=data.rename(columns={'PdgCode': 'PDG_ID'})
    data=data.rename(columns={'FEDRATrackID': 'FEDRA_Track_ID'})
    data=data.rename(columns={'VertexS': 'FEDRA_Vertex_ID'})
    data=data.rename(columns={'VertexE': 'FEDRA_Secondary_Vertex_ID'})
    data=data.rename(columns={'Z': 'z'})
    data=data.rename(columns={'ProcID': 'Process_ID'})
    if Test:
        output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/SND_Raw_Data_Test.csv'
    else:
        output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/SND_Raw_Data_Train.csv'
    data.to_csv(output_file_location,index=False) #Write the data
    print(UF.TimeStamp(), bcolors.OKGREEN+"The data was written to :"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
    print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)



