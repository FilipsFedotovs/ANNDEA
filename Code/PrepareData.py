# Part of <Work in progress package> package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
import argparse
import ast
import pandas as pd #We use Panda for a routine data processing
pd.options.mode.chained_assignment = None #Disable annoying Pandas warnings
import os
import psutil
import gc
import numpy as np
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
import Parameters as PM #This is where we keep framework global parameters

#Setting the parser
parser = argparse.ArgumentParser(description='This script compares the ouput of the previous step with the output of ANNDEA reconstructed data to calculate reconstruction performance.')
parser.add_argument('--f',help="Please enter the full path directory there the files are located", default='/eos/user/a/aiulian/sim_fedra/mergeneutrinos_260422_1e2nu_1e5mu/newtracking_110522/')
parser.add_argument('--TestBricks',help="What Names would you like to assign to the reconstruction methods that generated the tracks?", default="['11']")
parser.add_argument('--Gap',help="Offset along z?", default="50000")
parser.add_argument('--Test',help="Test?", default="Y")

args = parser.parse_args()

######################################## Welcome message  #############################################################
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################     Initialising Tracking  Quality     Evaluation module ########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(),bcolors.BOLD+'Stage 1:'+bcolors.ENDC+' Preparing the input data...')
######################################## Set variables  #############################################################

def MotherIDNorm(row):
        if row['MC_Track_Start_Z']>=row['Fiducial_Cut_z_LB'] and row['MC_Track_Start_Z']<=row['Fiducial_Cut_z_UB']:
          return row['MCMotherID']
        else:
          return -2

input_file_location=args.f
no_quadrants=4
no_brick_layers=5
req_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/SND_Raw_Data_Agg.csv'

if os.path.isfile(req_file_location)==False:
    columns_to_extract=['z','MCEvent','MCTrack']
    gap=int(args.Gap)
    Test=args.Test=='Y'
    TestBricks=ast.literal_eval(args.TestBricks)
    data=None
    for q in range(1,no_quadrants+1):
        for bl in range(1,no_brick_layers+1):
            input_file=input_file_location+'b0000'+str(bl)+str(q)+'/brick'+str(bl)+str(q)+'.csv'
            print(UF.TimeStamp(), 'Loading ',bcolors.OKBLUE+input_file+bcolors.ENDC)
            new_data=pd.read_csv(input_file,header=0,usecols=columns_to_extract)
            new_data['Z']=(new_data['z']+(bl*(77585+gap)))-gap
            new_data.drop(['z'],axis=1,inplace=True)
            new_data['MC_Track']=new_data['MCEvent'].astype(str)+'-'+new_data['MCTrack'].astype(str)

            new_data.drop(['MCEvent','MCTrack'],axis=1,inplace=True)
            data=pd.concat([data,new_data])
            print(data.memory_usage(index=True).sum()/(1024**2))
            print(round(process.memory_info().rss/(1024**2),0))
    del new_data
    gc.collect()
    data_agg=data.groupby(by=['MC_Track'])['Z'].min().reset_index()
    data_agg=data_agg.rename(columns={'Z': 'MC_Track_Start_Z'})
    data_agg.to_csv(req_file_location,index=False)
    print(UF.TimeStamp(), bcolors.OKGREEN+"The data was written to :"+bcolors.ENDC, bcolors.OKBLUE+req_file_location+bcolors.ENDC)
    print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
else:
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
                new_data['Fiducial_Cut_z_LB']=new_data['Z'].min()
                new_data['Fiducial_Cut_z_UB']=new_data['Z'].max()
                new_data.drop(['MCEvent'],axis=1,inplace=True)
                new_data=pd.merge(new_data,data_agg,how='inner',on=['MC_Track'])
                new_data['MC_Mother_ID']=new_data.apply(MotherIDNorm,axis=1)
                new_data.drop(['MCMotherID','MC_Track_Start_Z','MC_Track'],axis=1,inplace=True)
                data=pd.concat([data,new_data])
    data=data.rename(columns={'ID': 'Hit_ID'})
    data=data.rename(columns={'TX': 'tx'})
    data=data.rename(columns={'TY': 'ty'})
    data=data.rename(columns={'MCTrack': 'MC_Track_ID'})
    data=data.rename(columns={'PdgCode': 'PDG_ID'})
    data=data.rename(columns={'VertexS': 'FEDRA_Vertex_ID'})
    data=data.rename(columns={'VertexE': 'FEDRA_Secondary_Vertex_ID'})
    data=data.rename(columns={'Z': 'z'})
    if Test:
        output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/SND_Raw_Data_Test.csv'
    else:
        output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/SND_Raw_Data_Train.csv'
    data.to_csv(output_file_location,index=False)
    print(UF.TimeStamp(), bcolors.OKGREEN+"The data was written to :"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
    print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)



