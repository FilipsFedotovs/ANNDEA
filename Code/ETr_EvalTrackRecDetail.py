#!/usr/bin/env python
# coding: utf-8


#!/usr/bin/python3

import pandas as pd #for analysis
pd.options.mode.chained_assignment = None #Silence annoying warnings
import math 
import numpy as np
import argparse
from alive_progress import alive_bar

parser = argparse.ArgumentParser(description='This script compares the ouput of the previous step with the output of ANNDEA reconstructed data to calculate reconstruction performance.')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_FEDRA_Raw_UR.csv')
parser.add_argument('--ToolNames', type=str, choices=['FEDRA', 'ANN', 'ANN_Blank'], help="Please enter the computing tool name that you want to compare")
args = parser.parse_args()

input_file_location=args.f

#importing data - making sure we only use relevant columns
columns = ['Hit_ID','x','y','z','MC_Event_ID','MC_Track_ID','PDG_ID','MotherPDG','Brick_ID','FEDRA_Track_ID','SND_B31_3_2_2_Track_ID','SND_B31_3_2_2_Brick_ID','SND_B31_Blank_3_2_2_Track_ID','SND_B31_Blank_3_2_2_Brick_ID']
rowdata = pd.read_csv(input_file_location,usecols=columns)

#calculating overall density, coordinates initially in microns
columns = ['Hit_ID','x','y','z']
densitydata = rowdata[columns]

#using particle id columns
part_columns = ['Hit_ID','PDG_ID']
density_particles = rowdata[part_columns]

# number of Hit_ID's by specific particle ID's
density_particles = density_particles.groupby(['PDG_ID']).Hit_ID.nunique().reset_index() 

#binning x
densitydata['x'] = (densitydata['x']/10000) #going from microns to cms
densitydata['x'] = (densitydata['x']).apply(np.ceil) #rounding up to higher number

#binning y
densitydata['y'] = (densitydata['y']/10000)
densitydata['y'] = (densitydata['y']).apply(np.ceil)

#binning z
densitydata['z'] = (densitydata['z']/10000)
densitydata['z'] = (densitydata['z']).apply(np.ceil)

# number of Hit_ID's by specific coordinates
densitydata = densitydata.groupby(['x','y','z']).Hit_ID.nunique().reset_index() 
densitydata = densitydata.rename(columns={'Hit_ID':'Hit_Density'})

# starting an if loop to match the choice of Computing tool in the arguments
# Get precision and recall for ANNDEA with GNN
if args.ToolNames == 'ANN':
    ANN_test_columns = ['Hit_ID','x','y','z','MC_Event_ID','MC_Track_ID','SND_B31_3_2_2_Track_ID','SND_B31_3_2_2_Brick_ID']
    ANN = rowdata[ANN_test_columns]
    ANN_base = None
    
    ANN['z_coord'] = ANN['z']
    
    #binning x
    ANN['x'] = (ANN['x']/10000) #going from microns to cms
    ANN['x'] = (ANN['x']).apply(np.ceil).astype(int) #rounding up to higher number

    #binning y
    ANN['y'] = (ANN['y']/10000)
    ANN['y'] = (ANN['y']).apply(np.ceil).astype(int)

    #binning z
    ANN['z'] = (ANN['z']/10000)
    ANN['z'] = (ANN['z']).apply(np.ceil).astype(int)
    ANN['MC_Track_ID'] = ANN['MC_Track_ID'].astype(str)
    ANN['MC_Event_ID'] = ANN['MC_Event_ID'].astype(str)
    ANN['MC_Track'] = ANN['MC_Track_ID'] + '-' + ANN['MC_Event_ID']
    #print(ANN_test)
    
    #delete unwanted columns
    ANN.drop(['MC_Track_ID','MC_Event_ID'], axis=1, inplace=True)
    
    # create a loop for all x, y and z ranges to be evaluated
   
    xmin = math.floor(densitydata['x'].min())
    #print(xmin)
    xmax = math.ceil(densitydata['x'].max())
    #print(xmax)
    ymin = math.floor(densitydata['y'].min())
    #print(ymin)
    ymax = math.ceil(densitydata['y'].max())
    #print(ymax)
    zmin = math.floor(densitydata['z'].min())
    #print(zmin)
    zmax = math.ceil(densitydata['z'].max())
    #print(zmax)
    
    iterations = (xmax - xmin)*(ymax - ymin)*(zmax - zmin)
    with alive_bar(iterations,force_tty=True, title = 'Calculating densities.') as bar:
        for i in range(xmin,xmax):
            ANN_test_i = ANN[ANN.x==i]
            for  j in range(ymin,ymax):
                ANN_test_j = ANN_test_i[ANN_test_i.y==j]
                for k in range(zmin,zmax):
                    bar()
                    ANN_test = ANN_test_j[ANN_test_j.z==k]
                    if len(ANN_test)>0:
                        print(ANN_test.memory_usage)
                        
                    ANN_test = ANN_test.drop(['y','z'], axis=1)
                    if len(ANN_test)>0:
                        print(ANN_test.memory_usage)  
                        exit()

                    ANN_test_right = ANN_test
                    ANN_test_right = ANN_test_right.rename(columns={'Hit_ID':'Hit_ID_right','SND_B31_3_2_2_Track_ID':'SND_B31_3_2_2_Track_ID_right','MC_Track':'MC_Track_right','z_coord':'z_coord_right'})
                    ANN_test_all = pd.merge(ANN_test,ANN_test_right,how='inner',on=['x'])
                    #print(ANN_test_all)

                    ANN_test_all = ANN_test_all[ANN_test_all.Hit_ID!=ANN_test_all.Hit_ID_right]
                    #print(ANN_test_all)

                    ANN_test_all = ANN_test_all[ANN_test_all.z_coord>ANN_test_all.z_coord_right]
                    #print(ANN_test_all)

                    ANN_test_all['MC_true'] = (ANN_test_all['MC_Track']==ANN_test_all['MC_Track_right']).astype(int)
                    #print(ANN_test_all)

                    ANN_test_all['ANN_true'] = (ANN_test_all['SND_B31_3_2_2_Track_ID']==ANN_test_all['SND_B31_3_2_2_Track_ID_right']).astype(int)
                    #print(ANN_test_all)

                    ANN_test_all['True'] = ANN_test_all['MC_true'] + ANN_test_all['ANN_true']
                    ANN_test_all['True'] = (ANN_test_all['True']>1).astype(int)
                    #print(ANN_test_all[['SND_B31_3_2_2_Track_ID','SND_B31_3_2_2_Track_ID_right','ANN_true']])
                    
                    ANN_test_all['y'] = j
                    ANN_test_all['z'] = k

                    ANN_test_all = ANN_test_all[['MC_true','ANN_true','True','x','y','z']]
                    ANN_test_all = ANN_test_all.groupby(['x', 'y','z']).agg({'ANN_true':'sum','True':'sum','MC_true':'sum'}).reset_index()

                    ANN_test_all['ANN_recall'] = ANN_test_all['True']/ANN_test_all['MC_true']

                    ANN_test_all['ANN_precision'] = ANN_test_all['True']/ANN_test_all['ANN_true']
                    ANN_base = pd.concat([ANN_base,ANN_test_all])
                
    #create a table with all the wanted columns
    #print(ANN_base)
    ANN_analysis = pd.merge(densitydata,ANN_base, how='inner', on=['x','y','z'])
    #print(ANN_analysis)
    exit()
    
    #average of precision and recall
    recall_average = ANN_test_all.loc[:, 'ANN_recall'].mean()
    print(recall_average)
    precision_average = ANN_test_all.loc[:, 'ANN_precision'].mean()
    print(precision_average)

# Get precision and recall from ANNDEA without GNN
elif args.ToolNames == 'ANN_Blank':
    ANN_blank_test_columns = ['Hit_ID','x','y','z','MC_Event_ID','MC_Track_ID','SND_B31_Blank_3_2_2_Track_ID','SND_B31_Blank_3_2_2_Brick_ID']
    ANN_blank = rowdata[ANN_blank_test_columns]
    ANN_blank_base = None
    
    ANN_blank['z_coord'] = ANN_blank['z']
    
    #binning x
    ANN_blank['x'] = (ANN_blank['x']/10000) #going from microns to cms
    ANN_blank['x'] = (ANN_blank['x']).apply(np.ceil).astype(int) #rounding up to higher number

    #binning y
    ANN_blank['y'] = (ANN_blank['y']/10000)
    ANN_blank['y'] = (ANN_blank['y']).apply(np.ceil).astype(int)

    #binning z
    ANN_blank['z'] = (ANN_blank['z']/10000)
    ANN_blank['z'] = (ANN_blank['z']).apply(np.ceil).astype(int)
    ANN_blank['MC_Track_ID'] = ANN_blank['MC_Track_ID'].astype(str)
    ANN_blank['MC_Event_ID'] = ANN_blank['MC_Event_ID'].astype(str)
    ANN_blank['MC_Track'] = ANN_blank['MC_Track_ID'] + '-' + ANN_blank['MC_Event_ID']
    #print(ANN_blank_test)
    
    #drop unwanted columns 
    ANN_blank.drop(['MC_Track_ID','MC_Event_ID','Brick_ID'], axis=1, inplace=True)
    
    # create a loop for all x, y and z ranges to be evaluated
    
    xmin = math.floor(densitydata['x'].min())
    #print(xmin)
    xmax = math.ceil(densitydata['x'].max())
    #print(xmax)
    ymin = math.floor(densitydata['y'].min())
    #print(ymin)
    ymax = math.ceil(densitydata['y'].max())
    #print(ymax)
    zmin = math.floor(densitydata['z'].min())
    #print(zmin)
    zmax = math.ceil(densitydata['z'].max())
    #print(zmax)
    
    iterations = (xmax - xmin)*(ymax - ymin)*(zmax - zmin)
    with alive_bar(iterations,force_tty=True, title = 'Calculating densities.') as bar:
        for i in range(xmin,xmax):
            ANN_blank_test_i = ANN_blank[ANN_blank.x==i]
            for  j in range(ymin,ymax):
                ANN_blank_test_j = ANN_blank_test_i[ANN_blank_test_i.y==j]
                for k in range(zmin,zmax):
                    #print(i,j,k)
                    bar()
                    ANN_blank_test = ANN_blank_test_j[ANN_blank_test_j.z==k]

                    ANN_blank_test_right = ANN_blank_test
                    ANN_blank_test_right = ANN_blank_test_right.rename(columns={'Hit_ID':'Hit_ID_right','SND_B31_Blank_3_2_2_Track_ID':'SND_B31_Blank_3_2_2_Track_ID_right','MC_Track':'MC_Track_right','z_coord':'z_coord_right'})
                    ANN_blank_test_all = pd.merge(ANN_blank_test,ANN_blank_test_right,how='inner',on=['x','y','z'])
                    #print(ANN_blank_test_all)

                    ANN_blank_test_all = ANN_blank_test_all[ANN_blank_test_all.Hit_ID!=ANN_blank_test_all.Hit_ID_right]
                    #print(ANN_blank_test_all)

                    ANN_blank_test_all = ANN_blank_test_all[ANN_blank_test_all.z_coord>ANN_blank_test_all.z_coord_right]
                    #print(ANN_blank_test_all)

                    ANN_blank_test_all['MC_true'] = (ANN_blank_test_all['MC_Track']==ANN_blank_test_all['MC_Track_right']).astype(int)
                    #print(ANN_blank_test_all)

                    ANN_blank_test_all['ANN_blank_true'] = (ANN_blank_test_all['SND_B31_Blank_3_2_2_Track_ID']==ANN_blank_test_all['SND_B31_Blank_3_2_2_Track_ID_right']).astype(int)
                    #print(ANN_blank_test_all)

                    ANN_blank_test_all['True'] = ANN_blank_test_all['MC_true'] + ANN_blank_test_all['ANN_blank_true']
                    ANN_blank_test_all['True'] = (ANN_blank_test_all['True']>1).astype(int)
                    #print(ANN_blank_test_all[['SND_B31_Blank_3_2_2_Track_ID','SND_B31_Blank_3_2_2_Track_ID_right','ANN_blank_true']])

                    ANN_blank_test_all = ANN_blank_test_all[['MC_true','ANN_blank_true','True','x','y','z']]
                    ANN_blank_test_all = ANN_blank_test_all.groupby(['x', 'y','z']).agg({'ANN_blank_true':'sum','True':'sum','MC_true':'sum'}).reset_index()

                    ANN_blank_test_all['ANN_blank_recall'] = ANN_blank_test_all['True']/ANN_blank_test_all['MC_true']

                    ANN_blank_test_all['ANN_blank_precision'] = ANN_blank_test_all['True']/ANN_blank_test_all['ANN_blank_true']
                    ANN_blank_base = pd.concat([ANN_blank_base,ANN_blank_test_all])
                
    #create a table with all the wanted columns
    #print(ANN_blank_base)
    ANN_blank_analysis = pd.merge(densitydata,ANN_blank_base, how='inner', on=['x','y','z'])
    #print(ANN_blank_analysis)
    exit()
    
    #average of precision and recall
    recall_average = ANN_blank_test_all.loc[:, 'ANN_blank_recall'].mean()
    print(recall_average)
    precision_average = ANN_blank_test_all.loc[:, 'ANN_blank_precision'].mean()
    print(precision_average)
    
# Get precision and recall for FEDRA
else:
    FEDRA_test_columns = ['Hit_ID','x','y','z','MC_Event_ID','MC_Track_ID','Brick_ID','FEDRA_Track_ID']
    FEDRA = rowdata[FEDRA_test_columns]
    FEDRA_base = None

    FEDRA['z_coord'] = FEDRA['z']

    #binning x
    FEDRA['x'] = (FEDRA['x']/10000) #going from microns to cms
    FEDRA['x'] = (FEDRA['x']).apply(np.ceil).astype(int) #rounding up to higher number

    #binning y
    FEDRA['y'] = (FEDRA['y']/10000)
    FEDRA['y'] = (FEDRA['y']).apply(np.ceil).astype(int)

    #binning z
    FEDRA['z'] = (FEDRA['z']/10000)
    FEDRA['z'] = (FEDRA['z']).apply(np.ceil).astype(int)
    FEDRA['MC_Track_ID'] = FEDRA['MC_Track_ID'].astype(str)
    FEDRA['MC_Event_ID'] = FEDRA['MC_Event_ID'].astype(str)
    FEDRA['MC_Track'] = FEDRA['MC_Track_ID'] + '-' + FEDRA['MC_Event_ID']
    #print(FEDRA_test)
    
    #drop unwanted columns
    FEDRA.drop(['MC_Track_ID','MC_Event_ID','Brick_ID'], axis=1, inplace=True)
    
    # create a loop for all x, y and z ranges to be evaluated
    
    xmin = math.floor(densitydata['x'].min())
    #print(xmin)
    xmax = math.ceil(densitydata['x'].max())
    #print(xmax)
    ymin = math.floor(densitydata['y'].min())
    #print(ymin)
    ymax = math.ceil(densitydata['y'].max())
    #print(ymax)
    zmin = math.floor(densitydata['z'].min())
    #print(zmin)
    zmax = math.ceil(densitydata['z'].max())
    #print(zmax)
    
    iterations = (xmax - xmin)*(ymax - ymin)*(zmax - zmin)
    with alive_bar(iterations,force_tty=True, title = 'Calculating densities.') as bar:
        for i in range(xmin,xmax):
            FEDRA_test_i = FEDRA[FEDRA.x==i]
            for  j in range(ymin,ymax):
                FEDRA_test_j = FEDRA_test_i[FEDRA_test_i.y==j]
                for k in range(zmin,zmax):
                    #print(i,j,k)
                    FEDRA_test = FEDRA_test_j[FEDRA_test_j.z==k]
                    bar()
                    #print(FEDRA_test)

                    FEDRA_test_right = FEDRA_test
                    FEDRA_test_right = FEDRA_test_right.rename(columns={'Hit_ID':'Hit_ID_right','FEDRA_Track_ID':'FEDRA_Track_ID_right','MC_Track':'MC_Track_right','z_coord':'z_coord_right'})
                    FEDRA_test_all = pd.merge(FEDRA_test,FEDRA_test_right,how='inner',on=['x','y','z'])
                    #print(FEDRA_test_all)

                    FEDRA_test_all = FEDRA_test_all[FEDRA_test_all.Hit_ID!=FEDRA_test_all.Hit_ID_right]
                    #print(FEDRA_test_all)

                    FEDRA_test_all = FEDRA_test_all[FEDRA_test_all.z_coord>FEDRA_test_all.z_coord_right]
                    #print(FEDRA_test_all)
                    #print(len(FEDRA_test_all))

                    FEDRA_test_all['MC_true'] = (FEDRA_test_all['MC_Track']==FEDRA_test_all['MC_Track_right']).astype(int)
                    #print(FEDRA_test_all)

                    FEDRA_test_all['FEDRA_true'] = (FEDRA_test_all['FEDRA_Track_ID']==FEDRA_test_all['FEDRA_Track_ID_right']).astype(int)
                    #print(FEDRA_test_all)

                    FEDRA_test_all['True'] = FEDRA_test_all['MC_true'] + FEDRA_test_all['FEDRA_true']
                    FEDRA_test_all['True'] = (FEDRA_test_all['True']>1).astype(int)
                    #print(FEDRA_test_all[['FEDRA_Track_ID','FEDRA_Track_ID_right','FEDRA_true']])

                    FEDRA_test_all = FEDRA_test_all[['MC_true','FEDRA_true','True','x','y','z']]
                    FEDRA_test_all = FEDRA_test_all.groupby(['x', 'y','z']).agg({'FEDRA_true':'sum','True':'sum','MC_true':'sum'}).reset_index()
                    #print(FEDRA_test_all)

                    FEDRA_test_all['FEDRA_recall'] = FEDRA_test_all['True']/FEDRA_test_all['MC_true']

                    FEDRA_test_all['FEDRA_precision'] = FEDRA_test_all['True']/FEDRA_test_all['FEDRA_true']
                    FEDRA_base = pd.concat([FEDRA_base,FEDRA_test_all])
    
    #create a table with all the wanted columns
    #print(FEDRA_base)
    FEDRA_analysis = pd.merge(densitydata,FEDRA_base, how='inner', on=['x','y','z'])
    #print(FEDRA_analysis)
    exit()

    #average of precision and recall
    recall_average = FEDRA_test_all.loc[:, 'FEDRA_recall'].mean()
    print(recall_average)
    precision_average = FEDRA_test_all.loc[:, 'FEDRA_precision'].mean()
    print(precision_average)

# end of script #
