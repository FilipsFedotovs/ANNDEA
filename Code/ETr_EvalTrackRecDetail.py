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
parser.add_argument('--TrackName', type=str, default='FEDRA_Track_ID', help="Please enter the computing tool name that you want to compare")
args = parser.parse_args()

input_file_location=args.f

#importing data - making sure we only use relevant columns
columns = ['Hit_ID','x','y','z','MC_Event_ID','MC_Track_ID','PDG_ID','MotherPDG',args.TrackName]
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
ANN_test_columns = ['Hit_ID','x','y','z','MC_Event_ID','MC_Track_ID',args.TrackName]
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
                ANN_test = ANN_test.drop(['y','z'], axis=1)
                
                
                if len(ANN_test) > 0:          
                    ANN_test['SND_B31_3_2_2_Track_ID'] = pd.to_numeric(ANN_test['SND_B31_3_2_2_Track_ID'],errors='coerce').fillna(-2).astype('int')
                    ANN_test['z_coord'] = ANN_test['z_coord'].astype('int')
                    ANN_test = ANN_test.astype({col: 'int8' for col in ANN_test.select_dtypes('int64').columns})
                    #print(ANN_test.dtypes)
                    #exit()

                ANN_test_right = ANN_test
                ANN_test_right = ANN_test_right.rename(columns={'Hit_ID':'Hit_ID_right',args.TrackName:args.TrackName+'_right','MC_Track':'MC_Track_right','z_coord':'z_coord_right'})
                ANN_test_all = pd.merge(ANN_test,ANN_test_right,how='inner',on=['x'])
                #print(ANN_test_all)

                ANN_test_all = ANN_test_all[ANN_test_all.Hit_ID!=ANN_test_all.Hit_ID_right]
                #print(ANN_test_all)

                ANN_test_all = ANN_test_all[ANN_test_all.z_coord>ANN_test_all.z_coord_right]
                #print(ANN_test_all)

                ANN_test_all['MC_true'] = (ANN_test_all['MC_Track']==ANN_test_all['MC_Track_right']).astype(int)
                #print(ANN_test_all)

                ANN_test_all['ANN_true'] = (ANN_test_all[args.TrackName]==ANN_test_all[args.TrackName+'_right']).astype(int)
                #print(ANN_test_all)

                ANN_test_all['True'] = ANN_test_all['MC_true'] + ANN_test_all['ANN_true']
                ANN_test_all['True'] = (ANN_test_all['True']>1).astype(int)
                #print(ANN_test_all[[args.TrackName,args.TrackName+'_right','ANN_true']])

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
#exit()

#average of precision and recall
ANN_analysis['ANN_recall'] = pd.to_numeric(ANN_analysis['ANN_recall'],errors='coerce').fillna(0).astype('int')
ANN_analysis['ANN_precision'] = pd.to_numeric(ANN_analysis['ANN_precision'],errors='coerce').fillna(0).astype('int')
TotalMCtrue = ANN_analysis['MC_true'].sum()
TotalANNtrue = ANN_analysis['ANN_true'].sum()
Totaltrue = ANN_analysis['True'].sum()

Average_recall = Totaltrue/TotalMCtrue
Average_precision = Totaltrue/TotalANNtrue
print('Average recall is', Average_recall)
print('Average precision is', Average_precision)

#average of precision and recall
#recall_average = ANN_test_all.loc[:, 'ANN_recall'].mean()
#print('Average recall is', recall_average)
#precision_average = ANN_test_all.loc[:, 'ANN_precision'].mean()
#print('Average precision is', precision_average)

# end of script #
