#!/usr/bin/env python
# coding: utf-8

import ast
#!/usr/bin/python3

import pandas as pd #for analysis
pd.options.mode.chained_assignment = None #Silence annoying warnings
import math
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt #in order to create histograms
import numpy as np
import argparse
from alive_progress import alive_bar

parser = argparse.ArgumentParser(description='This script compares the ouput of the previous step with the output of ANNDEA reconstructed data to calculate reconstruction performance.')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_FEDRA_Raw_UR.csv')
parser.add_argument('--TrackName', type=str, default='FEDRA_Track_ID', help="Please enter the computing tool name that you want to compare")
parser.add_argument('--MotherPDG', type=str, default='[]', help="Please enter the computing tool name that you want to compare")
parser.add_argument('--MotherGroup', type=str, default='[]', help="Please enter the computing tool name that you want to compare")
args = parser.parse_args()


MotherPDG=ast.literal_eval(args.MotherPDG)
MotherGroup=ast.literal_eval(args.MotherGroup)

def JoinHits(_H1,_H2):
          if _H1[0]==_H2[0]:
              return False
          elif _H1[3]>=_H2[3]:
              return False
          return True

if len(MotherGroup)>0:
    GroupData=[]
    for mpg in range(len(MotherGroup)):
        for mp in MotherPDG[mpg]:
            GroupData.append([mp,MotherGroup[mpg]])

    Group_Df=pd.DataFrame(GroupData,columns=['MotherPDG','Mother_Group'])

input_file_location=args.f

#importing data - making sure we only use relevant columns
columns = ['Hit_ID','x','y','z','MC_Event_ID','MC_Track_ID','PDG_ID','MotherPDG',args.TrackName]
rowdata = pd.read_csv(input_file_location,usecols=columns)

if len(MotherGroup)>0:

   rowdata=pd.merge(rowdata,Group_Df,how='left',on=['MotherPDG'])
   rowdata['Mother_Group']=rowdata['Mother_Group'].fillna('Other')
   MotherGroup.append('Other')


else:
   rowdata['Mother_Group']='Other'
   MotherGroup.append('Other')
rowdata = rowdata.drop(['MotherPDG'],axis=1)

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
ANN_test_columns = ['Hit_ID','x','y','z','MC_Event_ID','MC_Track_ID',args.TrackName,'Mother_Group']
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
if os.path.isfile(args.TrackName+'_FinalData_WP.csv'):
    check_point = pd.read_csv(args.TrackName+'_FinalData_WP.csv',usecols=['x','y','z']).values.tolist()
    print(check_point)

iterations = (xmax - xmin)*(ymax - ymin)*(zmax - zmin)
with alive_bar(iterations,force_tty=True, title = 'Calculating densities.') as bar:
    for i in range(xmin,xmax):
        ANN_test_i = ANN[ANN.x==i]
        for  j in range(ymin,ymax):
            ANN_test_j = ANN_test_i[ANN_test_i.y==j]
            for k in range(zmin,zmax):
                bar()
                ANN_test = ANN_test_j[ANN_test_j.z==k]
                ANN_test = ANN_test.drop(['y','z','x'], axis=1)

                if len(ANN_test) > 0:
                    ANN_test[args.TrackName] = pd.to_numeric(ANN_test[args.TrackName],errors='coerce').fillna(-5).astype('int')
                    ANN_test['z_coord'] = ANN_test['z_coord'].astype('int')
                    ANN_test = ANN_test.astype({col: 'int8' for col in ANN_test.select_dtypes('int64').columns})

                ANN_test_right = ANN_test.rename(columns={'Hit_ID':'Hit_ID_right',args.TrackName:args.TrackName+'_right','MC_Track':'MC_Track_right','z_coord':'z_coord_right','Mother_Group':'Mother_Group_right'})
                ANN_test=ANN_test.values.tolist()
                ANN_test_right=ANN_test_right.values.tolist()
                _hit_count=0
                ANN_res=[]
                for l in ANN_test:
                    _hit_count+=1
                    for r in ANN_test_right:
                       if JoinHits(l,r):
                           ANN_res.append(l+r)


                ANN_test_all=pd.DataFrame(ANN_res,columns=['Hit_ID','SND_B31_3_2_2_Track_ID','Mother_Group','z_coord','MC_Track','Hit_ID_right','SND_B31_3_2_2_Track_ID_right','Mother_Group_right','z_coord_right','MC_Track_right'])
                #Little data trick to assess only the relevant connections
                MC_Block=ANN_test_all[['Hit_ID','Hit_ID_right','Mother_Group','MC_Track','MC_Track_right']]

                ANN_base_temp=pd.DataFrame([],columns=['Mother_Group','MC_true','ANN_true','True','x','y','z'])
                for mp in MotherGroup:
                    ANN_test_temp = ANN_test_all.drop(['MC_Track','MC_Track_right'],axis=1)
                    MC_Block_temp = MC_Block[MC_Block.MC_Track==MC_Block.MC_Track_right]
                    MC_Block_temp=MC_Block_temp.drop(['MC_Track','MC_Track_right'],axis=1)
                    MC_Block_temp=MC_Block_temp[MC_Block_temp.Mother_Group==mp]
                    MC_Block_temp=MC_Block_temp.drop(['Mother_Group'],axis=1)
                    MC_Block_temp['MC_true']=1
                    ANN_test_temp=pd.merge(ANN_test_temp,MC_Block_temp,how='left',on=['Hit_ID','Hit_ID_right'])
                    ANN_test_temp['MC_true']=ANN_test_temp['MC_true'].fillna(0)
                    ANN_test_temp=ANN_test_temp.drop(['Hit_ID','Hit_ID_right','z_coord','z_coord_right'],axis=1)
                    ANN_test_temp['Left_Check'] = (ANN_test_temp['Mother_Group']==mp).astype(int)
                    ANN_test_temp['Right_Check'] = (ANN_test_temp['Mother_Group_right']==mp).astype(int)
                    ANN_test_temp['Check'] = ANN_test_temp['Left_Check']+ANN_test_temp['Right_Check']
                    ANN_test_temp=ANN_test_temp.drop(ANN_test_temp.index[ANN_test_temp['Check'] < 1])
                    ANN_test_temp=ANN_test_temp.drop(['Mother_Group','Mother_Group_right','Left_Check','Right_Check','Check'],axis=1)

                    ANN_test_temp['ANN_true'] = ((ANN_test_temp[args.TrackName]==ANN_test_temp[args.TrackName+'_right']) & (ANN_test_temp[args.TrackName]!=-5))
                    ANN_test_temp['ANN_true'] = ANN_test_temp['ANN_true'].astype(int)
                    #print(ANN_test_temp)

                    ANN_test_temp['True'] = ANN_test_temp['MC_true'] + ANN_test_temp['ANN_true']
                    ANN_test_temp['True'] = (ANN_test_temp['True']>1).astype(int)
                    #print(ANN_test_temp[[args.TrackName,args.TrackName+'_right','ANN_true']])

                    ANN_test_temp['y'] = j
                    ANN_test_temp['z'] = k
                    ANN_test_temp['x'] = i

                    ANN_test_temp = ANN_test_temp[['MC_true','ANN_true','True','x','y','z']]
                    ANN_test_temp['Mother_Group'] =mp
                    ANN_base_temp = pd.concat([ANN_base_temp,ANN_test_temp])

                ANN_base_temp = ANN_base_temp.groupby(['Mother_Group','x', 'y','z']).agg({'ANN_true':'sum','True':'sum','MC_true':'sum'}).reset_index()

                ANN_base_temp['ANN_recall'] = ANN_base_temp['True']/ANN_base_temp['MC_true']

                ANN_base_temp['ANN_precision'] = ANN_base_temp['True']/ANN_base_temp['ANN_true']
            ANN_base = pd.concat([ANN_base,ANN_base_temp])
            if len(ANN_base)==0:
                continue
            ANN_analysis = pd.merge(densitydata,ANN_base, how='inner', on=['x','y','z'])
            print(ANN_analysis)
            ANN_analysis.to_csv(args.TrackName+'_FinalData_WP.csv', mode='a', header=not os.path.exists(args.TrackName+'_FinalData_WP.csv'))
            print(args.TrackName+'_FinalData_WP.csv', 'was updated')
            exit()
print('All good')

