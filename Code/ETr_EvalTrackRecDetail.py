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
parser.add_argument('--o', default='Output', help="Please enter the computing tool name that you want to compare")
parser.add_argument('--MotherGroup', type=str, default='[]', help="Please enter the computing tool name that you want to compare")
parser.add_argument('--MotherPDG', type=str, default='[]', help="Please enter the computing tool name that you want to compare")
args = parser.parse_args()
out=args.TrackName+'_'+args.o+'.csv'

MotherPDG=ast.literal_eval(args.MotherPDG)
MotherGroup=ast.literal_eval(args.MotherGroup)

def MeasureHitPair(_H1,_H2,_G):
          if _H1[0]==_H2[0]:
              return (0,0,0)
          elif _H1[3]>=_H2[3]:
              return (0,0,0)
          else:
              T=int((_H1[4]==_H2[4]) and _H1[2]==_G)
              P= int((_H1[1]==_H2[1]) and (_H1[2]==_G or _H2[2]==_G) and _H1[1]!=-5)
              TP=int(T==P==1)
              return T,P,TP


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
new_list=[]
if os.path.isfile(out):
    check_point = pd.read_csv(out,usecols=['x','y','z','Mother_Group']).values.tolist()

    for el in check_point:
        string=str(int(el[0]))+'-'+str(int(el[1]))+'-'+str(int(el[2]))+'-'+el[3]
        new_list.append(string)



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
                    ANN_test[args.TrackName] = pd.to_numeric(ANN_test[args.TrackName],errors='coerce').fillna(-2).astype('int')
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

                ANN_test_all['ANN_true'] = ((ANN_test_all[args.TrackName]==ANN_test_all[args.TrackName+'_right']) & (ANN_test_all[args.TrackName]!=-2))
                ANN_test_all['ANN_true'] = ANN_test_all['ANN_true'].astype(int)
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
output = args.TrackName+'_FinalData.csv'
ANN_analysis.to_csv(output,index=False)
print(output, 'was saved.')
#print(ANN_analysis)
#exit()

#creating an histogram of recall and precision by hit density
#plt.hist2d(ANN_analysis['Hit_Density']/100,ANN_analysis['ANN_recall'])
#plt.xlabel('Density of Hits')
#plt.ylabel('Recall Average')
#plt.title('Recall for Hit density')
#plt.show()
#exit()


#average of precision and recall
ANN_analysis['ANN_recall'] = pd.to_numeric(ANN_analysis['ANN_recall'],errors='coerce').fillna(0).astype('int')
ANN_analysis['ANN_precision'] = pd.to_numeric(ANN_analysis['ANN_precision'],errors='coerce').fillna(0).astype('int')
TotalMCtrue = ANN_analysis['MC_true'].sum()
TotalANNtrue = ANN_analysis['ANN_true'].sum()
Totaltrue = ANN_analysis['True'].sum()

                for mp in MotherGroup:
                    ANN_test = ANN_test_j[ANN_test_j.z==k]
                    ANN_test = ANN_test.drop(['y','z','x'], axis=1)

                    if len(ANN_test) > 0:
                        ANN_test[args.TrackName] = pd.to_numeric(ANN_test[args.TrackName],errors='coerce').fillna(-5).astype('int')
                        ANN_test['z_coord'] = ANN_test['z_coord'].astype('int')
                        ANN_test['Hit_ID'] = ANN_test['Hit_ID'].astype('str')
                        #ANN_test = ANN_test.astype({col: 'int16' for col in ANN_test.select_dtypes('int64').columns})
                    else:
                        continue
                    ANN_test_right = ANN_test.rename(columns={'Hit_ID':'Hit_ID_right',args.TrackName:args.TrackName+'_right','MC_Track':'MC_Track_right','z_coord':'z_coord_right','Mother_Group':'Mother_Group_right'})
                    ANN_test=ANN_test.values.tolist()
                    ANN_test_right=ANN_test_right.values.tolist()
                    _hit_count=0
                    T=0
                    P=0
                    TP=0

                    for l in ANN_test:
                            _hit_count+=1
                            for r in ANN_test_right:
                               result=(MeasureHitPair(l,r,mp))
                               T+=result[0]
                               P+=result[1]
                               TP+=result[2]
                    if T==P==TP==0:
                       continue
                    ANN_base_temp=pd.DataFrame([[mp,T,P,TP,i,j,k]],columns=['Mother_Group','MC_true','ANN_true','True','x','y','z'])
                    ANN_base_temp['ANN_recall'] = ANN_base_temp['True']/ANN_base_temp['MC_true']

                    ANN_base_temp['ANN_precision'] = ANN_base_temp['True']/ANN_base_temp['ANN_true']
                    ANN_analysis = pd.merge(densitydata,ANN_base_temp, how='inner', on=['x','y','z'])
                    ANN_analysis.to_csv(out, mode='a', header=not os.path.exists(out))
                    print(out, 'was updated')


    print('Success, the file has been saved as ',out)


