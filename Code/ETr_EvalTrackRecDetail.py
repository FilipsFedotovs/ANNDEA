#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/python3

import pandas as pd #for analysis
pd.options.mode.chained_assignment = None #Silence annoying warnings
import math 
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='This script compares the ouput of the previous step with the output of ANNDEA reconstructed data to calculate reconstruction performance.')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_FEDRA_Raw_UR.csv')
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
#print(densitydata)

# number of Hit_ID's by specific coordinates
densitydata = densitydata.groupby(['x','y','z']).Hit_ID.nunique().reset_index() 


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

print(FEDRA)
print('------')
print(FEDRA[FEDRA.x==14])
exit()

for i in range(14,49):
    for j in range(-9,26):
        for k in range(26,34):
            FEDRA_test = FEDRA[FEDRA.x==i]
            FEDRA_test = FEDRA[FEDRA.y==j]
            FEDRA_test = FEDRA[FEDRA.z==k]
            print(FEDRA_test)
            print(FEDRA_test.memory_usage())
            x=input()
            print(i,j,k)


            FEDRA_test['MC_Track_ID'] = FEDRA_test['MC_Track_ID'].astype(str)
            FEDRA_test['MC_Event_ID'] = FEDRA_test['MC_Event_ID'].astype(str)
            FEDRA_test['MC_Track'] = FEDRA_test['MC_Track_ID'] + '-' + FEDRA_test['MC_Event_ID']
            #print(FEDRA_test)



            FEDRA_test.drop(['MC_Track_ID','MC_Event_ID','Brick_ID'], axis=1, inplace=True)
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
            print(FEDRA_test_all)




            FEDRA_test_all['FEDRA_precision'] = FEDRA_test_all['True']/FEDRA_test_all['FEDRA_true']
            FEDRA_base = pd.concat([FEDRA_base,FEDRA_test_all])

            print(FEDRA_base)


# In[ ]:


FEDRA_analysis = pd.merge(densitydata,FEDRA_base, how='inner', on=['x','y','z'])
print(FEDRA_analysis)
exit()

# In[ ]:


#average of precision and recall
recall_average = FEDRA_test_all.loc[:, 'FEDRA_recall'].mean()
print(recall_average)
precision_average = FEDRA_test_all.loc[:, 'FEDRA_precision'].mean()
print(precision_average)

