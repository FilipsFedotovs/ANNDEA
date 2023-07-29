#!/usr/bin/env python
# coding: utf-8


#!/usr/bin/python3

import pandas as pd #for analysis
pd.options.mode.chained_assignment = None #Silence annoying warnings
import math 
import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns
from alive_progress import alive_bar

parser = argparse.ArgumentParser(description='This script compares the ouput of the previous step with the output of ANNDEA reconstructed data to calculate reconstruction performance.')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_FEDRA_Raw_UR.csv')
parser.add_argument('--TrackName', type=str, default='FEDRA_Track_ID', help="Please enter the computing tool name that you want to compare")
parser.add_argument('--BrickName', type=str, default='Brick_ID', help="Please enter the computing tool name that you want to compare")
args = parser.parse_args()

input_file_location=args.f

#importing data - making sure we only use relevant columns
columns = ['Hit_ID','x','y','z','tx','ty','MC_Event_ID','MC_Track_ID','PDG_ID','MotherPDG',args.BrickName,args.TrackName]
rowdata = pd.read_csv(input_file_location,usecols=columns)


insert = rowdata[['z','tx','ty']]
#heatmap plot for tx and ty
hitdata = rowdata[['Hit_ID','tx','ty']]
hitdata['tx'] = hitdata['tx'].round(decimals = 2)
hitdata['ty'] = hitdata['ty'].round(decimals = 2)
hitdata=hitdata.groupby(['tx','ty']).Hit_ID.nunique().reset_index()
print('Creating plot')

if args.TrackName == 'MC_Track_ID':
  
  rowdata['MC_Track_ID'] = rowdata['MC_Track_ID'].astype(str)
  rowdata['MC_Event_ID'] = rowdata['MC_Event_ID'].astype(str)
  rowdata['MC_Track'] = rowdata['MC_Track_ID'] + '-' + rowdata['MC_Event_ID']
  rowdata.drop(['MC_Track_ID','MC_Event_ID'], axis=1, inplace=True)
  
  
  #calculate the Track Length
  z_min = rowdata.groupby(['MC_Track']).z.min().reset_index() 
  z_max = rowdata.groupby(['MC_Track']).z.max().reset_index() 
  z_min = z_min.rename(columns={'z':'z_min'})
  z_max = z_max.rename(columns={'z':'z_max'})
  newdata = pd.merge(z_max,z_min,how='inner',on=['MC_Track'])

  newdata['Track_length'] = newdata['z_max'] - newdata['z_min']
  #newdata = pd.merge(newdata,mother,how='inner',on=['MC_Track'])
  #newdata = newdata.loc[newdata['Track_length'] > 0]
  print(newdata)

  #I have simplified the code and it works:
  newdata = pd.merge(newdata, rowdata[['x','MC_Track','z']].rename(columns={'x':'x_min'}), how='inner', left_on=['MC_Track','z_min'], right_on=['MC_Track','z'])
  newdata.drop(['z'], axis=1, inplace=True)

  newdata = pd.merge(newdata, rowdata[['x','MC_Track','z']].rename(columns={'x':'x_max'}), how='inner', left_on=['MC_Track','z_max'], right_on=['MC_Track','z'])
  newdata.drop(['z'], axis=1, inplace=True)

  newdata = pd.merge(newdata, rowdata[['y','MC_Track','z']].rename(columns={'y':'y_min'}), how='inner', left_on=['MC_Track','z_min'], right_on=['MC_Track','z'])
  newdata.drop(['z'], axis=1, inplace=True)

  newdata = pd.merge(newdata, rowdata[['y','MC_Track','z']].rename(columns={'y':'y_max'}), how='inner', left_on=['MC_Track','z_max'], right_on=['MC_Track','z'])
  newdata.drop(['z'], axis=1, inplace=True)


  newdata = newdata.loc[newdata['Track_length'] > 0]

  
  
  newdata['delta_x'] = newdata['x_max'] - newdata['x_min']
  newdata['delta_y'] = newdata['y_max'] - newdata['y_min']
  #print(newdata)

  newdata['TX'] = newdata['delta_x']/newdata['Track_length']
  newdata['TY'] = newdata['delta_y']/newdata['Track_length']
  #print(newdata)

  print('Maximum angle TX is', newdata['TX'].max())
  print('Minimum angle TX is', newdata['TX'].min())
  print('Maximum angle TY is', newdata['TY'].max())
  print('Minimum angle TY is', newdata['TY'].min())
  
  output1 = 'MC_Track'+'_smallangledata.csv'
  output = 'MC_Track'+'_AngleData.csv'
  newdata.to_csv(output,index=False)
  #insert.to_csv(output1,index=False)
  print(output, 'was saved.') 
  #print(output1, 'was saved.')

  
else:

  #calculate the Track Length
  z_min = rowdata.groupby([args.TrackName]).z.min().reset_index() 
  z_max = rowdata.groupby([args.TrackName]).z.max().reset_index() 
  z_min = z_min.rename(columns={'z':'z_min'})
  z_max = z_max.rename(columns={'z':'z_max'})
  newdata = pd.merge(z_max,z_min,how='inner',on=[args.TrackName])
  newdata['Track_length'] = newdata['z_max'] - newdata['z_min']
  #print(newdata)


  x_max = pd.merge(newdata, rowdata, how='inner', left_on=[args.TrackName,'z_max'], right_on=[args.TrackName,'z'])
  x_max = x_max.rename(columns={'x':'x_max'})
  x_max = x_max[['x_max', args.TrackName]]
  newdata = pd.merge(newdata,x_max,how='inner',on=[args.TrackName])
  #print(newdata)

  x_min = pd.merge(newdata, rowdata, how='inner', left_on=[args.TrackName,'z_min'], right_on=[args.TrackName,'z'])
  x_min = x_min.rename(columns={'x':'x_min'})
  x_min = x_min[['x_min', args.TrackName]]
  newdata = pd.merge(newdata,x_min,how='inner',on=[args.TrackName])
  #print(newdata)

  y_max = pd.merge(newdata, rowdata, how='inner', left_on=[args.TrackName,'z_max'], right_on=[args.TrackName,'z'])
  y_max = y_max.rename(columns={'y':'y_max'})
  y_max = y_max[['y_max', args.TrackName]]
  newdata = pd.merge(newdata,y_max,how='inner',on=[args.TrackName])
  #print(newdata)

  y_min = pd.merge(newdata, rowdata, how='inner', left_on=[args.TrackName,'z_min'], right_on=[args.TrackName,'z'])
  y_min = y_min.rename(columns={'y':'y_min'})
  y_min = y_min[['y_min', args.TrackName]]
  newdata = pd.merge(newdata,y_min,how='inner',on=[args.TrackName])
  #print(newdata)

  newdata['delta_x'] = newdata['x_max'] - newdata['x_min']
  newdata['delta_y'] = newdata['y_max'] - newdata['y_min']
  #print(newdata)

  newdata['TX'] = newdata['delta_x']/newdata['Track_length']
  newdata['TY'] = newdata['delta_y']/newdata['Track_length']
  #print(newdata)

  mother = rowdata[['MotherPDG',args.TrackName]]
  newdata=pd.merge(newdata,mother,how='inner',on=[args.TrackName])
  #print(newdata)

  print('Maximum angle TX is', newdata['TX'].max())
  print('Minimum angle TX is', newdata['TX'].min())
  print('Maximum angle TY is', newdata['TY'].max())
  print('Minimum angle TY is', newdata['TY'].min())

  output1 = args.TrackName+'_smallangledata.csv'
  output = args.TrackName+'_AngleData.csv'
  newdata.to_csv(output,index=False)
  #insert.to_csv(output1,index=False)
  print(output, 'was saved.') 
  #print(output1, 'was saved.')

plt.hist2d(hitdata['tx'],hitdata['ty'],bins=100)
