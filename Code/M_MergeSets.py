#This simple script is design for studying and selecting an optimal size and resolution of the images

########################################    Import libraries    #############################################
import csv
import argparse
import math
import ast
import numpy as np
import logging
import pickle

class bcolors:
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
import UtilityFunctions as UF
from UtilityFunctions import EMO
import Parameters as PM

parser = argparse.ArgumentParser(description='This script helps to visualise the seeds by projecting their hit coordinates to the 2-d screen.')
parser.add_argument('--TrainSampleID',help="What training sample to visualise?", default='SHIP_UR_v1')
parser.add_argument('--NewTrainSampleID',help="What training sample to visualise?", default='SHIP_UR_v1')
parser.add_argument('--f',help="Where are the sets located??", default='/eos/user/')
parser.add_argument('--Sets',help="Name of the training sets?", default='[]')
parser.add_argument('--Type',help="Please enter the sample type: VAL or TRAIN", default='1')
########################################     Main body functions    #########################################
args = parser.parse_args()
TrainSampleID=args.TrainSampleID
NewSampleID=args.NewTrainSampleID
Sets=ast.literal_eval(args.Sets)
input_location=args.f
Type=args.Type



print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################   Initialising ANNDEA Training Set Merger            #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
CombinedObject=[]
for s in Sets:
    input_file_location=input_location+'/'+s+'.pkl'
    ObjectSet=UF.PickleOperations(input_file_location,'r', 'N/A')[0]
    CombinedObject+=ObjectSet
print(len(CombinedObject))
exit()
# if args.Type=='VAL':
#  input_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_TRACK_OUTPUT_'+args.Type+'.pkl'
#
# ImageObjectSet=UF.PickleOperations(input_file_location,'r', 'N/A')[0]
#
# if args.Label=='Truth':
#      ImageObjectSet=[im for im in ImageObjectSet if im.Label == 1]
# if args.Label=='Fake':
#      ImageObjectSet=[im for im in ImageObjectSet if im.Label == 0]
# ImageObjectSet=ImageObjectSet[StartImage-1:min(TrackNo,len(ImageObjectSet))]
# TrainSampleInputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
# print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+TrainSampleInputMeta+bcolors.ENDC)
# MetaInput=UF.PickleOperations(TrainSampleInputMeta,'r', 'N/A')
# print(MetaInput[1])
# Meta=MetaInput[0]
# DummyModelObj=UF.ModelMeta("Dummy'")
# DummyModelObj.IniModelMeta([[],[],[],[],[],[],[],[],[],[],[],[MaxX,MaxY,MaxZ,resolution]], 'Tensorflow', Meta, EImg, 'CNN')
#
# #ImageObjectSet[0].PrepareSeedPrint(DummyModelObj)
# #ImageObjectSet[0].Plot('XZ')
#
# if args.PlotType=='XZ':
#   InitialData=[]
#   Index=-1
#   for x in range(-boundsX,boundsX):
#           for z in range(0,boundsZ):
#             InitialData.append(0.0)
#   Matrix = np.array(InitialData)
#   Matrix=np.reshape(Matrix,(H,L))
# if args.PlotType=='YZ':
#  InitialData=[]
#  Index=-1
#  for y in range(-boundsY,boundsY):
#           for z in range(0,boundsZ):
#             InitialData.append(0.0)
#
#  Matrix = np.array(InitialData)
#  Matrix=np.reshape(Matrix,(W,L))
# if args.PlotType=='XY':
#   InitialData=[]
#   Index=-1
#   for x in range(-boundsX,boundsX):
#           for y in range(-boundsY,boundsY):
#             InitialData.append(0.0)
#   Matrix = np.array(InitialData)
#   Matrix=np.reshape(Matrix,(H,W))
#
#
# Title=args.Label+' Track Image'
#
#
# counter=0
# for sd in ImageObjectSet:
#  progress=int( round( (float(counter)/float(len(ImageObjectSet))*100),1)  )
#  print('Rendering images, progress is ',progress, end="\r", flush=True)
#  counter+=1
#  sd.PrepareSeedPrint(DummyModelObj)
#  if args.PlotType=='XZ':
#   for Hits in sd.TrackPrint:
#       if abs(Hits[0])<boundsX and abs(Hits[2])<boundsZ:
#                    Matrix[Hits[0]+boundsX][Hits[2]]+=1
#  if args.PlotType=='YZ':
#         for Hits in sd.TrackPrint:
#                  if abs(Hits[1])<boundsY and abs(Hits[2])<boundsZ:
#                    Matrix[Hits[1]+boundsY][Hits[2]]+=1
#  if args.PlotType=='XY':
#      for Hits in sd.TrackPrint:
#        if abs(Hits[0])<boundsX and abs(Hits[1])<boundsY:
#          Matrix[Hits[0]+boundsX][Hits[1]+boundsY]+=1
# image_no=len(ImageObjectSet)
# del ImageObjectSet
# import matplotlib as plt
# from matplotlib.colors import LogNorm
# import numpy as np
# from matplotlib import pyplot as plt
# if args.PlotType=='XZ':
#  plt.title(Title)
#  plt.xlabel('Z [microns /'+str(int(resolution))+']')
#  plt.ylabel('X [microns /'+str(int(resolution))+']')
#  if image_no==1:
#     image=plt.imshow(Matrix,cmap='gray_r',extent=[0,boundsZ,boundsX,-boundsX])#,norm=LogNorm())
#  else:
#     image=plt.imshow(Matrix,cmap='gray_r',extent=[0,boundsZ,boundsX,-boundsX],norm=LogNorm())
#  plt.gca().invert_yaxis()
#  plt.show()
# if args.PlotType=='YZ':
#  import numpy as np
#  from matplotlib import pyplot as plt
#  plt.title(Title)
#  plt.xlabel('Z [microns /'+str(int(resolution))+']')
#  plt.ylabel('Y [microns /'+str(int(resolution))+']')
#  if image_no==1:
#     image=plt.imshow(Matrix,cmap='gray_r',extent=[0,boundsZ,boundsY,-boundsY])#,norm=LogNorm())
#  else:
#     image=plt.imshow(Matrix,cmap='gray_r',extent=[0,boundsZ,boundsY,-boundsY],norm=LogNorm())
#  plt.gca().invert_yaxis()
#  plt.show()
# if args.PlotType=='XY':
#  import numpy as np
#  from matplotlib import pyplot as plt
#  plt.title(Title)
#  plt.xlabel('X [microns /'+str(int(resolution))+']')
#  plt.ylabel('Y [microns /'+str(int(resolution))+']')
#  if image_no==1:
#     image=plt.imshow(Matrix,cmap='gray_r',extent=[boundsX,-boundsX,-boundsY,boundsY])#,norm=LogNorm())
#  else:
#     image=plt.imshow(Matrix,cmap='gray_r',extent=[boundsX,-boundsX,-boundsY,boundsY],norm=LogNorm())
#  plt.gca().invert_xaxis()
#  plt.show()
# exit()




