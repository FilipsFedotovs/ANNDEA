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
parser.add_argument('--StartImage',help="Please select the beginning Image", default='1')
parser.add_argument('--Images',help="Please select the number of Images", default='1')
parser.add_argument('--PlotType',help="Enter plot type: XZ/YZ/3D", default='XZ')
parser.add_argument('--Res',help="Please enter the scaling resolution in microns", default=10)
parser.add_argument('--EImg',help="Enhance image?", default='Y')
parser.add_argument('--MaxX',help="Enter max half height of the image in microns", default=3500)
parser.add_argument('--MaxY',help="Enter max half width of the image in microns", default=3500)
parser.add_argument('--MaxZ',help="Enter max length of the image in microns", default=20000)
parser.add_argument('--TrainSampleID',help="What training sample to visualise?", default='SHIP_UR_v1')
parser.add_argument('--Rescale',help="Rescale the images : Y/N ?", default='N')
parser.add_argument('--Type',help="Please enter the sample type: Val or number for the training set", default='1')
parser.add_argument('--Label',help="Which labels would you like to see: 'ANY/Fake/Truth", default='ANY')
parser.add_argument('--SeedType',help="", default='TRACK')
########################################     Main body functions    #########################################
args = parser.parse_args()
TrackNo=int(args.Images)
TrainSampleID=args.TrainSampleID
SeedType=args.SeedType
resolution=float(args.Res)
MaxX=float(args.MaxX)
MaxY=float(args.MaxY)
MaxZ=float(args.MaxZ)
if args.EImg=='Y':
    EImg='CNN-E'
else:
    EImg='CNN'
StartImage=int(args.StartImage)
if StartImage>TrackNo:
    TrackNo=StartImage
if args.Rescale=='Y':
    Rescale=True
else:
    Rescale=False
boundsX=int(round(MaxX/resolution,0))
boundsY=int(round(MaxY/resolution,0))
boundsZ=int(round(MaxZ/resolution,0))
H=(boundsX)*2
W=(boundsY)*2
L=(boundsZ)

print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################   Initialising ANNDEA image visualisation module     #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
if args.Type=='Val':
 input_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_'+SeedType+'_SEEDS_OUTPUT.pkl'
else:
 input_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_'+SeedType+'_SEEDS_OUTPUT_'+args.Type+'.pkl'

ImageObjectSet=UF.PickleOperations(input_file_location,'r', 'N/A')[0]

if args.Label=='Truth':
     ImageObjectSet=[im for im in ImageObjectSet if im.Label == 1]
if args.Label=='Fake':
     ImageObjectSet=[im for im in ImageObjectSet if im.Label == 0]
ImageObjectSet=ImageObjectSet[StartImage-1:min(TrackNo,len(ImageObjectSet))]
TrainSampleInputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+TrainSampleInputMeta+bcolors.ENDC)
MetaInput=UF.PickleOperations(TrainSampleInputMeta,'r', 'N/A')
print(MetaInput[1])
Meta=MetaInput[0]
DummyModelObj=UF.ModelMeta("Dummy'")
DummyModelObj.IniModelMeta([[],[],[],[],[],[],[],[],[],[],[],[MaxX,MaxY,MaxZ,resolution]], 'Tensorflow', Meta, EImg, 'CNN')

#ImageObjectSet[0].PrepareSeedPrint(DummyModelObj)
#ImageObjectSet[0].Plot('XZ')

if args.PlotType=='XZ':
  InitialData=[]
  Index=-1
  for x in range(-boundsX,boundsX):
          for z in range(0,boundsZ):
            InitialData.append(0.0)
  Matrix = np.array(InitialData)
  Matrix=np.reshape(Matrix,(H,L))
if args.PlotType=='YZ':
 InitialData=[]
 Index=-1
 for y in range(-boundsY,boundsY):
          for z in range(0,boundsZ):
            InitialData.append(0.0)

 Matrix = np.array(InitialData)
 Matrix=np.reshape(Matrix,(W,L))
if args.PlotType=='XY':
  InitialData=[]
  Index=-1
  for x in range(-boundsX,boundsX):
          for y in range(-boundsY,boundsY):
            InitialData.append(0.0)
  Matrix = np.array(InitialData)
  Matrix=np.reshape(Matrix,(H,W))


Title=args.Label+' Track Image'


counter=0
for sd in ImageObjectSet:
 progress=int( round( (float(counter)/float(len(ImageObjectSet))*100),1)  )
 print('Rendering images, progress is ',progress, end="\r", flush=True)
 counter+=1
 sd.PrepareSeedPrint(DummyModelObj)
 if args.PlotType=='XZ':
  for Hits in sd.TrackPrint:
      if abs(Hits[0])<boundsX and abs(Hits[2])<boundsZ:
                   Matrix[Hits[0]+boundsX][Hits[2]]+=1
 if args.PlotType=='YZ':
        for Hits in sd.TrackPrint:
                 if abs(Hits[1])<boundsY and abs(Hits[2])<boundsZ:
                   Matrix[Hits[1]+boundsY][Hits[2]]+=1
 if args.PlotType=='XY':
     for Hits in sd.TrackPrint:
       if abs(Hits[0])<boundsX and abs(Hits[1])<boundsY:
         Matrix[Hits[0]+boundsX][Hits[1]+boundsY]+=1
image_no=len(ImageObjectSet)
del ImageObjectSet
import matplotlib as plt
from matplotlib.colors import LogNorm
import numpy as np
from matplotlib import pyplot as plt
if args.PlotType=='XZ':
 plt.title(Title)
 plt.xlabel('Z [microns /'+str(int(resolution))+']')
 plt.ylabel('X [microns /'+str(int(resolution))+']')
 if image_no==1:
    image=plt.imshow(Matrix,cmap='gray_r',extent=[0,boundsZ,boundsX,-boundsX])#,norm=LogNorm())
 else:
    image=plt.imshow(Matrix,cmap='gray_r',extent=[0,boundsZ,boundsX,-boundsX],norm=LogNorm())
 plt.gca().invert_yaxis()
 plt.show()
if args.PlotType=='YZ':
 import numpy as np
 from matplotlib import pyplot as plt
 plt.title(Title)
 plt.xlabel('Z [microns /'+str(int(resolution))+']')
 plt.ylabel('Y [microns /'+str(int(resolution))+']')
 if image_no==1:
    image=plt.imshow(Matrix,cmap='gray_r',extent=[0,boundsZ,boundsY,-boundsY])#,norm=LogNorm())
 else:
    image=plt.imshow(Matrix,cmap='gray_r',extent=[0,boundsZ,boundsY,-boundsY],norm=LogNorm())
 plt.gca().invert_yaxis()
 plt.show()
if args.PlotType=='XY':
 import numpy as np
 from matplotlib import pyplot as plt
 plt.title(Title)
 plt.xlabel('X [microns /'+str(int(resolution))+']')
 plt.ylabel('Y [microns /'+str(int(resolution))+']')
 if image_no==1:
    image=plt.imshow(Matrix,cmap='gray_r',extent=[boundsX,-boundsX,-boundsY,boundsY])#,norm=LogNorm())
 else:
    image=plt.imshow(Matrix,cmap='gray_r',extent=[boundsX,-boundsX,-boundsY,boundsY],norm=LogNorm())
 plt.gca().invert_xaxis()
 plt.show()
exit()




