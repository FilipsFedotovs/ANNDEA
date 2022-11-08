#Made by Filips Fedotovs
#Current version 1.0

########################################    Import libraries    #############################################
import argparse
import pandas as pd #We use Panda for a routine data processing
import math #We use it for data manipulation
import random

#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--Z_ID',help="Enter Z id", default='0')
parser.add_argument('--X_ID',help="Enter X id", default='0')
parser.add_argument('--stepX',help="Enter X step size", default='0')
parser.add_argument('--stepY',help="Enter Y step size", default='0')
parser.add_argument('--stepZ',help="Enter Z step size", default='0')
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')
parser.add_argument('--zOffset',help="Data offset on z", default='0.0')
parser.add_argument('--yOffset',help="Data offset on y", default='0.0')
parser.add_argument('--xOffset',help="Data offset on x", default='0.0')
parser.add_argument('--valRatio',help="Fraction of validation edges", default='0.1')
parser.add_argument('--testRatio',help="Fraction of test edges", default='0.05')
parser.add_argument('--cut_dt',help="Cut on angle difference", default='1.0')
parser.add_argument('--cut_dr',help="Cut on angle difference", default='4000')
parser.add_argument('--TrainSampleID',help="Give name to this train sample", default='SHIP_TrainSample_v1')
parser.add_argument('--Z_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along z-axis.", default='1')
parser.add_argument('--Y_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along y-axis.", default='1')
parser.add_argument('--X_overlap',help="Enter the level of overlap in integer number between reconstruction blocks along x-axis.", default='1')

######################################## Set variables  #############################################################
args = parser.parse_args()

Z_overlap,Y_overlap,X_overlap=int(args.Z_overlap),int(args.Y_overlap),int(args.X_overlap)

Z_ID=int(args.Z_ID)/Z_overlap
X_ID=int(args.X_ID)/X_overlap

Z_ID_n=int(args.Z_ID)
X_ID_n=int(args.X_ID)

stepX=float(args.stepX) #The coordinate of the st plate in the current scope
stepZ=float(args.stepZ)
stepY=float(args.stepY)
z_offset=float(args.zOffset)
y_offset=float(args.yOffset)
x_offset=float(args.xOffset)
cut_dt=float(args.cut_dt)
cut_dr=float(args.cut_dr)
val_ratio=float(args.valRatio)
test_ratio=float(args.testRatio)

#Loading Directory locations
EOS_DIR=args.EOS
AFS_DIR=args.AFS
TrainSampleID=args.TrainSampleID
import UtilityFunctions as UF #This is where we keep routine utility functions

#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MTr1_'+TrainSampleID+'_hits.csv'
print(UF.TimeStamp(), "Modules Have been imported successfully...")
print(UF.TimeStamp(),'Loading pre-selected data from ',input_file_location)

data=pd.read_csv(input_file_location,header=0,
            usecols=["Hit_ID","x","y","z","tx","ty"])

data["x"] = pd.to_numeric(data["x"],downcast='float')
data["y"] = pd.to_numeric(data["y"],downcast='float')
data["z"] = pd.to_numeric(data["z"],downcast='float')
data['x']=data['x']-x_offset
data['y']=data['y']-y_offset
data["Hit_ID"] = data["Hit_ID"].astype(str)
data['z']=data['z']-z_offset
x_max=data['x'].max()
y_max=data['y'].max()

print(UF.TimeStamp(),'Creating clusters... ')
data.drop(data.index[data['z'] >= ((Z_ID+1)*stepZ)], inplace = True)  #Keeping the relevant z slice
data.drop(data.index[data['z'] < (Z_ID*stepZ)], inplace = True)  #Keeping the relevant z slice

data.drop(data.index[data['x'] >= ((X_ID+1)*stepX)], inplace = True)  #Keeping the relevant z slice
data.drop(data.index[data['x'] < (X_ID*stepX)], inplace = True)  #Keeping the relevant z slice
print(data)
exit()
data_list=data.values.tolist()

if Y_overlap==1:
    Ysteps=math.ceil((y_max)/stepY)
else:
    Ysteps=(math.ceil((y_max)/stepY)*(Y_overlap))-1



input_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/ETr1_'+TrainSampleID+'_hits.csv'
MCdata=pd.read_csv(input_file_location,header=0,
                            usecols=["Hit_ID","x","y","z","tx","ty",'MC_Mother_Track_ID'])
MCdata["x"] = pd.to_numeric(MCdata["x"],downcast='float')
MCdata["y"] = pd.to_numeric(MCdata["y"],downcast='float')
MCdata["z"] = pd.to_numeric(MCdata["z"],downcast='float')
MCdata["Hit_ID"] = MCdata["Hit_ID"].astype(str)
MCdata['z']=MCdata['z']-z_offset
MCdata['x']=MCdata['x']-x_offset
MCdata['y']=MCdata['y']-y_offset
MCdata.drop(MCdata.index[MCdata['z'] >= ((Z_ID+1)*stepZ)], inplace = True)  #Keeping the relevant z slice
MCdata.drop(MCdata.index[MCdata['z'] < (Z_ID*stepZ)], inplace = True)  #Keeping the relevant z slice
MCdata.drop(MCdata.index[MCdata['x'] >= ((X_ID+1)*stepX)], inplace = True)  #Keeping the relevant z slice
MCdata.drop(MCdata.index[MCdata['x'] < (X_ID*stepX)], inplace = True)  #Keeping the relevant z slice
MCdata_list=MCdata.values.tolist()
LoadedClusters=[]
for j in range(0,Ysteps):
        progress=round((float(j)/float(Ysteps))*100,2)
        print(UF.TimeStamp(),"progress is ",progress,' %') #Progress display
        HC=UF.HitCluster([X_ID,(j/Y_overlap),Z_ID],[stepX,stepY,stepZ])
        HC.LoadClusterHits(data_list)
        GraphStatus = HC.GenerateTrainData(MCdata_list,cut_dt, cut_dr)
        if GraphStatus:
            LoadedClusters.append(HC)
random.shuffle(LoadedClusters)
output_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/MTr1a_'+TrainSampleID+'_SelectedTrainClusters_'+str(Z_ID_n)+'_' +str(X_ID_n)+'.pkl'
UF.PickleOperations(output_file_location,'w', LoadedClusters)
#End of the script



