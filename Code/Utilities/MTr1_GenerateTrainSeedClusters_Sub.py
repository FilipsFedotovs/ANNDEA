#Made by Filips Fedotovs
#Part of ANNDEA package - this is the script that usually operates in HTCondor



########################################    Import libraries    #############################################
import argparse
#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--i',help="Enter the job number id", default='0')
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
parser.add_argument('--cut_dz',help="Cut on a distance across z-axis", default='4000')
parser.add_argument('--BatchID',help="Give name to this train sample", default='SHIP_TrainSample_v1')
parser.add_argument('--Yoverlap',help="Enter the level of overlap in integer number between reconstruction blocks along y-axis.", default='1')
parser.add_argument('--Xoverlap',help="Enter the level of overlap in integer number between reconstruction blocks along x-axis.", default='1')
parser.add_argument('--Zoverlap',help="Enter the level of overlap in integer number between reconstruction blocks along z-axis.", default='1')
parser.add_argument('--jobs',help="The list of jobs", default='')
parser.add_argument('--PY',help="Python libraries directory location", default='.')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
parser.add_argument('--SeedFlowLog',help="Track the seed cutflow?", default='N')
######################################## Set variables  #############################################################
args = parser.parse_args()

PY_DIR=args.PY
EOS_DIR=args.EOS
AFS_DIR=args.AFS

p=args.p
o=args.o
sfx=args.sfx
pfx=args.pfx

import sys
if PY_DIR!='': #Temp solution
    sys.path=['',PY_DIR]
    sys.path.append('/usr/lib64/python39.zip')
    sys.path.append('/usr/lib64/python3.9')
    sys.path.append('/usr/lib64/python3.9/lib-dynload')
    sys.path.append('/usr/lib64/python3.9/site-packages')
    sys.path.append('/usr/lib/python3.9/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')

########################################    Import libraries    #############################################
import pandas as pd #We use Panda for a routine data processing
import ast


jobs=ast.literal_eval(args.jobs)
SeedFlowLog=args.SeedFlowLog=='Y'
l=int(args.i)
i,j,k=jobs[l][0],jobs[l][1],jobs[l][2]
Yoverlap,Xoverlap,Zoverlap=int(args.Yoverlap),int(args.Xoverlap),int(args.Zoverlap)
X_ID=int(i)/Xoverlap #Renormalising the index of the cluster along x-axis
X_ID_n=int(i)
Y_ID=int(j)/Yoverlap #Renormalising the index of the cluster along x-axis
Y_ID_n=int(j)
Z_ID=int(k)/Zoverlap #Renormalising the index of the cluster along x-axis
Z_ID_n=int(k)

stepX=float(args.stepX) #The size of the cluster along x-direction
stepY=float(args.stepY) #The size of the cluster along y-direction
stepZ=float(args.stepZ) #The size of the cluster along z-direction (for normalisation)

y_offset=float(args.yOffset)
x_offset=float(args.xOffset)

cut_dt=float(args.cut_dt) #Simple geometric cuts that help reduce number of hit combinations within the cluster for classification
cut_dr=float(args.cut_dr) #Simple geometric cuts that help reduce number of hit combinations within the cluster for classification
cut_dz=float(args.cut_dz) #Simple geometric cuts that help reduce number of hit combinations within the cluster for classification

val_ratio=float(args.valRatio)
test_ratio=float(args.testRatio)

#Loading Directory locations
TrainSampleID=args.BatchID
import U_UI as UF #This is where we keep routine utility functions
import U_HC as HC #This is where we keep routine utility functions

#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'/MTr1_'+TrainSampleID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_hits.csv'
print(input_file_location)
print(UF.TimeStamp(), "Modules Have been imported successfully...")
print(UF.TimeStamp(),'Loading pre-selected data from ',input_file_location)
#Load the file with Hit detailed information
data=pd.read_csv(input_file_location,header=0,usecols=["Hit_ID","x","y","z","tx","ty","MC_Track_ID"])[["Hit_ID","x","y","z","tx","ty","MC_Track_ID"]]
print(UF.TimeStamp(),'Preparing data... ')
data["x"] = pd.to_numeric(data["x"],downcast='float')
data["y"] = pd.to_numeric(data["y"],downcast='float')
data["z"] = pd.to_numeric(data["z"],downcast='float')
data["Hit_ID"] = data["Hit_ID"].astype(str)
data_list=data.values.tolist()
#Specifying the full path to input/output files
print(UF.TimeStamp(),'Seeding the cluster.. ')

HC=HC.HitCluster([X_ID,Y_ID, Z_ID],[stepX,stepY, stepZ]) #Initialise HitCluster instance
HC.LoadClusterHits(data_list) #Decorate hot cluster with hit detailed data
GraphStatus = HC.GenerateSeeds(cut_dt, cut_dr, cut_dz, -1, -1, SeedFlowLog) #Creating Hit Cluster graph (using PyTorch Graph calss). We add Labels too sicnce it is Train data
#There are nodes + graph is generated. Add it to the container
output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+TrainSampleID+'_'+str(X_ID_n)+'/'+pfx+'_'+TrainSampleID+'_'+o+'_'+str(X_ID_n)+sfx
UF.PickleOperations(output_file_location,'w', HC) #Write the output
#End of the script



