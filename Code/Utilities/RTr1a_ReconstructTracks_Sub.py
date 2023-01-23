#Current version 1.1 - add change sys path capability

########################################    Import essential libriries    #############################################
import argparse
import sys


#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--i',help="Set number", default='1')
parser.add_argument('--j',help="Subset number", default='1')
parser.add_argument('--Z_ID_Max',help="SubSubset number", default='1')
parser.add_argument('--Z_overlap',help="Enter Z id", default='1')
parser.add_argument('--Y_overlap',help="Enter Y id", default='1')
parser.add_argument('--X_overlap',help="Enter X id", default='1')
parser.add_argument('--stepX',help="Enter X step size", default='0')
parser.add_argument('--stepY',help="Enter Y step size", default='0')
parser.add_argument('--stepZ',help="Enter Z step size", default='0')
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')
parser.add_argument('--PY',help="Python libraries directory location", default='.')
parser.add_argument('--zOffset',help="Data offset on z", default='0.0')
parser.add_argument('--yOffset',help="Data offset on y", default='0.0')
parser.add_argument('--xOffset',help="Data offset on x", default='0.0')
parser.add_argument('--cut_dt',help="Cut on angle difference", default='1.0')
parser.add_argument('--cut_dr',help="Cut on angle difference", default='4000')
parser.add_argument('--Log',help="Pull out stats?", default='No')
parser.add_argument('--ModelName',help="Name of the model to use?", default='0')
parser.add_argument('--BatchID',help="Give name to this train sample", default='')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')

#Working out where are the Py libraries
args = parser.parse_args()
#Loading Directory locations
EOS_DIR=args.EOS
AFS_DIR=args.AFS
PY_DIR=args.PY

if PY_DIR!='': #Temp solution
    sys.path=['',PY_DIR]
    sys.path.append('/usr/lib64/python36.zip')
    sys.path.append('/usr/lib64/python3.6')
    sys.path.append('/usr/lib64/python3.6/lib-dynload')
    sys.path.append('/usr/lib64/python3.6/site-packages')
    sys.path.append('/usr/lib/python3.6/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')
import pandas as pd #We use Panda for a routine data processing
######################################## Set variables  #############################################################
Z_overlap=int(args.Z_overlap)
Y_overlap=int(args.Y_overlap)
X_overlap=int(args.X_overlap)
Z_ID_Max=int(args.Z_ID_Max)
Y_ID=int(args.j)/Y_overlap
X_ID=int(args.i)/X_overlap
Y_ID_n=int(args.j)
X_ID_n=int(args.i)
stepX=float(args.stepX)
stepZ=float(args.stepZ)
stepY=float(args.stepY)
z_offset=float(args.zOffset)
y_offset=float(args.yOffset)
x_offset=float(args.xOffset)
cut_dt=float(args.cut_dt)
cut_dr=float(args.cut_dr)
Log=args.Log.upper()
ModelName=args.ModelName
RecBatchID=args.BatchID
p=args.p
o=args.o
sfx=args.sfx
pfx=args.pfx

import UtilityFunctions as UF #This is where we keep routine utility functions

#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1_'+RecBatchID+'_hits.csv'
output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(X_ID_n)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(X_ID_n)+'_'+str(Y_ID_n)+sfx

print(UF.TimeStamp(), "Modules Have been imported successfully...")
print(UF.TimeStamp(),'Loading pre-selected data from ',input_file_location)

#Load the file with Hit detailed information
data=pd.read_csv(input_file_location,header=0,usecols=["Hit_ID","x","y","z","tx","ty"])[["Hit_ID","x","y","z","tx","ty"]]
data["x"] = pd.to_numeric(data["x"],downcast='float')
data["y"] = pd.to_numeric(data["y"],downcast='float')
data["z"] = pd.to_numeric(data["z"],downcast='float')
data['x']=data['x']-x_offset
data['y']=data['y']-y_offset
data["Hit_ID"] = data["Hit_ID"].astype(str)
data['z']=data['z']-z_offset
print(UF.TimeStamp(),'Preparing data... ')
#Keeping only sections of the Hit data relevant to the volume being reconstructed to use less memory

data.drop(data.index[data['x'] >= ((X_ID+1)*stepX)], inplace = True)  #Keeping the relevant z slice
data.drop(data.index[data['x'] < (X_ID*stepX)], inplace = True)  #Keeping the relevant z slice
data.drop(data.index[data['y'] >= ((Y_ID+1)*stepY)], inplace = True)  #Keeping the relevant z slice
data.drop(data.index[data['y'] < (Y_ID*stepY)], inplace = True)  #Keeping the relevant z slice


#Additional options to include reconstruction stats. Require MC and possibly FEDRA reconstruction.
if Log!='NO':
    print(UF.TimeStamp(),'Preparing MC data... ')
    input_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/ETr1_'+RecBatchID+'_hits.csv'
    print(UF.TimeStamp(),'Loading pre-selected data from ',input_file_location)
    MCdata=pd.read_csv(input_file_location,header=0,
                                usecols=["Hit_ID","x","y","z","tx","ty",'MC_Mother_Track_ID'])[["Hit_ID","x","y","z","tx","ty",'MC_Mother_Track_ID']]
    MCdata["x"] = pd.to_numeric(MCdata["x"],downcast='float')
    MCdata["y"] = pd.to_numeric(MCdata["y"],downcast='float')
    MCdata["z"] = pd.to_numeric(MCdata["z"],downcast='float')
    MCdata["Hit_ID"] = MCdata["Hit_ID"].astype(str)
    MCdata['z']=MCdata['z']-z_offset
    MCdata['x']=MCdata['x']-x_offset
    MCdata['y']=MCdata['y']-y_offset

    MCdata.drop(MCdata.index[MCdata['x'] >= ((X_ID+1)*stepX)], inplace = True)  #Keeping the relevant z slice
    MCdata.drop(MCdata.index[MCdata['x'] < (X_ID*stepX)], inplace = True)  #Keeping the relevant z slice
    MCdata.drop(MCdata.index[MCdata['y'] >= ((Y_ID+1)*stepY)], inplace = True)  #Keeping the relevant z slice
    MCdata.drop(MCdata.index[MCdata['y'] < (Y_ID*stepY)], inplace = True)  #Keeping the relevant z slice

if Log=='KALMAN':
    print(UF.TimeStamp(),'Preparing KALMAN data... ')
    input_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/KTr1_'+RecBatchID+'_hits.csv'
    print(UF.TimeStamp(),'Loading pre-selected data from ',input_file_location)
    FEDRAdata=pd.read_csv(input_file_location,header=0,
                                usecols=["Hit_ID","x","y","z","tx","ty",'Rec_Track_ID'])[["Hit_ID","x","y","z","tx","ty",'Rec_Track_ID']]
    FEDRAdata["x"] = pd.to_numeric(FEDRAdata["x"],downcast='float')
    FEDRAdata["y"] = pd.to_numeric(FEDRAdata["y"],downcast='float')
    FEDRAdata["z"] = pd.to_numeric(FEDRAdata["z"],downcast='float')
    FEDRAdata["Hit_ID"] = FEDRAdata["Hit_ID"].astype(str)
    FEDRAdata['z']=FEDRAdata['z']-z_offset
    FEDRAdata['x']=FEDRAdata['x']-x_offset
    FEDRAdata['y']=FEDRAdata['y']-y_offset

    FEDRAdata.drop(FEDRAdata.index[FEDRAdata['x'] >= ((X_ID+1)*stepX)], inplace = True)  #Keeping the relevant z slice
    FEDRAdata.drop(FEDRAdata.index[FEDRAdata['x'] < (X_ID*stepX)], inplace = True)  #Keeping the relevant z slice
    FEDRAdata.drop(FEDRAdata.index[FEDRAdata['y'] >= ((Y_ID+1)*stepY)], inplace = True)  #Keeping the relevant z slice
    FEDRAdata.drop(FEDRAdata.index[FEDRAdata['y'] < (Y_ID*stepY)], inplace = True)  #Keeping the relevant z slice

torch_import=True
cluster_output=[]
for k in range(0,Z_ID_Max):
    Z_ID=int(k)/Z_overlap
    temp_data=data.drop(data.index[data['z'] >= ((Z_ID+1)*stepZ)])  #Keeping the relevant z slice
    temp_data=temp_data.drop(temp_data.index[temp_data['z'] < (Z_ID*stepZ)])  #Keeping the relevant z slice
    temp_data_list=temp_data.values.tolist()
    if Log!='NO':
       temp_MCData=MCdata.drop(MCdata.index[MCdata['z'] >= ((Z_ID+1)*stepZ)])  #Keeping the relevant z slice
       temp_MCData=temp_MCData.drop(temp_MCData.index[temp_MCData['z'] < (Z_ID*stepZ)])  #Keeping the relevant z slice
       temp_MCdata_list=MCdata.values.tolist()
    if Log=='KALMAN':
       temp_FEDRAdata=FEDRAdata.drop(FEDRAdata.index[FEDRAdata['z'] >= ((Z_ID+1)*stepZ)])  #Keeping the relevant z slice
       temp_FEDRAdata=FEDRAdata.drop(temp_FEDRAdata.index[temp_FEDRAdata['z'] < (Z_ID*stepZ)])  #Keeping the relevant z slice
       temp_FEDRAdata_list=FEDRAdata.values.tolist()
    print(UF.TimeStamp(),'Creating the cluster', X_ID,Y_ID,Z_ID)
    HC=UF.HitCluster([X_ID,Y_ID,Z_ID],[stepX,stepY,stepZ]) #Initializing the cluster
    print(UF.TimeStamp(),'Decorating the clusters')
    HC.LoadClusterHits(temp_data_list) #Decorating the Clusters with Hit information
    if len(HC.RawClusterGraph)>1: #If we have at least 2 Hits in the cluster that can create
        print(UF.TimeStamp(),'Generating the edges...')
        GraphStatus = HC.GenerateEdges(cut_dt, cut_dr)
        combined_weight_list=[]
        if GraphStatus:
            if HC.ClusterGraph.num_edges>0: #We only bring torch and GNN if we have some edges to classify
                        print(UF.TimeStamp(),'Classifying the edges...')
                        print(UF.TimeStamp(),'Preparing the model')
                        if torch_import:
                            import torch
                            EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
                            EOSsubModelDIR=EOSsubDIR+'/'+'Models'
                            #Load the model meta file
                            Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
                            #Specify the model path
                            Model_Path=EOSsubModelDIR+'/'+args.ModelName
                            ModelMeta=UF.PickleOperations(Model_Meta_Path, 'r', 'N/A')[0]
                            #Meta file contatins training session stats. They also record the optimal acceptance.
                            Acceptance=ModelMeta.TrainSessionsData[-1][-1][3]
                            device = torch.device('cpu')
                            #In PyTorch we don't save the actual model like in Tensorflow. We just save the weights, hence we have to regenerate the model again. The recepy is in the Model Meta file
                            model = UF.GenerateModel(ModelMeta).to(device)
                            model.load_state_dict(torch.load(Model_Path))
                            model.eval() #In Pytorch this function sets the model into the evaluation mode.
                            torch_import=False
                        w = model(HC.ClusterGraph.x, HC.ClusterGraph.edge_index, HC.ClusterGraph.edge_attr) #Here we use the model to assign the weights between Hit edges
                        w=w.tolist()
                        for edge in range(len(HC.edges)):
                            combined_weight_list.append(HC.edges[edge]+w[edge])
                        print(UF.TimeStamp(),'Tracking the cluster...')
                        HC.LinkHits(combined_weight_list,False,[],cut_dt,cut_dr,Acceptance) #We use the weights assigned by the model to perform microtracking within the volume
                        HC.UnloadClusterGraph() #Remove the Graph that we do not need anymore to reduce the object size
                        print(UF.TimeStamp(),'Current cLuster tracking is finished, adding it to the output container...')
                        cluster_output.append(HC)
                        if Log!='NO':
                            print(UF.TimeStamp(),'Tracking the cluster...')
                            HC.LinkHits(combined_weight_list,True,temp_MCdata_list,cut_dt,cut_dr,Acceptance) #We use the weights assigned by the model to perform microtracking within the volume
                        if Log=='KALMAN':
                               HC.TestKalmanHits(temp_FEDRAdata_list,temp_MCdata_list)
                        continue
            if HC.ClusterGraph.num_edges==0:
                 HC.RecHits=pd.DataFrame([], columns = ['HitID','z','Segment_ID'])
                 cluster_output.append(HC)




    else:
        HC.RecHits=pd.DataFrame([], columns = ['HitID','z','Segment_ID'])
        cluster_output.append(HC)
print(len(cluster_output))
for hc in cluster_output:
    print(HC.RecHits)
print(UF.TimeStamp(),'Writing the output...')
print(UF.PickleOperations(output_file_location,'w', cluster_output)[1])


