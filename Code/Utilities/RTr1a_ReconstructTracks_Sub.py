#Current version 1.1 - add change sys path capability

########################################    Import essential libriries    #############################################
import argparse
import sys
import copy
from statistics import mean
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

#This function is used for the actual tracking (combines hit-pairs into tracks
def InjectHit(Predator,Prey, Soft):
          if Soft==False:
             OverlapDetected=False
             New_Predator=copy.deepcopy(Predator)
             print(New_Predator)
             print(Prey)

             _prey_trigger_count=0
             for el in range (len(Prey[0])):
                 if Prey[0][el]!='_' and Predator[0][el]!='_':
                     if Prey[0][el]!=Predator[0][el]:
                        return(Predator,False)
                     elif Prey[0][el]==Predator[0][el]:
                        OverlapDetected=True
                        New_Predator[1][el]+=Prey[1][el]
                        _prey_trigger_count+=1

                 elif Predator[0][el]=='_' and Prey[0][el]!=Predator[0][el]:
                     New_Predator[0][el]=Prey[0][el]
                     New_Predator[1][el]+=Prey[1][el]
                 print(New_Predator)
                 print(Prey)
                 print(el,_prey_trigger_count)
                 x=input()
             if OverlapDetected:
                return(New_Predator,True)
             else:
                return(Predator,False)
          if Soft==True:
             for el1 in Prey[0]:
                 for el2 in Predator:
                  if el1==el2:
                     return True
             return False

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

torch_import=True
cluster_output=[]
import datetime
Before=datetime.datetime.now()
z_clusters_results=[]
for k in range(0,Z_ID_Max):
    Z_ID=int(k)/Z_overlap
    temp_data=data.drop(data.index[data['z'] >= ((Z_ID+1)*stepZ)])  #Keeping the relevant z slice
    temp_data=temp_data.drop(temp_data.index[temp_data['z'] < (Z_ID*stepZ)])  #Keeping the relevant z slice
    temp_data_list=temp_data.values.tolist()
    if Log!='NO':
       temp_MCData=MCdata.drop(MCdata.index[MCdata['z'] >= ((Z_ID+1)*stepZ)])  #Keeping the relevant z slice
       temp_MCData=temp_MCData.drop(temp_MCData.index[temp_MCData['z'] < (Z_ID*stepZ)])  #Keeping the relevant z slice
       temp_MCdata_list=MCdata.values.tolist()
    print(UF.TimeStamp(),'Creating the cluster', X_ID,Y_ID,Z_ID)
    HC=UF.HitCluster([X_ID,Y_ID,Z_ID],[stepX,stepY,stepZ]) #Initializing the cluster
    print(UF.TimeStamp(),'Decorating the clusters')
    HC.LoadClusterHits(temp_data_list) #Decorating the Clusters with Hit information
    if len(HC.RawClusterGraph)>1: #If we have at least 2 Hits in the cluster that can create
        print(UF.TimeStamp(),'Generating the edges...')
        print(UF.TimeStamp(),"Hit density of the Cluster",X_ID,Y_ID,Z_ID, "is  {} hits per cm\u00b3".format(round(len(HC.RawClusterGraph)/(0.6*0.6*1.2)),2))
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
                        combined_weight_list=pd.DataFrame(combined_weight_list, columns = ['l_HitID','r_HitID','link_strength'])
                        _Tot_Hits=pd.merge(HC.HitPairs, combined_weight_list, how="inner", on=['l_HitID','r_HitID'])
                        _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['link_strength'] <= Acceptance], inplace = True)
                        _Tot_Hits=_Tot_Hits[['r_HitID','l_HitID','r_z','l_z','link_strength']]
                        z_clusters_results.append(_Tot_Hits)
                        del HC
                        continue
import gc
gc.collect

if len(z_clusters_results)>0:
    _Tot_Hits=pd.concat(z_clusters_results)
    print(UF.TimeStamp(),'Collating the individual cluster results...')
    _Tot_Hits=pd.concat(z_clusters_results)
    ini_len=len(_Tot_Hits)
    print(UF.TimeStamp(),'Compressing the output...')
    _Tot_Hits=_Tot_Hits.groupby(['r_HitID','l_HitID','r_z','l_z']).link_strength.agg(['mean']).reset_index()
    _Tot_Hits=_Tot_Hits.rename(columns={'mean': "link_strength"})
    print(UF.TimeStamp(),'The compression ratio is ',round(100*len(_Tot_Hits)/ini_len,2), '%')
    print(UF.TimeStamp(),'Preparing the weighted hits for tracking...')
    _Tot_Hits.sort_values(by = ['r_HitID', 'l_z','link_strength'], ascending=[True,True, False],inplace=True)
    _Loc_Hits_r=_Tot_Hits[['r_z']].rename(columns={'r_z': 'z'})
    _Loc_Hits_l=_Tot_Hits[['l_z']].rename(columns={'l_z': 'z'})
    _Loc_Hits=pd.concat([_Loc_Hits_r,_Loc_Hits_l])
    _Loc_Hits.sort_values(by = ['z'], ascending=[True],inplace=True)
    _Loc_Hits.drop_duplicates(subset=['z'], keep='first', inplace=True)
    _Loc_Hits=_Loc_Hits.reset_index(drop=True)
    _Loc_Hits=_Loc_Hits.reset_index()

    _z_map_r=_Tot_Hits[['r_HitID','r_z']].rename(columns={'r_z': 'z','r_HitID': 'HitID'})
    _z_map_l=_Tot_Hits[['l_HitID','l_z']].rename(columns={'l_z': 'z','l_HitID': 'HitID'})
    _z_map=pd.concat([_z_map_r,_z_map_l])
    _z_map.drop_duplicates(subset=['HitID','z'], keep='first', inplace=True)

    _Loc_Hits_r=_Loc_Hits.rename(columns={'index': 'r_index', 'z': 'r_z'})
    _Loc_Hits_l=_Loc_Hits.rename(columns={'index': 'l_index', 'z': 'l_z'})
    _Tot_Hits=pd.merge(_Tot_Hits,_Loc_Hits_r, how='inner', on=['r_z'])
    _Tot_Hits=pd.merge(_Tot_Hits,_Loc_Hits_l, how='inner', on=['l_z'])
    _Tot_Hits=_Tot_Hits[['r_HitID','l_HitID','r_index','l_index','link_strength']]
    _Tot_Hits.sort_values(by = ['r_HitID', 'l_index','link_strength'], ascending=[True,True, False],inplace=True)
    _Tot_Hits.drop_duplicates(subset=['r_HitID', 'l_index','link_strength'], keep='first', inplace=True)
    _Tot_Hits.sort_values(by = ['l_HitID', 'r_index','link_strength'], ascending=[True,True, False],inplace=True)
    _Tot_Hits.drop_duplicates(subset=['l_HitID', 'r_index','link_strength'], keep='first', inplace=True)

    print(UF.TimeStamp(),'Tracking the cluster...')
    _Tot_Hits=_Tot_Hits.values.tolist()
    _Temp_Tot_Hits=[]
    for el in _Tot_Hits:
                    _Temp_Tot_Hit_El = [[],[]]
                    for pos in range(len(_Loc_Hits)):
                        if pos==el[2]:
                            _Temp_Tot_Hit_El[0].append(el[0])
                            _Temp_Tot_Hit_El[1].append(el[4])
                        elif pos==el[3]:
                            _Temp_Tot_Hit_El[0].append(el[1])
                            _Temp_Tot_Hit_El[1].append(el[4])
                        else:
                            _Temp_Tot_Hit_El[0].append('_')
                            _Temp_Tot_Hit_El[1].append(0.0)
                    _Temp_Tot_Hits.append(_Temp_Tot_Hit_El)
    _Tot_Hits=_Temp_Tot_Hits

    _Rec_Hits_Pool=[]
    _intital_size=len(_Tot_Hits)
    while len(_Tot_Hits)>0:
                    _Tot_Hits_PCopy=copy.deepcopy(_Tot_Hits)
                    _Tot_Hits_Predator=[]
                    for prd in range(0,len(_Tot_Hits_PCopy)):
                        print(UF.TimeStamp(),'Progress is ',round(100*prd/len(_Tot_Hits_PCopy),2), '%',end="\r", flush=True)
                        Predator=_Tot_Hits_PCopy[prd]
                        for pry in range(prd+1,len(_Tot_Hits_PCopy)):
                               Result=InjectHit(Predator,_Tot_Hits_PCopy[pry],False)
                               Predator=Result[0]
                        _Tot_Hits_Predator.append(Predator)
                    for s in _Tot_Hits_Predator:
                        s=s[0].append(mean(s.pop(1)))
                    _Tot_Hits_Predator = [item for l in _Tot_Hits_Predator for item in l]
                    for s in range(len(_Tot_Hits_Predator)):
                        for h in range(len(_Tot_Hits_Predator[s])):
                            if _Tot_Hits_Predator[s][h] =='_':
                                _Tot_Hits_Predator[s][h]='H_'+str(s)

                    column_no=len(_Tot_Hits_Predator[0])-1
                    columns=[]

                    for c in range(column_no):
                        columns.append(str(c))
                    columns.append('average_link_strength')
                    _Tot_Hits_Predator=pd.DataFrame(_Tot_Hits_Predator, columns = columns)
                    _Tot_Hits_Predator.sort_values(by = ['average_link_strength'], ascending=[False],inplace=True)
                    for c in range(column_no):
                        _Tot_Hits_Predator.drop_duplicates(subset=[str(c)], keep='first', inplace=True)
                    _Tot_Hits_Predator=_Tot_Hits_Predator.drop(['average_link_strength'],axis=1)

                    _Tot_Hits_Predator=_Tot_Hits_Predator.values.tolist()
                    for seg in range(len(_Tot_Hits_Predator)):
                        _Tot_Hits_Predator[seg]=[s for s in _Tot_Hits_Predator[seg] if ('H' in s)==False]
                    _Rec_Hits_Pool+=_Tot_Hits_Predator
                    for seg in _Tot_Hits_Predator:
                        _itr=0
                        while _itr<len(_Tot_Hits):
                            if InjectHit(seg,_Tot_Hits[_itr],True):
                                del _Tot_Hits[_itr]
                            else:
                                _itr+=1
    #Transpose the rows
    _track_list=[]
    _segment_id=RecBatchID+'_'+str(X_ID_n)+'_'+str(Y_ID_n)
    _no_tracks=len(_Rec_Hits_Pool)
    for t in range(len(_Rec_Hits_Pool)):
                  for h in _Rec_Hits_Pool[t]:
                         _track_list.append([_segment_id+'-'+str(t+1),h])
    _Rec_Hits_Pool=pd.DataFrame(_track_list, columns = ['Segment_ID','HitID'])
    _Rec_Hits_Pool=pd.merge(_z_map, _Rec_Hits_Pool, how="right", on=['HitID'])
    print(UF.TimeStamp(),_no_tracks, 'have been reconstructed in this cluster set ...')
else:
    print(UF.TimeStamp(),'No suitable hit pairs in the cluster set, just writing the empty one...')
    _Rec_Hits_Pool=pd.DataFrame([], columns = ['HitID','z','Segment_ID'])

print(UF.TimeStamp(),'Writing the output...')
After=datetime.datetime.now()
print('Final Time lapse', After-Before)
_Rec_Hits_Pool.to_csv(output_file_location,index=False)
print(UF.TimeStamp(),'Output is written to ',output_file_location)
exit()


