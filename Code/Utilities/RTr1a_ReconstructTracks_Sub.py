#Current version 1.1 - add change sys path capability

########################################    Import essential libriries    #############################################
import argparse
import sys
import copy
from statistics import mean
import os


#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--i',help="Set number", default='1')
parser.add_argument('--j',help="Subset number", default='1')
parser.add_argument('--Z_ID_Max',help="SubSubset number", default='1')
parser.add_argument('--TrackFitCutRes',help="Track Fit cut Residual", default=1000,type=int)
parser.add_argument('--TrackFitCutSTD',help="Track Fit cut", default=10,type=int)
parser.add_argument('--TrackFitCutMRes',help="Track Fit cut", default=200,type=int)
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
parser.add_argument('--ModelName',help="Name of the model to use?", default='0')
parser.add_argument('--BatchID',help="Give name to this train sample", default='')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
parser.add_argument('--CheckPoint',help="Save cluster sets during individual cluster tracking.", default='N')

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
import numpy as np
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
ModelName=args.ModelName
CheckPoint=args.CheckPoint.upper()=='Y'
RecBatchID=args.BatchID
TrackFitCutRes=args.TrackFitCutRes
TrackFitCutSTD=args.TrackFitCutSTD
TrackFitCutMRes=args.TrackFitCutMRes
p=args.p
o=args.o
sfx=args.sfx
pfx=args.pfx

import UtilityFunctions as UF #This is where we keep routine utility functions

#This function combines two segment object. Example: Segment 1 is [[a, _ ,b ,_ ,_ ][0.9,0.0,0.9,0.0,0.0]];  Segment 2 is [[a, _ ,c ,_ ,_ ][0.9,0.0,0.8,0.0,0.0]]; Segment 3 is [[_, d ,b ,_ ,_ ][0.0,0.8,0.8,0.0,0.0]]
#In order to combine segments we have to have at least one common hit and no clashes. Segment 1 and 2 have a common hit a, but their third plates clash. Segment 1 can be combined with segment 3 which yields: [[a, d ,b ,_ ,_ ][0.8,0.0,1.7,0.0,0.0]]
#Please note that if combination occurs then the hit weights combine together too
def InjectHit(Predator,Prey, Soft):
          if Soft==False:
             OverlapDetected=False
             _intersection=list(set(Predator[0]) & set(Prey[0])) #Do we have any common hits between two track segment skeletons?
             if len(_intersection)<2: #In order for hits to be combined there should always be at least two hits overlapping. Usually it will be '_' and a hit. It can be two hits overlap (Enforcing the existing connection)
                return(Predator,False)
             elif len(_intersection)==3: #This is the case when we have two overlapping hits (reinforcement) and an overlapping hole
                _intersection=[value for value in _intersection if value != "_"] #Remove the hole since we are not interested
                _index1 = Predator[0].index(_intersection[0]) #Find the index of the first overlapping hit
                _index2 = Predator[0].index(_intersection[1]) #Find the index of the second overlapping hit
                New_Predator=copy.deepcopy(Predator)
                New_Predator[1][_index1]+=Prey[1][_index1] #Add additional weights (Reinforcement of the hit connection)
                New_Predator[1][_index2]+=Prey[1][_index2] #Add additional weifght (Reinforcement of the hit connection)
                return(New_Predator,True) #Return the enriched track segment
             New_Predator=copy.deepcopy(Predator)
             _prey_trigger_count=0
             for el in range (len(Prey[0])): #If we have one overlapping hit and one overlapping hole
                 if Prey[0][el]!='_' and Predator[0][el]!='_': #Comparing hits not holes
                     if Prey[0][el]!=Predator[0][el]: #Reject if there is a clash
                        return(Predator,False)
                     elif Prey[0][el]==Predator[0][el]: #We found that there there is an overlapping hit
                        OverlapDetected=True
                        New_Predator[1][el]+=Prey[1][el] #Add new weight for the overlapping hit
                        _prey_trigger_count+=1
                 elif Predator[0][el]=='_' and Prey[0][el]!=Predator[0][el]: #If there is an overlap we fill the hole with a new hit from the subject pair
                     New_Predator[0][el]=Prey[0][el]
                     New_Predator[1][el]+=Prey[1][el] #Add new weigth for the new hit
                 if _prey_trigger_count>=2: #Max overlap can be 2 only, break the process to save time
                     break

             if OverlapDetected:
                return(New_Predator,True) #If overalp detected we return the updated track segment with incorporated hit
             else:
                return(Predator,False) #Return the same segment unchanged
          if Soft==True: #This is jsut a loop to check the overlap between two track segments (used later)
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

torch_import=True
cluster_output=[]
import datetime
Before=datetime.datetime.now()
z_clusters_results=[]
for k in range(0,Z_ID_Max):
    if CheckPoint:
        CheckPointFile=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(X_ID_n)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(X_ID_n)+'_'+str(Y_ID_n) +'_' +str(k)+'_CP'+sfx
        if os.path.isfile(CheckPointFile):
            continue
    Z_ID=int(k)/Z_overlap
    temp_data=data.drop(data.index[data['z'] >= ((Z_ID+1)*stepZ)])  #Keeping the relevant z slice
    temp_data=temp_data.drop(temp_data.index[temp_data['z'] < (Z_ID*stepZ)])  #Keeping the relevant z slice
    temp_data_list=temp_data.values.tolist()
    print(UF.TimeStamp(),'Creating the cluster', X_ID,Y_ID,Z_ID)
    HC=UF.HitCluster([X_ID,Y_ID,Z_ID],[stepX,stepY,stepZ]) #Initializing the cluster
    print(UF.TimeStamp(),'Decorating the clusters')
    HC.LoadClusterHits(temp_data_list) #Decorating the Clusters with Hit information
    if len(HC.RawClusterGraph)>1: #If we have at least 2 Hits in the cluster that can create
        print(UF.TimeStamp(),'Generating the edges...')
        print(UF.TimeStamp(),"Hit density of the Cluster",round(X_ID,1),round(Y_ID,1),round(Z_ID,1), "is  {} hits per cm\u00b3".format(round(len(HC.RawClusterGraph)/(0.6*0.6*1.2)),2))
        GraphStatus = HC.GenerateEdges(cut_dt, cut_dr)
        combined_weight_list=[]
        if GraphStatus:
            if HC.ClusterGraph.num_edges>0: #We only bring torch and GNN if we have some edges to classify
                        print(UF.TimeStamp(),'Classifying the edges...')
                        if args.ModelName!='blank':
                            if torch_import:
                                print(UF.TimeStamp(),'Preparing the model')
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
                                combined_weight_list.append(HC.edges[edge]+w[edge]) #Join the Hit Pair classification back to the hit pairs
                            combined_weight_list=pd.DataFrame(combined_weight_list, columns = ['l_HitID','r_HitID','link_strength'])
                            _Tot_Hits=pd.merge(HC.HitPairs, combined_weight_list, how="inner", on=['l_HitID','r_HitID'])
                            _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['link_strength'] <= Acceptance], inplace = True) #Remove all hit pairs that fail GNN classification
                        else:
                            _Tot_Hits=HC.HitPairs
                            _Tot_Hits['link_strength']=1.0

                        print(UF.TimeStamp(),'Number of all  hit combinations passing GNN selection:',len(_Tot_Hits))
                        _Tot_Hits=_Tot_Hits[['r_HitID','l_HitID','r_z','l_z','link_strength']]
                        print(UF.TimeStamp(),'Preparing the weighted hits for tracking...')
                        #Bellow just some data prep before the tracking procedure
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

                        #Bellow we build the track segment structure object that is made of two arrays:
                        #First array contains a list of hits arranged in z-order
                        #Second array contains the weights assighned to the hit combination

                        #Example if we have a cluster with 5 plates and a hit pair a and b at first and third plate and the weight is 0.9 then the object will look like [[a, _ ,b ,_ ,_ ][0.9,0.0,0.9,0.0,0.0]]. '_' represents a missing hit at a plate
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
                        KeepTracking=True
                        while len(_Tot_Hits)>0 and KeepTracking:
                                        _Tot_Hits_PCopy=copy.deepcopy(_Tot_Hits)
                                        _Tot_Hits_Predator=[]
                                        #Bellow we build all possible hit combinations that can occur in the data
                                        print(UF.TimeStamp(),'Building all possible track combinations...')
                                        for prd in range(len(_Tot_Hits_PCopy)):
                                            print(UF.TimeStamp(),'Progress is ',round(100*prd/len(_Tot_Hits_PCopy),2), '%',end="\r", flush=True)
                                            Predator=_Tot_Hits_PCopy[prd]

                                            for pry in range(len(_Tot_Hits_PCopy)):
                                                   #This function combines two segment object. Example: Segment 1 is [[a, _ ,b ,_ ,_ ][0.9,0.0,0.9,0.0,0.0]];  Segment 2 is [[a, _ ,c ,_ ,_ ][0.9,0.0,0.8,0.0,0.0]]; Segment 3 is [[_, d ,b ,_ ,_ ][0.0,0.8,0.8,0.0,0.0]]
                                                   #In order to combine segments we have to have at least one common hit and no clashes. Segment 1 and 2 have a common hit a, but their third plates clash. Segment 1 can be combined with segment 3 which yields: [[a, d ,b ,_ ,_ ][0.8,0.0,1.7,0.0,0.0]]
                                                   #Please note that if combination occurs then the hit weights combine together too
                                                   Result=InjectHit(Predator,_Tot_Hits_PCopy[pry],False)
                                                   Predator=Result[0]
                                            _Tot_Hits_Predator.append(Predator)
                                        #We calculate the average value of the segment weight
                                        for s in _Tot_Hits_Predator:
                                            s=s[0].append(mean(s.pop(1)))
                                        _Tot_Hits_Predator = [item for l in _Tot_Hits_Predator for item in l]
                                        for s in range(len(_Tot_Hits_Predator)):
                                            for h in range(len(_Tot_Hits_Predator[s])):
                                                if _Tot_Hits_Predator[s][h] =='_':
                                                    _Tot_Hits_Predator[s][h]='H_'+str(s) #Giving holes a unique name to avoid problems later

                                        column_no=len(_Tot_Hits_Predator[0])-1
                                        columns=[]
                                        print(UF.TimeStamp(),'Applying physical assumptions...')
                                        #Here we making sure that the tracks satisfy minimum fit requirements
                                        q_itr=0
                                        for thp in _Tot_Hits_Predator:
                                            print(UF.TimeStamp(),'Progress is ',round(100*q_itr/len(_Tot_Hits_Predator),2), '%',end="\r", flush=True)
                                            q_itr+=1
                                            fit_data_x=[]
                                            fit_data_y=[]
                                            fit_data_z=[]
                                            for cc in range(column_no):
                                                for td in temp_data_list:
                                                    if td[0][0]=='H':
                                                        break
                                                    elif td[0]==thp[cc]:
                                                        fit_data_x.append(td[1])
                                                        fit_data_y.append(td[2])
                                                        fit_data_z.append(td[3])
                                                        break

                                            line_x=np.polyfit(fit_data_z,fit_data_x,1)
                                            line_y=np.polyfit(fit_data_z,fit_data_y,1)
                                            x_residual=[x * line_x[0] for x in fit_data_z]
                                            x_residual=[x + line_x[1] for x in x_residual]
                                            x_residual=(np.array(x_residual)-np.array(fit_data_x))
                                            x_residual=[x ** 2 for x in x_residual]

                                            y_residual=[y * line_y[0] for y in fit_data_z]
                                            y_residual=[y + line_y[1] for y in y_residual]
                                            y_residual=(np.array(y_residual)-np.array(fit_data_y))
                                            y_residual=[y ** 2 for y in y_residual]
                                            residual=np.array(y_residual)+np.array(x_residual)
                                            residual=np.sqrt(residual)
                                            RES=sum(residual)/len(fit_data_x)
                                            STD=np.std(residual)
                                            MRES=max(residual)
                                            _Tot_Hits_Predator[_Tot_Hits_Predator.index(thp)]+=[RES,STD,MRES]


                                        #converting the list objects into Pandas dataframe
                                        for c in range(column_no):
                                            columns.append(str(c))
                                        columns+=['average_link_strength','RES','STD','MRES']
                                        _Tot_Hits_Predator=pd.DataFrame(_Tot_Hits_Predator, columns = columns)
                                        _Tot_Hits_Predator=_Tot_Hits_Predator.drop(_Tot_Hits_Predator.index[(_Tot_Hits_Predator['RES'] > TrackFitCutRes) | (_Tot_Hits_Predator['STD'] > TrackFitCutSTD) | (_Tot_Hits_Predator['MRES'] > TrackFitCutMRes)]) #Remove tracks with a bad fit
                                        KeepTracking=len(_Tot_Hits_Predator)>0
                                        _Tot_Hits_Predator.sort_values(by = ['average_link_strength'], ascending=[False],inplace=True) #Keep all the best hit combinations at the top
                                        _Tot_Hits_Predator=_Tot_Hits_Predator.drop(['average_link_strength','RES','STD','MRES'],axis=1) #We don't need the segment fit anymore
                                        for c in range(column_no):
                                            _Tot_Hits_Predator.drop_duplicates(subset=[str(c)], keep='first', inplace=True) #Iterating over hits, make sure that their belong to the best fit track
                                        _Tot_Hits_Predator=_Tot_Hits_Predator.values.tolist()
                                        for seg in range(len(_Tot_Hits_Predator)):
                                            _Tot_Hits_Predator[seg]=[s for s in _Tot_Hits_Predator[seg] if ('H' in s)==False] #Remove holes from the track representation
                                        _Rec_Hits_Pool+=_Tot_Hits_Predator
                                        for seg in _Tot_Hits_Predator:
                                            _itr=0
                                            while _itr<len(_Tot_Hits):
                                                if InjectHit(seg,_Tot_Hits[_itr],True): #We remove all the hitsthat become part of successful segments from the initial pool so we can rerun the process again with left over hits
                                                    del _Tot_Hits[_itr]
                                                else:
                                                    _itr+=1
                        #Transpose the rows
                        _track_list=[]
                        _segment_id=RecBatchID+'_'+str(X_ID)+'_'+str(Y_ID)+'_'+str(Z_ID) #Each segment name will have a relevant prefix (since numeration is only unique within an isolated cluster)
                        _no_tracks=len(_Rec_Hits_Pool)
                        for t in range(len(_Rec_Hits_Pool)):
                                      for h in _Rec_Hits_Pool[t]:
                                             _track_list.append([_segment_id+'-'+str(t+1),h])
                        _Rec_Hits_Pool=pd.DataFrame(_track_list, columns = ['Segment_ID','HitID'])
                        _Rec_Hits_Pool=pd.merge(_z_map, _Rec_Hits_Pool, how="right", on=['HitID'])
                        print(UF.TimeStamp(),_no_tracks, 'track segments have been reconstructed in this cluster set ...')
                        if CheckPoint:
                            _Rec_Hits_Pool.to_csv(CheckPointFile,index=False)
                        else:
                            z_clusters_results.append(_Rec_Hits_Pool) #Save all the reconstructed segments.
                        del HC
                        continue
import gc
gc.collect #Clean memory
print('Final Time lapse', datetime.datetime.now()-Before)


if CheckPoint:
    print(UF.TimeStamp(),'Loading all saved check points...')
    for k in range(0,Z_ID_Max):
        CheckPointFile=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(X_ID_n)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(X_ID_n)+'_'+str(Y_ID_n) +'_' +str(k)+'_CP'+sfx
        if os.path.isfile(CheckPointFile):
            ClusterData=pd.read_csv(CheckPointFile)
            z_clusters_results.append(ClusterData)

#Once we track all clusters we need to merge them along z-axis
if len(z_clusters_results)>0:
    print(UF.TimeStamp(),'Merging all clusters along z-axis...')
    ZContractedTable=z_clusters_results[0].rename(columns={"Segment_ID": "Master_Segment_ID","z": "Master_z" }) #First cluster is like a Pacman: it absorbes proceeding clusters and gets bigger
    for i in range(1,len(z_clusters_results)):
        SecondFileTable=z_clusters_results[i]
        FileClean=pd.merge(ZContractedTable,SecondFileTable,how='inner', on=['HitID']) #Join segments based on the common hits
        FileClean["Segment_No"]= FileClean["Segment_ID"]
        FileClean=FileClean.groupby(by=["Master_Segment_ID","Segment_ID"])["Segment_No"].count().reset_index() #If multiple segments share hit with the master segment we decide the best one by a level of the overlap
        FileClean=FileClean.sort_values(["Master_Segment_ID","Segment_No"],ascending=[1,0]) #Keep the best matching segment
        FileClean.drop_duplicates(subset=["Master_Segment_ID"],keep='first',inplace=True)
        FileClean=FileClean.drop(['Segment_No'],axis=1)
        FileClean=pd.merge(FileClean,SecondFileTable,how='right', on=['Segment_ID'])
        FileClean["Master_Segment_ID"] = FileClean["Master_Segment_ID"].fillna(FileClean["Segment_ID"]) #All segments that did not have overlapping hits with the master segment become masters themselves and become part of the Pacman
        FileClean=FileClean.rename(columns={"z": "Master_z" })
        FileClean=FileClean.drop(['Segment_ID'],axis=1)
        ZContractedTable=pd.concat([ZContractedTable,FileClean]) #Absorbing proceeding cluster
        ZContractedTable.drop_duplicates(subset=["Master_Segment_ID","HitID",'Master_z'],keep='first',inplace=True)
    ZContractedTable=ZContractedTable.sort_values(["Master_Segment_ID",'Master_z'],ascending=[1,1])
else: #If Cluster tracking yielded no segments we just create an empty array for consistency
     print(UF.TimeStamp(),'No suitable hit pairs in the cluster set, just writing the empty one...')
     ZContractedTable=pd.DataFrame([], columns = ['HitID','Master_z','Master_Segment_ID'])
output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(X_ID_n)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(X_ID_n)+'_'+str(Y_ID_n)+sfx
print(UF.TimeStamp(),'Writing the output...')
ZContractedTable.to_csv(output_file_location,index=False) #Write the final result
print(UF.TimeStamp(),'Output is written to ',output_file_location)
exit()
