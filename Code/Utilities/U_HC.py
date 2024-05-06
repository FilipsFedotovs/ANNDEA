###This file contains the utility functions that are commonly used in ANNDEA packages


import math

import numpy as np





class HitCluster:
      def __init__(self,ClusterID, Step):
          self.ClusterID=ClusterID
          self.Step=Step
      def __eq__(self, other):
        return ('-'.join(str(self.ClusterID))) == ('-'.join(str(other.ClusterID)))
      def __hash__(self):
        return hash(('-'.join(str(self.ClusterID))))
      def LoadClusterHits(self,RawHits): #Decorate hit information
           self.ClusterHits=[]
           self.ClusterHitIDs=[]
           __ClusterHitsTemp=[]
           for s in RawHits:
               if s[1]>=self.ClusterID[0]*self.Step[0] and s[1]<((self.ClusterID[0]+1)*self.Step[0]):
                   if s[2]>=self.ClusterID[1]*self.Step[1] and s[2]<((self.ClusterID[1]+1)*self.Step[1]):
                       if s[3]>=self.ClusterID[2]*self.Step[2] and s[3]<((self.ClusterID[2]+1)*self.Step[2]):
                          __ClusterHitsTemp.append([(s[1]-(self.ClusterID[0]*self.Step[0]))/self.Step[2],(s[2]-(self.ClusterID[1]*self.Step[1]))/self.Step[2], (s[3]-(self.ClusterID[2]*self.Step[2]))/self.Step[2],((s[4])+1)/2, ((s[5])+1)/2])
                          self.ClusterHitIDs.append(s[0])
                          self.ClusterHits.append(s)
           self.ClusterSize=len(__ClusterHitsTemp)
           self.RawClusterGraph=__ClusterHitsTemp #Avoiding importing torch without a good reason (reduce load on the HTCOndor initiative)
           del __ClusterHitsTemp
      def GenerateTrainData(self, MCHits,cut_dt, cut_dr): #Decorate hit information
           import pandas as pd
           _MCClusterHits=[]
           for s in MCHits:
               if s[1]>=self.ClusterID[0]*self.Step[0] and s[1]<((self.ClusterID[0]+1)*self.Step[0]):
                   if s[2]>=self.ClusterID[1]*self.Step[1] and s[2]<((self.ClusterID[1]+1)*self.Step[1]):
                       if s[3]>=self.ClusterID[2]*self.Step[2] and s[3]<((self.ClusterID[2]+1)*self.Step[2]):
                          _MCClusterHits.append([s[0],s[6]])
           #Preparing Raw and MC combined data 1
           _l_MCHits=pd.DataFrame(_MCClusterHits, columns = ['l_HitID','l_MC_ID'])
           _l_Hits=pd.DataFrame(self.ClusterHits, columns = ['l_HitID','l_x','l_y','l_z','l_tx','l_ty'])
           #Join hits + MC truth
           _l_Tot_Hits=pd.merge(_l_MCHits, _l_Hits, how="right", on=['l_HitID'])
           _l_Tot_Hits['join_key'] = 'join_key'

           #Preparing Raw and MC combined data 2
           _r_MCHits=pd.DataFrame(_MCClusterHits, columns = ['r_HitID','r_MC_ID'])
           _r_Hits=pd.DataFrame(self.ClusterHits, columns = ['r_HitID','r_x','r_y','r_z','r_tx','r_ty'])
           #Join hits + MC truth
           _r_Tot_Hits=pd.merge(_r_MCHits, _r_Hits, how="right", on=['r_HitID'])
           _r_Tot_Hits['join_key'] = 'join_key'

           #Combining data 1 and 2
           _Tot_Hits=pd.merge(_l_Tot_Hits, _r_Tot_Hits, how="inner", on=['join_key'])

           _Tot_Hits.l_MC_ID= _Tot_Hits.l_MC_ID.fillna(_Tot_Hits.l_HitID)
           _Tot_Hits.r_MC_ID= _Tot_Hits.r_MC_ID.fillna(_Tot_Hits.r_HitID)
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_HitID'] == _Tot_Hits['r_HitID']], inplace = True)
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_z'] <= _Tot_Hits['r_z']], inplace = True)
           _Tot_Hits['d_tx'] = _Tot_Hits['l_tx']-_Tot_Hits['r_tx']
           _Tot_Hits['d_tx'] = _Tot_Hits['d_tx'].abs()
           _Tot_Hits['d_ty'] = _Tot_Hits['l_ty']-_Tot_Hits['r_ty']
           _Tot_Hits['d_ty'] = _Tot_Hits['d_ty'].abs()
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_tx'] >= cut_dt], inplace = True)
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_ty'] >= cut_dt], inplace = True)
           _Tot_Hits['d_x'] = (_Tot_Hits['r_x']-(_Tot_Hits['l_x']+(_Tot_Hits['l_tx']*(_Tot_Hits['r_z']-_Tot_Hits['l_z']))))
           _Tot_Hits['d_x'] = _Tot_Hits['d_x'].abs()
           _Tot_Hits['d_y'] = (_Tot_Hits['r_y']-(_Tot_Hits['l_y']+(_Tot_Hits['l_ty']*(_Tot_Hits['r_z']-_Tot_Hits['l_z']))))
           _Tot_Hits['d_y'] = _Tot_Hits['d_y'].abs()
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_x'] >= cut_dr], inplace = True)
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_y'] >= cut_dr], inplace = True)

           #_Tot_Hits = _Tot_Hits.drop(['d_tx','d_ty','d_x','d_y','join_key','l_tx','l_ty','r_tx','r_ty'],axis=1)
           _Tot_Hits = _Tot_Hits.drop(['d_x','d_y','join_key','l_tx','l_ty','r_tx','r_ty'],axis=1)
           _Tot_Hits['l_x']=_Tot_Hits['l_x']/self.Step[2]
           _Tot_Hits['l_y']=_Tot_Hits['l_y']/self.Step[2]
           _Tot_Hits['l_z']=_Tot_Hits['l_z']/self.Step[2]
           _Tot_Hits['r_x']=_Tot_Hits['r_x']/self.Step[2]
           _Tot_Hits['r_y']=_Tot_Hits['r_y']/self.Step[2]
           _Tot_Hits['r_z']=_Tot_Hits['r_z']/self.Step[2]

           _Tot_Hits['label']=(_Tot_Hits['l_MC_ID']==_Tot_Hits['r_MC_ID']).astype('int8')
           _Tot_Hits_tr = _Tot_Hits[_Tot_Hits['r_MC_ID'].str.contains("--") & _Tot_Hits['l_MC_ID'].str.contains("--")]
           _Tot_Hits_fr = _Tot_Hits[(_Tot_Hits['r_MC_ID'].str.contains("--")==False) | (_Tot_Hits['l_MC_ID'].str.contains("--")==False)]
           _Tot_Hits_tr['label']=0
           _Tot_Hits=pd.concat([_Tot_Hits_tr,_Tot_Hits_fr])
           _Tot_Hits['d_l'] = (np.sqrt(((_Tot_Hits['r_y']-_Tot_Hits['l_y'])**2) + ((_Tot_Hits['r_x']-_Tot_Hits['l_x'])**2) + ((_Tot_Hits['r_z']-_Tot_Hits['l_z'])**2)))
           _Tot_Hits['d_t'] = np.sqrt(((_Tot_Hits['r_y']-_Tot_Hits['l_y'])**2) + ((_Tot_Hits['r_x']-_Tot_Hits['l_x'])**2))
           _Tot_Hits['d_z'] = (_Tot_Hits['r_z']-_Tot_Hits['l_z']).abs()

           _Tot_Hits = _Tot_Hits.drop(['r_x','r_y','r_z','l_x','l_y','l_z','l_MC_ID','r_MC_ID'],axis=1)
           _Tot_Hits=_Tot_Hits[['l_HitID','r_HitID','label','d_l','d_t','d_z','d_tx','d_ty']]
           _Tot_Hits=_Tot_Hits.values.tolist()
           import torch
           import torch_geometric
           from torch_geometric.data import Data
           self.ClusterGraph=Data(x=torch.Tensor(self.RawClusterGraph), edge_index=None, y=None)
           self.ClusterGraph.edge_index=torch.tensor((HitCluster.GenerateLinks(_Tot_Hits,self.ClusterHitIDs)))
           self.ClusterGraph.edge_attr=torch.tensor((HitCluster.GenerateEdgeAttributes(_Tot_Hits)))
           self.ClusterGraph.y=torch.tensor((HitCluster.GenerateEdgeLabels(_Tot_Hits)))
           if len(self.ClusterGraph.x)>0:
               return True
           else:
               return False
      def GenerateEdges(self, cut_dt, cut_dr): #Decorate hit information
           #New workaround: instead of a painful Pandas outer join a loop over list is perfromed
           _l_Hits=self.ClusterHits
           _r_Hits=self.ClusterHits
           #Combining data 1 and 2
           _Tot_Hits=[]
           _hit_count=0
           print('Initial number of all possible hit combinations is:',len(_l_Hits)**2)
           print('Number of all possible hit combinations without self-permutations:',(len(_l_Hits)**2)-len(_l_Hits))
           print('Number of all possible hit  combinations with enforced one-directionality:',int(((len(_l_Hits)**2)-len(_l_Hits))/2))
           for l in _l_Hits:
               _hit_count+=1
               print('Edge generation progress is ',round(100*_hit_count/len(_l_Hits),2), '%',end="\r", flush=True)
               for r in _r_Hits:
                  if HitCluster.JoinHits(l,r,cut_dt,cut_dr):
                      _Tot_Hits.append(l+r)
           print('Number of all  hit combinations passing fiducial cuts:',len(_Tot_Hits))
           import pandas as pd
           _Tot_Hits=pd.DataFrame(_Tot_Hits, columns = ['l_HitID','l_x','l_y','l_z','l_tx','l_ty','r_HitID','r_x','r_y','r_z','r_tx','r_ty'])
           self.HitPairs=_Tot_Hits[['l_HitID','l_z','r_HitID','r_z']]
           _Tot_Hits['l_x']=_Tot_Hits['l_x']/self.Step[2]
           _Tot_Hits['l_y']=_Tot_Hits['l_y']/self.Step[2]
           _Tot_Hits['l_z']=_Tot_Hits['l_z']/self.Step[2]
           _Tot_Hits['r_x']=_Tot_Hits['r_x']/self.Step[2]
           _Tot_Hits['r_y']=_Tot_Hits['r_y']/self.Step[2]
           _Tot_Hits['r_z']=_Tot_Hits['r_z']/self.Step[2]
           _Tot_Hits['label']='N/A'
           _Tot_Hits['d_l'] = (np.sqrt(((_Tot_Hits['r_y']-_Tot_Hits['l_y'])**2) + ((_Tot_Hits['r_x']-_Tot_Hits['l_x'])**2) + ((_Tot_Hits['r_z']-_Tot_Hits['l_z'])**2)))
           _Tot_Hits['d_t'] = np.sqrt(((_Tot_Hits['r_y']-_Tot_Hits['l_y'])**2) + ((_Tot_Hits['r_x']-_Tot_Hits['l_x'])**2))
           _Tot_Hits['d_z'] = (_Tot_Hits['r_z']-_Tot_Hits['l_z']).abs()
           _Tot_Hits['d_tx'] = _Tot_Hits['l_tx']-_Tot_Hits['r_tx']
           _Tot_Hits['d_tx'] = _Tot_Hits['d_tx'].abs()
           _Tot_Hits['d_ty'] = _Tot_Hits['l_ty']-_Tot_Hits['r_ty']
           _Tot_Hits['d_ty'] = _Tot_Hits['d_ty'].abs()
           _Tot_Hits = _Tot_Hits.drop(['r_x','r_y','r_z','l_x','l_y','l_z'],axis=1)
           _Tot_Hits=_Tot_Hits[['l_HitID','r_HitID','label','d_l','d_t','d_z','d_tx','d_ty']]

           _Tot_Hits=_Tot_Hits.values.tolist()
           if len(_Tot_Hits)>0:
               import torch
               from torch_geometric.data import Data
               self.ClusterGraph=Data(x=torch.Tensor(self.RawClusterGraph), edge_index=None, y=None)
               self.ClusterGraph.edge_index=torch.tensor((HitCluster.GenerateLinks(_Tot_Hits,self.ClusterHitIDs)))
               self.ClusterGraph.edge_attr=torch.tensor((HitCluster.GenerateEdgeAttributes(_Tot_Hits)))
               self.edges=[]
               for r in _Tot_Hits:
                   self.edges.append(r[:2])
               if len(self.ClusterGraph.edge_attr)>0:
                   return True
               else:
                   return False
           else:
               return False


      @staticmethod
      def GenerateLinks(_input,_ClusterID):
          _Top=[]
          _Bottom=[]
          for ip in _input:
              _Top.append(_ClusterID.index(ip[0]))
              _Bottom.append(_ClusterID.index(ip[1]))
          return [_Top,_Bottom]

      def JoinHits(_H1,_H2, _cdt, _cdr):
          if _H1[0]==_H2[0]:
              return False
          elif _H1[3]<=_H2[3]:
              return False
          else:
              _dtx=abs(_H1[4]-_H2[4])
              if _dtx>=_cdt:
                  return False
              else:
                  _dty=abs(_H1[5]-_H2[5])
                  if _dty>=_cdt:
                      return False
                  else:
                      _d_x = abs(_H2[1]-(_H1[1]+(_H1[4]*(_H2[3]-_H1[3]))))
                      if _d_x>=_cdr:
                         return False
                      else:
                          _d_y = abs(_H2[2]-(_H1[2]+(_H1[5]*(_H2[3]-_H1[3]))))
                          if _d_y>=_cdr:
                             return False
          return True
      def GenerateEdgeAttributes(_input):
          _EdgeAttr=[]
          for ip in _input:
              _EdgeAttr.append(ip[3:])
          return _EdgeAttr
      def GenerateEdgeLabels(_input):
          _EdgeLbl=[]
          for ip in _input:
              _EdgeLbl.append(ip[2])
          return _EdgeLbl
      def UnloadClusterGraph(self):
          del self.ClusterGraph
          del self.HitLinks

def GenerateModel(ModelMeta,TrainParams=None):
      if ModelMeta.ModelFramework=='PyTorch':
         import torch
         import torch.nn as nn

         from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid,Softmax
         import torch.nn.functional as F

         from torch import Tensor
         import torch_geometric
         from torch_geometric.nn import MessagePassing
         from torch_geometric.nn import global_mean_pool

         if ModelMeta.ModelArchitecture=='TCN':
            from MTr_IN import InteractionNetwork as IN
            class MLP(nn.Module):
                  def __init__(self, input_size, output_size, hidden_size):
                     super(MLP, self).__init__()

                     if ModelMeta.ModelParameters[0]==3:
                         self.layers = nn.Sequential(
                             nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, output_size),
                         )
                     elif ModelMeta.ModelParameters[0]==4:
                         self.layers = nn.Sequential(
                             nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, output_size),
                         )

                     elif ModelMeta.ModelParameters[0]==5:
                         self.layers = nn.Sequential(
                             nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, output_size),
                         )

                     elif ModelMeta.ModelParameters[0]==6:
                         self.layers = nn.Sequential(
                             nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, output_size),
                         )

                     elif ModelMeta.ModelParameters[0]==7:
                         self.layers = nn.Sequential(
                             nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, output_size),
                         )

                     elif ModelMeta.ModelParameters[0]==8:
                         self.layers = nn.Sequential(
                             nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, output_size),
                         )

                     elif ModelMeta.ModelParameters[0]==2:
                         self.layers = nn.Sequential(
                             nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, output_size),
                         )
                     elif ModelMeta.ModelParameters[0]==1:
                         self.layers = nn.Sequential(
                             nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, output_size),
                         )
                  def forward(self, C):
                     return self.layers(C)

            class TCN(nn.Module):
                 def __init__(self, node_indim, edge_indim, hs):
                     super(TCN, self).__init__()
                     if ModelMeta.ModelParameters[2]==2:
                         self.in_w1 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w2 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)

                     elif ModelMeta.ModelParameters[2]==1:
                         self.in_w1 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)

                     elif ModelMeta.ModelParameters[2]==3:
                         self.in_w1 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w2 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w3 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)

                     elif ModelMeta.ModelParameters[2]==4:
                         self.in_w1 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w2 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w3 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w4 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                     elif ModelMeta.ModelParameters[2]==5:
                         self.in_w1 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w2 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w3 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w4 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w5 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                     elif ModelMeta.ModelParameters[2]==6:
                         self.in_w1 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w2 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w3 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w4 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w5 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w6 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                     elif ModelMeta.ModelParameters[2]==7:
                         self.in_w1 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w2 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w3 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w4 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w5 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w6 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w7 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                     elif ModelMeta.ModelParameters[2]==8:
                         self.in_w1 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w2 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w3 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w4 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w5 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w6 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w7 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w8 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)

                     self.W = MLP(edge_indim*(ModelMeta.ModelParameters[2]+1), 1, ModelMeta.ModelParameters[1])

                 def forward(self, x: Tensor, edge_index: Tensor,
                             edge_attr: Tensor) -> Tensor:

                     # re-embed the graph twice with add aggregation
                     if ModelMeta.ModelParameters[2]==2:
                         x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)

                         x2, edge_attr_2 = self.in_w2(x1, edge_index, edge_attr_1)

                         # combine all edge features, use to predict edge weights
                         initial_edge_attr = torch.cat([edge_attr, edge_attr_1,
                                                        edge_attr_2], dim=1)
                     if ModelMeta.ModelParameters[2]==3:
                         x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)

                         x2, edge_attr_2 = self.in_w2(x1, edge_index, edge_attr_1)

                         x3, edge_attr_3 = self.in_w3(x2, edge_index, edge_attr_2)

                         # combine all edge features, use to predict edge weights
                         initial_edge_attr = torch.cat([edge_attr, edge_attr_1,
                                                        edge_attr_2, edge_attr_3], dim=1)
                     if ModelMeta.ModelParameters[2]==4:
                         x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)

                         x2, edge_attr_2 = self.in_w2(x1, edge_index, edge_attr_1)

                         x3, edge_attr_3 = self.in_w3(x2, edge_index, edge_attr_2)

                         x4, edge_attr_4 = self.in_w4(x3, edge_index, edge_attr_3)

                         # combine all edge features, use to predict edge weights
                         initial_edge_attr = torch.cat([edge_attr, edge_attr_1,
                                                        edge_attr_2, edge_attr_3, edge_attr_4], dim=1)

                     if ModelMeta.ModelParameters[2]==5:
                         x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)

                         x2, edge_attr_2 = self.in_w2(x1, edge_index, edge_attr_1)

                         x3, edge_attr_3 = self.in_w3(x2, edge_index, edge_attr_2)

                         x4, edge_attr_4 = self.in_w4(x3, edge_index, edge_attr_3)

                         x5, edge_attr_5 = self.in_w5(x4, edge_index, edge_attr_4)

                         # combine all edge features, use to predict edge weights
                         initial_edge_attr = torch.cat([edge_attr, edge_attr_1,
                                                        edge_attr_2, edge_attr_3, edge_attr_4,edge_attr_5], dim=1)

                     if ModelMeta.ModelParameters[2]==6:
                         x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)

                         x2, edge_attr_2 = self.in_w2(x1, edge_index, edge_attr_1)

                         x3, edge_attr_3 = self.in_w3(x2, edge_index, edge_attr_2)

                         x4, edge_attr_4 = self.in_w4(x3, edge_index, edge_attr_3)

                         x5, edge_attr_5 = self.in_w5(x4, edge_index, edge_attr_4)

                         x6, edge_attr_6 = self.in_w6(x5, edge_index, edge_attr_5)

                         # combine all edge features, use to predict edge weights
                         initial_edge_attr = torch.cat([edge_attr, edge_attr_1,
                                                        edge_attr_2, edge_attr_3, edge_attr_4,edge_attr_5,edge_attr_6], dim=1)
                     if ModelMeta.ModelParameters[2]==7:
                         x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)

                         x2, edge_attr_2 = self.in_w2(x1, edge_index, edge_attr_1)

                         x3, edge_attr_3 = self.in_w3(x2, edge_index, edge_attr_2)

                         x4, edge_attr_4 = self.in_w4(x3, edge_index, edge_attr_3)

                         x5, edge_attr_5 = self.in_w5(x4, edge_index, edge_attr_4)

                         x6, edge_attr_6 = self.in_w6(x5, edge_index, edge_attr_5)

                         x7, edge_attr_7 = self.in_w7(x6, edge_index, edge_attr_6)

                         # combine all edge features, use to predict edge weights
                         initial_edge_attr = torch.cat([edge_attr, edge_attr_1,
                                                        edge_attr_2, edge_attr_3, edge_attr_4,edge_attr_5,edge_attr_6,edge_attr_7], dim=1)
                     if ModelMeta.ModelParameters[2]==8:
                         x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)

                         x2, edge_attr_2 = self.in_w2(x1, edge_index, edge_attr_1)

                         x3, edge_attr_3 = self.in_w3(x2, edge_index, edge_attr_2)

                         x4, edge_attr_4 = self.in_w4(x3, edge_index, edge_attr_3)

                         x5, edge_attr_5 = self.in_w5(x4, edge_index, edge_attr_4)

                         x6, edge_attr_6 = self.in_w6(x5, edge_index, edge_attr_5)

                         x7, edge_attr_7 = self.in_w7(x6, edge_index, edge_attr_6)

                         x8, edge_attr_8 = self.in_w8(x7, edge_index, edge_attr_7)

                         # combine all edge features, use to predict edge weights
                         initial_edge_attr = torch.cat([edge_attr, edge_attr_1,
                                                        edge_attr_2, edge_attr_3, edge_attr_4,edge_attr_5,edge_attr_6,edge_attr_7,edge_attr_8], dim=1)

                     if ModelMeta.ModelParameters[2]==1:
                         x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)
                         # combine all edge features, use to predict edge weights
                         initial_edge_attr = torch.cat([edge_attr, edge_attr_1], dim=1)
                     edge_weights = torch.sigmoid(self.W(initial_edge_attr))
                     return edge_weights
            model = TCN(ModelMeta.num_node_features, ModelMeta.num_edge_features, ModelMeta.ModelParameters[3])
            return model
         elif ModelMeta.ModelArchitecture=='GCN-4N-IC':
            from torch_geometric.nn import GCNConv
            HiddenLayer=[]
            OutputLayer=[]
            for el in ModelMeta.ModelParameters:
              if ModelMeta.ModelParameters.index(el)<=4 and len(el)>0:
                 HiddenLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)==10:
                 OutputLayer=el

            class GCN(torch.nn.Module):
                def __init__(self):
                    super(GCN, self).__init__()
                    torch.manual_seed(12345)
                    if len(HiddenLayer)==3:
                        self.conv1 = GCNConv(4 , HiddenLayer[0][0])
                        self.conv2 = GCNConv(HiddenLayer[0][0],HiddenLayer[1][0])
                        self.conv3 = GCNConv(HiddenLayer[1][0],HiddenLayer[2][0])
                        self.lin = Linear(HiddenLayer[2][0],OutputLayer[1])
                    self.softmax = Softmax(dim=-1)

                def forward(self, x, edge_index, edge_attr, batch):
                    # 1. Obtain node embeddings
                    if len(HiddenLayer)==3:

                        x = self.conv1(x, edge_index)
                        x = x.relu()
                        x = self.conv2(x, edge_index)
                        x = x.relu()
                        x = self.conv3(x, edge_index)

                    # 2. Readout layer
                    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

                    # 3. Apply a final classifier
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = self.lin(x)
                    x = self.softmax(x)
                    return x
            return GCN()
         elif ModelMeta.ModelArchitecture=='GCN-6N-IC':
            from torch_geometric.nn import GCNConv
            HiddenLayer=[]
            OutputLayer=[]
            for el in ModelMeta.ModelParameters:
              if ModelMeta.ModelParameters.index(el)<=4 and len(el)>0:
                 HiddenLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)==10:
                 OutputLayer=el

            class GCN(torch.nn.Module):
                def __init__(self):
                    super(GCN, self).__init__()
                    torch.manual_seed(12345)
                    if len(HiddenLayer)==3:
                        self.conv1 = GCNConv(6 , HiddenLayer[0][0])
                        self.conv2 = GCNConv(HiddenLayer[0][0],HiddenLayer[1][0])
                        self.conv3 = GCNConv(HiddenLayer[1][0],HiddenLayer[2][0])
                        self.lin = Linear(HiddenLayer[2][0],OutputLayer[1])
                    self.softmax = Softmax(dim=-1)

                def forward(self, x, edge_index, edge_attr, batch):
                    # 1. Obtain node embeddings
                    if len(HiddenLayer)==3:

                        x = self.conv1(x, edge_index)
                        x = x.relu()
                        x = self.conv2(x, edge_index)
                        x = x.relu()
                        x = self.conv3(x, edge_index)

                    # 2. Readout layer
                    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

                    # 3. Apply a final classifier
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = self.lin(x)
                    x = self.softmax(x)
                    return x
            return GCN()
         elif ModelMeta.ModelArchitecture=='GCN-5N-FC':
            from torch_geometric.nn import GCNConv
            HiddenLayer=[]
            OutputLayer=[]
            for el in ModelMeta.ModelParameters:
              if ModelMeta.ModelParameters.index(el)<=4 and len(el)>0:
                 HiddenLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)==10:
                 OutputLayer=el

            class GCN(torch.nn.Module):
                def __init__(self):
                    super(GCN, self).__init__()
                    torch.manual_seed(12345)
                    if len(HiddenLayer)==3:
                        self.conv1 = GCNConv(5 , HiddenLayer[0][0])
                        self.conv2 = GCNConv(HiddenLayer[0][0],HiddenLayer[1][0])
                        self.conv3 = GCNConv(HiddenLayer[1][0],HiddenLayer[2][0])
                        self.lin = Linear(HiddenLayer[2][0],OutputLayer[1])
                    elif len(HiddenLayer)==4:
                        self.conv1 = GCNConv(5 , HiddenLayer[0][0])
                        self.conv2 = GCNConv(HiddenLayer[0][0],HiddenLayer[1][0])
                        self.conv3 = GCNConv(HiddenLayer[1][0],HiddenLayer[2][0])
                        self.conv4 = GCNConv(HiddenLayer[2][0],HiddenLayer[3][0])
                        self.lin = Linear(HiddenLayer[3][0],OutputLayer[1])
                    self.softmax = Softmax(dim=-1)

                def forward(self, x, edge_index, edge_attr, batch):
                    # 1. Obtain node embeddings
                    if len(HiddenLayer)==3:
                        x = self.conv1(x, edge_index)
                        x = x.relu()
                        x = self.conv2(x, edge_index)
                        x = x.relu()
                        x = self.conv3(x, edge_index)
                    elif len(HiddenLayer)==4:
                        x = self.conv1(x, edge_index)
                        x = x.relu()
                        x = self.conv2(x, edge_index)
                        x = x.relu()
                        x = self.conv3(x, edge_index)
                        x = x.relu()
                        x = self.conv4(x, edge_index)

                    # 2. Readout layer
                    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

                    # 3. Apply a final classifier
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = self.lin(x)
                    x = self.softmax(x)
                    return x
            return GCN()
         elif ModelMeta.ModelArchitecture=='TAG-4N-IC':
            from torch_geometric.nn import TAGConv
            HiddenLayer=[]
            OutputLayer=[]
            for el in ModelMeta.ModelParameters:
              if ModelMeta.ModelParameters.index(el)<=4 and len(el)>0:
                 HiddenLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)==10:
                 OutputLayer=el

            class TAG(torch.nn.Module):
                def __init__(self):
                    super(TAG, self).__init__()
                    torch.manual_seed(12345)
                    if len(HiddenLayer)==3:
                        self.conv1 = TAGConv(4 , HiddenLayer[0][0])
                        self.conv2 = TAGConv(HiddenLayer[0][0],HiddenLayer[1][0])
                        self.conv3 = TAGConv(HiddenLayer[1][0],HiddenLayer[2][0])
                        self.lin = Linear(HiddenLayer[2][0],OutputLayer[1])
                    elif len(HiddenLayer)==4:
                        self.conv1 = TAGConv(4 , HiddenLayer[0][0])
                        self.conv2 = TAGConv(HiddenLayer[0][0],HiddenLayer[1][0])
                        self.conv3 = TAGConv(HiddenLayer[1][0],HiddenLayer[2][0])
                        self.conv4 = TAGConv(HiddenLayer[2][0],HiddenLayer[3][0])
                        self.lin = Linear(HiddenLayer[3][0],OutputLayer[1])
                    self.softmax = Softmax(dim=-1)

                def forward(self, x, edge_index, edge_attr, batch):
                    # 1. Obtain node embeddings
                    if len(HiddenLayer)==3:

                        x = self.conv1(x, edge_index)
                        x = x.relu()
                        x = self.conv2(x, edge_index)
                        x = x.relu()
                        x = self.conv3(x, edge_index)
                    elif len(HiddenLayer)==4:

                        x = self.conv1(x, edge_index)
                        x = x.relu()
                        x = self.conv2(x, edge_index)
                        x = x.relu()
                        x = self.conv3(x, edge_index)
                        x = x.relu()
                        x = self.conv4(x, edge_index)
                    # 2. Readout layer
                    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

                    # 3. Apply a final classifier
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = self.lin(x)
                    x = self.softmax(x)
                    return x
            return TAG()
         elif ModelMeta.ModelArchitecture=='TAG-5N-FC':
            from torch_geometric.nn import TAGConv
            HiddenLayer=[]
            OutputLayer=[]
            for el in ModelMeta.ModelParameters:
              if ModelMeta.ModelParameters.index(el)<=4 and len(el)>0:
                 HiddenLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)==10:
                 OutputLayer=el

            class TAG(torch.nn.Module):
                def __init__(self):
                    super(TAG, self).__init__()
                    torch.manual_seed(12345)
                    if len(HiddenLayer)==3:
                        self.conv1 = TAGConv(5 , HiddenLayer[0][0])
                        self.conv2 = TAGConv(HiddenLayer[0][0],HiddenLayer[1][0])
                        self.conv3 = TAGConv(HiddenLayer[1][0],HiddenLayer[2][0])
                        self.lin = Linear(HiddenLayer[2][0],OutputLayer[1])
                    elif len(HiddenLayer)==4:
                        self.conv1 = TAGConv(5 , HiddenLayer[0][0])
                        self.conv2 = TAGConv(HiddenLayer[0][0],HiddenLayer[1][0])
                        self.conv3 = TAGConv(HiddenLayer[1][0],HiddenLayer[2][0])
                        self.conv4 = TAGConv(HiddenLayer[2][0],HiddenLayer[3][0])
                        self.lin = Linear(HiddenLayer[3][0],OutputLayer[1])
                    self.softmax = Softmax(dim=-1)

                def forward(self, x, edge_index, edge_attr, batch):
                    # 1. Obtain node embeddings
                    if len(HiddenLayer)==3:
                        x = self.conv1(x, edge_index)
                        x = x.relu()
                        x = self.conv2(x, edge_index)
                        x = x.relu()
                        x = self.conv3(x, edge_index)
                    elif len(HiddenLayer)==4:
                        x = self.conv1(x, edge_index)
                        x = x.relu()
                        x = self.conv2(x, edge_index)
                        x = x.relu()
                        x = self.conv3(x, edge_index)
                        x = x.relu()
                        x = self.conv4(x, edge_index)

                    # 2. Readout layer
                    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

                    # 3. Apply a final classifier
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = self.lin(x)
                    x = self.softmax(x)
                    return x
            return TAG()
         elif ModelMeta.ModelArchitecture=='GMM-5N-FC':
            from torch_geometric.nn import GMMConv
            HiddenLayer=[]
            OutputLayer=[]
            for el in ModelMeta.ModelParameters:
              if ModelMeta.ModelParameters.index(el)<=4 and len(el)>0:
                 HiddenLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)==10:
                 OutputLayer=el

            class GMM(torch.nn.Module):
                def __init__(self):
                    super(GMM, self).__init__()
                    torch.manual_seed(12345)
                    if len(HiddenLayer)==3:
                        self.conv1 = GMMConv(5 , HiddenLayer[0][0],dim=3,kernel_size=HiddenLayer[0][1])
                        self.conv2 = GMMConv(HiddenLayer[0][0],HiddenLayer[1][0],dim=3,kernel_size=HiddenLayer[1][1])
                        self.conv3 = GMMConv(HiddenLayer[1][0],HiddenLayer[2][0],dim=3,kernel_size=HiddenLayer[2][1])
                        self.lin = Linear(HiddenLayer[2][0],OutputLayer[1])
                    elif len(HiddenLayer)==4:
                        self.conv1 = GMMConv(5 , HiddenLayer[0][0],dim=3,kernel_size=HiddenLayer[0][1])
                        self.conv2 = GMMConv(HiddenLayer[0][0],HiddenLayer[1][0],dim=3,kernel_size=HiddenLayer[1][1])
                        self.conv3 = GMMConv(HiddenLayer[1][0],HiddenLayer[2][0],dim=3,kernel_size=HiddenLayer[2][1])
                        self.conv4 = GMMConv(HiddenLayer[2][0],HiddenLayer[3][0],dim=3,kernel_size=HiddenLayer[3][1])
                        self.lin = Linear(HiddenLayer[3][0],OutputLayer[1])
                    self.softmax = Softmax(dim=-1)

                def forward(self, x, edge_index, edge_attr, batch):
                    # 1. Obtain node embeddings
                    if len(HiddenLayer)==3:
                        x = self.conv1(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv2(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv3(x, edge_index,edge_attr)
                    elif len(HiddenLayer)==4:
                        x = self.conv1(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv2(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv3(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv4(x, edge_index,edge_attr)

                    # 2. Readout layer
                    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
                    # 3. Apply a final classifier
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = self.lin(x)
                    x = self.softmax(x)
                    return x
            return GMM()
         elif ModelMeta.ModelArchitecture=='GMM-4N-IC':
            from torch_geometric.nn import GMMConv
            HiddenLayer=[]
            OutputLayer=[]
            for el in ModelMeta.ModelParameters:
              if ModelMeta.ModelParameters.index(el)<=4 and len(el)>0:
                 HiddenLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)==10:
                 OutputLayer=el

            class GMM(torch.nn.Module):
                def __init__(self):
                    super(GMM, self).__init__()
                    torch.manual_seed(12345)
                    if len(HiddenLayer)==3:
                        self.conv1 = GMMConv(4 , HiddenLayer[0][0],dim=4,kernel_size=HiddenLayer[0][1])
                        self.conv2 = GMMConv(HiddenLayer[0][0],HiddenLayer[1][0],dim=4,kernel_size=HiddenLayer[1][1])
                        self.conv3 = GMMConv(HiddenLayer[1][0],HiddenLayer[2][0],dim=4,kernel_size=HiddenLayer[2][1])
                        self.lin = Linear(HiddenLayer[2][0],OutputLayer[1])
                    elif len(HiddenLayer)==4:
                        self.conv1 = GMMConv(4 , HiddenLayer[0][0],dim=4,kernel_size=HiddenLayer[0][1])
                        self.conv2 = GMMConv(HiddenLayer[0][0],HiddenLayer[1][0],dim=4,kernel_size=HiddenLayer[1][1])
                        self.conv3 = GMMConv(HiddenLayer[1][0],HiddenLayer[2][0],dim=4,kernel_size=HiddenLayer[2][1])
                        self.conv4 = GMMConv(HiddenLayer[2][0],HiddenLayer[3][0],dim=4,kernel_size=HiddenLayer[3][1])
                        self.lin = Linear(HiddenLayer[3][0],OutputLayer[1])
                    self.softmax = Softmax(dim=-1)

                def forward(self, x, edge_index, edge_attr, batch):
                    # 1. Obtain node embeddings
                    if len(HiddenLayer)==3:
                        x = self.conv1(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv2(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv3(x, edge_index,edge_attr)
                    elif len(HiddenLayer)==4:
                        x = self.conv1(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv2(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv3(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv4(x, edge_index,edge_attr)

                    # 2. Readout layer
                    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
                    # 3. Apply a final classifier
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = self.lin(x)
                    x = self.softmax(x)
                    return x
            return GMM()
         elif ModelMeta.ModelArchitecture=='GMM-6N-IC':
            from torch_geometric.nn import GMMConv
            HiddenLayer=[]
            OutputLayer=[]
            for el in ModelMeta.ModelParameters:
              if ModelMeta.ModelParameters.index(el)<=4 and len(el)>0:
                 HiddenLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)==10:
                 OutputLayer=el

            class GMM(torch.nn.Module):
                def __init__(self):
                    super(GMM, self).__init__()
                    torch.manual_seed(12345)
                    if len(HiddenLayer)==3:
                        self.conv1 = GMMConv(6 , HiddenLayer[0][0],dim=4,kernel_size=HiddenLayer[0][1])
                        self.conv2 = GMMConv(HiddenLayer[0][0],HiddenLayer[1][0],dim=4,kernel_size=HiddenLayer[1][1])
                        self.conv3 = GMMConv(HiddenLayer[1][0],HiddenLayer[2][0],dim=4,kernel_size=HiddenLayer[2][1])
                        self.lin = Linear(HiddenLayer[2][0],OutputLayer[1])
                    elif len(HiddenLayer)==1:
                        self.conv1 = GMMConv(6 , HiddenLayer[0][0],dim=4,kernel_size=HiddenLayer[0][1])
                        self.lin = Linear(HiddenLayer[0][0],OutputLayer[1])
                    elif len(HiddenLayer)==4:
                        self.conv1 = GMMConv(6 , HiddenLayer[0][0],dim=4,kernel_size=HiddenLayer[0][1])
                        self.conv2 = GMMConv(HiddenLayer[0][0],HiddenLayer[1][0],dim=4,kernel_size=HiddenLayer[1][1])
                        self.conv3 = GMMConv(HiddenLayer[1][0],HiddenLayer[2][0],dim=4,kernel_size=HiddenLayer[2][1])
                        self.conv4 = GMMConv(HiddenLayer[2][0],HiddenLayer[3][0],dim=4,kernel_size=HiddenLayer[3][1])
                        self.lin = Linear(HiddenLayer[3][0],OutputLayer[1])
                    self.softmax = Softmax(dim=-1)

                def forward(self, x, edge_index, edge_attr, batch):
                    # 1. Obtain node embeddings
                    if len(HiddenLayer)==3:
                        x = self.conv1(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv2(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv3(x, edge_index,edge_attr)
                    elif len(HiddenLayer)==1:
                        x = self.conv1(x, edge_index,edge_attr)
                        x = x.relu()
                    elif len(HiddenLayer)==4:
                        x = self.conv1(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv2(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv3(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv4(x, edge_index,edge_attr)

                    # 2. Readout layer
                    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
                    # 3. Apply a final classifier
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = self.lin(x)
                    x = self.softmax(x)
                    return x
            return GMM()
      elif ModelMeta.ModelFramework=='Tensorflow':
          if ModelMeta.ModelType=='CNN':
            act_fun_list=['N/A','linear','exponential','elu','relu', 'selu','sigmoid','softmax','softplus','softsign','tanh']
            import tensorflow as tf
            from tensorflow import keras
            from keras.models import Sequential
            from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
            from keras.optimizers import Adam
            from keras import callbacks
            from keras import backend as K
            HiddenLayer=[]
            FullyConnectedLayer=[]
            OutputLayer=[]
            ImageLayer=[]
            for el in ModelMeta.ModelParameters:
              if len(el)==7:
                 HiddenLayer.append(el)
              elif len(el)==3:
                 FullyConnectedLayer.append(el)
              elif len(el)==2:
                 OutputLayer=el
              elif len(el)==4:
                 ImageLayer=el
            H=int(round(ImageLayer[0]/ImageLayer[3],0))*2
            W=int(round(ImageLayer[1]/ImageLayer[3],0))*2
            L=int(round(ImageLayer[2]/ImageLayer[3],0))
            model = Sequential()
            for HL in HiddenLayer:
                     Nodes=HL[0]*16
                     KS=HL[1]
                     PS=HL[3]
                     DR=float(HL[6]-1)/10.0
                     if HiddenLayer.index(HL)==0:
                        model.add(Conv3D(Nodes, activation=act_fun_list[HL[2]],kernel_size=(KS[0],KS[1],KS[2]),kernel_initializer='he_uniform', input_shape=(H,W,L,1)))
                     else:
                        model.add(Conv3D(Nodes, activation=act_fun_list[HL[2]],kernel_size=(KS[0],KS[1],KS[2]),kernel_initializer='he_uniform'))
                     if PS[0]>1 and PS[1]>1 and PS[2]>1:
                        model.add(MaxPooling3D(pool_size=(PS[0], PS[1], PS[2])))
                     model.add(BatchNormalization(center=HL[4]>1, scale=HL[5]>1))
                     model.add(Dropout(DR))
            model.add(Flatten())
            for FC in FullyConnectedLayer:
                         Nodes=4**FC[0]
                         DR=float(FC[2]-1)/10.0
                         model.add(Dense(Nodes, activation=act_fun_list[FC[1]], kernel_initializer='he_uniform'))
                         model.add(Dropout(DR))
            model.add(Dense(OutputLayer[1], activation=act_fun_list[OutputLayer[0]]))
            opt = Adam(learning_rate=TrainParams[0])
     # Compile the model
            model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
            return model

def ErrorOperations(a,b,a_e,b_e,mode):
    if mode=='+' or mode == '-':
        c_e=math.sqrt((a_e**2) + (b_e**2))
        return(c_e)
    if mode=='*':
        c_e=a*b*math.sqrt(((a_e/a)**2) + ((b_e/b)**2))
        return(c_e)
    if mode=='/':
        c_e=(a/b)*math.sqrt(((a_e/a)**2) + ((b_e/b)**2))
        return(c_e)



