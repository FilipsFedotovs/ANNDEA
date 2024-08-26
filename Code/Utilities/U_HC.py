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
      def GenerateEdges(self, cut_dt, cut_dr, cut_dz, MCHits): #Decorate hit information
           #New workaround: instead of a painful Pandas outer join a loop over list is performed
           _Hits=self.ClusterHits
           _Hits= sorted(_Hits, key=lambda x: x[3], reverse=True) #Sorting by z
           _Tot_Hits=[]
           print('Initial number of all possible hit combinations is:',len(_Hits)**2)
           print('Number of all possible hit combinations without self-permutations:',(len(_Hits)**2)-len(_Hits))
           print('Number of all possible hit  combinations with enforced one-directionality:',int(((len(_Hits)**2)-len(_Hits))/2))


           for l in range(0,len(_Hits)-1):
               for r in range(l+1,len(_Hits)):
                   if HitCluster.JoinHits(_Hits[l],_Hits[r],cut_dt,cut_dr):
                          _Tot_Hits.append(_Hits[l]+_Hits[r])

           print('Number of all  hit combinations passing fiducial cuts:',len(_Tot_Hits))
           self.HitPairs=[]
           for TH in _Tot_Hits:
               self.HitPairs.append([TH[0],TH[3], TH[6],TH[9]])
           for TH in _Tot_Hits:
               for i in range(1,4):TH[i]=TH[i]/self.Step[2]
               for i in range(7,10):TH[i]=TH[i]/self.Step[2]
               TH.append('N/A')
               TH.append((math.sqrt(((TH[8]-TH[2])**2) + ((TH[7]-TH[1])**2) + ((TH[9]-TH[3])**2))))
               TH.append(math.sqrt(((TH[8]-TH[2])**2) + ((TH[7]-TH[1])**2)))
               TH.append(abs(TH[9]-TH[3]))
               TH.append(abs(TH[4]-TH[10]))
               TH.append(abs(TH[5]-TH[11]))
               del TH[1:6]
               del TH[2:7]
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

      def JoinHits(_H1, _H2, _cdt, _cdr, _cdz):
          if _H1[3]==_H2[3]: #Ensuring hit combinations are on different plates
              return False
          else:
              if abs(_H1[4]-_H2[4])>=_cdt:
                  return False
              else:
                  if abs(_H1[5]-_H2[5])>=_cdt:
                      return False
                  else:
                      if abs(_H2[1]-(_H1[1]+(_H1[4]*(_H2[3]-_H1[3]))))>=_cdr:
                         return False
                      else:
                          if abs(_H2[2]-(_H1[2]+(_H1[5]*(_H2[3]-_H1[3]))))>=_cdr:
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



