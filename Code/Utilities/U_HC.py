###This file contains the utility functions that are commonly used in ANNDEA packages


import math






class HitCluster:
      def __init__(self,ClusterID, Step):
          self.ClusterID=ClusterID
          self.Step=Step
      def __eq__(self, other):
        return ('-'.join(str(self.ClusterID))) == ('-'.join(str(other.ClusterID)))
      def __hash__(self):
        return hash(('-'.join(str(self.ClusterID))))
      def LoadClusterHits(self,RawHits): #Decorate hit information
           self.Hits=[]
           self.HitIDs=[]
           __ClusterHitsTemp=[]
           _scale_factor=8
           for s in RawHits:
               if s[1]>=self.ClusterID[0]*self.Step[0] and s[1]<((self.ClusterID[0]+1)*self.Step[0]):
                   if s[2]>=self.ClusterID[1]*self.Step[1] and s[2]<((self.ClusterID[1]+1)*self.Step[1]):
                          if s[3]>=self.ClusterID[2]*self.Step[2] and s[3]<((self.ClusterID[2]+1)*self.Step[2]):
                              __ClusterHitsTemp.append([(s[1]-(self.ClusterID[0]*self.Step[0]))/self.Step[2],(s[2]-(self.ClusterID[1]*self.Step[1]))/self.Step[2], (s[3]-(self.ClusterID[2]*self.Step[2]))/self.Step[2],((s[4])+1)/2, ((s[5])+1)/2])
                              self.HitIDs.append(s[0])
                              self.Hits.append(s)
           self.ClusterSize=len(__ClusterHitsTemp)
           self.RawNodes=__ClusterHitsTemp #Avoiding importing torch without a good reason (reduce load on the HTCOndor initiative)
           del __ClusterHitsTemp


      def GenerateSeeds(self, cut_dt, cut_dr, cut_dz, l, MaxEdges, SeedFlowLog, EOS_DIR, ModelName=None): #Decorate hit information
           #New workaround: instead of a painful Pandas outer join a loop over list is performed
           _Hits=self.Hits
           _Hits= sorted(_Hits, key=lambda x: x[3], reverse=True) #Sorting by z
           self.Seeds=[] #Initiate the empty container for seeds
           _sp,_ep=HitCluster.SplitJob(l,MaxEdges,self.ClusterSize)

           if ModelName!=None:
               EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
               EOSsubModelDIR=EOSsubDIR+'/'+'Models'
               Model_Meta_Path=EOSsubModelDIR+'/'+ModelName+'_Meta'
               Model_Path=EOSsubModelDIR+'/'+ModelName
               import U_UI as UI
               import U_ML as ML
               ModelMeta=UI.PickleOperations(Model_Meta_Path, 'r', 'N/A')[0]
               import torch
               device = torch.device('cpu')
               model = ML.GenerateModel(ModelMeta).to(device)
               model.load_state_dict(torch.load(Model_Path))
               model_acceptance=ModelMeta.TrainSessionsData[-1][-1][3]


           self.SeedFlowLabels=['All','Excluding self-permutations', 'Excluding duplicates','Excluding seeds on the same plate', 'Cut on dz', 'Cut on dtx', 'Cut on dty' , 'Cut on drx', 'Cut on dry', 'MLP filter', 'GNN filter', 'Tracking process' ]
           self.SeedFlowValuesAll=[len(_Hits)**2,(len(_Hits)**2)-len(_Hits), int(((len(_Hits)**2)-len(_Hits))/2), 0, 0, 0, 0, 0, 0, 0, 0, 0]
           self.SeedFlowValuesTrue=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

           if SeedFlowLog:
               for l in range(_sp,min(_ep,self.ClusterSize-1)):
                    for r in range(l+1,len(_Hits)):
                        FitSeed=HitCluster.FitTrackSeed(_Hits[l],_Hits[r],cut_dt,cut_dr,cut_dz)
                        self.SeedFlowValuesAll = [a + b for a, b in zip(self.SeedFlowValuesAll, FitSeed[1])]
                        self.SeedFlowValuesTrue = [a + b for a, b in zip(self.SeedFlowValuesTrue, FitSeed[2])]
                        if FitSeed[0]:
                               if ModelName==None:
                                    self.Seeds.append(HitCluster.NormaliseSeed(self,_Hits[r], _Hits[l], cut_dt))
                               else:
                                    _refined_seed=HitCluster.NormaliseSeed(self,_Hits[r], _Hits[l], cut_dt)
                                    _refined_seed_vector=self.GenerateSeedVectors([_refined_seed])
                                    y=_refined_seed_vector[1][0]
                                    _refined_seed_vector_tensor=torch.tensor(_refined_seed_vector[0], dtype=torch.float32)

                                    model.eval()
                                    with torch.no_grad():
                                        x = _refined_seed_vector_tensor[0].unsqueeze(0)
                                        o = model(x)
                                    if o.item()>model_acceptance:
                                        self.SeedFlowValuesTrue = [a + b for a, b in zip(self.SeedFlowValuesTrue, [0, 0, 0, 0, 0, 0, 0, 0, 0, y, 0, 0])]
                                        self.SeedFlowValuesAll = [a + b for a, b in zip(self.SeedFlowValuesAll, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])]
                                        self.Seeds.append(HitCluster.NormaliseSeed(self,_Hits[r], _Hits[l], cut_dt))
                                    else:
                                        continue


           else:
               for l in range(_sp,min(_ep,self.ClusterSize-1)):
                    for r in range(l+1,len(_Hits)):
                        FitSeed=HitCluster.FitSeed(_Hits[l],_Hits[r],cut_dt,cut_dr,cut_dz)
                        if FitSeed:
                            if ModelName==None:
                                    self.Seeds.append(HitCluster.NormaliseSeed(self,_Hits[r], _Hits[l], cut_dt))
                            else:
                                    _refined_seed=HitCluster.NormaliseSeed(self,_Hits[r], _Hits[l], cut_dt)
                                    _refined_seed_vector=self.GenerateSeedVectors([_refined_seed])
                                    _refined_seed_vector_tensor=torch.tensor(_refined_seed_vector[0], dtype=torch.float32)
                                    model.eval()
                                    with torch.no_grad():
                                        x = _refined_seed_vector_tensor[0].unsqueeze(0)
                                        o = model(x)
                                    if o.item()>model_acceptance:
                                        self.Seeds.append(HitCluster.NormaliseSeed(self,_Hits[r], _Hits[l], cut_dt))
                                    else:
                                        continue
           return True

      def GenerateSeedGraph(self): #Decorate hit information
           import torch
           from torch_geometric.data import Data
           self.ClusterGraph=Data(x=torch.Tensor(self.RawNodes), edge_index=None, y=None)
           self.ClusterGraph.edge_index=torch.tensor((HitCluster.GenerateEdgeLinks(self.Seeds,self.HitIDs)))
           self.ClusterGraph.edge_attr=torch.tensor((HitCluster.GenerateEdgeAttributes(self.Seeds)))
           print(self.ClusterGraph.edge_attr)
           exit()
           # if len(MCHits)>0:
           #  self.ClusterGraph.y=torch.tensor((HitCluster.GenerateEdgeLabels(self.RawEdgeGraph)))
           # self.edges=[]
           # for r in self.RawEdgeGraph:
           #     self.edges.append(r[:2])
           # if len(self.ClusterGraph.edge_attr)>0:
           #     return True
           # else:
           #     return False



      @staticmethod

      def SplitJob(_l,_MaxEdges, _n_hits):
        if _l>-1 and _MaxEdges>-1:
               _job_iter=0
               _acc_edg=0
               _start_pos=0
               _end_pos=_n_hits
               for _n_e in range(1,_n_hits+1):
                   _acc_edg+=_n_hits-_n_e
                   if _acc_edg>=_MaxEdges:
                       _job_iter+=1
                       _acc_edg=0
                       if _job_iter==_l+1:
                          _end_pos=_n_e
                          break
                       else:
                          _start_pos=_n_e
               return _start_pos, _end_pos
        else:
            return 0, _n_hits-1

      @staticmethod
      def GenerateEdgeLinks(_input,_ID):
          _Top, _Bottom =[], []
          for ip in _input:
              _Top.append(_ID.index(ip[0]))
              _Bottom.append(_ID.index(ip[1]))
          return [_Top,_Bottom]

      def NormaliseSeed(self,_Hit2, _Hit1, _cut_dt):
          _scale_factor=8
          h1,h2,x1,x2,y1,y2,z1,z2, tx1, tx2, ty1, ty2, l1, l2=_Hit2[0],_Hit1[0],_Hit2[1],_Hit1[1],_Hit2[2],_Hit1[2],_Hit2[3],_Hit1[3],_Hit2[4],_Hit1[4],_Hit2[5],_Hit1[5],_Hit2[6],_Hit1[6]
          _dl= math.sqrt(((x2-x1)**2) + ((y2-y1)**2) + ((z2-z1)**2))/self.Step[2]
          _dr= math.sqrt(((x2-x1)**2) + ((y2-y1)**2))/(self.Step[0]/_scale_factor)
          _dz=abs(z2-z1)/self.Step[2]
          _dtx=abs(tx2-tx1)/(_cut_dt/_scale_factor)
          _dty=abs(ty2-ty1)/(_cut_dt/_scale_factor)
          _ts=int(((l1==l2) and ('--' not in l1)))
          return [h1, h2, _ts, _dl,_dr,_dz,_dtx,_dty]


      def FitSeed(_H1, _H2, _cdt, _cdr, _cdz):
          if _H1[3]==_H2[3]: #Ensuring hit combinations are on different plates
              return False
          elif abs(_H1[3]-_H2[3])>=_cdz:
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
      @staticmethod
      def GenerateSeedVectors(Seeds):
            #Split samples into X and Y sets
            SeedsX = []
            SeedsY = []
            for s in Seeds:
                SeedsX.append(s[3:])
                SeedsY.append(s[2])
            return SeedsX,SeedsY
      def FitTrackSeed(_H1, _H2, _cdt, _cdr, _cdz): #A more involved option that involves producing the seed cutflow and the truth distribution if the MC data available.
          _ts=int(((_H1[6]==_H2[6]) and ('--' not in _H1[6])))
          if _H1[3]==_H2[3]: #Ensuring hit combinations are on different plates
                return False, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [_ts, _ts, _ts, 0, 0, 0, 0, 0, 0, 0, 0, 0]
          elif abs(_H1[3]-_H2[3])>=_cdz:
              return False, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [_ts, _ts, _ts, _ts, 0, 0, 0, 0, 0, 0, 0, 0]
          else:
              if abs(_H1[4]-_H2[4])>=_cdt:
                  return False, [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], [_ts, _ts, _ts, _ts, _ts, 0, 0, 0, 0, 0, 0, 0]
              else:
                  if abs(_H1[5]-_H2[5])>=_cdt:
                      return False, [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], [_ts, _ts, _ts, _ts, _ts, _ts, 0, 0, 0, 0, 0, 0]
                  else:
                      if abs(_H2[1]-(_H1[1]+(_H1[4]*(_H2[3]-_H1[3]))))>=_cdr:
                         return False, [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0], [_ts, _ts, _ts, _ts, _ts, _ts, _ts, 0, 0, 0, 0, 0]
                      else:
                          if abs(_H2[2]-(_H1[2]+(_H1[5]*(_H2[3]-_H1[3]))))>=_cdr:
                             return False, [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0], [_ts, _ts, _ts, _ts, _ts, _ts, _ts, _ts, 0, 0, 0, 0]
          return True, [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0], [_ts, _ts, _ts, _ts, _ts, _ts, _ts, _ts, _ts, 0, 0, 0]

      @staticmethod
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



