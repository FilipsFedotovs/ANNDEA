###This file contains the utility functions that are commonly used in ANNDEA packages

import csv
import sys
import math
import os
import subprocess
import datetime
import numpy as np
import copy

import ast
import shutil

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
#from scipy.stats import chisquare

#This utility provides Timestamps for print messages
def TimeStamp():
 return "["+datetime.datetime.now().strftime("%D")+' '+datetime.datetime.now().strftime("%H:%M:%S")+"]"

class TrainingSampleMeta:
      def __init__(self,TrainDataID):
          self.TrainDataID=TrainDataID
      def __eq__(self, other):
        return (self.TrainDataID) == (other.TrainDataID)
      def __hash__(self):
        return hash(self.TrainDataID)
      def IniHitClusterMetaData(self,stepX,stepY,stepZ,cut_dt,cut_dr,testRatio,valRatio,z_offset,y_offset,x_offset, Xsteps, Zsteps,X_overlap,Y_overlap,Z_overlap):
          self.stepX=stepX
          self.stepY=stepY
          self.stepZ=stepZ
          self.cut_dt=cut_dt
          self.cut_dr=cut_dr
          self.testRatio=testRatio
          self.valRatio=valRatio
          self.z_offset=z_offset
          self.y_offset=y_offset
          self.x_offset=x_offset
          self.Xsteps=Xsteps
          self.Zsteps=Zsteps
          self.X_overlap=X_overlap
          self.Y_overlap=Y_overlap
          self.Z_overlap=Z_overlap
      def IniTrackMetaData(self,ClassHeaders,ClassNames,ClassValues,MaxSegments,JobSets):
          self.ClassHeaders=ClassHeaders
          self.ClassNames=ClassNames
          self.ClassValues=ClassValues
          self.MaxSegments=MaxSegments
          self.JobSets=JobSets
      def IniTrackSeedMetaData(self,MaxSLG,MaxSTG,MaxDOCA,MaxAngle,JobSets,MaxSegments,VetoMotherTrack,MaxSeeds):
          self.MaxSLG=MaxSLG
          self.MaxSTG=MaxSTG
          self.MaxDOCA=MaxDOCA
          self.MaxAngle=MaxAngle
          self.JobSets=JobSets
          self.MaxSegments=MaxSegments
          self.MaxSeeds=MaxSeeds
          self.VetoMotherTrack=VetoMotherTrack
      def UpdateHitClusterMetaData(self,NoS,NoNF,NoEF,NoSets):
          self.num_node_features=NoNF
          self.num_edge_features=NoEF
          self.tot_sample_size=NoS
          self.no_sets=NoSets
      def UpdateStatus(self, status):
          if hasattr(self,'Status'):
            self.Status.append(status)
          else:
            self.Status=[status]
def GetEquationOfLine(Data):
          x=[]
          for i in range(0,len(Data)):
              x.append(i)
          line=np.polyfit(x,Data,1)
          return line
class ModelMeta:
      def __init__(self,ModelID):
          self.ModelID=ModelID
      def __eq__(self, other):
        return (self.ModelID) == (other.ModelID)
      def __hash__(self):
        return hash(self.ModelID)
      def IniModelMeta(self, ModelParams, framework, DataMeta, architecture, type):
          self.ModelParameters=ModelParams
          self.ModelFramework=framework
          self.ModelArchitecture=architecture
          self.ModelType=type
          self.TrainSessionsDataID=[]
          self.TrainSessionsDateTime=[]
          self.TrainSessionsParameters=[]
          self.TrainSessionsData=[]
          if hasattr(DataMeta,'ClassHeaders'):
              self.ClassHeaders=DataMeta.ClassHeaders
          if hasattr(DataMeta,'ClassNames'):
              self.ClassNames=DataMeta.ClassNames
          if hasattr(DataMeta,'ClassValues'):
              self.ClassValues=DataMeta.ClassValues

          if (self.ModelFramework=='PyTorch') and (self.ModelArchitecture=='TCN'):
              self.num_node_features=DataMeta.num_node_features
              self.num_edge_features=DataMeta.num_edge_features
              self.stepX=DataMeta.stepX
              self.stepY=DataMeta.stepY
              self.stepZ=DataMeta.stepZ
              self.cut_dt=DataMeta.cut_dt
              self.cut_dr=DataMeta.cut_dr
          else:
              if hasattr(DataMeta,'MaxSLG'):
                  self.MaxSLG=DataMeta.MaxSLG
              if hasattr(DataMeta,'MaxSTG'):
                  self.MaxSTG=DataMeta.MaxSTG
              if hasattr(DataMeta,'MaxDOCA'):
                  self.MaxDOCA=DataMeta.MaxDOCA
              if hasattr(DataMeta,'MaxAngle'):
                  self.MaxAngle=DataMeta.MaxAngle

      def IniTrainingSession(self, TrainDataID, DateTime, TrainParameters):
          self.TrainSessionsDataID.append(TrainDataID)
          self.TrainSessionsDateTime.append(DateTime)
          self.TrainSessionsParameters.append(TrainParameters)
      def CompleteTrainingSession(self, TrainData):
          self.TrainSessionsData.append(TrainData)


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
           for l in _l_Hits:
               _hit_count+=1
               print(UF.TimeStamp(),'Edge generation progress is ',round(100*_hit_count/len(_l_Hits),2), '%',end="\r", flush=True)
               for r in _r_Hits:
                  if HitCluster.JoinHits(l,r,cut_dt,cut_dr):
                      _Tot_Hits.append(l+r)
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



class EMO:
      def __init__(self,parts):
          self.Header=sorted(parts, key=str.lower)
          self.Partition=len(self.Header)
      def __eq__(self, other):
        return ('-'.join(self.Header)) == ('-'.join(other.Header))
      def __hash__(self):
        return hash(('-'.join(self.Header)))
      def Decorate(self,RawHits): #Decorate hit information
          self.Hits=[]
          for s in range(len(self.Header)):
              self.Hits.append([])
              for t in RawHits:
                   if self.Header[s]==t[5]:
                      self.Hits[s].append(t[:5])
          for Hit in range(0, len(self.Hits)):
             self.Hits[Hit]=sorted(self.Hits[Hit],key=lambda x: float(x[2]),reverse=False)
      def LabelSeed(self,label):
          self.Label=label
      def LabelTrack(self,label):
          self.Label=label
      def GetTrInfo(self):
          if hasattr(self,'Hits'):
             if self.Partition==2:
                __XZ1=EMO.GetEquationOfTrack(self.Hits[0])[0]
                __XZ2=EMO.GetEquationOfTrack(self.Hits[1])[0]
                __YZ1=EMO.GetEquationOfTrack(self.Hits[0])[1]
                __YZ2=EMO.GetEquationOfTrack(self.Hits[1])[1]
                __vector_1_st = np.array([np.polyval(__XZ1,self.Hits[0][0][2]),np.polyval(__YZ1,self.Hits[0][0][2]),self.Hits[0][0][2]])
                __vector_1_end = np.array([np.polyval(__XZ1,self.Hits[0][len(self.Hits[0])-1][2]),np.polyval(__YZ1,self.Hits[0][len(self.Hits[0])-1][2]),self.Hits[0][len(self.Hits[0])-1][2]])
                __vector_2_st = np.array([np.polyval(__XZ2,self.Hits[0][0][2]),np.polyval(__YZ2,self.Hits[0][0][2]),self.Hits[0][0][2]])
                __vector_2_end = np.array([np.polyval(__XZ2,self.Hits[0][len(self.Hits[0])-1][2]),np.polyval(__YZ2,self.Hits[0][len(self.Hits[0])-1][2]),self.Hits[0][len(self.Hits[0])-1][2]])
                __result=EMO.closestDistanceBetweenLines(__vector_1_st,__vector_1_end,__vector_2_st,__vector_2_end,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False)
                __midpoint=(__result[0]+__result[1])/2
                __v1=np.subtract(__vector_1_end,__midpoint)
                __v2=np.subtract(__vector_2_end,__midpoint)
                if self.Hits[0][len(self.Hits)-1][2]>self.Hits[1][len(self.Hits)-1][2]: #Workout which track is leading (has highest z-coordinate)
                    __leading_seg=0
                    __subleading_seg=1
                else:
                    __leading_seg=1
                    __subleading_seg=0
                self.Opening_Angle=EMO.angle_between(__v1, __v2)
                self.DOCA=__result[2]
                self.SLG=float(self.Hits[__leading_seg][0][2])-float(self.Hits[__subleading_seg][len(self.Hits[__subleading_seg])-1][2])
                __x2=float(self.Hits[__leading_seg][0][0])
                __x1=self.Hits[__subleading_seg][len(self.Hits[__subleading_seg])-1][0]
                __y2=float(self.Hits[__leading_seg][0][1])
                __y1=self.Hits[__subleading_seg][len(self.Hits[__subleading_seg])-1][1]
                self.STG=math.sqrt(((__x2-__x1)**2)+((__y2-__y1)**2))
             else:
                 raise ValueError("Method 'DecorateTrackGeoInfo' currently works for seeds with partition of 2 only")
          else:
                raise ValueError("Method 'DecorateTrackGeoInfo' works only if 'Decorate' method has been acted upon the seed before")
      def TrackQualityCheck(self,MaxDoca,MaxSLG, MaxSTG,MaxAngle):
                    return (self.DOCA<=MaxDoca and self.SLG<=MaxSLG and self.STG<=(MaxSTG+(self.SLG*0.96)) and abs(self.Opening_Angle)<=MaxAngle)

      def PrepareSeedPrint(self,MM):
          __TempTrack=copy.deepcopy(self.Hits)

          self.Resolution=MM.ModelParameters[11][3]
          self.bX=int(round(MM.ModelParameters[11][0]/self.Resolution,0))
          self.bY=int(round(MM.ModelParameters[11][1]/self.Resolution,0))
          self.bZ=int(round(MM.ModelParameters[11][2]/self.Resolution,0))
          self.H=(self.bX)*2
          self.W=(self.bY)*2
          self.L=(self.bZ)
          __StartTrackZ=6666666666
          __EndTrackZ=-6666666666
          for __Track in __TempTrack:
            __CurrentZ=float(__Track[0][2])
            if __CurrentZ<=__StartTrackZ:
                __StartTrackZ=__CurrentZ
                __FinX=float(__Track[0][0])
                __FinY=float(__Track[0][1])
                __FinZ=float(__Track[0][2])
                self.PrecedingTrackInd=__TempTrack.index(__Track)
            if __CurrentZ>=__EndTrackZ:
                __EndTrackZ=__CurrentZ
                self.LagTrackInd=__TempTrack.index(__Track)
          for __Tracks in __TempTrack:
              for __Hits in __Tracks:
                  __Hits[0]=float(__Hits[0])-__FinX
                  __Hits[1]=float(__Hits[1])-__FinY
                  __Hits[2]=float(__Hits[2])-__FinZ

          if MM.ModelArchitecture=='CNN-E':

              #Lon Rotate x
              __Track=__TempTrack[self.LagTrackInd]
              __Vardiff=float(__Track[len(__Track)-1][0])
              __Zdiff=float(__Track[len(__Track)-1][2])
              __vector_1 = [__Zdiff, 0]
              __vector_2 = [__Zdiff, __Vardiff]
              __Angle=EMO.angle_between(__vector_1, __vector_2)
              if np.isnan(__Angle)==True:
                        __Angle=0.0
              for __Tracks in __TempTrack:
                for __hits in __Tracks:
                     __Z=float(__hits[2])
                     __Pos=float(__hits[0])
                     __hits[2]=(__Z*math.cos(-__Angle)) - (__Pos * math.sin(-__Angle))
                     __hits[0]=(__Z*math.sin(-__Angle)) + (__Pos * math.cos(-__Angle))
              #Lon Rotate y
              __Track=__TempTrack[self.LagTrackInd]
              __Vardiff=float(__Track[len(__Track)-1][1])
              __Zdiff=float(__Track[len(__Track)-1][2])
              __vector_1 = [__Zdiff, 0]
              __vector_2 = [__Zdiff, __Vardiff]
              __Angle=EMO.angle_between(__vector_1, __vector_2)
              if np.isnan(__Angle)==True:
                         __Angle=0.0
              for __Tracks in __TempTrack:
                for __hits in __Tracks:
                     __Z=float(__hits[2])
                     __Pos=float(__hits[1])
                     __hits[2]=(__Z*math.cos(-__Angle)) - (__Pos * math.sin(-__Angle))
                     __hits[1]=(__Z*math.sin(-__Angle)) + (__Pos * math.cos(-__Angle))
             #  Phi rotate print

              __LongestDistance=0.0
              for __Track in __TempTrack:
                     __X=float(__Track[len(__Track)-1][0])
                     __Y=float(__Track[len(__Track)-1][1])
                     __Distance=math.sqrt((__X**2)+(__Y**2))
                     if __Distance>=__LongestDistance:
                      __LongestDistance=__Distance
                      __vector_1 = [__Distance, 0]
                      __vector_2 = [__X, __Y]
                      __Angle=-EMO.angle_between(__vector_1,__vector_2)
              if np.isnan(__Angle)==True:
                         __Angle=0.0
              for __Tracks in __TempTrack:
                 for __hits in __Tracks:
                     __X=float(__hits[0])
                     __Y=float(__hits[1])
                     __hits[0]=(__X*math.cos(__Angle)) - (__Y * math.sin(__Angle))
                     __hits[1]=(__X*math.sin(__Angle)) + (__Y * math.cos(__Angle))

              __X=[]
              __Y=[]
              __Z=[]
              for __Tracks in __TempTrack:
                 for __hits in __Tracks:
                     __X.append(__hits[0])
                     __Y.append(__hits[1])
                     __Z.append(__hits[2])
              __dUpX=MM.ModelParameters[11][0]-max(__X)
              __dDownX=MM.ModelParameters[11][0]+min(__X)
              __dX=(__dUpX+__dDownX)/2
              __xshift=__dUpX-__dX
              __X=[]
              for __Tracks in __TempTrack:
                 for __hits in __Tracks:
                     __hits[0]=__hits[0]+__xshift
                     __X.append(__hits[0])
             ##########Y
              __dUpY=MM.ModelParameters[11][1]-max(__Y)
              __dDownY=MM.ModelParameters[11][1]+min(__Y)
              __dY=(__dUpY+__dDownY)/2
              __yshift=__dUpY-__dY
              __Y=[]
              for __Tracks in __TempTrack:
                 for __hits in __Tracks:
                     __hits[1]=__hits[1]+__yshift
                     __Y.append(__hits[1])
              __min_scale=max(max(__X)/(MM.ModelParameters[11][0]-(2*self.Resolution)),max(__Y)/(MM.ModelParameters[11][1]-(2*self.Resolution)), max(__Z)/(MM.ModelParameters[11][2]-(2*self.Resolution)))
              for __Tracks in __TempTrack:
                 for __hits in __Tracks:
                     __hits[0]=int(round(__hits[0]/__min_scale,0))
                     __hits[1]=int(round(__hits[1]/__min_scale,0))
                     __hits[2]=int(round(__hits[2]/__min_scale,0))
          else:
            for __Tracks in __TempTrack:
                  for __Hits in __Tracks:
                      __Hits[2]=__Hits[2]*self.Resolution/1315
          __TempEnchTrack=[]
          for __Tracks in __TempTrack:
               for h in range(0,len(__Tracks)-1):
                   __deltaX=float(__Tracks[h+1][0])-float(__Tracks[h][0])
                   __deltaZ=float(__Tracks[h+1][2])-float(__Tracks[h][2])
                   __deltaY=float(__Tracks[h+1][1])-float(__Tracks[h][1])
                   try:
                    __vector_1 = [__deltaZ,0]
                    __vector_2 = [__deltaZ, __deltaX]
                    __ThetaAngle=EMO.Opening_Angle(__vector_1, __vector_2)
                   except:
                     __ThetaAngle=0.0
                   try:
                     __vector_1 = [__deltaZ,0]
                     __vector_2 = [__deltaZ, __deltaY]
                     __PhiAngle=EMO.Opening_Angle(__vector_1, __vector_2)
                   except:
                     __PhiAngle=0.0
                   __TotalDistance=math.sqrt((__deltaX**2)+(__deltaY**2)+(__deltaZ**2))
                   __Distance=(float(self.Resolution)/3)
                   if __Distance>=0 and __Distance<1:
                      __Distance=1.0
                   if __Distance<0 and __Distance>-1:
                      __Distance=-1.0
                   __Iterations=int(round(__TotalDistance/__Distance,0))
                   for i in range(1,__Iterations):
                       __New_Hit=[]
                       if math.isnan(float(__Tracks[h][0])+float(i)*__Distance*math.sin(__ThetaAngle)):
                          continue
                       if math.isnan(float(__Tracks[h][1])+float(i)*__Distance*math.sin(__PhiAngle)):
                          continue
                       if math.isnan(float(__Tracks[h][2])+float(i)*__Distance*math.cos(__ThetaAngle)):
                          continue
                       __New_Hit.append(float(__Tracks[h][0])+float(i)*__Distance*math.sin(__ThetaAngle))
                       __New_Hit.append(float(__Tracks[h][1])+float(i)*__Distance*math.sin(__PhiAngle))
                       __New_Hit.append(float(__Tracks[h][2])+float(i)*__Distance*math.cos(__ThetaAngle))
                       __TempEnchTrack.append(__New_Hit)


          self.TrackPrint=[]
          for __Tracks in __TempTrack:
               for __Hits in __Tracks:
                   __Hits[0]=int(round(float(__Hits[0])/self.Resolution,0))
                   __Hits[1]=int(round(float(__Hits[1])/self.Resolution,0))
                   __Hits[2]=int(round(float(__Hits[2])/self.Resolution,0))
                   self.TrackPrint.append(str(__Hits[:3]))
          for __Hits in __TempEnchTrack:
                   __Hits[0]=int(round(float(__Hits[0])/self.Resolution,0))
                   __Hits[1]=int(round(float(__Hits[1])/self.Resolution,0))
                   __Hits[2]=int(round(float(__Hits[2])/self.Resolution,0))
                   self.TrackPrint.append(str(__Hits[:3]))
          self.TrackPrint=list(set(self.TrackPrint))
          for p in range(len(self.TrackPrint)):
               self.TrackPrint[p]=ast.literal_eval(self.TrackPrint[p])
          self.TrackPrint=[p for p in self.TrackPrint if (abs(p[0])<self.bX and abs(p[1])<self.bY and abs(p[2])<self.bZ)]
          del __TempEnchTrack
          del __TempTrack
      def AssignANNTrUID(self,ID):
          self.UTrID=ID
      def PrepareSeedGraph(self,MM):
          if MM.ModelArchitecture=='GCN-4N-FC':

                      __TempTrack=copy.deepcopy(self.Hits)
                      for __Tracks in __TempTrack:
                              for h in range(len(__Tracks)):
                                  __Tracks[h]=__Tracks[h][:3]

                      __LongestDistance=0.0
                      for __Track in __TempTrack:
                        __Xdiff=float(__Track[len(__Track)-1][0])-float(__Track[0][0])
                        __Ydiff=float(__Track[len(__Track)-1][1])-float(__Track[0][1])
                        __Zdiff=float(__Track[len(__Track)-1][2])-float(__Track[0][2])
                        __Distance=math.sqrt(__Xdiff**2+__Ydiff**2+__Zdiff**2)
                        if __Distance>=__LongestDistance:
                            __LongestDistance=__Distance
                            __FinX=float(__Track[0][0])
                            __FinY=float(__Track[0][1])
                            __FinZ=float(__Track[0][2])
                            self.LongestTrackInd=__TempTrack.index(__Track)
                      # Shift
                      for __Tracks in __TempTrack:
                          for h in range(len(__Tracks)):
                              __Tracks[h][0]=float(__Tracks[h][0])-__FinX
                              __Tracks[h][1]=float(__Tracks[h][1])-__FinY
                              __Tracks[h][2]=float(__Tracks[h][2])-__FinZ

                      # Rescale
                      for __Tracks in __TempTrack:
                              for h in range(len(__Tracks)):
                                  __Tracks[h][0]=__Tracks[h][0]/MM.ModelParameters[11][0]
                                  __Tracks[h][1]=__Tracks[h][1]/MM.ModelParameters[11][1]
                                  __Tracks[h][2]=__Tracks[h][2]/MM.ModelParameters[11][2]

                      import pandas as pd
# Graph representation v2 (fully connected)
                      for el in range(len(__TempTrack[0])):
                         __TempTrack[0][el].append('0')
                         __TempTrack[0][el].append(el)

                      try:
                          for el in range(len(__TempTrack[1])):
                             __TempTrack[1][el].append('1')
                             __TempTrack[1][el].append(el)

                          __graphData_x =__TempTrack[0]+__TempTrack[1]
                      except:
                          __graphData_x =__TempTrack[0]

                      __graphData_x = pd.DataFrame (__graphData_x, columns = ['x', 'y', 'z', 'TrackID', 'NodeIndex'])
                      __graphData_x['dummy'] = 'dummy'
                      __graphData_x_r = __graphData_x

                      __graphData_join = pd.merge(
                      __graphData_x,
                      __graphData_x_r,
                      how="inner",
                      on="dummy",
                      suffixes=('_l','_r'),
                      )
                      __graphData_join = __graphData_join.drop(__graphData_join.index[__graphData_join['TrackID_l']==__graphData_join['TrackID_r']] & __graphData_join.index[__graphData_join['NodeIndex_l']==__graphData_join['NodeIndex_r']])

                      __graphData_join['d_z'] = np.sqrt((__graphData_join['z_l'] - __graphData_join['z_r'])**2)
                      __graphData_join['d_xy'] = np.sqrt((__graphData_join['x_l'] - __graphData_join['x_r'])**2 + (__graphData_join['y_l'] - __graphData_join['y_r'])**2)
                      __graphData_join['d_xyz'] = np.sqrt((__graphData_join['x_l'] - __graphData_join['x_r'])**2 + (__graphData_join['y_l'] - __graphData_join['y_r'])**2 + (__graphData_join['z_l'] - __graphData_join['z_r'])**2)
                      __graphData_join['ConnectionType'] = __graphData_join['TrackID_l'] == __graphData_join['TrackID_r']
                      __graphData_join.drop(['x_l', 'y_l', 'z_l', 'x_r', 'y_r', 'z_r', 'dummy'], axis = 1, inplace = True)

                      __graphData_join[['ConnectionType']] = __graphData_join[['ConnectionType']].astype(float)
                      __graphData_join[['NodeIndex_l']] = __graphData_join[['NodeIndex_l']].astype(str)
                      __graphData_join[['NodeIndex_r']] = __graphData_join[['NodeIndex_r']].astype(str)

                      __graphData_join['LeftKey'] = __graphData_join['TrackID_l'] +'-'+ __graphData_join['NodeIndex_l']
                      __graphData_join['RightKey'] = __graphData_join['TrackID_r'] +'-'+ __graphData_join['NodeIndex_r']

                      __graphData_join.drop(['NodeIndex_l', 'TrackID_l', 'NodeIndex_r', 'TrackID_r'], axis = 1, inplace = True)

                      __graphData_list = __graphData_join.values.tolist()


                      try:
                        __graphData_nodes =__TempTrack[0]+__TempTrack[1]
                      except:
                        __graphData_nodes =__TempTrack[0]

                    # position of nodes
                      __graphData_pos = []
                      for node in __graphData_nodes:
                        __graphData_pos.append(node[0:3])

                      for g in __graphData_nodes:
                        g.append(g[3]+'-'+str(g[4]))
                        g[3]=float(g[3])


                      Data_x = []
                      for g in __graphData_nodes:
                        Data_x.append(g[:4])


                      node_ind_list=[]
                      for g in __graphData_nodes:
                        node_ind_list.append(g[5])


                      top_edge = []
                      bottom_edge = []
                      edge_attr = []



                      for h in __graphData_list:
                        top_edge.append(node_ind_list.index(h[5]))

                      for h in __graphData_list:
                        bottom_edge.append(node_ind_list.index(h[4]))

                      for h in __graphData_list:
                        edge_attr.append(h[:4])


                      try:
                          __y=[]
                          for i in range(MM.ModelParameters[10][1]):
                            if self.Label==i:
                                __y.append(1)
                            else:
                                __y.append(0)
                          __graphData_y = (__y)
                          import torch
                          import torch_geometric
                          from torch_geometric.data import Data
                          self.GraphSeed = Data(x=torch.Tensor(Data_x), edge_index = torch.Tensor([top_edge, bottom_edge]).long(), edge_attr = torch.Tensor(edge_attr),y=torch.Tensor([__graphData_y]), pos = torch.Tensor(__graphData_pos))
                      except:
                          import torch
                          import torch_geometric
                          from torch_geometric.data import Data
                          self.GraphSeed = Data(x=torch.Tensor(Data_x), edge_index = torch.Tensor([top_edge, bottom_edge]).long(), edge_attr = torch.Tensor(edge_attr), pos = torch.Tensor(__graphData_pos))
      def Plot(self,PlotType):
        if PlotType=='XZ' or PlotType=='ZX':
          __InitialData=[]
          __Index=-1
          for x in range(-self.bX,self.bX):
             for z in range(0,self.bZ):
                 __InitialData.append(0.0)
          __Matrix = np.array(__InitialData)
          __Matrix=np.reshape(__Matrix,(self.H,self.L))
          for __Hits in self.TrackPrint:
                   __Matrix[int(__Hits[0])+self.bX][int(__Hits[2])]=1
          import matplotlib as plt
          from matplotlib.colors import LogNorm
          from matplotlib import pyplot as plt
          plt.title('Seed '+':'.join(self.Header))
          plt.xlabel('Z [microns /'+str(int(self.Resolution))+']')
          plt.ylabel('X [microns /'+str(int(self.Resolution))+']')
          __image=plt.imshow(__Matrix,cmap='gray_r',extent=[0,self.bZ,self.bX,-self.bX])
          plt.gca().invert_yaxis()
          plt.show()
        elif PlotType=='YZ' or PlotType=='ZY':
          __InitialData=[]
          __Index=-1
          for y in range(-self.bY,self.bY):
             for z in range(0,self.bZ):
                 __InitialData.append(0.0)
          __Matrix = np.array(__InitialData)
          __Matrix=np.reshape(__Matrix,(self.W,self.L))
          for __Hits in self.TrackPrint:
                   __Matrix[int(__Hits[1])+self.bY][int(__Hits[2])]=1
          import matplotlib as plt
          from matplotlib.colors import LogNorm
          from matplotlib import pyplot as plt
          plt.title('Seed '+':'.join(self.Header))
          plt.xlabel('Z [microns /'+str(int(self.Resolution))+']')
          plt.ylabel('Y [microns /'+str(int(self.Resolution))+']')
          __image=plt.imshow(__Matrix,cmap='gray_r',extent=[0,self.bZ,self.bY,-self.bY])
          plt.gca().invert_yaxis()
          plt.show()
        elif PlotType=='XY' or PlotType=='YX':
          __InitialData=[]
          __Index=-1
          for x in range(-self.bX,self.bX):
             for y in range(-self.bY,self.bY):
                 __InitialData.append(0.0)
          __Matrix = np.array(__InitialData)
          __Matrix=np.reshape(__Matrix,(self.H,self.W))
          for __Hits in self.TrackPrint:
                   __Matrix[int(__Hits[0])+self.bX][int(__Hits[1]+self.bY)]=1
          import matplotlib as plt
          from matplotlib.colors import LogNorm
          from matplotlib import pyplot as plt
          plt.title('Seed '+':'.join(self.Header))
          plt.xlabel('Y [microns /'+str(int(self.Resolution))+']')
          plt.ylabel('X [microns /'+str(int(self.Resolution))+']')
          __image=plt.imshow(__Matrix,cmap='gray_r',extent=[-self.bY,self.bY,self.bX,-self.bX])
          plt.gca().invert_yaxis()
          plt.show()
        else:
          print('Invalid plot type input value! Should be XZ, YZ or XY')

      def FitSeed(self,Mmeta,M):
          if Mmeta.ModelType=='CNN':
             EMO.PrepareSeedPrint(self,Mmeta)
             __Image=LoadRenderImages([self],1,1)[0]
             self.Fit=M.predict(__Image)[0][1]
             self.FIT=[self.Fit,self.Fit]
             del __Image
          elif Mmeta.ModelType=='GNN':
             EMO.PrepareSeedGraph(self,Mmeta)
             M.eval()
             import torch
             graph = self.GraphSeed
             graph.batch = torch.zeros(len(graph.x),dtype=torch.int64)
             self.Fit=M(graph.x, graph.edge_index, graph.edge_attr,graph.batch)[0][1].item()
             self.FIT=[self.Fit,self.Fit]
          return self.Fit>=0.5
      def ClassifySeed(self,Mmeta,M):
          if Mmeta.ModelType=='CNN':
             EMO.PrepareSeedPrint(self,Mmeta)
             __Image=LoadRenderImages([self],1,1)[0]
             self.Class=M.predict(__Image)[0]
             self.ClassHeaders=Mmeta.ClassHeaders.append('Other')
             del __Image
          elif Mmeta.ModelType=='GNN':
             EMO.PrepareSeedGraph(self,Mmeta)
             M.eval()
             import torch
             graph = self.GraphSeed
             graph.batch = torch.zeros(len(graph.x),dtype=torch.int64)
             #self.Fit=M(graph.x, graph.edge_index, graph.edge_attr,graph.batch)[0][1].item()
             self.Class=M(graph.x, graph.edge_index, graph.edge_attr,graph.batch).tolist()[0]
             self.ClassHeaders=Mmeta.ClassHeaders+['Other']
      def InjectTrackSeed(self,OtherSeed):
          self_matx=EMO.DensityMatrix(OtherSeed.Header,self.Header)
          if EMO.Overlap(self_matx)==False:
              return EMO.Overlap(self_matx)

          new_seed_header=EMO.ProjectVectorElements(self_matx,self.Header)
          new_self_hits=EMO.ProjectVectorElements(self_matx,self.Hits)
          new_self_fit=EMO.ProjectVectorElements(self_matx,self.FIT)
          remain_1_s = EMO.GenerateInverseVector(self.Header,new_seed_header)
          remain_1_o = EMO.GenerateInverseVector(OtherSeed.Header,new_seed_header)
          OtherSeed.Header=EMO.ProjectVectorElements([remain_1_o],OtherSeed.Header)
          self.Header=EMO.ProjectVectorElements([remain_1_s],self.Header)
          OtherSeed.Hits=EMO.ProjectVectorElements([remain_1_o],OtherSeed.Hits)
          self.Hits=EMO.ProjectVectorElements([remain_1_s],self.Hits)
          OtherSeed.FIT=EMO.ProjectVectorElements([remain_1_o],OtherSeed.FIT)
          self.FIT=EMO.ProjectVectorElements([remain_1_s],self.FIT)
          if (len(OtherSeed.Header))==0:
              self.Header+=new_seed_header
              self.Hits+=new_self_hits
              self.FIT+=new_self_fit
              self.Fit=sum(self.FIT)/len(self.FIT)
              self.Partition=len(self.Header)
              if len(self.FIT)!=len(self.Header):
                  raise Exception('Fit error')
                  exit()

              return True
          if (len(self.Header))==0:
              self.Header+=new_seed_header
              self.Hits+=new_self_hits
              self.FIT+=new_self_fit
              self.Header+=OtherSeed.Header
              self.Hits+=OtherSeed.Hits
              self.FIT+=OtherSeed.FIT
              self.Fit=sum(self.FIT)/len(self.FIT)
              self.Partition=len(self.Header)
              if len(self.FIT)!=len(self.Header):
                  raise Exception('Fit error')
                  exit()
              return True
          self_2_matx=EMO.DensityMatrix(OtherSeed.Hits,self.Hits)
          other_2_matx=EMO.DensityMatrix(self.Hits,OtherSeed.Hits)
          #print('Test',self_2_matx,self_2_matx)
          last_s_seed_header=EMO.ProjectVectorElements(self_2_matx,self.Header)
          last_o_seed_header=EMO.ProjectVectorElements(other_2_matx,OtherSeed.Header)
          remain_2_s = EMO.GenerateInverseVector(self.Header,last_s_seed_header)
          remain_2_o = EMO.GenerateInverseVector(OtherSeed.Header,last_o_seed_header)

          new_seed_header+=EMO.ProjectVectorElements([remain_2_s],self.Header)
          new_seed_header+=EMO.ProjectVectorElements([remain_2_o],OtherSeed.Header)
          new_self_fit+=EMO.ProjectVectorElements([remain_2_s],self.FIT)
          new_self_fit+=EMO.ProjectVectorElements([remain_2_o],OtherSeed.FIT)
          new_self_hits+=EMO.ProjectVectorElements([remain_2_s],self.Hits)
          new_self_hits+=EMO.ProjectVectorElements([remain_2_o],OtherSeed.Hits)


          last_remain_headers_s = EMO.GenerateInverseVector(self.Header,new_seed_header)
          last_remain_headers_o = EMO.GenerateInverseVector(OtherSeed.Header,new_seed_header)
          last_self_headers=EMO.ProjectVectorElements([last_remain_headers_s],self.Header)
          last_other_headers=EMO.ProjectVectorElements([last_remain_headers_o],OtherSeed.Header)
          if (len(last_other_headers))==0:
              self.Header=new_seed_header
              self.Hits=new_self_hits
              self.FIT=new_self_fit
              self.Fit=sum(self.FIT)/len(self.FIT)
              self.Partition=len(self.Header)
              if len(self.FIT)!=len(self.Header):
                  raise Exception('Fit error')
                  exit()
              return True

          last_self_hits=EMO.ProjectVectorElements([last_remain_headers_s],self.Hits)
          last_other_hits=EMO.ProjectVectorElements([last_remain_headers_o],OtherSeed.Hits)
          last_self_fits=EMO.ProjectVectorElements([last_remain_headers_s],self.FIT)
          last_other_fits=EMO.ProjectVectorElements([last_remain_headers_o],OtherSeed.FIT)
          last_remain_matr=EMO.DensityMatrix(last_other_hits,last_self_hits)

          new_seed_header+=EMO.ReplaceWeakerTracks(last_remain_matr,last_other_headers,last_self_headers,last_other_fits,last_self_fits)
          new_self_fit+=EMO.ReplaceWeakerFits(new_seed_header,last_self_headers,last_other_headers,last_other_fits,last_self_fits)[0:len(EMO.ReplaceWeakerFits(new_seed_header,last_self_headers,last_other_headers,last_other_fits,last_self_fits))]
          new_self_hits+=EMO.ReplaceWeakerTracks(last_remain_matr,last_other_hits,last_self_hits,last_other_fits,last_self_fits)
          self.Header=new_seed_header
          self.Hits=new_self_hits
          self.FIT=new_self_fit
          self.Avg_Fit=sum(self.FIT)/len(self.FIT)
          self.Partition=len(self.Header)
          if len(self.FIT)!=len(self.Header):
                  raise Exception('Fit error')
                  exit()

          return True
      @staticmethod
      def unit_vector(vector):
          return vector / np.linalg.norm(vector)
      def Overlap(a):
            overlap=0
            for j in a:
                for i in j:
                    overlap+=i
            return(overlap>0)
      def angle_between(v1, v2):
            v1_u = EMO.unit_vector(v1)
            v2_u = EMO.unit_vector(v2)
            dot = v1_u[0]*v2_u[0] + v1_u[1]*v2_u[1]      # dot product
            det = v1_u[0]*v2_u[1] - v1_u[1]*v2_u[0]      # determinant
            return np.arctan2(det, dot)

      def GetEquationOfTrack(EMO):
          Xval=[]
          Yval=[]
          Zval=[]
          for Hits in EMO:
              Xval.append(Hits[0])
              Yval.append(Hits[1])
              Zval.append(Hits[2])
          XZ=np.polyfit(Zval,Xval,1)
          YZ=np.polyfit(Zval,Yval,1)
          return (XZ,YZ, 'N/A',Xval[0],Yval[0],Zval[0])

      def closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False):
            a0=np.array(a0)
            a1=np.array(a1)
            b0=np.array(b0)
            b1=np.array(b1)
            # If clampAll=True, set all clamps to True
            if clampAll:
                clampA0=True
                clampA1=True
                clampB0=True
                clampB1=True


            # Calculate denomitator
            A = a1 - a0
            B = b1 - b0
            magA = np.linalg.norm(A)
            magB = np.linalg.norm(B)

            _A = A / magA
            _B = B / magB

            cross = np.cross(_A, _B);
            denom = np.linalg.norm(cross)**2


            # If lines are parallel (denom=0) test if lines overlap.
            # If they don't overlap then there is a closest point solution.
            # If they do overlap, there are infinite closest positions, but there is a closest distance
            if not denom:
                d0 = np.dot(_A,(b0-a0))

                # Overlap only possible with clamping
                if clampA0 or clampA1 or clampB0 or clampB1:
                    d1 = np.dot(_A,(b1-a0))

                    # Is segment B before A?
                    if d0 <= 0 >= d1:
                        if clampA0 and clampB1:
                            if np.absolute(d0) < np.absolute(d1):
                                return a0,b0,np.linalg.norm(a0-b0)
                            return a0,b1,np.linalg.norm(a0-b1)


                    # Is segment B after A?
                    elif d0 >= magA <= d1:
                        if clampA1 and clampB0:
                            if np.absolute(d0) < np.absolute(d1):
                                return a1,b0,np.linalg.norm(a1-b0)
                            return a1,b1,np.linalg.norm(a1-b1)


                # Segments overlap, return distance between parallel segments
                return None,None,np.linalg.norm(((d0*_A)+a0)-b0)



            # Lines criss-cross: Calculate the projected closest points
            t = (b0 - a0);
            detA = np.linalg.det([t, _B, cross])
            detB = np.linalg.det([t, _A, cross])

            t0 = detA/denom;
            t1 = detB/denom;

            pA = a0 + (_A * t0) # Projected closest point on segment A
            pB = b0 + (_B * t1) # Projected closest point on segment B


            # Clamp projections
            if clampA0 or clampA1 or clampB0 or clampB1:
                if clampA0 and t0 < 0:
                    pA = a0
                elif clampA1 and t0 > magA:
                    pA = a1

                if clampB0 and t1 < 0:
                    pB = b0
                elif clampB1 and t1 > magB:
                    pB = b1

                # Clamp projection A
                if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
                    dot = np.dot(_B,(pA-b0))
                    if clampB0 and dot < 0:
                        dot = 0
                    elif clampB1 and dot > magB:
                        dot = magB
                    pB = b0 + (_B * dot)

                # Clamp projection B
                if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
                    dot = np.dot(_A,(pB-a0))
                    if clampA0 and dot < 0:
                        dot = 0
                    elif clampA1 and dot > magA:
                        dot = magA
                    pA = a0 + (_A * dot)
            return pA,pB,np.linalg.norm(pA-pB)
      
      def Product(a,b):
         if type(a) is str:
             if type(b) is str:
                 return(int(a==b))
             if type(b) is int:
                 return(b)
         if type(b) is str:
             if type(a) is str:
                 return(int(a==b))
             if type(a) is int:
                 return(a)
         if type(a) is list:
             if type(b) is list:
                 a_temp=[]
                 b_temp=[]
                 for el in a:
                     a_temp.append(el[2])
                 for el in b:
                     b_temp.append(el[2])
                 min_a=min(a_temp)
                 min_b=min(b_temp)
                 max_a=max(a_temp)
                 max_b=max(b_temp)
                 if (min_b>=min_a) and (max_b<=max_a):
                     return(1)
                 elif (min_a>=min_b) and (max_a<=max_b):
                     return(1)
                 elif (max_a>min_b) and (max_a<max_b):
                     return(1)
                 elif (max_b>min_a) and (max_b<max_a):
                     return(1)
                 return(0)
             elif b==1:
                 return(a)
             elif b==0:
                 return(b)
             else:
                 raise Exception('Value incompatibility error')
         if type(b) is list:
             if type(a) is list:
                 a_temp=[]
                 b_temp=[]
                 for el in a:
                     a_temp.append(el[2])
                 for el in b:
                     b_temp.append(el[2])
                 min_a=min(a_temp)
                 min_b=min(b_temp)
                 max_a=max(a_temp)
                 max_b=max(b_temp)
                 if (min_b>=min_a) and (max_b<=max_a):
                     return(1)
                 elif (min_a>=min_b) and (max_a<=max_b):
                     return(1)
                 elif (max_a>min_b) and (max_a<max_b):
                     return(1)
                 elif (max_b>min_a) and (max_b<max_a):
                     return(1)
                 return(0)
             elif a==1:
                 return(b)
             elif a==0:
                 return(a)
             else:
                 raise Exception('Value incompatibility error')
         if type(b) is int and type(a) is int:
             return(a*b)
         elif type(b) is int and ((type(a) is float) or (type(a) is np.float32)):
             return(a*b)
         elif type(a) is int and ((type(b) is float) or (type(b) is np.float32)):
             return(a*b)
      def DensityMatrix(m,f):
            matrix=[]
            for j in m:
                row=[]
                for i in f:
                    row.append(EMO.Product(j,i))
                matrix.append(row)
            return matrix
      def ReplaceWeakerTracks(matx,m,f,m_fit,f_fit):
                      res_vector=[]
                      delete_vec=[]
                      for j in range(len(m)):
                          accumulative_fit_f=0
                          accumulative_fit_m=m_fit[j]
                          del_temp_vec=[]
                          counter=0
                          for i in range(len(matx[j])):
                                  if matx[j][i]==1:
                                      accumulative_fit_f+=f_fit[i]
                                      del_temp_vec.append(f[i])
                                      counter+=1
                          if (accumulative_fit_m>accumulative_fit_f/counter):
                              res_vector.append(m[j])
                              delete_vec+=del_temp_vec
                          else:
                              res_vector+=del_temp_vec
                      final_vector=[]
                      for mel in m:
                          if (mel in res_vector):
                             final_vector.append(mel)
                      for fel in f:
                          if (fel in delete_vec)==False:
                             final_vector.append(fel)
                      return(final_vector)
      def ReplaceWeakerFits(h,l_f,l_m,m_fit,f_fit):
                      new_h=l_f+l_m
                      new_fit=f_fit+m_fit
                      res_fits=[]
                      for hd in range(len(new_h)):
                          if (new_h[hd] in h):
                              res_fits.append(new_fit[hd])
                      return res_fits
      def ProjectVectorElements(m,v):
                  if (len(m[0])!=len(v)):
                      raise Exception('Number of vector columns is not equal to number of acting matrix rows')
                  else:
                      res_vector=[]
                      for j in m:
                          for i in range(len(j)):
                              if (EMO.Product(j[i],v[i]))==1:
                                  res_vector.append(v[i])
                              elif (EMO.Product(j[i],v[i]))==v[i]:
                                  res_vector.append(v[i])
                      return(res_vector)
      def GenerateInverseVector(ov,v):
            inv_vector=[]
            for el in ov:
               if (el in v) == False:
                   inv_vector.append(1)
               elif (el in v):
                   inv_vector.append(0)
            return(inv_vector)
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
         elif ModelMeta.ModelArchitecture=='GCN-4N-FC':
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
              if ModelMeta.ModelParameters.index(el)<=4 and len(el)>0:
                 HiddenLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)<=9 and len(el)>0:
                 FullyConnectedLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)==10:
                 OutputLayer=el
              elif ModelMeta.ModelParameters.index(el)==11:
                 ImageLayer=el
            H=int(round(ImageLayer[0]/ImageLayer[3],0))*2
            W=int(round(ImageLayer[1]/ImageLayer[3],0))*2
            L=int(round(ImageLayer[2]/ImageLayer[3],0))
            model = Sequential()
            for HL in HiddenLayer:
                     Nodes=HL[0]*16
                     KS=(HL[2]*2)+1
                     PS=HL[3]
                     DR=float(HL[6]-1)/10.0
                     if HiddenLayer.index(HL)==0:
                        model.add(Conv3D(Nodes, activation=act_fun_list[HL[1]],kernel_size=(KS,KS,KS),kernel_initializer='he_uniform', input_shape=(H,W,L,1)))
                     else:
                        model.add(Conv3D(Nodes, activation=act_fun_list[HL[1]],kernel_size=(KS,KS,KS),kernel_initializer='he_uniform'))
                     if PS>1:
                        model.add(MaxPooling3D(pool_size=(PS, PS, PS)))
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
def CleanFolder(folder,key):
    if key=='':
      for the_file in os.listdir(folder):
                file_path=os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
    else:
      for the_file in os.listdir(folder):
                file_path=os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path) and (key in the_file):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
#This function automates csv read/write operations
def LogOperations(flocation,mode, message):
    if mode=='a':
        csv_writer_log=open(flocation,"a")
        log_writer = csv.writer(csv_writer_log)
        if len(message)>0:
         for m in message:
          log_writer.writerow(m)
        csv_writer_log.close()
    if mode=='w':
        csv_writer_log=open(flocation,"w")
        log_writer = csv.writer(csv_writer_log)
        if len(message)>0:
         for m in message:
           log_writer.writerow(m)
        csv_writer_log.close()
    if mode=='r':
        csv_reader_log=open(flocation,"r")
        log_reader = csv.reader(csv_reader_log)
        return list(log_reader)


def PickleOperations(flocation,mode, message):
    import pickle
    if mode=='w':
        pickle_writer_log=open(flocation,"wb")
        pickle.dump(message, pickle_writer_log)
        pickle_writer_log.close()
        return ('',"UF.PickleOperations Message: Data has been written successfully into "+flocation)
    if mode=='r':
        pickle_writer_log=open(flocation,'rb')
        result=pickle.load(pickle_writer_log)
        pickle_writer_log.close()
        return (result,"UF.PickleOperations Message: Data has been loaded successfully from "+flocation)

def RecCleanUp(AFS_DIR, EOS_DIR, Process, FileNames, ProcessId):
      subprocess.call(['condor_rm', '-constraint', ProcessId])
      EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
      EOSsubModelDIR=EOSsubDIR+'/'+'Data/REC_SET'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      folder =  AFS_DIR+'/HTCondor/SH'
      CleanFolder(folder,'SH_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/SUB'
      CleanFolder(folder,'SUB_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/MSG'
      CleanFolder(folder,'MSG_'+Process+'_')

def EvalCleanUp(AFS_DIR, EOS_DIR, Process, FileNames, ProcessId):
      subprocess.call(['condor_rm', '-constraint', ProcessId])
      EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
      EOSsubModelDIR=EOSsubDIR+'/'+'Data/TEST_SET'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      folder =  AFS_DIR+'/HTCondor/SH'
      CleanFolder(folder,'SH_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/SUB'
      CleanFolder(folder,'SUB_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/MSG'
      CleanFolder(folder,'MSG_'+Process+'_')

def TrainCleanUp(AFS_DIR, EOS_DIR, Process, FileNames, ProcessId):
      subprocess.call(['condor_rm', '-constraint', ProcessId])
      EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
      EOSsubModelDIR=EOSsubDIR+'/'+'Data/TRAIN_SET'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      EOSsubModelDIR=EOSsubDIR+'/'+'Models'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      folder =  AFS_DIR+'/HTCondor/SH'
      CleanFolder(folder,'SH_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/SUB'
      CleanFolder(folder,'SUB_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/MSG'
      CleanFolder(folder,'MSG_'+Process+'_')

def CreateCondorJobs(AFS,EOS,PY,path,o,pfx,sfx,ID,loop_params,OptionHeader,OptionLine,Sub_File,batch_sub=False,Exception=['',''], Log=False, GPU=False):
   if Exception[0]==" --PlateZ ":
    if batch_sub==False:
        from alive_progress import alive_bar
        bad_pop=[]
        TotJobs=0
        if type(loop_params) is int:
            nest_lvl=1
            TotJobs=loop_params
        elif type(loop_params[0]) is int:
            nest_lvl=2
            TotJobs=np.sum(loop_params)
        elif type(loop_params[0][0]) is int:
            nest_lvl=3
            for lp in loop_params:
                TotJobs+=np.sum(lp)
        OptionHeader+=[' --EOS '," --AFS ", " --PY ", " --BatchID "]
        OptionLine+=[EOS, AFS, PY, ID]
        TotJobs=int(TotJobs)
        with alive_bar(TotJobs,force_tty=True, title='Checking the results from HTCondor') as bar:
             if nest_lvl==2:
                 for i in range(len(loop_params)):
                     for j in range(loop_params[i]):
                               required_output_file_location=EOS+'/'+path+'/Temp_'+pfx+'_'+ID+'_'+str(i)+'/'+pfx+'_'+ID+'_'+o+'_'+str(i)+'_'+str(j)+sfx
                               bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               if os.path.isfile(required_output_file_location)!=True:
                                  bad_pop.append([OptionHeader+[' --i ', ' --j ', ' --p ', ' --o ',' --pfx ', ' --sfx ', Exception[0]], OptionLine+[i, j, path,o, pfx, sfx, Exception[1][j][0]], SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])

             if nest_lvl==3:
                 for i in range(len(loop_params)):
                     for j in range(len(loop_params[i])):
                         for k in range(loop_params[i][j]):
                               required_output_file_location=EOS+'/'+path+'/Temp_'+pfx+'_'+ID+'_'+str(i)+'/'+pfx+'_'+ID+'_'+o+'_'+str(i)+'_'+str(j)+sfx
                               bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '_' + str(k) +'.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '_' + str(k) +'.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '_' + str(k)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               if os.path.isfile(required_output_file_location)!=True:
                                  bad_pop.append([OptionHeader+[' --i ', ' --j ',' --k ', ' --p ', ' --o ',' --pfx ', ' --sfx ', Exception[0]], OptionLine+[i, j,k, path,o, pfx, sfx, Exception[1][j][0]], SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
        return(bad_pop)
    else:
        from alive_progress import alive_bar
        bad_pop=[]
        TotJobs=0
        if type(loop_params) is int:
            nest_lvl=1
            TotJobs=loop_params
        elif type(loop_params[0]) is int:
            nest_lvl=2
            TotJobs=np.sum(loop_params)
        elif type(loop_params[0][0]) is int:
            nest_lvl=3
            for lp in loop_params:
                TotJobs+=np.sum(lp)
        OptionHeader+=[' --EOS '," --AFS ", " --PY ", " --BatchID "]
        OptionLine+=[EOS, AFS, PY, ID]
        TotJobs=int(TotJobs)
        with alive_bar(TotJobs,force_tty=True, title='Checking the results from HTCondor') as bar:
             if nest_lvl==2:
                 for i in range(len(loop_params)):
                               bar.text = f'-> Preparing batch submission...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) + '.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               bad_pop.append([OptionHeader+[' --i ', ' --j ', ' --p ', ' --o ',' --pfx ', ' --sfx ', Exception[0]], OptionLine+[i, '$1', path,o, pfx, sfx, Exception[1][i][0]], SHName, SUBName, MSGName, ScriptName, loop_params[i], 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
             if nest_lvl==3:
                 for i in range(len(loop_params)):
                     for j in range(loop_params[i]):
                               bar.text = f'-> Preparing batch submission...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j)+'.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) +'_' + str(j)+'.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i) +'_' + str(j)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               bad_pop.append([OptionHeader+[' --i ', ' --j ',' --k ', ' --p ', ' --o ',' --pfx ', ' --sfx ', Exception[0]], OptionLine+[i, j, '$1', path,o, pfx, sfx, Exception[1][i][0]], SHName, SUBName, MSGName, ScriptName, loop_params[i], 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
        return(bad_pop)
   else:
    if batch_sub==False:
        from alive_progress import alive_bar
        bad_pop=[]
        TotJobs=0
        if type(loop_params) is int:
            nest_lvl=1
            TotJobs=loop_params
        elif type(loop_params[0]) is int:
            nest_lvl=2
            TotJobs=np.sum(loop_params)
        elif type(loop_params[0][0]) is int:
            nest_lvl=3
            for lp in loop_params:
                TotJobs+=np.sum(lp)
        OH=OptionHeader+[' --EOS '," --AFS ", " --PY ", " --BatchID "]
        OL=OptionLine+[EOS, AFS, PY, ID]
        TotJobs=int(TotJobs)
        with alive_bar(TotJobs,force_tty=True, title='Checking the results from HTCondor') as bar:
             if nest_lvl==1:
                 for i in range(loop_params):
                               required_output_file_location=EOS+'/'+path+'/Temp_'+pfx+'_'+ID+'_'+str(0)+'/'+pfx+'_'+ID+'_'+o+'_'+str(i)+sfx
                               bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(0)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(0)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) + '.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(0)+'/MSG_'+pfx+'_'+ ID+'_' + str(i)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               if os.path.isfile(required_output_file_location)!=True:
                                  bad_pop.append([OH+[' --i ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OL+[i, path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
             if nest_lvl==2:
                 for i in range(len(loop_params)):
                     for j in range(loop_params[i]):
                               required_output_file_location=EOS+'/'+path+'/Temp_'+pfx+'_'+ID+'_'+str(i)+'/'+pfx+'_'+ID+'_'+o+'_'+str(i)+'_'+str(j)+sfx
                               bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               if os.path.isfile(required_output_file_location)!=True:
                                  bad_pop.append([OH+[' --i ', ' --j ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OL+[i, j, path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])

             if nest_lvl==3:
                 for i in range(len(loop_params)):
                     for j in range(len(loop_params[i])):
                         for k in range(loop_params[i][j]):
                               required_output_file_location=EOS+'/'+path+'/Temp_'+pfx+'_'+ID+'_'+str(i)+'/'+pfx+'_'+ID+'_'+o+'_'+str(i)+'_'+str(j) + '_' + str(k)+sfx
                               bar.text = f'-> Checking whether the file : {required_output_file_location}, exists...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '_' + str(k) +'.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '_' + str(k) +'.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j) + '_' + str(k)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               if os.path.isfile(required_output_file_location)!=True:
                                  bad_pop.append([OH+[' --i ', ' --j ',' --k ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OL+[i, j,k, path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, 1, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
        return(bad_pop)
    else:
        from alive_progress import alive_bar
        bad_pop=[]
        TotJobs=0
        if type(loop_params) is int:
            nest_lvl=1
            TotJobs=loop_params
        elif type(loop_params[0]) is int:
            nest_lvl=2
            TotJobs=np.sum(loop_params)
        elif type(loop_params[0][0]) is int:
            nest_lvl=3
            for lp in loop_params:
                TotJobs+=np.sum(lp)
        OH=OptionHeader+[' --EOS '," --AFS ", " --PY ", " --BatchID "]
        OL=OptionLine+[EOS, AFS, PY, ID]
        TotJobs=int(TotJobs)
        with alive_bar(TotJobs,force_tty=True, title='Checking the results from HTCondor') as bar:
             if nest_lvl==2:
                 for i in range(len(loop_params)):
                               bar.text = f'-> Preparing batch submission...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) + '.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               bad_pop.append([OH+[' --i ', ' --j ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OL+[i, '$1', path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, loop_params[i], 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
             if nest_lvl==3:
                 for i in range(len(loop_params)):
                     for j in range(len(loop_params[i])):
                               bar.text = f'-> Preparing batch submission...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SH_'+pfx+'_'+ ID+'_' + str(i) + '_' + str(j)+'.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(i)+'/SUB_'+pfx+'_'+ ID+'_' + str(i) +'_' + str(j)+'.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(i)+'/MSG_'+pfx+'_'+ ID+'_' + str(i) +'_' + str(j)
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               bad_pop.append([OH+[' --i ', ' --j ',' --k ', ' --p ', ' --o ',' --pfx ', ' --sfx '],OL+[i, j, '$1', path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, loop_params[i][j], 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
             if nest_lvl==1:
                               bar.text = f'-> Preparing batch submission...'
                               bar()
                               SHName = AFS + '/HTCondor/SH/Temp_'+pfx+'_'+ID+'_'+str(0)+'/SH_'+pfx+'_'+ ID+'.sh'
                               SUBName = AFS + '/HTCondor/SUB/Temp_'+pfx+'_'+ID+'_'+str(0)+'/SUB_'+pfx+'_'+ ID+'.sub'
                               MSGName = AFS + '/HTCondor/MSG/Temp_'+pfx+'_'+ID+'_'+str(0)+'/MSG_'+pfx+'_'+ ID
                               ScriptName = AFS + '/Code/Utilities/'+Sub_File
                               bad_pop.append([OH+[' --i ', ' --p ', ' --o ',' --pfx ', ' --sfx '], OL+['$1', path,o, pfx, sfx], SHName, SUBName, MSGName, ScriptName, loop_params, 'ANNDEA-'+pfx+'-'+ID, Log,GPU])
        return(bad_pop)
   return []
def SubmitJobs2Condor(job,local=False,ExtCPU=False,JobFlavour='workday'):
    if local:
       OptionLine = job[0][0]+str(job[1][0])
       for line in range(1,len(job[0])):
                OptionLine+=job[0][line]
                OptionLine+=str(job[1][line])
                TotalLine = 'python3 ' + job[5] + OptionLine
       submission_line='python3 '+job[5]+OptionLine
       for j in range(0,job[6]):
         act_submission_line=submission_line.replace('$1',str(j))
         subprocess.call([act_submission_line],shell=True)
         print(bcolors.OKGREEN+act_submission_line+" has been successfully executed"+bcolors.ENDC)
    else:
        SHName = job[2]
        SUBName = job[3]
        if job[8]:
            MSGName=job[4]
        OptionLine = job[0][0]+str(job[1][0])
        for line in range(1,len(job[0])):
            OptionLine+=job[0][line]
            OptionLine+=str(job[1][line])
        f = open(SUBName, "w")
        f.write("executable = " + SHName)
        f.write("\n")
        if job[8]:
            f.write("output ="+MSGName+".out")
            f.write("\n")
            f.write("error ="+MSGName+".err")
            f.write("\n")
            f.write("log ="+MSGName+".log")
            f.write("\n")
        f.write('requirements = (CERNEnvironment =!= "qa")')
        f.write("\n")
        if job[9]:
            f.write('request_gpus = 1')
            f.write("\n")
        if ExtCPU and job[9]==False:
            f.write('RequestCpus = 4')
            f.write("\n")
        f.write('arguments = $(Process)')
        f.write("\n")
        f.write('+SoftUsed = '+'"'+job[7]+'"')
        f.write("\n")
        f.write('transfer_output_files = ""')
        f.write("\n")
        f.write('+JobFlavour = "'+JobFlavour+'"')
        f.write("\n")
        f.write('queue ' + str(job[6]))
        f.write("\n")
        f.close()
        TotalLine = 'python3 ' + job[5] + OptionLine
        f = open(SHName, "w")
        f.write("#!/bin/bash")
        f.write("\n")
        f.write("set -ux")
        f.write("\n")
        f.write(TotalLine)
        f.write("\n")
        f.close()
        subprocess.call(['condor_submit', SUBName])
        print(TotalLine, " has been successfully submitted")



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

def LoadRenderImages(Seeds,StartSeed,EndSeed,num_classes=2):
    import tensorflow as tf
    NewSeeds=Seeds[StartSeed-1:min(EndSeed,len(Seeds))]
    ImagesY=np.empty([len(NewSeeds),1])

    ImagesX=np.empty([len(NewSeeds),NewSeeds[0].H,NewSeeds[0].W,NewSeeds[0].L],dtype=np.bool)
    for im in range(len(NewSeeds)):
        if hasattr(NewSeeds[im],'Label'):
           ImagesY[im]=int(float(NewSeeds[im].Label))
        else:
           ImagesY[im]=0
        BlankRenderedImage=[]
        for x in range(-NewSeeds[im].bX,NewSeeds[im].bX):
          for y in range(-NewSeeds[im].bY,NewSeeds[im].bY):
            for z in range(0,NewSeeds[im].bZ):
             BlankRenderedImage.append(0)
        RenderedImage = np.array(BlankRenderedImage)
        RenderedImage = np.reshape(RenderedImage,(NewSeeds[im].H,NewSeeds[im].W,NewSeeds[im].L))
        for Hits in NewSeeds[im].TrackPrint:
                   RenderedImage[Hits[0]+NewSeeds[im].bX][Hits[1]+NewSeeds[im].bY][Hits[2]]=1
        ImagesX[im]=RenderedImage
    ImagesX= ImagesX[..., np.newaxis]
    ImagesY=tf.keras.utils.to_categorical(ImagesY,num_classes)
    return (ImagesX,ImagesY)

def ManageTempFolders(spi,op_type):
    if type(spi[1][8]) is int:
       _tot=spi[1][8]
    else:
       _tot=len(spi[1][8])
    if op_type=='Create':
       for i in range(_tot):
           try:
              os.mkdir(spi[1][1]+spi[1][3]+'Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i))
           except OSError as error:
              continue
           try:
              os.mkdir(spi[1][0]+'/HTCondor/SUB/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i))
           except OSError as error:
              continue
           try:
              os.mkdir(spi[1][0]+'/HTCondor/SH/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i))
           except OSError as error:
              continue
           try:
              os.mkdir(spi[1][0]+'/HTCondor/MSG/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i))
           except OSError as error:
              continue
       return 'Temporary folders have been created'
    if op_type=='Delete':
       for i in range(_tot):
           shutil.rmtree(spi[1][1]+spi[1][3]+'Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i),True)
           shutil.rmtree(spi[1][0]+'/HTCondor/SUB/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i),True)
           shutil.rmtree(spi[1][0]+'/HTCondor/SH/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i),True)
           shutil.rmtree(spi[1][0]+'/HTCondor/MSG/Temp_'+spi[1][5]+'_'+spi[1][7]+'_'+str(i),True)
       #return spi[1][8]
       return 'Temporary folders have been deleted'

