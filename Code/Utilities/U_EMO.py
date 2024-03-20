###This file contains the utility functions that are commonly used in ANNDEA packages

import math
import numpy as np
import copy
import ast

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
      def GetVXInfo(self):
          if hasattr(self,'Hits'):
             if self.Partition==2:
                __XZ1=EMO.GetEquationOfTrack(self.Hits[0])[0]
                __XZ2=EMO.GetEquationOfTrack(self.Hits[1])[0]
                __YZ1=EMO.GetEquationOfTrack(self.Hits[0])[1]
                __YZ2=EMO.GetEquationOfTrack(self.Hits[1])[1]
                __X1S=EMO.GetEquationOfTrack(self.Hits[0])[3]
                __X2S=EMO.GetEquationOfTrack(self.Hits[1])[3]
                __Y1S=EMO.GetEquationOfTrack(self.Hits[0])[4]
                __Y2S=EMO.GetEquationOfTrack(self.Hits[1])[4]
                __Z1S=EMO.GetEquationOfTrack(self.Hits[0])[5]
                __Z2S=EMO.GetEquationOfTrack(self.Hits[1])[5]
                __vector_1_st = np.array([np.polyval(__XZ1,self.Hits[0][0][2]),np.polyval(__YZ1,self.Hits[0][0][2]),self.Hits[0][0][2]])
                __vector_1_end = np.array([np.polyval(__XZ1,self.Hits[0][len(self.Hits[0])-1][2]),np.polyval(__YZ1,self.Hits[0][len(self.Hits[0])-1][2]),self.Hits[0][len(self.Hits[0])-1][2]])
                __vector_2_st = np.array([np.polyval(__XZ2,self.Hits[0][0][2]),np.polyval(__YZ2,self.Hits[0][0][2]),self.Hits[0][0][2]])
                __vector_2_end = np.array([np.polyval(__XZ2,self.Hits[0][len(self.Hits[0])-1][2]),np.polyval(__YZ2,self.Hits[0][len(self.Hits[0])-1][2]),self.Hits[0][len(self.Hits[0])-1][2]])
                __result=EMO.closestDistanceBetweenLines(__vector_1_st,__vector_1_end,__vector_2_st,__vector_2_end,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False)
                __midpoint=(__result[0]+__result[1])/2
                __D1M=math.sqrt(((__midpoint[0]-__X1S)**2) + ((__midpoint[1]-__Y1S)**2) + ((__midpoint[2]-__Z1S)**2))
                __D2M=math.sqrt(((__midpoint[0]-__X2S)**2) + ((__midpoint[1]-__Y2S)**2) + ((__midpoint[2]-__Z2S)**2))
                __v1=np.subtract(__vector_1_end,__midpoint)
                __v2=np.subtract(__vector_2_end,__midpoint)
                self.angle=EMO.angle_between(__v1, __v2)
                self.Vx=__midpoint[0]
                self.Vy=__midpoint[1]
                self.Vz=__midpoint[2]
                self.DOCA=__result[2]
                self.V_Tr=[__D1M,__D2M]
                self.Tr_Tr=math.sqrt(((float(self.Hits[0][0][0])-float(self.Hits[1][0][0]))**2)+((float(self.Hits[0][0][1])-float(self.Hits[1][0][1]))**2)+((float(self.Hits[0][0][2])-float(self.Hits[1][0][2]))**2))
             else:
                 raise ValueError("Method 'DecorateSeedGeoInfo' currently works for seeds with track multiplicity of 2 only")
          else:
                raise ValueError("Method 'DecorateSeedGeoInfo' works only if 'DecorateTracks' method has been acted upon the seed before")
        #         __XZ1=EMO.GetEquationOfTrack(self.Hits[0])[0]
        #         __XZ2=EMO.GetEquationOfTrack(self.Hits[1])[0]
        #         __YZ1=EMO.GetEquationOfTrack(self.Hits[0])[1]
        #         __YZ2=EMO.GetEquationOfTrack(self.Hits[1])[1]
        #         __vector_1_st = np.array([np.polyval(__XZ1,self.Hits[0][0][2]),np.polyval(__YZ1,self.Hits[0][0][2]),self.Hits[0][0][2]])
        #         __vector_1_end = np.array([np.polyval(__XZ1,self.Hits[0][len(self.Hits[0])-1][2]),np.polyval(__YZ1,self.Hits[0][len(self.Hits[0])-1][2]),self.Hits[0][len(self.Hits[0])-1][2]])
        #         __vector_2_st = np.array([np.polyval(__XZ2,self.Hits[0][0][2]),np.polyval(__YZ2,self.Hits[0][0][2]),self.Hits[0][0][2]])
        #         __vector_2_end = np.array([np.polyval(__XZ2,self.Hits[0][len(self.Hits[0])-1][2]),np.polyval(__YZ2,self.Hits[0][len(self.Hits[0])-1][2]),self.Hits[0][len(self.Hits[0])-1][2]])
        #         __result=EMO.closestDistanceBetweenLines(__vector_1_st,__vector_1_end,__vector_2_st,__vector_2_end,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False)
        #         __midpoint=(__result[0]+__result[1])/2
        #         __v1=np.subtract(__vector_1_end,__midpoint)
        #         __v2=np.subtract(__vector_2_end,__midpoint)
        #         if self.Hits[0][len(self.Hits)-1][2]>self.Hits[1][len(self.Hits)-1][2]: #Workout which track is leading (has highest z-coordinate)
        #             __leading_seg=0
        #             __subleading_seg=1
        #         else:
        #             __leading_seg=1
        #             __subleading_seg=0
        #         self.Opening_Angle=EMO.angle_between(__v1, __v2)
        #         self.DOCA=__result[2]
        #         __x2=float(self.Hits[__leading_seg][0][0])
        #         __x1=self.Hits[__subleading_seg][len(self.Hits[__subleading_seg])-1][0]
        #         __y2=float(self.Hits[__leading_seg][0][1])
        #         __y1=self.Hits[__subleading_seg][len(self.Hits[__subleading_seg])-1][1]
        #      else:
        #          raise ValueError("Method 'DecorateTrackGeoInfo' currently works for seeds with partition of 2 only")
        #   else:
        #         raise ValueError("Method 'DecorateTrackGeoInfo' works only if 'Decorate' method has been acted upon the seed before")  
      def TrackQualityCheck(self,MaxDoca,MaxSLG, MaxSTG,MaxAngle):
                    if self.DOCA>MaxDoca: #Check whether the seed passes the DOCA cut
                        return False
                    else:
                        if abs(self.Opening_Angle)>MaxAngle: #Check whether the seed passes the Angle cut
                            return False
                        else:
                            if MaxSLG>=0: #Non Overlapping track situation
                                if self.SLG<0: #We don't care about overlapping tracks
                                   return False
                                else:
                                    if self.SLG>=MaxSLG: #Non-overlapping tracks: max gap cut
                                        return False
                                    else:
                                        return self.STG <= MaxSTG+(self.SLG*0.96) #Final cut on transverse displacement
                            else: #Overalpping track situation
                                if self.SLG >= 0: #Discard non-overlapping tracks
                                   return False
                                else:
                                   if self.SLG < MaxSLG: #We apply the cut on the negative value
                                        return False
                                   else:
                                        return self.STG <= MaxSTG #Still apply the STG cut
                                   ######
      def VertexQualityCheck(self,MaxDoca, MaxVXT, MaxAngle, FiducialVolumeCut):
          if len(FiducialVolumeCut) >= 6:
                MinX = FiducialVolumeCut[0] 
                MaxX = FiducialVolumeCut[1]
                MinY = FiducialVolumeCut[2]
                MaxY = FiducialVolumeCut[3]
                MinZ = FiducialVolumeCut[4] 
                MaxZ = FiducialVolumeCut[5]
                return (self.DOCA<=MaxDoca and min(self.V_Tr)<=MaxVXT and self.Vx>=MinX and self.Vx<=MaxX and self.Vy>=MinY and self.Vy<=MaxY and self.Vz>=MinZ and self.Vz<=MaxZ and abs(self.angle)<=MaxAngle)
          return (self.DOCA<=MaxDoca and min(self.V_Tr)<=MaxVXT and abs(self.angle)<=MaxAngle)
    
      def PrepareSeedPrint(self,MM):
          __TempTrack=copy.deepcopy(self.Hits)

          self.Resolution=MM.ModelParameters[-1][3]
          self.bX=int(round(MM.ModelParameters[-1][0]/self.Resolution,0))
          self.bY=int(round(MM.ModelParameters[-1][1]/self.Resolution,0))
          self.bZ=int(round(MM.ModelParameters[-1][2]/self.Resolution,0))
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
              __dUpX=MM.ModelParameters[-1][0]-max(__X)
              __dDownX=MM.ModelParameters[-1][0]+min(__X)
              __dX=(__dUpX+__dDownX)/2
              __xshift=__dUpX-__dX
              __X=[]
              for __Tracks in __TempTrack:
                 for __hits in __Tracks:
                     __hits[0]=__hits[0]+__xshift
                     __X.append(__hits[0])
             ##########Y
              __dUpY=MM.ModelParameters[-1][1]-max(__Y)
              __dDownY=MM.ModelParameters[-1][1]+min(__Y)
              __dY=(__dUpY+__dDownY)/2
              __yshift=__dUpY-__dY
              __Y=[]
              for __Tracks in __TempTrack:
                 for __hits in __Tracks:
                     __hits[1]=__hits[1]+__yshift
                     __Y.append(__hits[1])
              __min_scale=max(max(__X)/(MM.ModelParameters[-1][0]-(2*self.Resolution)),max(__Y)/(MM.ModelParameters[-1][1]-(2*self.Resolution)), max(__Z)/(MM.ModelParameters[-1][2]-(2*self.Resolution)))
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
                    __ThetaAngle=EMO.angle_between(__vector_1, __vector_2)
                   except:
                     __ThetaAngle=0.0
                   try:
                     __vector_1 = [__deltaZ,0]
                     __vector_2 = [__deltaZ, __deltaY]
                     __PhiAngle=EMO.angle_between(__vector_1, __vector_2)
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
      def AssignANNVxUID(self,ID):
          self.UVxID=ID
      def PrepareSeedGraph(self,MM):
          if MM.ModelArchitecture[-5:]=='4N-IC':
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
          if MM.ModelArchitecture[-5:]=='6N-IC':
                      __TempTrack=copy.deepcopy(self.Hits)
                      for __Tracks in __TempTrack:
                              for h in range(len(__Tracks)):
                                  __Tracks[h]=__Tracks[h][:5]

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
                      __graphData_x = pd.DataFrame (__graphData_x, columns = ['x', 'y', 'z', 'tx' , 'ty' , 'TrackID', 'NodeIndex'])
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
                      __graphData_join.drop(['x_l', 'y_l', 'z_l', 'x_r', 'y_r', 'z_r', 'dummy','tx_l','ty_l','tx_r','ty_r'], axis = 1, inplace = True)

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
                        __graphData_pos.append(node[0:5])
                      for g in __graphData_nodes:
                        g.append(g[5]+'-'+str(g[6]))
                        g[5]=float(g[5])
                      Data_x = []
                      for g in __graphData_nodes:
                        Data_x.append(g[:6])
                      node_ind_list=[]
                      for g in __graphData_nodes:
                        node_ind_list.append(g[7])
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
          if MM.ModelArchitecture[-5:]=='5N-FC':
                      __TempTrack=copy.deepcopy(self.Hits)
                      for __Tracks in __TempTrack:
                              for h in range(len(__Tracks)):
                                  __Tracks[h]=__Tracks[h][:5]

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
# Graph representation v1 (not fully connected)
                      try:
                          __graphData_x =__TempTrack[0]+__TempTrack[1]
                      except:
                          __graphData_x =__TempTrack[0]

                      __graphData_pos = []
                      for node in __graphData_x:
                        # hit attributes [x,y,z,tx,ty]
                        # the (x,y,z) attributes are hit[0:3]
                        __graphData_pos.append(node[0:3])

                      # edge index and attributes
                      __graphData_edge_index = []
                      __graphData_edge_attr = []
                                #fully connected
                      for i in range(len(__TempTrack[0])+len(__TempTrack[1])):
                        for j in range(0,i):
                            # the edges are diretced, so i->j and j->i are different edges
                            __graphData_edge_index.append([i,j])
                            # the vector i->j in 3D space are set as edge attribute
                            __graphData_edge_attr.append(np.array(__graphData_pos[j]) - np.array(__graphData_pos[i]))
                            __graphData_edge_index.append([j,i])
                            __graphData_edge_attr.append(np.array(__graphData_pos[i]) - np.array(__graphData_pos[j]))
                      if(hasattr(self, 'Label')):
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
                          self.GraphSeed = Data(x=torch.Tensor(__graphData_x), edge_index = torch.Tensor(__graphData_edge_index).t().contiguous().long(), edge_attr = torch.Tensor(__graphData_edge_attr),y=torch.Tensor([__graphData_y]), pos = torch.Tensor(__graphData_pos))
                      else:
                          import torch
                          import torch_geometric
                          from torch_geometric.data import Data
                          self.GraphSeed = Data(x=torch.Tensor(__graphData_x), edge_index = torch.Tensor(__graphData_edge_index).t().contiguous().long(), edge_attr = torch.Tensor(__graphData_edge_attr), pos = torch.Tensor(__graphData_pos))
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

          plt.title('Seed:'.join(self.Header))
          if hasattr(self,'Label'):
              plt.suptitle('Label:'+str(self.Label))
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
             del self.GraphSeed
          return self.Fit>=0.5
      def ClassifySeed(self,Mmeta,M):
          if Mmeta.ModelType=='CNN':
             EMO.PrepareSeedPrint(self,Mmeta)
             __Image=LoadRenderImages([self],1,1)[0]
             self.Class=M.predict(__Image)[0]
             print(self.Class)
             x=input()
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
      def InjectSeed(self,OtherSeed):
          __overlap=False
          for t1 in self.Header:
              for t2 in OtherSeed.Header:
                  if t1==t2:
                      __overlap=True
                      break
          if __overlap:
              overlap_matrix=[]
              for t1 in range(len(self.Header)):
                 for t2 in range(len(OtherSeed.Header)):
                    if self.Header[t1]==OtherSeed.Header[t2]:
                       overlap_matrix.append(t2)

              for t2 in range(len(OtherSeed.Header)):
                if (t2 in overlap_matrix)==False:
                  self.Header.append(OtherSeed.Header[t2])
                  if hasattr(self,'Hits') and hasattr(OtherSeed,'Hits'):
                          self.Hits.append(OtherSeed.Hits[t2])
              if hasattr(self,'Label') and hasattr(OtherSeed,'Label'):
                         self.Label=(self.Label and OtherSeed.Label)
              if hasattr(self,'FIT') and hasattr(OtherSeed,'FIT'):
                        self.FIT+=OtherSeed.FIT
              elif hasattr(self,'FIT'):
                       self.FIT.append(OtherSeed.Fit)
              elif hasattr(OtherSeed,'FIT'):
                       self.FIT=[self.Fit]
                       self.FIT+=OtherSeed.FIT
              else:
                      self.FIT=[]
                      self.FIT.append(self.Fit)
                      self.FIT.append(OtherSeed.Fit)
              if hasattr(self,'VX_x') and hasattr(OtherSeed,'VX_x'):
                      self.VX_x+=OtherSeed.VX_x
              elif hasattr(self,'VX_x'):
                       self.VX_x.append(OtherSeed.Vx)
              elif hasattr(OtherSeed,'VX_x'):
                       self.VX_x=[self.Vx]
                       self.VX_x+=OtherSeed.VX_x
              else:
                      self.VX_x=[]
                      self.VX_x.append(self.Vx)
                      self.VX_x.append(OtherSeed.Vx)
                          
              if hasattr(self,'VX_y') and hasattr(OtherSeed,'VX_y'):
                      self.VX_y+=OtherSeed.VX_y
              elif hasattr(self,'VX_y'):
                       self.VX_y.append(OtherSeed.Vy)
              elif hasattr(OtherSeed,'VX_y'):
                       self.VX_y=[self.Vy]
                       self.VX_y+=OtherSeed.VX_y
              else:
                      self.VX_y=[]
                      self.VX_y.append(self.Vy)
                      self.VX_y.append(OtherSeed.Vy)
                  
              if hasattr(self,'VX_z') and hasattr(OtherSeed,'VX_z'):
                      self.VX_z+=OtherSeed.VX_z
              elif hasattr(self,'VX_z'):
                       self.VX_z.append(OtherSeed.Vz)
              elif hasattr(OtherSeed,'VX_z'):
                       self.VX_z=[self.Vz]
                       self.VX_z+=OtherSeed.VX_z
              else:
                      self.VX_z=[]
                      self.VX_z.append(self.Vz)
                      self.VX_z.append(OtherSeed.Vz)
              self.VX_z=list(set(self.VX_z))
              self.VX_x=list(set(self.VX_x))
              self.VX_y=list(set(self.VX_y))
              self.FIT=list(set(self.FIT))
              self.Partition=len(self.Header)
              self.Fit=sum(self.FIT)/len(self.FIT)
              self.Vx=sum(self.VX_x)/len(self.VX_x)
              self.Vy=sum(self.VX_y)/len(self.VX_y)
              self.Vz=sum(self.VX_z)/len(self.VX_z)
              if hasattr(self,'angle'):
                  delattr(self,'angle')
              if hasattr(self,'DOCA'):
                  delattr(self,'DOCA')
              if hasattr(self,'V_Tr'):
                  delattr(self,'V_Tr')
              if hasattr(self,'Tr_Tr'):
                  delattr(self,'Tr_Tr')



              return True

          else:
              return __overlap

      def InjectTrackSeed(self,OtherSeed):
          Overlap=list(set(self.Header) & set(OtherSeed.Header)) #Basic check - does the other seed have at least one track segment in common?

          if len(Overlap)==0: #If not, we don't care and no injection is required
                return False
          elif len(Overlap)==1: #The scenario where there is one track segment in common
                ovlp_index=[self.Header.index(Overlap[0]),OtherSeed.Header.index(Overlap[0])]
                sh_remove=[]
                oh_injest=[]
                lost_fit=0
                pot_fit=0
                for sh in range(len(self.Header)):
                    for oh in range(len(OtherSeed.Header)):
                        if sh!=ovlp_index[0] and oh!=ovlp_index[1]:
                            # print('Combo:',self.Header[sh],OtherSeed.Header[oh])
                            if EMO.HitOverlap(self.Hits[sh],OtherSeed.Hits[oh]):
                               lost_fit+=self.FIT[sh]
                               pot_fit+=OtherSeed.FIT[oh]
                               sh_remove.append(sh)
                               oh_injest.append(oh)
                            else:
                               oh_injest.append(oh)
                oh_injest=list(set(oh_injest))
                if lost_fit==pot_fit==0:
                   for oh in oh_injest:
                       self.Header.append(OtherSeed.Header[oh])
                       self.Hits.append(OtherSeed.Hits[oh])
                       self.FIT.append(OtherSeed.FIT[oh])
                elif lost_fit<pot_fit:
                    inx_drop=0
                    for sh in sh_remove:
                       self.Header.pop(sh-inx_drop)
                       self.Hits.pop(sh-inx_drop)
                       self.FIT.pop(sh-inx_drop)
                       inx_drop+=1
                    for oh in oh_injest:
                       self.Header.append(OtherSeed.Header[oh])
                       self.Hits.append(OtherSeed.Hits[oh])
                       self.FIT.append(OtherSeed.FIT[oh])
                return True
          else:
              self.FIT[(self.Header.index(Overlap[0]))]+=OtherSeed.FIT[(OtherSeed.Header.index(Overlap[0]))]
              self.FIT[(self.Header.index(Overlap[1]))]+=OtherSeed.FIT[(OtherSeed.Header.index(Overlap[1]))]
              return True
      def InjectDistantTrackSeed(self,OtherSeed):

          _IniTrace=[OtherSeed.Header,OtherSeed.Hits,OtherSeed.FIT]

          self_matx=EMO.DensityMatrix(OtherSeed.Header,self.Header)
          if EMO.Overlap(self_matx)==False:
              return EMO.Overlap(self_matx)
          _ovl=EMO.Overlap(self_matx)
          _smatr=self_matx
          _PostTrace=[self.Header,self.Hits,self.FIT]
          new_seed_header=EMO.ProjectVectorElements(self_matx,self.Header)
          _new_sd_hd=copy.deepcopy(new_seed_header)
          new_self_hits=EMO.ProjectVectorElements(self_matx,self.Hits)
          _new_seed_hits=copy.deepcopy(new_self_hits)
          new_self_fit=EMO.ProjectVectorElements(self_matx,self.FIT)
          _new_seed_fits=copy.deepcopy(new_self_fit)
          remain_1_s = EMO.GenerateInverseVector(self.Header,new_seed_header)
          _new_remain_s=copy.deepcopy(remain_1_s)
          remain_1_o = EMO.GenerateInverseVector(OtherSeed.Header,new_seed_header)
          _new_remain_o=copy.deepcopy(remain_1_o)
          OtherSeed.Header=EMO.ProjectVectorElements([remain_1_o],OtherSeed.Header)
          _other_seed_header2=copy.deepcopy(OtherSeed.Header)

          self.Header=EMO.ProjectVectorElements([remain_1_s],self.Header)
          _self_seed_header2=copy.deepcopy(self.Header)
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
          _other_seed_header3=copy.deepcopy(OtherSeed.Header)
          _other_seed_hits3=copy.deepcopy(OtherSeed.Hits)
          _self_seed_header3=copy.deepcopy(self.Header)
          _self_seed_hits3=copy.deepcopy(self.Hits)

          self_2_matx=EMO.DensityMatrix(OtherSeed.Hits,self.Hits)
          other_2_matx=EMO.DensityMatrix(self.Hits,OtherSeed.Hits)
          _self_m2=copy.deepcopy(self_2_matx)
          _other_m2=copy.deepcopy(other_2_matx)
          last_s_seed_header=EMO.ProjectVectorElements(self_2_matx,self.Header)
          last_o_seed_header=EMO.ProjectVectorElements(other_2_matx,OtherSeed.Header)
          remain_2_s = EMO.GenerateInverseVector(self.Header,last_s_seed_header)
          remain_2_o = EMO.GenerateInverseVector(OtherSeed.Header,last_o_seed_header)
          _new_remain2_s=copy.deepcopy(remain_2_s)
          _new_remain2_o=copy.deepcopy(remain_2_o)
          new_seed_header+=EMO.ProjectVectorElements([remain_2_s],self.Header)
          new_seed_header+=EMO.ProjectVectorElements([remain_2_o],OtherSeed.Header)
          _new_sd_hd2=copy.deepcopy(new_seed_header)
          new_self_fit+=EMO.ProjectVectorElements([remain_2_s],self.FIT)
          new_self_fit+=EMO.ProjectVectorElements([remain_2_o],OtherSeed.FIT)
          new_self_hits+=EMO.ProjectVectorElements([remain_2_s],self.Hits)
          new_self_hits+=EMO.ProjectVectorElements([remain_2_o],OtherSeed.Hits)
          _new_seed_hits2=copy.deepcopy(new_self_hits)


          last_remain_headers_s = EMO.GenerateInverseVector(self.Header,new_seed_header)
          last_remain_headers_o = EMO.GenerateInverseVector(OtherSeed.Header,new_seed_header)
          last_self_headers=EMO.ProjectVectorElements([last_remain_headers_s],self.Header)
          last_other_headers=EMO.ProjectVectorElements([last_remain_headers_o],OtherSeed.Header)
          _last_remaining_sheaders2=copy.deepcopy(last_self_headers)
          _last_remaining_oheaders2=copy.deepcopy(last_other_headers)
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
          _last_remaining_shits2=copy.deepcopy(last_self_hits)
          _last_remaining_ohits2=copy.deepcopy(last_other_hits)
          last_self_fits=EMO.ProjectVectorElements([last_remain_headers_s],self.FIT)
          last_other_fits=EMO.ProjectVectorElements([last_remain_headers_o],OtherSeed.FIT)
          last_remain_matr=EMO.DensityMatrix(last_other_hits,last_self_hits)
          _last_remaining_matr2=copy.deepcopy(last_remain_matr)
          _weak=EMO.ReplaceWeakerTracks(last_remain_matr,last_other_hits,last_self_hits,last_other_fits,last_self_fits)
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
          if len(self.Header)!=len(self.Hits):
              # print('Error',self.Header,self.Hits)
              # print('ErrorFit',self.FIT,OtherSeed.FIT)
              # print('IniTrace',_IniTrace)
              # print('PostTrace',_PostTrace)
              # print('Overlap',_ovl)
              # print('New Seed Header',_new_sd_hd)
              # print('New Seed Hits',_new_seed_hits)
              # print('New seed fits',_new_seed_fits)
              # print('Remain_s',_new_remain_s)
              # print('Remain_o',_new_remain_o)
              # print('_other_seed_header2',_other_seed_header2)
              # print('_self_seed_header3',_self_seed_header3)
              # print('_self_seed_hits3',_self_seed_hits3)
              # print('_other_seed_header3',_other_seed_header3)
              # print('_other_seed_hits3',_other_seed_hits3)
              # print('Remain2_s',_new_remain2_s)
              # print('Remain2_o',_new_remain2_o)
              # print('_self_m2',_self_m2)
              # print('_other_m2',_other_m2)
              # print('New Seed Header2',_new_sd_hd2)
              # print('New Seed Hits2',_new_seed_hits2)
              # print('_last_remaining_sheaders2',_last_remaining_sheaders2)
              # print('_last_remaining_oheaders2',_last_remaining_oheaders2)
              # print('_last_remaining_shits2',_last_remaining_shits2)
              # print('_last_remaining_ohits2',_last_remaining_ohits2)
              # print('_last_remaining_matr2',_last_remaining_matr2)
              # print('weak',_weak)
              # print('weak2',EMO.ReplaceWeakerTracksTest(last_remain_matr,last_other_hits,last_self_hits,last_other_fits,last_self_fits))
              # print('weakhdr',EMO.ReplaceWeakerTracks(last_remain_matr,last_other_headers,last_self_headers,last_other_fits,last_self_fits))
              # print('Matrx',EMO.ProjectVectorElements(_smatr,_PostTrace[0]))
              # print('matrix',_smatr)
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
      def HitOverlap(a,b):

                 min_a=min(a)
                 min_b=min(b)
                 max_a=max(a)
                 max_b=max(b)
                 if (min_b>=min_a) and (max_b<=max_a):
                     return(True)
                 elif (min_a>=min_b) and (max_a<=max_b):
                     return(True)
                 elif (max_a>min_b) and (max_a<max_b):
                     return(True)
                 elif (max_b>min_a) and (max_b<max_a):
                     return(True)
                 return(False)
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
                     a_temp.append(el)
                 for el in b:
                     b_temp.append(el)
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
                     a_temp.append(el)
                 for el in b:
                     b_temp.append(el)
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
      def ReplaceWeakerTracksTest(matx,m,f,m_fit,f_fit):
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
                          if (mel in res_vector) and (mel in final_vector)==False:
                             final_vector.append(mel)
                      for fel in f:
                          if (fel in delete_vec)==False and (fel in final_vector)==False:
                             final_vector.append(fel)
                      return(final_vector)
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
                          if (mel in res_vector) and (mel in final_vector)==False:
                             final_vector.append(mel)
                      for fel in f:
                          if (fel in delete_vec)==False and (fel in final_vector)==False:
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

def LoadRenderImages(Seeds,StartSeed,EndSeed,num_classes=2):
    import tensorflow as tf
    NewSeeds=Seeds[StartSeed-1:min(EndSeed,len(Seeds))]
    ImagesY=np.empty([len(NewSeeds),1])

    ImagesX=np.empty([len(NewSeeds),NewSeeds[0].H,NewSeeds[0].W,NewSeeds[0].L],dtype=np.float16)
    for im in range(len(NewSeeds)):
        if num_classes>1:
            if hasattr(NewSeeds[im],'Label'):
               ImagesY[im]=int(float(NewSeeds[im].Label))
            else:
               ImagesY[im]=0
        else:
            if hasattr(NewSeeds[im],'Label'):
               ImagesY[im]=float(NewSeeds[im].Label)
            else:
               ImagesY[im]=0.0
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
    if num_classes>1:
        ImagesY=tf.keras.utils.to_categorical(ImagesY,num_classes)
    return (ImagesX,ImagesY)


