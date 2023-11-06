###This file contains the standard UI utility functions that are commonly used in ANNDEA packages

#Libraries used
import numpy as np
import pandas as pd
from alive_progress import alive_bar

#Alignment Functions

#Defining some relevant functions
def FitPlate(PlateZ,dx,dy,input_data,Track_ID,ProgressBar):
    change_df = pd.DataFrame([[PlateZ,dx,dy]], columns = ['Plate_ID','dx','dy'])
    temp_data=input_data[[Track_ID,'x','y','z','Track_No','Plate_ID']]
    temp_data=pd.merge(temp_data,change_df,on='Plate_ID',how='left')
    temp_data['dx'] = temp_data['dx'].fillna(0.0)
    temp_data['dy'] = temp_data['dy'].fillna(0.0)
    temp_data['x']=temp_data['x']+temp_data['dx']
    temp_data['y']=temp_data['y']+temp_data['dy']
    temp_data=temp_data[[Track_ID,'x','y','z','Track_No']]
    Tracks_Head=temp_data[[Track_ID]]
    Tracks_Head.drop_duplicates(inplace=True)
    Tracks_List=temp_data.values.tolist() #I find it is much easier to deal with tracks in list format when it comes to fitting
    Tracks_Head=Tracks_Head.values.tolist()
    #Bellow we build the track representatation that we can use to fit slopes
    if ProgressBar:
        with alive_bar(len(Tracks_Head)*2,force_tty=True, title='Spatially fitting data...') as bar:
            for bth in Tracks_Head:
                       bth.append([])
                       bt=0
                       trigger=False
                       bar()
                       while bt<(len(Tracks_List)):
                           if bth[0]==Tracks_List[bt][0]:

                               bth[1].append(Tracks_List[bt][1:4])
                               del Tracks_List[bt]
                               bt-=1
                               trigger=True
                           elif trigger:
                                break
                           else:
                                continue
                           bt+=1
            for bth in Tracks_Head:
               x,y,z=[],[],[]
               for b in bth[1]:
                   x.append(b[0])
                   y.append(b[1])
                   z.append(b[2])
               tx=np.polyfit(z,x,1)[0]
               ax=np.polyfit(z,x,1)[1]
               ty=np.polyfit(z,y,1)[0]
               ay=np.polyfit(z,y,1)[1]
               bth.append(ax) #Append x intercept
               bth.append(tx) #Append x slope
               bth.append(0) #Append a placeholder slope (for polynomial case)
               bth.append(ay) #Append x intercept
               bth.append(ty) #Append x slope
               bth.append(0) #Append a placeholder slope (for polynomial case)
               del(bth[1])
               bar()
    else:
            for bth in Tracks_Head:
                       bth.append([])
                       bt=0
                       trigger=False
                       while bt<(len(Tracks_List)):
                           if bth[0]==Tracks_List[bt][0]:

                               bth[1].append(Tracks_List[bt][1:4])
                               del Tracks_List[bt]
                               bt-=1
                               trigger=True
                           elif trigger:
                                break
                           else:
                                continue
                           bt+=1
            for bth in Tracks_Head:
               x,y,z=[],[],[]
               for b in bth[1]:
                   x.append(b[0])
                   y.append(b[1])
                   z.append(b[2])
               tx=np.polyfit(z,x,1)[0]
               ax=np.polyfit(z,x,1)[1]
               ty=np.polyfit(z,y,1)[0]
               ay=np.polyfit(z,y,1)[1]
               bth.append(ax) #Append x intercept
               bth.append(tx) #Append x slope
               bth.append(0) #Append a placeholder slope (for polynomial case)
               bth.append(ay) #Append x intercept
               bth.append(ty) #Append x slope
               bth.append(0) #Append a placeholder slope (for polynomial case)
               del(bth[1])
    #Once we get coefficients for all tracks we convert them back to Pandas dataframe and join back to the data
    Tracks_Head=pd.DataFrame(Tracks_Head, columns = [Track_ID,'ax','t1x','t2x','ay','t1y','t2y'])

    temp_data=pd.merge(temp_data,Tracks_Head,how='inner',on = [Track_ID])
    #Calculating x and y coordinates of the fitted line for all plates in the track
    temp_data['new_x']=temp_data['ax']+(temp_data['z']*temp_data['t1x'])+((temp_data['z']**2)*temp_data['t2x'])
    temp_data['new_y']=temp_data['ay']+(temp_data['z']*temp_data['t1y'])+((temp_data['z']**2)*temp_data['t2y'])
    #Calculating how far hits deviate from the fit polynomial
    temp_data['d_x']=temp_data['x']-temp_data['new_x']
    temp_data['d_y']=temp_data['y']-temp_data['new_y']
    temp_data['d_r']=temp_data['d_x']**2+temp_data['d_y']**2
    temp_data['d_r'] = temp_data['d_r'].astype(float)
    temp_data['d_r']=np.sqrt(temp_data['d_r']) #Absolute distance
    temp_data=temp_data[[Track_ID,'Track_No','d_r']]
    temp_data=temp_data.groupby([Track_ID,'Track_No']).agg({'d_r':'sum'}).reset_index()

    temp_data=temp_data.agg({'d_r':'sum','Track_No':'sum'})
    temp_data=temp_data.values.tolist()
    fit=temp_data[0]/temp_data[1]
    return fit
def FitPlateAngle(PlateZ,dtx,dty,input_data,Track_ID,ProgressBar=False):
    change_df = pd.DataFrame([[PlateZ,dtx,dty]], columns = ['Plate_ID','dtx','dty'])
    temp_data=input_data[[Track_ID,'x','y','z','tx','ty','Track_No','Plate_ID']]
    temp_data=pd.merge(temp_data,change_df,on='Plate_ID',how='left')
    temp_data['dtx'] = temp_data['dtx'].fillna(0.0)
    temp_data['dty'] = temp_data['dty'].fillna(0.0)
    temp_data['tx']=temp_data['tx']+temp_data['dtx']
    temp_data['ty']=temp_data['ty']+temp_data['dty']
    temp_data=temp_data[[Track_ID,'x','y','z','tx','ty','Track_No']]
    Tracks_Head=temp_data[[Track_ID]]
    Tracks_Head.drop_duplicates(inplace=True)
    Tracks_List=temp_data.values.tolist() #I find it is much easier to deal with tracks in list format when it comes to fitting
    Tracks_Head=Tracks_Head.values.tolist()
    #Bellow we build the track representatation that we can use to fit slopes
    if ProgressBar:
        with alive_bar(len(Tracks_Head)*2,force_tty=True, title='Angularly fitting data...') as bar:
            for bth in Tracks_Head:
                       bth.append([])
                       bt=0
                       trigger=False
                       bar()
                       while bt<(len(Tracks_List)):
                           if bth[0]==Tracks_List[bt][0]:

                               bth[1].append(Tracks_List[bt][1:4])
                               del Tracks_List[bt]
                               bt-=1
                               trigger=True
                           elif trigger:
                                break
                           else:
                                continue
                           bt+=1
            for bth in Tracks_Head:
               x,y,z=[],[],[]
               for b in bth[1]:
                   x.append(b[0])
                   y.append(b[1])
                   z.append(b[2])
               tx=np.polyfit(z,x,1)[0]
               ty=np.polyfit(z,y,1)[0]
               bth.append(tx) #Append x slope
               bth.append(ty) #Append x slope
               del(bth[1])
               bar()
    else:
            for bth in Tracks_Head:
                       bth.append([])
                       bt=0
                       trigger=False
                       while bt<(len(Tracks_List)):
                           if bth[0]==Tracks_List[bt][0]:

                               bth[1].append(Tracks_List[bt][1:4])
                               del Tracks_List[bt]
                               bt-=1
                               trigger=True
                           elif trigger:
                                break
                           else:
                                continue
                           bt+=1
            for bth in Tracks_Head:
               x,y,z=[],[],[]
               for b in bth[1]:
                   x.append(b[0])
                   y.append(b[1])
                   z.append(b[2])
               tx=np.polyfit(z,x,1)[0]
               ty=np.polyfit(z,y,1)[0]
               bth.append(tx) #Append x slope
               bth.append(ty) #Append x slope
               del(bth[1])

    #Once we get coefficients for all tracks we convert them back to Pandas dataframe and join back to the data
    Tracks_Head=pd.DataFrame(Tracks_Head, columns = [Track_ID,'ntx','nty'])

    temp_data=pd.merge(temp_data,Tracks_Head,how='inner',on = [Track_ID])

    #Calculating x and y coordinates of the fitted line for all plates in the track
    #Calculating how far hits deviate from the fit polynomial
    temp_data['d_tx']=temp_data['tx']-temp_data['ntx']
    temp_data['d_ty']=temp_data['ty']-temp_data['nty']

    temp_data['d_tr']=temp_data['d_tx']**2+temp_data['d_ty']**2
    temp_data['d_tr'] = temp_data['d_tr'].astype(float)
    temp_data['d_tr']=np.sqrt(temp_data['d_tr']) #Absolute distance

    temp_data=temp_data[[Track_ID,'Track_No','d_tr']]
    temp_data=temp_data.groupby([Track_ID,'Track_No']).agg({'d_tr':'sum'}).reset_index()
    temp_data=temp_data.agg({'d_tr':'sum','Track_No':'sum'})
    temp_data=temp_data.values.tolist()
    fit=temp_data[0]/temp_data[1]
    return fit

