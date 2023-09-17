#This simple script prepares 2-segment track seeds for the initial CNN/GNN union
# Part of ANNDEA package
#Made by Filips Fedotovs
#Current version 1.0

########################################    Import libraries    #############################################
import argparse
import sys




######################################## Set variables  #############################################################
#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--PlateZ',help="The Z coordinate of the starting plate", default='-36820.0')
parser.add_argument('--i',help="Set number", default='1')
parser.add_argument('--j',help="Subset number", default='1')
parser.add_argument('--k',help="Subset number", default='1')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
parser.add_argument('--ValMinHits',help="Maximum allowed longitudinal gap value between segments", default='8000',type=int)
parser.add_argument('--MinHits',help="Maximum allowed transverse gap value between segments per SLG length", default='1000',type=int)
parser.add_argument('--Plate',help="Maximum allowed transverse gap value between segments per SLG length", default='[]')
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')
parser.add_argument('--BatchID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--Size',help="A size of the volume of the local alignment", default='20000')
parser.add_argument('--OptBound',help="A bound of the optimisation", default='20000')
parser.add_argument('--PY',help="Python libraries directory location", default='.')

######################################## Set variables  #############################################################
args = parser.parse_args()
i=int(args.i)    #This is just used to name the output file
j=int(args.j)  #The subset helps to determine what portion of the track list is used to create the Seeds
k=int(args.k)
p=args.p
o=args.o
sfx=args.sfx
pfx=args.pfx
Size=float(args.Size)
OptBound=float(args.OptBound)
BatchID=args.BatchID
MinHits=args.MinHits
ValMinHits=args.ValMinHits

########################################     Preset framework parameters    #########################################
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
import UtilityFunctions as UF #This is where we keep routine utility functions
import pandas as pd #We use Panda for a routine data processing
import numpy as np
from scipy.optimize import minimize_scalar
import ast
Plate=ast.literal_eval(args.Plate)
#Define some functions
def FitPlate(PlateZ,dx,dy,input_data,Track_ID):
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
def AlignPlate(PlateZ,dx,dy,input_data):
    change_df = pd.DataFrame([[PlateZ,dx,dy]], columns = ['Plate_ID','dx','dy'])
    temp_data=input_data
    temp_data=pd.merge(temp_data,change_df,on='Plate_ID',how='left')
    temp_data['dx'] = temp_data['dx'].fillna(0.0)
    temp_data['dy'] = temp_data['dy'].fillna(0.0)
    temp_data['x']=temp_data['x']+temp_data['dx']
    temp_data['y']=temp_data['y']+temp_data['dy']
    temp_data = temp_data.drop(['dx','dy'],axis=1)
    return temp_data

#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/R_'+BatchID+'_HITS_'+str(j)+'_'+str(k)+'.csv'
output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+BatchID+'_'+str(i)+'/'+pfx+'_'+BatchID+'_RawSeeds_'+str(i)+'_'+str(j)+'_'+str(k)+sfx
print(UF.TimeStamp(), "Modules Have been imported successfully...")
print(UF.TimeStamp(),'Loading pre-selected data from ',input_file_location)
data=pd.read_csv(input_file_location,header=0,
                    usecols=['x','y','z','Rec_Seg_ID','Hit_ID'])[['Rec_Seg_ID','Hit_ID','x','y','z']]
final_rows=len(data)
print(UF.TimeStamp(),'The cleaned data has',final_rows,'hits')
print(UF.TimeStamp(),'Removing tracks which have less than',ValMinHits,'hits...')
track_no_data=data.groupby(['Rec_Seg_ID'],as_index=False).count()
track_no_data=track_no_data.drop(['Hit_ID','y','z'],axis=1)
track_no_data=track_no_data.rename(columns={'x': "Track_No"})
new_combined_data=pd.merge(data, track_no_data, how="left", on=['Rec_Seg_ID'])
new_combined_data=new_combined_data.drop(['Hit_ID'],axis=1)
new_combined_data=new_combined_data.sort_values(['Rec_Seg_ID','z'],ascending=[1,1])
train_data = new_combined_data[new_combined_data.Track_No >= MinHits]
validation_data = new_combined_data[new_combined_data.Track_No >= ValMinHits]
validation_data = validation_data[validation_data.Track_No < MinHits]
print(train_data)
print(validation_data)


am=[i,j,k]
def FitPlateFixedX(x):
    return FitPlate(p[i],x,0,train_data,'Rec_Seg_ID')
def FitPlateFixedY(x):
    return FitPlate(p[i],0,x,train_data,'Rec_Seg_ID')
def FitPlateValX(x):
    return FitPlate(p[i],x,0,validation_data,'Rec_Seg_ID')
def FitPlateValY(x):
    return FitPlate(p[i],0,x,validation_data,'Rec_Seg_ID')
res = minimize_scalar(FitPlateFixedX, bounds=(-OptBound, OptBound), method='bounded')
print(res)
exit()
validation_data=AlignPlate(p[0],res.x,0,validation_data)
am.append(res.x)
FitFix = FitPlateFixedX(res.x)
FitVal = FitPlateValX(0)

res = minimize_scalar(FitPlateFixedY, bounds=(-OptBound, OptBound), method='bounded')
validation_data=AlignPlate(p[0],0,res.x,validation_data)
am.append(res.x)

FitFix = FitPlateFixedY(res.x)
FitVal = FitPlateValY(0)

bar.text('| Validation fit value changed from '+ini_val+' to '+str(round(FitVal,2)))
bar()
local_logdata = [c+1,"global vertical-horizontal plate alignment XY", iterator, p[0], FitFix, FitVal, MinHits,ValMinHits]
global_logdata.append(local_logdata)
alignment_map.append(am)
alignment_map_global.append(am)

#
# UF.LogOperations(output_file_location,'a',result_list) #Writing the remaining data into the csv
# UF.LogOperations(output_result_location,'w',[])
# print(UF.TimeStamp(), "Reconstruction seed generation is finished...")
# #End of the script



