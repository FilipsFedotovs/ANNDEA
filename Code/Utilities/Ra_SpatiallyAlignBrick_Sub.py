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
import U_UI as UI
import U_Alignment as UA
import pandas as pd #We use Panda for a routine data processing
from scipy.optimize import minimize_scalar
import ast
Plate=ast.literal_eval(args.Plate)
#Define some functions

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
output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+BatchID+'_'+str(i)+'/'+pfx+'_'+BatchID+'_'+o+'_'+str(i)+'_'+str(j)+'_'+str(k)+sfx
print(UI.TimeStamp(), "Modules Have been imported successfully...")
print(UI.TimeStamp(),'Loading pre-selected data from ',input_file_location)
data=pd.read_csv(input_file_location,header=0,
                    usecols=['x','y','z','Rec_Seg_ID'])[['Rec_Seg_ID','x','y','z']]
final_rows=len(data)
print(UI.TimeStamp(),'The cleaned data has',final_rows,'hits')
print(UI.TimeStamp(),'Removing tracks which have less than',ValMinHits,'hits...')
track_no_data=data.groupby(['Rec_Seg_ID'],as_index=False).count()
track_no_data=track_no_data.drop(['y','z'],axis=1)
track_no_data=track_no_data.rename(columns={'x': "Track_No"})
new_combined_data=pd.merge(data, track_no_data, how="left", on=['Rec_Seg_ID'])
new_combined_data=new_combined_data.sort_values(['Rec_Seg_ID','z'],ascending=[1,1])
new_combined_data['Plate_ID']=new_combined_data['z'].astype(int)
train_data = new_combined_data[new_combined_data.Track_No >= MinHits]
validation_data = new_combined_data[new_combined_data.Track_No >= ValMinHits]
validation_data = validation_data[validation_data.Track_No < MinHits]



am=['Spatial',Plate[i][0],j,k]

def FitPlateFixedX(x):
    return UA.FitPlate(Plate[i][0],x,0,train_data,'Rec_Seg_ID',False)
def FitPlateFixedY(x):
    return UA.FitPlate(Plate[i][0],0,x,train_data,'Rec_Seg_ID',False)
def FitPlateValX(x):
    return UA.FitPlate(Plate[i][0],x,0,validation_data,'Rec_Seg_ID',False)
def FitPlateValY(x):
    return UA.FitPlate(Plate[i][0],0,x,validation_data,'Rec_Seg_ID',False)
if len(train_data)>0:
    res = minimize_scalar(FitPlateFixedX, bounds=(-OptBound, OptBound), method='bounded')
    validation_data=AlignPlate(Plate[i][0],res.x,0,validation_data)
    am.append(res.x)
    FitFix = FitPlateFixedX(res.x)
    FitVal = FitPlateValX(res.x)
    am.append(FitFix)
    am.append(FitVal)
    res = minimize_scalar(FitPlateFixedY, bounds=(-OptBound, OptBound), method='bounded')
    validation_data=AlignPlate(Plate[i][0],0,res.x,validation_data)
    am.append(res.x)
    FitFix = FitPlateFixedY(res.x)
    FitVal = FitPlateValY(0)
    am.append(FitFix)
    am.append(FitVal)
else:
    am.append(0)
    am.append(0)
    am.append(0)
    am.append(0)
    am.append(0)
    am.append(0)
UI.LogOperations(output_file_location,'w',[am]) #Writing the remaining data into the csv
print(UI.TimeStamp(), "Optimisation is finished...")
#End of the script



