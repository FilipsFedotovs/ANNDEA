########################################################################################################################
#######################################  This simple script prepares data for CNN  #####################################




########################################    Import libraries    ########################################################
import argparse
import sys
########################## Visual Formatting #################################################
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

########################## Setting the parser ################################################
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--AFS',help="Please enter the user afs directory", default='.')
parser.add_argument('--EOS',help="Please enter the user eos directory", default='.')
parser.add_argument('--PY',help="Python libraries directory location", default='.')
parser.add_argument('--RecBatchID',help="Give this reconstruction batch an ID", default='Test_Slider')
parser.add_argument('--X_ID',help="Enter X id", default='0')
parser.add_argument('--Y_ID_Max',help="Enter Y id", default='0')

########################################     Initialising Variables    #########################################
args = parser.parse_args()
##################################   Loading Directory locations   ##################################################
AFS_DIR=args.AFS
EOS_DIR=args.EOS
PY_DIR=args.PY
if PY_DIR!='':
    sys.path=[PY_DIR]
    sys.path.append('/usr/lib64/python36.zip')
    sys.path.append('/usr/lib64/python3.6')
    sys.path.append('/usr/lib64/python3.6/lib-dynload')
sys.path.append(AFS_DIR+'/Code/Utilities')
RecBatchID=args.RecBatchID
#import the rest of the libraries
import pandas as pd
import UtilityFunctions as UF
#Load data configuration
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubDataDIR=EOSsubDIR+'/'+'Data'
X_ID=int(args.X_ID)
Y_ID_Max=int(args.Y_ID_Max)
##############################################################################################################################
######################################### Starting the program ################################################################
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
def zero_divide(a, b):
    if (b==0): return 0
    return a/b

FirstFile=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1b_'+RecBatchID+'_hit_cluster_rec_z_set_'+str(0)+'_' +str(X_ID)+'.pkl'
FirstFileRaw=UF.PickleOperations(FirstFile,'r', 'N/A')
FirstFile=FirstFileRaw[0]
ZContractedTable=FirstFile.RecSegments
ZContractedTable["Segment_No"]=0
ZContractedTable["Segment_No_Tot"]=0
for i in range(1,Y_ID_Max):
    SecondFile=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1b_'+RecBatchID+'_hit_cluster_rec_z_set_'+str(i)+'_' +str(X_ID)+'.pkl'
    SecondFileRaw=UF.PickleOperations(SecondFile,'r', 'N/A')
    print(SecondFileRaw[1])
    SecondFile=SecondFileRaw[0]
    SecondFileTable=SecondFile.RecSegments.rename(columns={"Master_Segment_ID":"Segment_ID","Master_z":"z" })
    FileClean=pd.merge(ZContractedTable.drop_duplicates(subset=["Master_Segment_ID","HitID",'Master_z'],keep='first'),SecondFileTable,how='inner', on=['HitID'])
    FileClean["Segment_No"]= FileClean["Segment_ID"]
    FileClean=FileClean.groupby(by=["Master_Segment_ID","Segment_ID"])["Segment_No"].count().reset_index()
    FileCleanTot=FileClean.groupby(by=["Master_Segment_ID"])["Segment_No"].sum().reset_index()
    FileCleanTot.rename(columns={"Segment_No":"Segment_No_Tot"},inplace=True)
    FileClean=pd.merge(FileClean,FileCleanTot,how='inner', on=["Master_Segment_ID"])
    FileClean=FileClean.sort_values(["Master_Segment_ID","Segment_No"],ascending=[1,0])
    FileClean.drop_duplicates(subset=["Master_Segment_ID"],keep='first',inplace=True)
    FileClean=pd.merge(FileClean,SecondFileTable,how='right', on=['Segment_ID'])
    FileClean["Master_Segment_ID"] = FileClean["Master_Segment_ID"].fillna(FileClean["Segment_ID"])
    FileClean["Segment_No"] = FileClean["Segment_No"].fillna(0)
    FileClean["Segment_No_Tot"] = FileClean["Segment_No_Tot"].fillna(0)
    FileClean=FileClean.rename(columns={"z": "Master_z" })
    FileClean=FileClean.drop(['Segment_ID'],axis=1)
    ZContractedTable=pd.concat([ZContractedTable,FileClean])
    ZContractedTable_r=ZContractedTable[['Master_Segment_ID','Segment_No','Segment_No_Tot']]
    ZContractedTable_r.drop_duplicates(subset=['Master_Segment_ID','Segment_No','Segment_No_Tot'],keep='first',inplace=True)
    ZContractedTable_r=ZContractedTable_r.groupby(['Master_Segment_ID']).agg({'Segment_No':'sum','Segment_No_Tot':'sum'}).reset_index()
    ZContractedTable=ZContractedTable.drop(['Segment_No','Segment_No_Tot'],axis=1)
    ZContractedTable=pd.merge(ZContractedTable,ZContractedTable_r,how='inner', on=["Master_Segment_ID"])
FirstFile.RecSegments=ZContractedTable.sort_values(["Master_Segment_ID",'Master_z'],ascending=[1,1])
OutputFile=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1c_'+RecBatchID+'_hit_cluster_rec_y_set_' +str(X_ID)+'.pkl'
print(UF.PickleOperations(OutputFile, 'w', FirstFile)[1])
exit()



