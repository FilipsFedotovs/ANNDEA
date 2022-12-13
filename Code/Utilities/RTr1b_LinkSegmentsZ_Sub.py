########################################################################################################################
#######################################  This simple script prepares data for CNN  #####################################




########################################    Import essential libraries    ########################################################
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
parser.add_argument('--BatchID',help="Give this reconstruction batch an ID", default='Test_Slider')
parser.add_argument('--j',help="Enter Y id", default='0')
parser.add_argument('--i',help="Enter X id", default='0')
parser.add_argument('--Z_ID_Max',help="Enter Max Z id", default='2')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
########################################     Initialising Variables    #########################################
args = parser.parse_args()
##################################   Loading Directory locations   ##################################################
AFS_DIR=args.AFS
EOS_DIR=args.EOS
PY_DIR=args.PY
if PY_DIR!='': #Temp solution
    sys.path=['',PY_DIR]
    sys.path.append('/usr/lib64/python36.zip')
    sys.path.append('/usr/lib64/python3.6')
    sys.path.append('/usr/lib64/python3.6/lib-dynload')
    sys.path.append('/usr/lib64/python3.6/site-packages')
    sys.path.append('/usr/lib/python3.6/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')
#import other libraries
import pandas as pd
RecBatchID=args.BatchID
import UtilityFunctions as UF
#Load data configuration
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubDataDIR=EOSsubDIR+'/'+'Data'
Y_ID=int(args.j)
X_ID=int(args.i)
Z_ID_Max=int(args.Z_ID_Max)
p=args.p
o=args.o
sfx=args.sfx
pfx=args.pfx
##############################################################################################################################
######################################### Starting the program ################################################################
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
def zero_divide(a, b):
    if (b==0): return 0
    return a/b

FirstFile=EOS_DIR+p+'/Temp_'+'RTr1a'+'_'+RecBatchID+'_'+str(X_ID)+'/'+'RTr1a'+'_'+RecBatchID+'_'+'hit_cluster_rec_z_set'+'_'+str(X_ID)+'_'+str(Y_ID)+'_'+str(0)+sfx
FirstFileRaw=UF.PickleOperations(FirstFile,'r', 'N/A')
FirstFile=FirstFileRaw[0]
ZContractedTable=FirstFile.RecHits.rename(columns={"Segment_ID": "Master_Segment_ID","z": "Master_z" })
for i in range(1,Z_ID_Max):
    SecondFile=EOS_DIR+p+'/Temp_'+'RTr1a'+'_'+RecBatchID+'_'+str(X_ID)+'/'+'RTr1a'+'_'+RecBatchID+'_'+'hit_cluster_rec_z_set'+'_'+str(X_ID)+'_'+str(Y_ID)+'_'+str(i)+sfx
    print('Opening ',SecondFile)
    SecondFileRaw=UF.PickleOperations(SecondFile,'r', 'N/A')
    SecondFile=SecondFileRaw[0]
    SecondFileTable=SecondFile.RecHits
    FileClean=pd.merge(ZContractedTable,SecondFileTable,how='inner', on=['HitID'])
    FileClean["Segment_No"]= FileClean["Segment_ID"]
    FileClean=FileClean.groupby(by=["Master_Segment_ID","Segment_ID"])["Segment_No"].count().reset_index()
    FileClean=FileClean.sort_values(["Master_Segment_ID","Segment_No"],ascending=[1,0])
    FileClean.drop_duplicates(subset=["Master_Segment_ID"],keep='first',inplace=True)
    FileClean=FileClean.drop(['Segment_No'],axis=1)
    FileClean=pd.merge(FileClean,SecondFileTable,how='right', on=['Segment_ID'])
    FileClean["Master_Segment_ID"] = FileClean["Master_Segment_ID"].fillna(FileClean["Segment_ID"])
    FileClean=FileClean.rename(columns={"z": "Master_z" })
    FileClean=FileClean.drop(['Segment_ID'],axis=1)
    ZContractedTable=pd.concat([ZContractedTable,FileClean])
    ZContractedTable.drop_duplicates(subset=["Master_Segment_ID","HitID",'Master_z'],keep='first',inplace=True)
FirstFile.RecSegments=ZContractedTable.sort_values(["Master_Segment_ID",'Master_z'],ascending=[1,1])
output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(X_ID)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(X_ID)+'_'+str(Y_ID)+sfx
print(UF.PickleOperations(output_file_location, 'w', FirstFile)[1])
exit()



