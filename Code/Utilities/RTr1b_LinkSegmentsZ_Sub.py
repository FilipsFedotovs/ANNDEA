########################################################################################################################
#######################################  This simple script prepares data for CNN  #####################################




########################################    Import libraries    ########################################################
import argparse
import pandas as pd
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
parser.add_argument('--RecBatchID',help="Give this reconstruction batch an ID", default='Test_Slider')
parser.add_argument('--Y_ID',help="Enter Y id", default='0')
parser.add_argument('--X_ID',help="Enter X id", default='0')
parser.add_argument('--Z_ID_Max',help="Enter Max Z id", default='2')
########################################     Initialising Variables    #########################################
args = parser.parse_args()
##################################   Loading Directory locations   ##################################################
AFS_DIR=args.AFS
EOS_DIR=args.EOS
RecBatchID=args.RecBatchID
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import UtilityFunctions as UF
#Load data configuration
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubDataDIR=EOSsubDIR+'/'+'Data'
Y_ID=int(args.Y_ID)
X_ID=int(args.X_ID)
Z_ID_Max=int(args.Z_ID_Max)
##############################################################################################################################
######################################### Starting the program ################################################################
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
def zero_divide(a, b):
    if (b==0): return 0
    return a/b

FirstFile=EOS_DIR+'/ANNDEA/Data/REC_SET/RH1a_'+RecBatchID+'_hit_cluster_rec_set_'+str(0)+'_' +str(Y_ID)+'_' +str(X_ID)+'.pkl'
FirstFileRaw=UF.PickleOperations(FirstFile,'r', 'N/A')
FirstFile=FirstFileRaw[0]
ZContractedTable=FirstFile.RecHits.rename(columns={"Segment_ID": "Master_Segment_ID","z": "Master_z" })
#ZContractedTable.to_csv('FirstFile.csv',index=False)
for i in range(1,Z_ID_Max):
    SecondFile=EOS_DIR+'/ANNDEA/Data/REC_SET/RH1a_'+RecBatchID+'_hit_cluster_rec_set_'+str(i)+'_' +str(Y_ID)+'_' +str(X_ID)+'.pkl'
    SecondFileRaw=UF.PickleOperations(SecondFile,'r', 'N/A')
    #print(SecondFileRaw[1])
    SecondFile=SecondFileRaw[0]
    SecondFileTable=SecondFile.RecHits
    #SecondFileTable.to_csv('SecondFile.csv',index=False)
    FileClean=pd.merge(ZContractedTable,SecondFileTable,how='inner', on=['HitID'])

    FileClean["Segment_No"]= FileClean["Segment_ID"]
    FileClean=FileClean.groupby(by=["Master_Segment_ID","Segment_ID"])["Segment_No"].count().reset_index()

    FileClean=FileClean.sort_values(["Master_Segment_ID","Segment_No"],ascending=[1,0])
    #print(FileClean)
    #FileClean.to_csv('FileClean1.csv',index=False)
    FileClean.drop_duplicates(subset=["Master_Segment_ID"],keep='first',inplace=True)

    FileClean=FileClean.drop(['Segment_No'],axis=1)
    #FileClean.to_csv('FileClean2.csv',index=False)

    FileClean=pd.merge(FileClean,SecondFileTable,how='right', on=['Segment_ID'])
    #print(FileClean)

    FileClean["Master_Segment_ID"] = FileClean["Master_Segment_ID"].fillna(FileClean["Segment_ID"])
    FileClean=FileClean.rename(columns={"z": "Master_z" })
    # SecondFileClean=FileClean[['Segment_ID']]
    # SecondFileClean=pd.merge(FileClean[['Segment_ID']],SecondFileTable,how='right', on=['Segment_ID'])
    FileClean=FileClean.drop(['Segment_ID'],axis=1)
    ZContractedTable=pd.concat([ZContractedTable,FileClean])
    ZContractedTable.drop_duplicates(subset=["Master_Segment_ID","HitID",'Master_z'],keep='first',inplace=True)
    #print(ZContractedTable.sort_values(["Master_Segment_ID",'Master_z'],ascending=[1,1]))


FirstFile.RecSegments=ZContractedTable.sort_values(["Master_Segment_ID",'Master_z'],ascending=[1,1])
OutputFile=EOS_DIR+'/ANNDEA/Data/REC_SET/RH1b_'+RecBatchID+'_hit_cluster_rec_z_set_'+str(Y_ID)+'_' +str(X_ID)+'.pkl'
print(UF.PickleOperations(OutputFile, 'w', FirstFile)[1])
exit()



