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
parser.add_argument('--X_ID_Max',help="Enter X id", default='0')

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
X_ID_Max=int(args.X_ID_Max)
##############################################################################################################################
######################################### Starting the program ################################################################
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
def zero_divide(a, b):
    if (b==0): return 0
    return a/b

FirstFile=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1c_'+RecBatchID+'_hit_cluster_rec_y_set_' +str(0)+'.pkl'
FirstFileRaw=UF.PickleOperations(FirstFile,'r', 'N/A')
FirstFile=FirstFileRaw[0]
ZContractedTable=FirstFile.RecSegments
#ZContractedTable.to_csv('FirstFile.csv',index=False)
for i in range(1,X_ID_Max):
    SecondFile=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1c_'+RecBatchID+'_hit_cluster_rec_y_set_'+str(i)+'.pkl'
    SecondFileRaw=UF.PickleOperations(SecondFile,'r', 'N/A')
    print(SecondFileRaw[1])
    SecondFile=SecondFileRaw[0]
    SecondFileTable=SecondFile.RecSegments.rename(columns={"Master_Segment_ID":"Segment_ID","Master_z":"z" })
    #SecondFileTable.to_csv('SecondFile.csv',index=False)
    FileClean=pd.merge(ZContractedTable.drop_duplicates(subset=["Master_Segment_ID","HitID",'Master_z'],keep='first'),SecondFileTable,how='inner', on=['HitID'])
    FileClean["Segment_No_z"]= FileClean["Segment_ID"]
    FileClean=FileClean.groupby(by=["Master_Segment_ID","Segment_ID","Segment_No_x","Segment_No_y","Segment_No_Tot_x","Segment_No_Tot_y"])["Segment_No_z"].count().reset_index()
    FileCleanTot=FileClean.groupby(by=["Master_Segment_ID"])["Segment_No_z"].sum().reset_index()
    FileCleanTot.rename(columns={"Segment_No_z":"Segment_No_Tot_z"},inplace=True)
    FileClean=pd.merge(FileClean,FileCleanTot,how='inner', on=["Master_Segment_ID"])
    FileClean['Segment_No']=FileClean['Segment_No_x']+FileClean['Segment_No_y']+FileClean['Segment_No_z']
    FileClean['Segment_No_Tot']=FileClean['Segment_No_Tot_x']+FileClean['Segment_No_Tot_y']+FileClean['Segment_No_Tot_z']
    FileClean=FileClean.drop(['Segment_No_x','Segment_No_y','Segment_No_z',"Segment_No_Tot_x","Segment_No_Tot_y","Segment_No_Tot_z"],axis=1)
    FileClean=FileClean.sort_values(["Master_Segment_ID","Segment_No"],ascending=[1,0])

    #FileClean.to_csv('FileClean1.csv',index=False)
    FileClean.drop_duplicates(subset=["Master_Segment_ID"],keep='first',inplace=True)
    #FileClean=FileClean.drop(['Segment_No'],axis=1)
    #FileClean.to_csv('FileClean2.csv',index=False)

    FileClean=pd.merge(FileClean,SecondFileTable,how='right', on=['Segment_ID'])
    FileCleanOrlp=FileClean.dropna(subset=["Master_Segment_ID"])
    FileCleanOrlp['Segment_No']=FileCleanOrlp['Segment_No_x']
    FileCleanOrlp['Segment_No_Tot']=FileCleanOrlp['Segment_No_Tot_x']
    FileCleanOrlp=FileCleanOrlp.drop(['Segment_No_x','Segment_No_y',"Segment_No_Tot_x","Segment_No_Tot_y"],axis=1)
    FileCleanR=FileClean[FileClean["Master_Segment_ID"].isnull()]
    FileCleanR["Master_Segment_ID"] = FileCleanR["Master_Segment_ID"].fillna(FileCleanR["Segment_ID"])
    FileCleanR['Segment_No']=FileCleanR['Segment_No_y']
    FileCleanR['Segment_No_Tot']=FileCleanR['Segment_No_Tot_y']
    FileCleanR=FileCleanR.drop(['Segment_No_x','Segment_No_y',"Segment_No_Tot_x","Segment_No_Tot_y",'Segment_ID'],axis=1)

    FileClean=pd.concat([FileCleanOrlp,FileCleanR])
    FileClean=FileClean.drop(['Segment_ID'],axis=1)
    FileClean=FileClean.rename(columns={"z": "Master_z" })
    ZContractedTable=pd.concat([ZContractedTable,FileClean])
    ZContractedTable_r=ZContractedTable[['Master_Segment_ID','Segment_No','Segment_No_Tot']]
    ZContractedTable_r.drop_duplicates(subset=['Master_Segment_ID','Segment_No','Segment_No_Tot'],keep='first',inplace=True)
    ZContractedTable_r=ZContractedTable_r.groupby(['Master_Segment_ID']).agg({'Segment_No':'sum','Segment_No_Tot':'sum'}).reset_index()
    ZContractedTable=ZContractedTable.drop(['Segment_No','Segment_No_Tot'],axis=1)
    ZContractedTable=pd.merge(ZContractedTable,ZContractedTable_r,how='inner', on=["Master_Segment_ID"])

ZContractedTable['Fit']=ZContractedTable['Segment_No']/ZContractedTable['Segment_No_Tot']
ZContractedTable['Fit'] = ZContractedTable['Fit'].fillna(1.0)

ZContractedTable=ZContractedTable.drop(['Segment_No','Segment_No_Tot'],axis=1)


ZContractedTable['Hit_No']=ZContractedTable['HitID']
ZContractedTable=ZContractedTable.groupby(by=["Master_Segment_ID","Master_z","HitID","Fit"])['Hit_No'].count().reset_index()
ZContractedTable.sort_values(["HitID",'Fit',"Hit_No"],ascending=[1,0,0],inplace=True)

ZContractedTable.drop_duplicates(subset=["HitID"],keep='first',inplace=True)
ZContractedTableIDs=ZContractedTable[["Master_Segment_ID"]]
ZContractedTableIDs=ZContractedTableIDs.drop_duplicates(keep='first')
ZContractedTableIDs=ZContractedTableIDs.reset_index().drop(['index'],axis=1)
ZContractedTableIDs=ZContractedTableIDs.reset_index()
ZContractedTableIDs.rename(columns={"index":"ANN_Track_ID"},inplace=True)
ZContractedTable.drop(['Fit',"Hit_No"],axis=1,inplace=True)
ZContractedTable=pd.merge(ZContractedTable,ZContractedTableIDs,how='inner',on=["Master_Segment_ID"])
ZContractedTable.drop(['Master_z',"Master_Segment_ID"],axis=1,inplace=True)
ZContractedTable['ANN_Brick_ID']=RecBatchID
FirstFile.RecTracks=ZContractedTable
OutputFile=EOS_DIR+'/ANNDEA/Data/REC_SET/RTr1d_'+RecBatchID+'_hit_cluster_rec_x_set.pkl'
print(UF.PickleOperations(OutputFile, 'w', FirstFile)[1])
exit()



