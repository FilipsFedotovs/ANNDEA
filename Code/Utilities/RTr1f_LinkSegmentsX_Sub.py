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
parser.add_argument('--BatchID',help="Give this reconstruction batch an ID", default='Test_Slider')
parser.add_argument('--i',help="Enter X id", default='0') #Dummy argument, just put there because the Standard Program framework requires it
parser.add_argument('--X_ID_Max',help="Enter X id", default='0')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
########################################     Initialising Variables    #########################################
args = parser.parse_args()
RecBatchID=args.BatchID
##################################   Loading Directory locations   ##################################################
AFS_DIR=args.AFS
EOS_DIR=args.EOS
PY_DIR=args.PY
if PY_DIR!='': #Temp solution
    sys.path=['',PY_DIR]
    sys.path.append('/usr/lib64/python39.zip')
    sys.path.append('/usr/lib64/python3.9')
    sys.path.append('/usr/lib64/python3.9/lib-dynload')
    sys.path.append('/usr/lib64/python3.9/site-packages')
    sys.path.append('/usr/lib/python3.9/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')

#import the rest of the libraries
import pandas as pd
pd.options.mode.chained_assignment = None #Silence annoying warnings
import U_UI as UI
from alive_progress import alive_bar
#Load data configuration
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubDataDIR=EOSsubDIR+'/'+'Data'
X_ID_Max=int(args.X_ID_Max)
X_ID=int(args.i)
p=args.p
o=args.o
sfx=args.sfx
pfx=args.pfx
##############################################################################################################################
######################################### Starting the program ################################################################
print(UI.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
def zero_divide(a, b):
    if (b==0): return 0
    return a/b
#Load the first file (on the y-axis) with reconstructed clusters that already have been merged along z-axis
FirstFileName=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_RTr1e_'+RecBatchID+'_'+str(0)+'/RTr1e_'+RecBatchID+'_hit_cluster_rec_y_set_' +str(0)+'.csv'

ZContractedTable=pd.read_csv(FirstFileName)  #First cluster is like a Pacman: it absorbes proceeding clusters and gets bigger
ZContractedTable.drop_duplicates(subset=['HitID', 'Master_z', 'Master_Segment_ID', 'Segment_No', 'Segment_No_Tot'],keep='first',inplace=True)
ZContractedTable["HitID"] = ZContractedTable["HitID"].astype(str)
with alive_bar(X_ID_Max-1,force_tty=True, title='Merging cluster sets along x-axis..') as bar:
    for i in range(1,X_ID_Max):
        bar()
        SecondFileName=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_RTr1e_'+RecBatchID+'_'+str(0)+'/RTr1e_'+RecBatchID+'_hit_cluster_rec_y_set_'+str(i)+'.csv' #keep loading subsequent files along y-xis with reconstructed clusters that already have been merged along z and y-axis
        SecondFile=pd.read_csv(SecondFileName)
        SecondFile.drop_duplicates(subset=['HitID', 'Master_z', 'Master_Segment_ID', 'Segment_No', 'Segment_No_Tot'],keep='first',inplace=True)
        SecondFile["HitID"] = SecondFile["HitID"].astype(str)
        # print('--------------Part 1------------------')
        # filtered_df = SecondFile[SecondFile['Master_Segment_ID'].isin(['ANNDEA_B41_ExclEM_Debug_19.5_18.5_1.6666666666666667-2','ANNDEA_B41_ExclEM_Debug_20.0_18.0_2.0-1'])]
        # print(filtered_df)
        #filtered_df = ZContractedTable[ZContractedTable['Master_Segment_ID'].isin(['ANNDEA_B41_ExclEM_Debug_19.5_18.5_1.6666666666666667-2','ANNDEA_B41_ExclEM_Debug_20.0_18.0_2.0-1'])]
        # print(filtered_df)

        # print('--------------Part 2------------------')
        SecondFileTable=SecondFile.rename(columns={"Master_Segment_ID":"Segment_ID","Master_z":"z" }) #Initally the following clusters are downgraded from the master status
        # filtered_df = SecondFileTable[SecondFileTable['Segment_ID'].isin(['ANNDEA_B41_ExclEM_Debug_19.5_18.5_1.6666666666666667-2','ANNDEA_B41_ExclEM_Debug_20.0_18.0_2.0-1'])]
        # print(filtered_df)
        #
        # print('--------------Part 3------------------')
        # test=ZContractedTable.drop_duplicates(subset=["Master_Segment_ID","HitID",'Master_z'])
        # filtered_df = test[test['Master_Segment_ID'].isin(['ANNDEA_B41_ExclEM_Debug_19.5_18.5_1.6666666666666667-2','ANNDEA_B41_ExclEM_Debug_20.0_18.0_2.0-1'])]
        # print(filtered_df)
        # SecondFileTable["HitID"] = SecondFileTable["HitID"].astype(str)
        # ZContractedTable["HitID"] = ZContractedTable["HitID"].astype(str)
        FileClean=pd.merge(ZContractedTable.drop_duplicates(subset=["Master_Segment_ID","HitID",'Master_z'],keep='first'),SecondFileTable,how='inner', on=['HitID']) #Join segments based on the common hits
        # filtered_df = FileClean[FileClean["Master_Segment_ID"].isin(['ANNDEA_B41_ExclEM_Debug_19.5_18.5_1.6666666666666667-2','ANNDEA_B41_ExclEM_Debug_20.0_18.0_2.0-1'])]
        # print(filtered_df)
        # #print(FileClean)

        # print('--------------Part 4------------------')

        FileClean["Segment_No_z"]= FileClean["Segment_ID"]
        FileClean=FileClean.groupby(by=["Master_Segment_ID","Segment_ID","Segment_No_x","Segment_No_y","Segment_No_Tot_x","Segment_No_Tot_y"])["Segment_No_z"].count().reset_index()
        filtered_df = FileClean[FileClean["Master_Segment_ID"].isin(['ANNDEA_B41_ExclEM_Debug_19.5_18.5_1.6666666666666667-2','ANNDEA_B41_ExclEM_Debug_20.0_18.0_2.0-1'])]
        # print(filtered_df)
        # print('--------------Part 5------------------')
        FileCleanTot=FileClean.groupby(by=["Master_Segment_ID"])["Segment_No_z"].sum().reset_index()
        FileCleanTot.rename(columns={"Segment_No_z":"Segment_No_Tot_z"},inplace=True)
        FileClean=pd.merge(FileClean,FileCleanTot,how='inner', on=["Master_Segment_ID"])
        FileClean['Segment_No']=FileClean['Segment_No_x']+FileClean['Segment_No_y']+FileClean['Segment_No_z']
        FileClean['Segment_No_Tot']=FileClean['Segment_No_Tot_x']+FileClean['Segment_No_Tot_y']+FileClean['Segment_No_Tot_z']
        FileClean=FileClean.drop(['Segment_No_x','Segment_No_y','Segment_No_z',"Segment_No_Tot_x","Segment_No_Tot_y","Segment_No_Tot_z"],axis=1)
        FileClean=FileClean.sort_values(["Master_Segment_ID","Segment_No"],ascending=[1,0])
        FileClean.drop_duplicates(subset=["Master_Segment_ID"],keep='first',inplace=True)  #Keep the best matching segment



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

        # print('--------------Part xx------------------')
        ZContractedTable.drop_duplicates(subset=['HitID', 'Master_z', 'Master_Segment_ID', 'Segment_No', 'Segment_No_Tot'],keep='first',inplace=True)
        ZContractedTable["HitID"] = ZContractedTable["HitID"].astype(str)
        # filtered_df = ZContractedTable[ZContractedTable['Master_Segment_ID'].isin(['ANNDEA_B41_ExclEM_Debug_19.5_18.5_1.6666666666666667-2','ANNDEA_B41_ExclEM_Debug_20.0_18.0_2.0-1'])]
        # print(filtered_df)
        # x=input()

ZContractedTable['Fit']=ZContractedTable['Segment_No']/ZContractedTable['Segment_No_Tot']
ZContractedTable['Fit'] = ZContractedTable['Fit'].fillna(1.0)

ZContractedTable=ZContractedTable.drop(['Segment_No','Segment_No_Tot'],axis=1)


ZContractedTable['Hit_No']=ZContractedTable['HitID']
ZContractedTable=ZContractedTable.groupby(by=["Master_Segment_ID","Master_z","HitID","Fit"])['Hit_No'].count().reset_index()
ZContractedTable.sort_values(["HitID",'Fit',"Hit_No"],ascending=[1,0,0],inplace=True)

ZContractedTable.drop_duplicates(subset=["HitID"],keep='first',inplace=True) #Ensure the hit fidelity the tracks are ready
ZContractedTableIDs=ZContractedTable[["Master_Segment_ID"]]
ZContractedTableIDs=ZContractedTableIDs.drop_duplicates(keep='first')
ZContractedTableIDs=ZContractedTableIDs.reset_index().drop(['index'],axis=1) #Create numerical track numbers
ZContractedTableIDs=ZContractedTableIDs.reset_index()
ZContractedTableIDs.rename(columns={"index":RecBatchID+'_Track_ID'},inplace=True) #These are the ANN Track IDs
ZContractedTable.drop(['Fit',"Hit_No"],axis=1,inplace=True) #Removing the info that is not used anymore
ZContractedTable=pd.merge(ZContractedTable,ZContractedTableIDs,how='inner',on=["Master_Segment_ID"])
ZContractedTable.drop(['Master_z',"Master_Segment_ID"],axis=1,inplace=True)
ZContractedTable[RecBatchID+'_Brick_ID']=RecBatchID #Creating the track prefix relevant to this particular reconstruction (to keep track IDs unique)
output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(0)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(0)+sfx
ZContractedTable.to_csv(output_file_location,index=False)
print(UI.TimeStamp(),'Output is written to ',output_file_location) #Write the output
exit()
