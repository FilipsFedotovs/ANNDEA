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
parser.add_argument('--i',help="Enter X id", default='0')
parser.add_argument('--j',help="Enter Y id", default='0')
parser.add_argument('--Z_ID_Max',help="Enter Z id", default='0')
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
    sys.path.append('/usr/lib64/python39.zip')
    sys.path.append('/usr/lib64/python3.9')
    sys.path.append('/usr/lib64/python3.9/lib-dynload')
    sys.path.append('/usr/lib64/python3.9/site-packages')
    sys.path.append('/usr/lib/python3.9/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')
RecBatchID=args.BatchID
#import the rest of the libraries
import pandas as pd
pd.options.mode.chained_assignment = None #Silence annoying warnings
import U_UI as UI
#Load data configuration
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubDataDIR=EOSsubDIR+'/'+'Data'
X_ID=int(args.i)
Y_ID=int(args.j)
Z_ID_Max=int(args.Z_ID_Max)
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
FirstFileName=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_'+'RTr1a'+'_'+RecBatchID+'_'+str(X_ID)+'_'+str(Y_ID)+'/RTr1a_'+RecBatchID+'_hit_cluster_rec_set_'+str(X_ID)+'_' +str(Y_ID)+'_' +str(0)+'.csv'
ZContractedTable=pd.read_csv(FirstFileName) #First cluster is like a Pacman: it absorbes proceeding clusters and gets bigger
print(ZContractedTable)
x=input()
exit()
# ZContractedTable=z_clusters_results[0].rename(columns={"Segment_ID": "Master_Segment_ID","z": "Master_z" }) #First cluster is like a Pacman: it absorbs proceeding clusters and gets bigger
# ZContractedTable.drop_duplicates(subset=['HitID', 'Master_z', 'Master_Segment_ID'],keep='first',inplace=True)
# ZContractedTable["HitID"] = ZContractedTable["HitID"].astype(str)
#     for i in range(1,len(z_clusters_results)):
#         SecondFileTable=z_clusters_results[i]
#         SecondFileTable.drop_duplicates(subset=['HitID', 'z', 'Segment_ID'],keep='first',inplace=True)
#
#         SecondFileTable["HitID"] = SecondFileTable["HitID"].astype(str)
#         FileClean=pd.merge(ZContractedTable,SecondFileTable,how='inner', on=['HitID']) #Join segments based on the common hits
#         FileClean["Segment_No"]= FileClean["Segment_ID"]
#         FileClean=FileClean.groupby(by=["Master_Segment_ID","Segment_ID"])["Segment_No"].count().reset_index() #If multiple segments share hit with the master segment we decide the best one by a level of the overlap
#         FileClean=FileClean.sort_values(["Master_Segment_ID","Segment_No"],ascending=[1,0]) #Keep the best matching segment
#         FileClean.drop_duplicates(subset=["Master_Segment_ID"],keep='first',inplace=True)
#         FileClean=FileClean.drop(['Segment_No'],axis=1)
#         FileClean=pd.merge(FileClean,SecondFileTable,how='right', on=['Segment_ID'])
#         FileClean["Master_Segment_ID"] = FileClean["Master_Segment_ID"].fillna(FileClean["Segment_ID"]) #All segments that did not have overlapping hits with the master segment become masters themselves and become part of the Pacman
#         FileClean=FileClean.rename(columns={"z": "Master_z" })
#         FileClean=FileClean.drop(['Segment_ID'],axis=1)
#         ZContractedTable=pd.concat([ZContractedTable,FileClean]) #Absorbing proceeding cluster
#         ZContractedTable.drop_duplicates(subset=["Master_Segment_ID","HitID",'Master_z'],keep='first',inplace=True)
#         ZContractedTable["HitID"] = ZContractedTable["HitID"].astype(str)
#     ZContractedTable=ZContractedTable.sort_values(["Master_Segment_ID",'Master_z'],ascending=[1,1])
# else: #If Cluster tracking yielded no segments we just create an empty array for consistency
#      print(UI.TimeStamp(),'No suitable hit pairs in the cluster set, just writing the empty one...')
#      ZContractedTable=pd.DataFrame([], columns = ['HitID','Master_z','Master_Segment_ID'])
# output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(X_ID_n)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(X_ID_n)+'_'+str(Y_ID_n)+sfx
# print(UI.TimeStamp(),'Writing the output...')
# ZContractedTable.to_csv(output_file_location,index=False) #Write the final result
# print(UI.TimeStamp(),'Output is written to ',output_file_location)
# exit()
#
#
# for i in range(1,Y_ID_Max):
#     SecondFileName=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'/Temp_'+'RTr1a'+'_'+RecBatchID+'_'+str(X_ID)+'/RTr1a_'+RecBatchID+'_hit_cluster_rec_set_'+str(X_ID)+'_' +str(i)+'.csv' #keep loading subsequent files along y-xis with reconstructed clusters that already have been merged along z-axis
#     SecondFile=pd.read_csv(SecondFileName)
#     SecondFile.drop_duplicates(subset=['HitID', 'Master_z', 'Master_Segment_ID'],keep='first',inplace=True)
#     SecondFile["HitID"] = SecondFile["HitID"].astype(str)
#     SecondFileTable=SecondFile.rename(columns={"Master_Segment_ID":"Segment_ID","Master_z":"z" }) #Initally the following clusters are downgraded from the master status
#     FileClean=pd.merge(ZContractedTable.drop_duplicates(subset=["Master_Segment_ID","HitID",'Master_z'],keep='first'),SecondFileTable,how='inner', on=['HitID']) #Join segments based on the common hits
#     FileClean["Segment_No"]= FileClean["Segment_ID"]
#     FileClean=FileClean.groupby(by=["Master_Segment_ID","Segment_ID"])["Segment_No"].count().reset_index()
#     FileCleanTot=FileClean.groupby(by=["Master_Segment_ID"])["Segment_No"].sum().reset_index()
#     FileCleanTot.rename(columns={"Segment_No":"Segment_No_Tot"},inplace=True)
#     FileClean=pd.merge(FileClean,FileCleanTot,how='inner', on=["Master_Segment_ID"])
#     FileClean=FileClean.sort_values(["Master_Segment_ID","Segment_No"],ascending=[1,0])
#     FileClean.drop_duplicates(subset=["Master_Segment_ID"],keep='first',inplace=True)  #Keep the best matching segment
#     FileClean=pd.merge(FileClean,SecondFileTable,how='right', on=['Segment_ID'])
#     FileClean["Master_Segment_ID"] = FileClean["Master_Segment_ID"].fillna(FileClean["Segment_ID"])  #All segments that did not have overlapping hits with the master segment become masters themselves and become part of the Pacman
#     FileClean["Segment_No"] = FileClean["Segment_No"].fillna(0)
#     FileClean["Segment_No_Tot"] = FileClean["Segment_No_Tot"].fillna(0)
#     FileClean=FileClean.rename(columns={"z": "Master_z" })
#     FileClean=FileClean.drop(['Segment_ID'],axis=1)
#     ZContractedTable=pd.concat([ZContractedTable,FileClean])
#     ZContractedTable_r=ZContractedTable[['Master_Segment_ID','Segment_No','Segment_No_Tot']]
#     ZContractedTable_r.drop_duplicates(subset=['Master_Segment_ID','Segment_No','Segment_No_Tot'],keep='first',inplace=True) #Working the fit of the track (The overlapping clusters reinforce the fit of the final track)
#     ZContractedTable_r=ZContractedTable_r.groupby(['Master_Segment_ID']).agg({'Segment_No':'sum','Segment_No_Tot':'sum'}).reset_index()
#     ZContractedTable=ZContractedTable.drop(['Segment_No','Segment_No_Tot'],axis=1)
#     ZContractedTable=pd.merge(ZContractedTable,ZContractedTable_r,how='inner', on=["Master_Segment_ID"])
#     ZContractedTable.drop_duplicates(subset=['HitID', 'Master_z', 'Master_Segment_ID', 'Segment_No', 'Segment_No_Tot'],keep='first',inplace=True)
#     ZContractedTable["HitID"] = ZContractedTable["HitID"].astype(str)
# ZContractedTable=ZContractedTable.sort_values(["Master_Segment_ID",'Master_z'],ascending=[1,1])
# output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(0)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(X_ID)+sfx
# ZContractedTable.to_csv(output_file_location,index=False)
# print(UI.TimeStamp(),'Output is written to ',output_file_location) #Write the final result
# exit()
