
#This simple script prepares data for CNN
########################################    Import libraries    #############################################
import sys
import argparse

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--i',help="Set number", default='1')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
parser.add_argument('--EOS',help="EOS location", default='')
parser.add_argument('--AFS',help="AFS location", default='')
parser.add_argument('--PY',help="Py lib location", default='')
parser.add_argument('--MaxSLG',help="Maximum allowed longitudinal gap value between segments", default='8000')
parser.add_argument('--BatchID',help="Give this training sample batch an ID", default='SHIP_UR_v1')

########################################     Main body functions    #########################################
args = parser.parse_args()
i=int(args.i)    #This is just used to name the output file
p=args.p
o=args.o
sfx=args.sfx
pfx=args.pfx
AFS_DIR=args.AFS
EOS_DIR=args.EOS
PY_DIR=args.PY
BatchID=args.BatchID
MaxSLG=float(args.MaxSLG)
if PY_DIR!='': #Temp solution
    sys.path=['',PY_DIR]
    sys.path.append('/usr/lib64/python36.zip')
    sys.path.append('/usr/lib64/python3.6')
    sys.path.append('/usr/lib64/python3.6/lib-dynload')
    sys.path.append('/usr/lib64/python3.6/site-packages')
    sys.path.append('/usr/lib/python3.6/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')
import U_UI as UI
import pandas as pd

input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+BatchID+'/RUTr1d_'+BatchID+'_Fit_Filtered_Seeds.pkl'

output_file_location=EOS_DIR+'/'+p+'/Temp_'+pfx+'_'+BatchID+'_0/'+pfx+'_'+BatchID+'_'+o+'_'+str(i)+sfx

print(UI.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(UI.TimeStamp(), "Loading fit track seeds from the file",bcolors.OKBLUE+input_file_location+bcolors.ENDC)

base_data=UI.PickleOperations(input_file_location,'r', 'N/A')[0]
rec_list=[]
for rd in base_data:
    rec_list.append([rd.Header[0],rd.Header[1]])
    rec = pd.DataFrame(rec_list, columns = ['Segment_1','Segment_2'])
r1_rec=rec[['Segment_1']].rename(columns={'Segment_1':"Segment"})
r2_rec=rec[['Segment_2']].rename(columns={'Segment_2':"Segment"})
r1_rec['count']=1
r2_rec['count']=1
r_rec=pd.concat([r1_rec,r2_rec])
r_rec=r_rec.groupby(['Segment'])['count'].sum().reset_index()
r_rec=r_rec.rename(columns={'Segment':"Segment_2",'count':'r_count'})
l_rec=r_rec.rename(columns={'Segment_2':"Segment_1",'r_count':'l_count'})

rec=pd.merge(rec,r_rec,how='left',on='Segment_2')
rec=pd.merge(rec,l_rec,how='left',on='Segment_1')
rec['tot_count']=rec['l_count']+rec['r_count']
rec.drop(['l_count','r_count'],axis=1,inplace=True)
print(rec)
if i==0:
    rec = rec[rec.tot_count == 2]
rec.drop(['tot_count'],axis=1,inplace=True)
print(rec)
exit()
# rec=pd.merge(rec,r2_rec,how='left',on='Segment_2')
#
#
# print(rec)
# exit()
print(UI.TimeStamp(), bcolors.OKGREEN+"Loading is successful, there are total of "+str(len(base_data))+" glued tracks..."+bcolors.ENDC)
print(UI.TimeStamp(), "Initiating the  track merging...")
InitialDataLength=len(base_data)
SeedCounter=0
SeedCounterContinue=True
while SeedCounterContinue:
         if SeedCounter==len(base_data):
                           SeedCounterContinue=False
                           break
         SubjectSeed=base_data[SeedCounter]
         for ObjectSeed in base_data[SeedCounter+1:]:
                  if MaxSLG>=0:
                    if SubjectSeed.InjectDistantTrackSeed(ObjectSeed):
                        base_data.pop(base_data.index(ObjectSeed))
                  else:
                    if SubjectSeed.InjectTrackSeed(ObjectSeed):
                        base_data.pop(base_data.index(ObjectSeed))
         SeedCounter+=1
         print(SeedCounter)
print(str(InitialDataLength), "segment pairs from different files were merged into", str(len(base_data)), 'tracks...')
exit()
print(UI.PickleOperations(output_file_location,'w', base_data)[1])

