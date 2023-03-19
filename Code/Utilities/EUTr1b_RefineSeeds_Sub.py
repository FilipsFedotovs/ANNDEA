
#This simple script prepares data for CNN

import argparse
import sys
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
parser.add_argument('--EOS',help="EOS location", default='')
parser.add_argument('--AFS',help="AFS location", default='')
parser.add_argument('--BatchID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--PY',help="Python libraries directory location", default='.')
########################################     Main body functions    #########################################
args = parser.parse_args()

AFS_DIR=args.AFS
EOS_DIR=args.EOS
PY_DIR=args.PY

BatchID=args.BatchID

########################################    Import libraries    #############################################

if PY_DIR!='': #Temp solution
    sys.path=['',PY_DIR]
    sys.path.append('/usr/lib64/python36.zip')
    sys.path.append('/usr/lib64/python3.6')
    sys.path.append('/usr/lib64/python3.6/lib-dynload')
    sys.path.append('/usr/lib64/python3.6/site-packages')
    sys.path.append('/usr/lib/python3.6/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')
import pandas as pd #We use Panda for a routine data processing
import gc  #Helps to clear memory
import UtilityFunctions as UF
from UtilityFunctions import EMO

input_segment_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/EUTr1_'+BatchID+'_TRACK_SEGMENTS.csv'
input_track_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/EUTr1b_'+BatchID+'_SEED_TRUTH_COMBINATIONS.csv'
output_file_location=EOS_DIR+'/ANNDEA/Data/TEST_SET/'+BatchID+'_SEED_TRUTH_ANALYSIS.csv'
print(UF.TimeStamp(),'Loading the data')
tracks=pd.read_csv(input_track_file_location)
tracks_1=tracks.drop(['Segment_2'],axis=1)
tracks_1=tracks_1.rename(columns={"Segment_1": "Rec_Seg_ID"})
tracks_2=tracks.drop(['Segment_1'],axis=1)
tracks_2=tracks_2.rename(columns={"Segment_2": "Rec_Seg_ID"})
track_list=result = pd.concat([tracks_1,tracks_2])
track_list=track_list.sort_values(['Rec_Seg_ID'])
track_list.drop_duplicates(subset="Rec_Seg_ID",keep='first',inplace=True)
segments=pd.read_csv(input_segment_file_location)
print(UF.TimeStamp(),'Analysing the data')
segments=pd.merge(segments, track_list, how="inner", on=["Rec_Seg_ID"]) #Shrinking the Track data so just a star hit for each segment is present.
segments["x"] = pd.to_numeric(segments["x"],downcast='float')
segments["y"] = pd.to_numeric(segments["y"],downcast='float')
segments["z"] = pd.to_numeric(segments["z"],downcast='float')
segments["tx"] = pd.to_numeric(segments["tx"],downcast='float')
segments["ty"] = pd.to_numeric(segments["ty"],downcast='float')

# reorder the columns
segments = segments[['x','y','z','tx','ty', 'Rec_Seg_ID']]
segments = segments.values.tolist() #Convirting the result to List data type
tracks = tracks.values.tolist() #Convirting the result to List data type

del tracks_1
del tracks_2
del track_list
gc.collect()
limit=len(tracks)
track_counter=0
print(UF.TimeStamp(),bcolors.OKGREEN+'Data has been successfully loaded and prepared..'+bcolors.ENDC)
#create seeds
GoodTracks=[['DOCA','SLG','STG','Opening_Angle']]
print(UF.TimeStamp(),'Beginning the sample analysis part...')
from alive_progress import alive_bar
with alive_bar(limit,force_tty=True, title='Analysing data...') as bar:
    for s in range(0,limit):
     bar()
     track=tracks.pop(0)

     track=EMO(track[:2])

     track.Decorate(segments)
     try:
       track.GetTrInfo()
     except:
       continue
     GoodTracks.append([track.DOCA,track.SLG,track.STG,track.Opening_Angle])

print(UF.TimeStamp(),bcolors.OKGREEN+'The sample generation has been completed..'+bcolors.ENDC)
del tracks
del segments
gc.collect()
UF.LogOperations(output_file_location,'w', GoodTracks)

