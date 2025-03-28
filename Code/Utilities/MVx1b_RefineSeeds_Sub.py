
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
parser.add_argument('--MaxDOCA',help="Maximum DOCA allowed", default='50')
parser.add_argument('--MaxAngle',help="Maximum magnitude of angle allowed", default='1')
parser.add_argument('--MaxDST',help="", default='50')
parser.add_argument('--MaxVXT',help="", default='4000')
parser.add_argument('--i',help="Set number", default='1')
parser.add_argument('--j',help="Subset number", default='1')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
parser.add_argument('--EOS',help="EOS location", default='')
parser.add_argument('--AFS',help="AFS location", default='')
parser.add_argument('--BatchID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--PY',help="Python libraries directory location", default='.')
parser.add_argument('--FiducialVolumeCut',help="", default='')
########################################     Main body functions    #########################################
args = parser.parse_args()
i=int(args.i)    #This is just used to name the output file
j=int(args.j)  #The subset helps to determine what portion of the track list is used to create the Seeds
p=args.p
o=args.o
sfx=args.sfx
pfx=args.pfx
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
import U_UI as UI
from U_EMO import EMO
import ast

import pandas as pd #We use Panda for a routine data processing
import gc  #Helps to clear memory
FiducialVolumeCut=ast.literal_eval(args.FiducialVolumeCut)
BatchID=args.BatchID
Metas=[]
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
#
MaxDOCA=float(args.MaxDOCA)
MaxDST=float(args.MaxDST)
MaxVXT=float(args.MaxVXT)
MaxAngle=float(args.MaxAngle)
input_segment_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+BatchID+'/MVx1_'+BatchID+'_TRACK_SEGMENTS_'+str(i)+'.csv'
input_track_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+BatchID+'/Temp_MVx1a_'+BatchID+'_'+str(i)+'/MVx1a_'+BatchID+'_SelectedSeeds_'+str(i)+'_'+str(j)+'.csv'
output_file_location=EOS_DIR+'/'+p+'/Temp_'+pfx+'_'+BatchID+'_'+str(i)+'/'+pfx+'_'+BatchID+'_'+o+'_'+str(i)+'_'+str(j)+sfx
print(UI.TimeStamp(),'Loading the data')
tracks=pd.read_csv(input_track_file_location)
tracks_1=tracks.drop(['Segment_2'],axis=1)
tracks_1=tracks_1.rename(columns={"Segment_1": "Rec_Seg_ID"})
tracks_2=tracks.drop(['Segment_1'],axis=1)
tracks_2=tracks_2.rename(columns={"Segment_2": "Rec_Seg_ID"})
track_list=result = pd.concat([tracks_1,tracks_2])
track_list=track_list.sort_values(['Rec_Seg_ID'])
track_list.drop_duplicates(subset="Rec_Seg_ID",keep='first',inplace=True)
segments=pd.read_csv(input_segment_file_location)
print(UI.TimeStamp(),'Analysing the data')
segments=pd.merge(segments, track_list, how="inner", on=["Rec_Seg_ID"]) #Shrinking the Track data so just a star hit for each segment is present.
segments["x"] = pd.to_numeric(segments["x"],downcast='float')
segments["y"] = pd.to_numeric(segments["y"],downcast='float')
segments["z"] = pd.to_numeric(segments["z"],downcast='float')
segments["tx"] = pd.to_numeric(segments["tx"],downcast='float')
segments["ty"] = pd.to_numeric(segments["ty"],downcast='float')

# reorder the columns
segments = segments[['x','y','z','tx','ty', 'Rec_Seg_ID', 'MC_VX_ID', 'Seed_Type']]
segments = segments.values.tolist() #Convirting the result to List data type
tracks = tracks.values.tolist() #Convirting the result to List data type

del tracks_1
del tracks_2
del track_list
gc.collect()
limit=len(tracks)
track_counter=0
print(UI.TimeStamp(),bcolors.OKGREEN+'Data has been successfully loaded and prepared..'+bcolors.ENDC)
#create seeds
GoodTracks=[]
print(UI.TimeStamp(),'Beginning the sample generation part...')
for s in range(0,limit):
      
        
     track=tracks.pop(0)

     label=track[2]
     track=EMO(track[:2])
     if label:
         num_label = 1
     else:
         num_label = 0
     track.LabelSeed(num_label)
     track.Decorate(segments)
     try:
        track.GetVXInfo()
     except:
       continue

     keep_seed=True

     if track.VertexQualityCheck(MaxDOCA, MaxVXT, MaxAngle, FiducialVolumeCut):
         if keep_seed:
            GoodTracks.append(track)
     else:
         del track
         continue

print(UI.TimeStamp(),bcolors.OKGREEN+'The sample generation has been completed..'+bcolors.ENDC)
del tracks
del segments
gc.collect()
print(UI.PickleOperations(output_file_location,'w', GoodTracks)[1])

