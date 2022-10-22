
#This simple script prepares data for CNN
########################################    Import libraries    #############################################
import UtilityFunctions as UF
from UtilityFunctions import EMO
import ast
import argparse
import pandas as pd #We use Panda for a routine data processing
import gc  #Helps to clear memory
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
parser.add_argument('--MaxSTG',help="Maximum Segment Transverse gap per SLG", default='50')
parser.add_argument('--MaxSLG',help="Maximum Segment Longitudinal Gap", default='4000')
parser.add_argument('--i',help="Set number", default='1')
parser.add_argument('--j',help="Subset number", default='1')
parser.add_argument('--k',help="Fraction number", default='1')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
parser.add_argument('--EOS',help="EOS location", default='')
parser.add_argument('--AFS',help="AFS location", default='')
parser.add_argument('--ModelName',help="WHat GNN model would you like to use?", default="['MH_GNN_5FTR_4_120_4_120']")
parser.add_argument('--BatchID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
########################################     Main body functions    #########################################
args = parser.parse_args()
i=int(args.i)    #This is just used to name the output file
j=int(args.j)  #The subset helps to determine what portion of the track list is used to create the Seeds
k=int(args.k)  #The subset helps to determine what portion of the track list is used to create the Seeds
p=args.p
o=args.o
sfx=args.sfx
pfx=args.pfx
AFS_DIR=args.AFS
EOS_DIR=args.EOS
ModelName=ast.literal_eval(args.ModelName)
BatchID=args.BatchID
# if PreFit:
#     resolution=float(args.resolution)
#     acceptance=float(args.acceptance)
#     MaxX=float(args.MaxX)
#     MaxY=float(args.MaxY)
#     MaxZ=float(args.MaxZ)
#     print(UF.TimeStamp(),'Loading the model...')
#     #Load the model
#     import tensorflow as tf
#     from tensorflow import keras
#     model_name=EOS_DIR+'/EDER-TSU/Models/'+args.ModelName
#     model=tf.keras.models.load_model(model_name)
MaxDOCA=float(args.MaxDOCA)
MaxSTG=float(args.MaxSTG)
MaxSLG=float(args.MaxSLG)
MaxAngle=float(args.MaxAngle)
input_segment_file_location=EOS_DIR+'/ANNADEA/Data/TRAIN_SET/MUTr1_'+BatchID+'_TRACK_SEGMENTS.csv'
input_track_file_location=EOS_DIR+'/ANNADEA/Data/TRAIN_SET/MUTr1a_'+BatchID+'_SelectedSeeds_'+str(i)+'_'+str(j)+'_'+str(k)+'.csv'
output_file_location=EOS_DIR+'/'+p+'/'+pfx+'_'+BatchID+'_'+o+'_'+str(i)+'_'+str(j)+'_'+str(k)+sfx
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
segments = segments[['x','y','z','tx','ty', 'Rec_Seg_ID', 'MC_Mother_Track_ID', 'Seed_Type']]
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
GoodTracks=[]
print(UF.TimeStamp(),'Beginning the sample generation part...')
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
       track.GetTrInfo()
     except:
       continue
#     if track.GeoFit and PreFit:
#                track.PrepareTrackPrint(2000.0,500.0,20000.0,50,True)
#                TrackImage=UF.LoadRenderImages([track],1,1,numClasses)[0]
#                track.UnloadTrackPrint()
#                track.CNNFitTrack(model.predict(TrackImage)[0][1])
#                if track.Track_CNN_Fit>=acceptance:
#                   GoodTracks.append(track)
     if track.TrackQualityCheck(MaxDOCA,MaxSLG,MaxSTG, MaxAngle):
           if track.Label:
              print(track.Label)
           GoodTracks.append(track.TrackQualityCheck)
     else:
         del track
         continue
print(UF.TimeStamp(),bcolors.OKGREEN+'The sample generation has been completed..'+bcolors.ENDC)
del tracks
del segments
gc.collect()
print(UF.PickleOperations(output_file_location,'w', GoodTracks)[1])

