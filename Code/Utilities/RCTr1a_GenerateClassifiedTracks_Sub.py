#This simple script prepares 2-segment track seeds for the initial CNN/GNN union
# Part of ANNDEA package
#Made by Filips Fedotovs
#Current version 1.0

########################################    Import libraries    #############################################
import argparse
import sys
########################################    Import libraries    #############################################
import argparse





######################################## Set variables  #############################################################
#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--i',help="Set number", default='1')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')
parser.add_argument('--BatchID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--MaxSegments',help="A maximum number of track combinations that will be used in a particular HTCondor job for this script", default='20000')
parser.add_argument('--ModelName',help="WHat ANN model would you like to use?", default='MH_GNN_5FTR_4_120_4_120')
parser.add_argument('--PY',help="Python libraries directory location", default='.')

######################################## Set variables  #############################################################
args = parser.parse_args()
i=int(args.i)    #This is just used to name the output file
p=args.p
o=args.o
sfx=args.sfx
pfx=args.pfx
BatchID=args.BatchID
ModelName=args.ModelName

########################################     Preset framework parameters    #########################################
MaxSegments=int(args.MaxSegments)
#Loading Directory locations
EOS_DIR=args.EOS
AFS_DIR=args.AFS
PY_DIR=args.PY
if PY_DIR!='': #Temp solution
    sys.path=['',PY_DIR]
    sys.path.append('/usr/lib64/python39.zip')
    sys.path.append('/usr/lib64/python3.9')
    sys.path.append('/usr/lib64/python3.9/lib-dynload')
    sys.path.append('/usr/lib64/python3.9/site-packages')
    sys.path.append('/usr/lib/python3.9/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')
import U_UI as UI #This is where we keep routine utility functions
import U_ML as ML #This is where we keep routine utility functions
import pandas as pd #We use Panda for a routine data processing
import gc  #Helps to clear memory
from U_EMO import EMO
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
Model_Meta_Path=EOSsubModelDIR+'/'+ModelName+'_Meta'
Model_Path=EOSsubModelDIR+'/'+ModelName
ModelMeta=UI.PickleOperations(Model_Meta_Path, 'r', 'N/A')[0]
if ModelMeta.ModelFramework=='Tensorflow':
        import tensorflow as tf
        from tensorflow import keras
        Model_Path=EOSsubModelDIR+'/'+ModelName+'.keras'
        model=tf.keras.models.load_model(Model_Path)
if ModelMeta.ModelFramework=='PyTorch':
        import torch
        from torch import optim
        Model_Meta_Path=EOSsubModelDIR+'/'+ModelName+'_Meta'
        Model_Path=EOSsubModelDIR+'/'+ModelName
        ModelMeta=UI.PickleOperations(Model_Meta_Path, 'r', 'N/A')[0]
        device = torch.device('cpu')
        model = ML.GenerateModel(ModelMeta).to(device)
        model.load_state_dict(torch.load(Model_Path))


#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+BatchID+'/RCTr1_'+BatchID+'_TRACKS.csv'
output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+BatchID+'_0/'+pfx+'_'+BatchID+'_'+o+'_'+str(i)+sfx
print(UI.TimeStamp(), "Modules Have been imported successfully...")
print(UI.TimeStamp(),'Loading pre-selected data from ',input_file_location)
data=pd.read_csv(input_file_location,header=0,
                    usecols=['x','y','z','tx','ty','Rec_Seg_ID'])
track_headers = data[['Rec_Seg_ID']]
track_headers = track_headers.drop_duplicates(subset=['Rec_Seg_ID'],keep='first')
track_column_headers=track_headers.columns.values.tolist()
track_headers=track_headers.values.tolist()
track_data = data[['x','y','z','tx','ty','Rec_Seg_ID']].values.tolist() #Convirting the result to List data type
track_headers = track_headers[int(i)*MaxSegments : min((int(i)+1)*MaxSegments, len(track_headers))]

gc.collect()
track_counter=0
print('Data has been successfully loaded and prepared..')
#create seeds
GoodTracks=[]
print(UI.TimeStamp(),'Beginning the image generation part...')
limit = len(track_headers)

for s in range(0,limit):

     track=track_headers.pop(0)
     track=EMO([track[0]])
     track.Decorate(track_data)
     track.ClassifySeed(ModelMeta,model)
     GoodTracks.append(track)

print(UI.TimeStamp(),'The track classification has been completed..')
print(UI.TimeStamp(),'Saving the results..')
print(UI.PickleOperations(output_file_location,'w', GoodTracks)[1])
#End of the script



