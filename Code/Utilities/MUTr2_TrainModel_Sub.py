########################################################################################################################
#######################################  This simple script prepares data for CNN  #####################################




########################################    Import libraries    ########################################################
import argparse
import math
import ast
# import torch
# from torch import optim
# from torch.optim.lr_scheduler import StepLR
# import torch.nn.functional as F
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"
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
parser.add_argument('--TrainParams',help="Please enter the train params: '[<Session No>, <Learning Rate>, <Batch size>, <Epochs>]'", default='[1, 0.0001, 4, 10]')
parser.add_argument('--AFS',help="Please enter the user afs directory", default='.')
parser.add_argument('--EOS',help="Please enter the user eos directory", default='.')
parser.add_argument('--BatchID',help="Give name of the training ", default='SHIP_TrainSample_v1')
parser.add_argument('--ModelName',help="Name of the model", default='1T_MC_1_model')
parser.add_argument('--i',help="Set number", default='1')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
########################################     Initialising Variables    #########################################
args = parser.parse_args()
TrainParams=ast.literal_eval(args.TrainParams)
TrainSampleID=args.BatchID
ModelName=args.ModelName
##################################   Loading Directory locations   ##################################################
AFS_DIR=args.AFS
EOS_DIR=args.EOS
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import UtilityFunctions as UF
#Load data configuration
EOSsubDIR=EOS_DIR+'/'+'ANNADEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'


##############################################################################################################################
######################################### Starting the program ################################################################
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################  Initialising     ANNADEA   model creation module   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
def zero_divide(a, b):
    if (b==0): return 0
    return a/b

def CNNtrain(model, Sample, Batches):

    for ib in range(2480,Batches):
        StartSeed=(ib*TrainParams[1])+1
        EndSeed=StartSeed+TrainParams[1]-1
        BatchImages=UF.LoadRenderImages(Sample,StartSeed,EndSeed)
        t=model.train_on_batch(BatchImages[0],BatchImages[1],reset_metrics=False)
    return t

def CNNvalidate(model, Sample, Batches):
    for ib in range(2350,Batches):
        StartSeed=(ib*TrainParams[1])+1
        EndSeed=StartSeed+TrainParams[1]-1
        BatchImages=UF.LoadRenderImages(Sample,StartSeed,EndSeed)
        v=model.test_on_batch(BatchImages[0],BatchImages[1],reset_metrics=False)
    return v



TrainSampleInputMeta=EOS_DIR+'/ANNADEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+TrainSampleInputMeta+bcolors.ENDC)
MetaInput=UF.PickleOperations(TrainSampleInputMeta,'r', 'N/A')
print(MetaInput[1])
Meta=MetaInput[0]
Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
Model_Path=EOSsubModelDIR+'/'+args.ModelName
ModelMeta=UF.PickleOperations(Model_Meta_Path, 'r', 'N/A')[0]
ValSamples=UF.PickleOperations(EOS_DIR+'/ANNADEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_TRACK_SEEDS_OUTPUT.pkl','r', 'N/A')[0]
if ModelMeta.ModelType=='CNN':
   if len(ModelMeta.TrainSessionsData)==0:
       TrainSamples=UF.PickleOperations(EOS_DIR+'/ANNADEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_TRACK_SEEDS_OUTPUT_1.pkl','r', 'N/A')[0]
       train_set=1
   else:
       print(ModelMeta.TrainSessionsData)
       print(ModelMeta.TrainSessionsDataID)
       exit()
   NTrainBatches=math.ceil(float(len(TrainSamples))/float(TrainParams[1]))
   NValBatches=math.ceil(float(len(ValSamples))/float(TrainParams[1]))
   for ts in TrainSamples:
       ts.PrepareTrackPrint(ModelMeta)
   for vs in ValSamples:
       vs.PrepareTrackPrint(ModelMeta)

print(UF.TimeStamp(), bcolors.OKGREEN+"Train and Validation data has loaded and analysed successfully..."+bcolors.ENDC)

def main(self):
    print(UF.TimeStamp(),'Starting the training process... ')
    if ModelMeta.ModelType=='CNN':
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
        import logging
        logging.getLogger('tensorflow').setLevel(logging.FATAL)
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        import tensorflow as tf
        from tensorflow import keras
        from keras import backend as K
        try:
            print(EOSsubModelDIR+'/'+ModelName)
            model=tf.keras.models.load_model(EOSsubModelDIR+ModelName)
            K.set_value(model.optimizer.learning_rate, TrainParams[1])
        except:
            print(UF.TimeStamp(), bcolors.WARNING+"Model/state data files are missing, skipping this step..." +bcolors.ENDC)
            model = UF.GenerateModel(ModelMeta,TrainParams)
        model.summary()
        records=[]
        for epoch in range(0, TrainParams[2]):
            train_loss, itr=CNNtrain(model, TrainSamples, NTrainBatches),len(TrainSamples)
            val_loss=CNNvalidate(model, ValSamples, NValBatches)
            test_loss=val_loss
            print(UF.TimeStamp(),'Epoch ',epoch, ' is completed')
            records.append([epoch,itr,train_loss[0],0.5,val_loss[0],val_loss[1],test_loss[0],test_loss[1],train_set])
        Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
        Model_Path=EOSsubModelDIR+'/'+args.ModelName
        model.save(Model_Path)
        Header=[['Epoch','# Samples','Train Loss','Optimal Threshold','Validation Loss','Validation Accuracy','Test Loss','Test Accuracy','Training Set']]
        Header+=records
        ModelMeta.CompleteTrainingSession(Header)
        print(UF.PickleOperations(Model_Meta_Path, 'w', ModelMeta)[1])
        exit()
if __name__ == '__main__':
     main(sys.argv[1:])


