########################################################################################################################
#######################################  This simple script prepares data for CNN  #####################################




########################################    Import libraries    ########################################################
import argparse
import math
import ast
import os
import copy
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
parser.add_argument('--Sampling',help="How much sampling?", default='1.0')
parser.add_argument('--BatchID',help="Give name of the training ", default='SHIP_TrainSample_v1')
parser.add_argument('--TrainSampleID',help="Train Sample ID", default='1T_MC_1_model')
parser.add_argument('--i',help="Set number", default='1')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
parser.add_argument('--PY',help="Python libraries directory location", default='.')
########################################     Initialising Variables    #########################################
args = parser.parse_args()
TrainParams=ast.literal_eval(args.TrainParams)
TrainSampleID=args.TrainSampleID
ModelName=args.BatchID
##################################   Loading Directory locations   ##################################################
AFS_DIR=args.AFS
EOS_DIR=args.EOS
PY_DIR=args.PY
import sys
if PY_DIR!='': #Temp solution
    sys.path=['',PY_DIR]
    sys.path.append('/usr/lib64/python39.zip')
    sys.path.append('/usr/lib64/python3.9')
    sys.path.append('/usr/lib64/python3.9/lib-dynload')
    sys.path.append('/usr/lib64/python3.9/site-packages')
    sys.path.append('/usr/lib/python3.9/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')

import U_UI as UI
import U_ML as ML
import U_EMO as EMO
#Load data configuration
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'

##############################################################################################################################
TrainSampleInputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
print(UI.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+TrainSampleInputMeta+bcolors.ENDC)
MetaInput=UI.PickleOperations(TrainSampleInputMeta,'r', 'N/A')
print(MetaInput[1])
Meta=MetaInput[0]
Model_Meta_Path=EOSsubModelDIR+'/'+ModelName+'_Meta'
ModelMeta=UI.PickleOperations(Model_Meta_Path, 'r', 'N/A')[0]
ValSamples=UI.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_TRACK_OUTPUT.pkl','r', 'N/A')[0]
print(UI.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_TRACK_OUTPUT.pkl','r', 'N/A')[1])
train_set=1
if ModelMeta.ModelType=='CNN':
   Model_Path=EOSsubModelDIR+'/'+ModelName+'.keras'
   if len(ModelMeta.TrainSessionsData)==0:
       TrainSamples=UI.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_TRACK_OUTPUT_1.pkl','r', 'N/A')[0]
       print(UI.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_TRACK_OUTPUT_1.pkl','r', 'N/A')[1])
       train_set=1
   else:

       for el in range(max(len(ModelMeta.TrainSessionsDataID)-2,0),-1,-1):
        if ModelMeta.TrainSessionsDataID[el]==TrainSampleID:
           train_set=ModelMeta.TrainSessionsData[el][-1][8]+1

           next_file=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_TRACK_OUTPUT_'+str(train_set)+'.pkl'
           if os.path.isfile(next_file):
               TrainSamples=UI.PickleOperations(next_file,'r', 'N/A')[0]
               print(UI.PickleOperations(next_file,'r', 'N/A')[1])
           else:
               train_set=1
               TrainSamples=UI.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_TRACK_OUTPUT_1.pkl','r', 'N/A')[0]
               print(UI.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_TRACK_OUTPUT_1.pkl','r', 'N/A')[1])
        break
   NTrainBatches=math.ceil(float(len(TrainSamples))/float(TrainParams[1]))
   NValBatches=math.ceil(float(len(ValSamples))/float(TrainParams[1]))
   for ts in TrainSamples:
       ts.PrepareSeedPrint(ModelMeta)
   for vs in ValSamples:
       vs.PrepareSeedPrint(ModelMeta)

elif ModelMeta.ModelType=='GNN':
       import torch

       if len(ModelMeta.TrainSessionsData)==0:
           TrainSamples=UI.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_TRACK_OUTPUT_1.pkl','r', 'N/A')[0]
           print(UI.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_TRACK_OUTPUT_1.pkl','r', 'N/A')[1])
           train_set=1

       else:
           for el in range(max(len(ModelMeta.TrainSessionsDataID)-2,0),-1,-1):
            if ModelMeta.TrainSessionsDataID[el]==TrainSampleID:
               train_set=ModelMeta.TrainSessionsData[el][-1][8]+1
               next_file=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_TRACK_OUTPUT_'+str(train_set)+'.pkl'
               if os.path.isfile(next_file):
                   TrainSamples=UI.PickleOperations(next_file,'r', 'N/A')[0]
                   print(UI.PickleOperations(next_file,'r', 'N/A')[1])
               else:
                   train_set=1
                   TrainSamples=UI.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_TRACK_OUTPUT_1.pkl','r', 'N/A')[0]
                   print(UI.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_TRACK_OUTPUT_1.pkl','r', 'N/A')[1])
            break
       NTrainBatches=math.ceil(float(len(TrainSamples))/float(TrainParams[1]))
       NValBatches=math.ceil(float(len(ValSamples))/float(TrainParams[1]))
       for ts in TrainSamples:
           ts.PrepareSeedGraph(ModelMeta)
       train_dataset = []
       for smpl1 in TrainSamples:
        smpl1.GraphSeed.y = smpl1.GraphSeed.y
        train_dataset.append(copy.deepcopy(smpl1.GraphSeed))
       del TrainSamples

       for vs in ValSamples:
           vs.PrepareSeedGraph(ModelMeta)
       val_dataset = []
       for smpl in ValSamples:
        smpl.GraphSeed.y = smpl.GraphSeed.y
        val_dataset.append(copy.deepcopy(smpl.GraphSeed))
       del ValSamples
       import torch_geometric
       from torch_geometric.loader import DataLoader

       TrainSamples = DataLoader(train_dataset, batch_size=TrainParams[1], shuffle=True)
       ValSamples = DataLoader(val_dataset, batch_size=TrainParams[1], shuffle=False)

print(UI.TimeStamp(), bcolors.OKGREEN+"Train and Validation data has loaded and analysed successfully..."+bcolors.ENDC)
def main(self):
    Model_Meta_Path=EOSsubModelDIR+'/'+ModelName+'_Meta'
    ModelMeta=UI.PickleOperations(Model_Meta_Path, 'r', 'N/A')[0]
    print(UI.TimeStamp(),'Starting the training process... ')
    if ModelMeta.ModelType=='CNN':
        Model_Path=EOSsubModelDIR+'/'+ModelName+'.keras'
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
        import logging
        logging.getLogger('tensorflow').setLevel(logging.FATAL)
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        import tensorflow as tf
        import keras
        from keras import backend as K
        try:
            model=tf.keras.models.load_model(Model_Path)
            K.set_value(model.optimizer.learning_rate, TrainParams[0])
        except:
             print(UI.TimeStamp(), bcolors.WARNING+"Model/state data files are missing, skipping this step..." +bcolors.ENDC)
             model = ML.GenerateModel(ModelMeta,TrainParams)
        model.summary()
        for el in ModelMeta.ModelParameters:
          if len(el)==2:
             OutputSize=el[1]
        records=[]
        for epoch in range(0, TrainParams[2]):
            train_loss, itr=ML.CNNtrain(model, TrainSamples, NTrainBatches,OutputSize,TrainParams[1]),len(TrainSamples)
            val_loss=ML.CNNvalidate(model, ValSamples, NValBatches,OutputSize,TrainParams[1])
            test_loss=val_loss
            print(UI.TimeStamp(),'Epoch ',epoch, ' is completed')
            records.append([epoch,itr,train_loss[0],0.5,val_loss[0],val_loss[1],test_loss[0],test_loss[1],train_set])
        model.save(Model_Path)
        Header=[['Epoch','# Samples','Train Loss','Optimal Threshold','Validation Loss','Validation Accuracy','Test Loss','Test Accuracy','Training Set']]
        Header+=records
        ModelMeta.CompleteTrainingSession(Header)
        print(UI.PickleOperations(Model_Meta_Path, 'w', ModelMeta)[1])
        exit()
    elif ModelMeta.ModelType=='GNN':
        from torch import optim
        from torch.optim.lr_scheduler import StepLR
        print(UI.TimeStamp(),'Starting the training process... ')
        State_Save_Path=EOSsubModelDIR+'/'+ModelName+'_State'
        Model_Meta_Path=EOSsubModelDIR+'/'+ModelName+'_Meta'
        Model_Path=EOSsubModelDIR+'/'+ModelName
        ModelMeta=UI.PickleOperations(Model_Meta_Path, 'r', 'N/A')[0]
        device = torch.device('cpu')
        model = ML.GenerateModel(ModelMeta).to(device)
        optimizer = optim.Adam(model.parameters(), lr=TrainParams[0])
        scheduler = StepLR(optimizer, step_size=0.1,gamma=0.1)
        print(UI.TimeStamp(),'Try to load the previously saved model/optimiser state files ')
        try:
               model.load_state_dict(torch.load(Model_Path))
               checkpoint = torch.load(State_Save_Path)
               optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
               scheduler.load_state_dict(checkpoint['scheduler'])
        except:
               print(UI.TimeStamp(), bcolors.WARNING+"Model/state data files are missing, skipping this step..." +bcolors.ENDC)
        records=[]
        if ModelMeta.ModelParameters[10][1]==1:
           criterion=torch.nn.MSELoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(0, TrainParams[2]):
            train_loss,itr= ML.GNNtrain(model,TrainSamples, optimizer,criterion),len(TrainSamples.dataset)
            print(train_loss)
            val=ML.GNNvalidate(model,  ValSamples, criterion)
            val_loss=val[1]
            val_acc=val[0]
            test_loss=val_loss
            test_acc=val_acc
            scheduler.step()
            print(UI.TimeStamp(),'Epoch ',epoch, ' is completed')
            records.append([epoch,itr,train_loss.item(),0.5,val_loss,val_acc,test_loss,test_acc,train_set])
            torch.save({    'epoch': epoch,
                          'optimizer_state_dict': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict(),    # HERE IS THE CHANGE
                          }, State_Save_Path)
        torch.save(model.state_dict(), Model_Path)
        Header=[['Epoch','# Samples','Train Loss','Optimal Threshold','Validation Loss','Validation Accuracy','Test Loss','Test Accuracy','Training Set']]
        Header+=records
        ModelMeta.CompleteTrainingSession(Header)
        print(UI.PickleOperations(Model_Meta_Path, 'w', ModelMeta)[1])
        exit()
if __name__ == '__main__':
     main(sys.argv[1:])


