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
print(TrainParams[0])
TrainSampleID=args.TrainSampleID
ModelName=args.BatchID
##################################   Loading Directory locations   ##################################################
AFS_DIR=args.AFS
EOS_DIR=args.EOS
PY_DIR=args.PY
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import UtilityFunctions as UF
#Load data configuration
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'


##############################################################################################################################
######################################### Starting the program ################################################################
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################  Initialising     ANNDEA   model creation module   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
def zero_divide(a, b):
    if (b==0): return 0
    return a/b

def CNNtrain(model, Sample, Batches,num_classes=2):

    for ib in range(Batches):
        StartSeed=(ib*TrainParams[1])+1
        EndSeed=StartSeed+TrainParams[1]-1
        BatchImages=UF.LoadRenderImages(Sample,StartSeed,EndSeed,num_classes)
        t=model.train_on_batch(BatchImages[0],BatchImages[1],reset_metrics=False)
    return t

def GNNtrain(model, Sample,optimizer):
    model.train()
    for data in Sample:
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y)
        print(loss)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()

    return loss

def GNNvalidate(model, Sample):
    model.eval()
    correct = 0
    loss_accumulative = 0
    for data in Sample:
         out = model(data.x, data.edge_index, data.edge_attr, data.batch)
         pred = out.argmax(dim=1)  # Use the class with the highest probability.
         y_index = data.y.argmax(dim=1)
         correct += int((pred == y_index).sum())  # Check against ground-truth labels.
         loss = criterion(out, data.y)
         loss_accumulative += float(loss)
    return (correct / len(Sample.dataset), loss_accumulative/len(Sample.dataset))

def CNNvalidate(model, Sample, Batches,num_classes=2):
    for ib in range(Batches):
        StartSeed=(ib*TrainParams[1])+1
        EndSeed=StartSeed+TrainParams[1]-1
        BatchImages=UF.LoadRenderImages(Sample,StartSeed,EndSeed,num_classes)
        v=model.test_on_batch(BatchImages[0],BatchImages[1],reset_metrics=False)
    return v



TrainSampleInputMeta=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_info.pkl'
print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+TrainSampleInputMeta+bcolors.ENDC)
MetaInput=UF.PickleOperations(TrainSampleInputMeta,'r', 'N/A')
print(MetaInput[1])
Meta=MetaInput[0]
Model_Meta_Path=EOSsubModelDIR+'/'+ModelName+'_Meta'
Model_Path=EOSsubModelDIR+'/'+ModelName
ModelMeta=UF.PickleOperations(Model_Meta_Path, 'r', 'N/A')[0]
ValSamples=UF.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_VERTEX_SEEDS_OUTPUT.pkl','r', 'N/A')[0]
print(UF.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_VERTEX_SEEDS_OUTPUT.pkl','r', 'N/A')[1])
train_set=1
if ModelMeta.ModelType=='CNN':
   if len(ModelMeta.TrainSessionsData)==0:
       TrainSamples=UF.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_VERTEX_SEEDS_OUTPUT_1.pkl','r', 'N/A')[0]
       print(UF.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_VERTEX_SEEDS_OUTPUT_1.pkl','r', 'N/A')[1])
       train_set=1
   else:

       for el in range(max(len(ModelMeta.TrainSessionsDataID)-2,0),-1,-1):
        if ModelMeta.TrainSessionsDataID[el]==TrainSampleID:
           train_set=ModelMeta.TrainSessionsData[el][-1][8]+1

           next_file=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_VERTEX_SEEDS_OUTPUT_'+str(train_set)+'.pkl'
           if os.path.isfile(next_file):
               TrainSamples=UF.PickleOperations(next_file,'r', 'N/A')[0]
               print(UF.PickleOperations(next_file,'r', 'N/A')[1])
           else:
               train_set=1
               TrainSamples=UF.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_VERTEX_SEEDS_OUTPUT_1.pkl','r', 'N/A')[0]
               print(UF.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_VERTEX_SEEDS_OUTPUT_1.pkl','r', 'N/A')[1])
        break
   NTrainBatches=math.ceil(float(len(TrainSamples))/float(TrainParams[1]))
   NValBatches=math.ceil(float(len(ValSamples))/float(TrainParams[1]))
   for ts in TrainSamples:
       ts.PrepareSeedPrint(ModelMeta)
   for vs in ValSamples:
       vs.PrepareSeedPrint(ModelMeta)

elif ModelMeta.ModelType=='GNN':
       import torch
       criterion = torch.nn.CrossEntropyLoss()
       if len(ModelMeta.TrainSessionsData)==0:
           TrainSamples=UF.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_VERTEX_SEEDS_OUTPUT_1.pkl','r', 'N/A')[0]
           print(UF.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_VERTEX_SEEDS_OUTPUT_1.pkl','r', 'N/A')[1])
           train_set=1

       else:
           for el in range(max(len(ModelMeta.TrainSessionsDataID)-2,0),-1,-1):
            if ModelMeta.TrainSessionsDataID[el]==TrainSampleID:
               train_set=ModelMeta.TrainSessionsData[el][-1][8]+1
               next_file=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_VERTEX_SEEDS_OUTPUT_'+str(train_set)+'.pkl'
               if os.path.isfile(next_file):
                   TrainSamples=UF.PickleOperations(next_file,'r', 'N/A')[0]
                   print(UF.PickleOperations(next_file,'r', 'N/A')[1])
               else:
                   train_set=1
                   TrainSamples=UF.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_VERTEX_SEEDS_OUTPUT_1.pkl','r', 'N/A')[0]
                   print(UF.PickleOperations(EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_VERTEX_SEEDS_OUTPUT_1.pkl','r', 'N/A')[1])
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

print(UF.TimeStamp(), bcolors.OKGREEN+"Train and Validation data has loaded and analysed successfully..."+bcolors.ENDC)
print(len(TrainSamples),len(ValSamples))
def main(self):
    Model_Meta_Path=EOSsubModelDIR+'/'+ModelName+'_Meta'
    Model_Path=EOSsubModelDIR+'/'+ModelName
    ModelMeta=UF.PickleOperations(Model_Meta_Path, 'r', 'N/A')[0]
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
            model=tf.keras.models.load_model(Model_Path)
            K.set_value(model.optimizer.learning_rate, TrainParams[0])
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
        model.save(Model_Path)
        Header=[['Epoch','# Samples','Train Loss','Optimal Threshold','Validation Loss','Validation Accuracy','Test Loss','Test Accuracy','Training Set']]
        Header+=records
        ModelMeta.CompleteTrainingSession(Header)
        print(UF.PickleOperations(Model_Meta_Path, 'w', ModelMeta)[1])
        exit()
    elif ModelMeta.ModelType=='GNN':
        from torch import optim
        from torch.optim.lr_scheduler import StepLR
        print(UF.TimeStamp(),'Starting the training process... ')
        State_Save_Path=EOSsubModelDIR+'/'+ModelName+'_State'
        Model_Meta_Path=EOSsubModelDIR+'/'+ModelName+'_Meta'
        Model_Path=EOSsubModelDIR+'/'+ModelName
        ModelMeta=UF.PickleOperations(Model_Meta_Path, 'r', 'N/A')[0]
        device = torch.device('cpu')
        model = UF.GenerateModel(ModelMeta).to(device)
        print(model)
        optimizer = optim.Adam(model.parameters(), lr=TrainParams[0])

        scheduler = StepLR(optimizer, step_size=0.1,gamma=0.1)
        print(UF.TimeStamp(),'Try to load the previously saved model/optimiser state files ')
        try:
               model.load_state_dict(torch.load(Model_Path))
               checkpoint = torch.load(State_Save_Path)
               optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
               scheduler.load_state_dict(checkpoint['scheduler'])
        except:
               print(UF.TimeStamp(), bcolors.WARNING+"Model/state data files are missing, skipping this step..." +bcolors.ENDC)
        records=[]
        for epoch in range(0, TrainParams[2]):
            train_loss,itr= GNNtrain(model,TrainSamples, optimizer),len(TrainSamples.dataset)
            val=GNNvalidate(model,  ValSamples)
            val_loss=val[1]
            val_acc=val[0]
            test_loss=val_loss
            test_acc=val_acc
            scheduler.step()
            print(UF.TimeStamp(),'Epoch ',epoch, ' is completed')
            records.append([epoch,itr,train_loss.item(),0.5,val_loss,val_acc,test_loss,test_acc,train_set])
            print(train_loss)
            print(itr)
            torch.save({    'epoch': epoch,
                          'optimizer_state_dict': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict(),    # HERE IS THE CHANGE
                          }, State_Save_Path)
        torch.save(model.state_dict(), Model_Path)
        Header=[['Epoch','# Samples','Train Loss','Optimal Threshold','Validation Loss','Validation Accuracy','Test Loss','Test Accuracy','Training Set']]
        Header+=records
        ModelMeta.CompleteTrainingSession(Header)
        print(UF.PickleOperations(Model_Meta_Path, 'w', ModelMeta)[1])
        exit()
if __name__ == '__main__':
     main(sys.argv[1:])


