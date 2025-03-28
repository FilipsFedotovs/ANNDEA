########################################################################################################################
#######################################  This simple script prepares data for CNN  #####################################


########################################    Import libraries    ########################################################
import argparse
import math
import ast

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
parser.add_argument('--ModelParams',help="Please enter the model params: '[<Number of MLP layers>, <'MLP hidden size'>, <Number of IN layers>, <'IN hidden size'>]'", default='[3,80,3,80]')
parser.add_argument('--TrainParams',help="Please enter the train params: '[<Learning Rate>, <Batch size>, <Epochs>, <Fraction>]'", default='[ 0.0001, 4, 10 ,1]')
parser.add_argument('--AFS',help="Please enter the user afs directory", default='.')
parser.add_argument('--EOS',help="Please enter the user eos directory", default='.')
parser.add_argument('--TrainSampleID',help="Give name of the training ", default='SHIP_TrainSample_v1')
parser.add_argument('--BatchID',help="Name of the model", default='1T_MC_1_model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--i',help="Set number", default='1')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
parser.add_argument('--PY',help="Python libraries directory location", default='.')
########################################     Initialising Variables    #########################################
args = parser.parse_args()
ModelParams=ast.literal_eval(args.ModelParams)
TrainParams=ast.literal_eval(args.TrainParams)
TrainSampleID=args.TrainSampleID
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
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

import U_UI as UI
import U_ML as ML
import U_HC as HC
#Load data configuration
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'

b_parameter=2

##############################################################################################################################
######################################### Starting the program ################################################################

print(UI.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)

#Ptotect from division by zero
def zero_divide(a, b):
    if (b==0): return 0
    return a/b

#The function bellow calculates binary classification stats
def BinaryClassifierStats(O, Y, thld, b):

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for o, y in zip(O, Y):
        TP += int((y==1) & (o>thld))
        TN += int((y==0) & (o<thld))
        FP += int((y==0) & (o>thld))
        FN += int((y==1) & (o<thld))

    acc = zero_divide(TP+TN, TP+TN+FP+FN)
    R = zero_divide(TP, TP+FN)
    P = zero_divide(TP, TP+FP)
    Fb=zero_divide((1+b**2) * (P * R), ((b**2*P)+R)) #F2 score

    return acc, Fb




def train(model,  sampleX, sampleY, optimizer, criterion):
    """ train routine, loss and accumulated gradients used to update
        the model via the ADAM optimizer externally
    """
    model.train()
    losses = [] # edge weight loss
    iterator=0
    for x, y in zip(sampleX, sampleY):
        if len(x)==0: continue
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        iterator+=1
        o = model(x)
        #edge weight loss
        loss = criterion(o, y)
        # optimize total loss
        if iterator%TrainParams[1]==0: #Update gradients by batch
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
        # store losses
        losses.append(loss.item())
    return np.nanmean(losses),iterator

#Deriving validation metrics. Please note that in this function the optimal acceptance is calculated. This is unique to the tracker module
def validate(model,  sampleX, sampleY, criterion, b):
    model.eval() #Specific feature of pytorch - it has 2 modes eval and train that need to be selected depending on the evaluation.
    losses = []

    #Generate predictions
    Y=[]
    O=[]

    with torch.no_grad():
        for x, y in zip(sampleX, sampleY):
                x, y = x.unsqueeze(0), y.unsqueeze(0)
                o = model(x)
                loss = criterion(o, y).item()
                losses.append(loss)
                O.append(o.item())
                Y.append(y.item())

    #Optimise acceptance
    best_F2=0.0
    best_thresh=0.0
    best_acc=0.0
    for thld in range(0, 101):
        result=BinaryClassifierStats(O, Y, thld/100, b)
        if result[1]>best_F2:
            best_F2=result[1]
            best_thresh=thld/100
            best_acc=result[0]

    return best_thresh,np.nanmean(losses),best_acc

#Deriving testing metrics
def test(model,  sampleX, sampleY, criterion, thld, b):
    model.eval()
    losses, Y, O = [], [], []
    with torch.no_grad():
        for x, y in zip(sampleX, sampleY):
            x, y = x.unsqueeze(0), y.unsqueeze(0)
            o = model(x)
            loss = criterion(o, y).item()
            losses.append(loss)
            O.append(o.item())
            Y.append(y.item())
    result=BinaryClassifierStats(O, Y, thld, b)
    return np.nanmean(losses), result[0]

########## importing and preparing data samples
#Training sample
output_train_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_SEEDS'+'.pkl' #Path
TrainSamplesX=torch.tensor(HC.HitCluster.GenerateSeedVectors(UI.PickleOperations(output_train_file_location,'r', 'N/A')[0])[0], dtype=torch.float32) #Loading features
TrainSamplesY=torch.tensor(HC.HitCluster.GenerateSeedVectors(UI.PickleOperations(output_train_file_location,'r', 'N/A')[0])[1], dtype=torch.float32).view(-1, 1) #Loading labels

TrainSamplesX=TrainSamplesX[0:TrainParams[4]]
TrainSamplesY=TrainSamplesY[0:TrainParams[4]]


#Validation sample
output_val_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_SEEDS'+'.pkl' #Path
ValSamplesX=torch.tensor(HC.HitCluster.GenerateSeedVectors(UI.PickleOperations(output_val_file_location,'r', 'N/A')[0])[0], dtype=torch.float32) #Loading features
ValSamplesY=torch.tensor(HC.HitCluster.GenerateSeedVectors(UI.PickleOperations(output_val_file_location,'r', 'N/A')[0])[1], dtype=torch.float32).view(-1, 1) #Loading labels


#Test sample
output_test_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TEST_SEEDS'+'.pkl' #Path
TestSamplesX=torch.tensor(HC.HitCluster.GenerateSeedVectors(UI.PickleOperations(output_test_file_location,'r', 'N/A')[0])[0], dtype=torch.float32) #Loading features
TestSamplesY=torch.tensor(HC.HitCluster.GenerateSeedVectors(UI.PickleOperations(output_test_file_location,'r', 'N/A')[0])[1], dtype=torch.float32).view(-1, 1) #Loading labels


def main(self):
    print(UI.TimeStamp(),'Starting the training process... ')
    State_Save_Path=EOSsubModelDIR+'/'+args.BatchID+'_State'
    Model_Meta_Path=EOSsubModelDIR+'/'+args.BatchID+'_Meta'
    Model_Path=EOSsubModelDIR+'/'+args.BatchID
    ModelMeta=UI.PickleOperations(Model_Meta_Path, 'r', 'N/A')[0]

    model = ML.GenerateModel(ModelMeta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=TrainParams[0])
    scheduler = StepLR(optimizer, step_size=0.1,gamma=0.1)
    criterion = nn.BCELoss()
    print(UI.TimeStamp(),'Try to load the previously saved model/optimiser state files ')
    try:
           model.load_state_dict(torch.load(Model_Path))
           checkpoint = torch.load(State_Save_Path)
           optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
           scheduler.load_state_dict(checkpoint['scheduler'])
    except:
           print(UI.TimeStamp(), bcolors.WARNING+"Model/state data files are missing, skipping this step..." +bcolors.ENDC)
    records=[]
    TrainSampleSize=len(TrainSamplesX)
    fraction_size=math.ceil(TrainSampleSize/TrainParams[3])
    for epoch in range(0, TrainParams[2]):
       for fraction in range(0, TrainParams[3]):
         sp=fraction*fraction_size
         ep=min((fraction+1)*fraction_size,TrainSampleSize)
         train_loss, itr= train(model, TrainSamplesX[sp:ep], TrainSamplesY[sp:ep], optimizer,criterion)
         thld, val_loss,val_acc = validate(model, ValSamplesX, ValSamplesY, criterion, b_parameter)
         test_loss, test_acc = test(model, TestSamplesX, TestSamplesY, criterion, thld, b_parameter)
         scheduler.step()
         print(UI.TimeStamp(),'Epoch ',epoch, ' and fraction ', fraction, ' is completed')
         records.append([epoch,itr,train_loss,thld,val_loss,val_acc,test_loss,test_acc])
         torch.save({    'epoch': epoch,
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(),    # HERE IS THE CHANGE
                      }, State_Save_Path)
         print(records)
    torch.save(model.state_dict(), Model_Path)
    Header=[['Epoch','# Samples','Train Loss','Optimal Threshold','Validation Loss','Validation Accuracy','Test Loss','Test Accuracy']]
    Header+=records
    ModelMeta.CompleteTrainingSession(Header)
    print(UI.PickleOperations(Model_Meta_Path, 'w', ModelMeta)[1])
    exit()
if __name__ == '__main__':
     main(sys.argv[1:])


