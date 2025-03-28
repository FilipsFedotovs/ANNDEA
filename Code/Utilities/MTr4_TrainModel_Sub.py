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
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
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
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

import U_UI as UI
import U_ML as ML
import U_HC as HC
#Load data configuration
EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'


##############################################################################################################################
######################################### Starting the program ################################################################

print(UI.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)

#Ptotect from division by zero
def zero_divide(a, b):
    if (b==0): return 0
    return a/b

#The function bellow calculates binary classification stats
def binary_classification_stats(output, y, thld):
    TP = torch.sum((y==1) & (output>thld))
    TN = torch.sum((y==0) & (output<thld))
    FP = torch.sum((y==0) & (output>thld))
    FN = torch.sum((y==1) & (output<thld))
    acc = zero_divide(TP+TN, TP+TN+FP+FN)
    TPR = zero_divide(TP, TP+FN)
    TNR = zero_divide(TN, TN+FP)
    return acc, TPR, TNR

def train(model, device, sample, optimizer):
    """ train routine, loss and accumulated gradients used to update
        the model via the ADAM optimizer externally
    """
    model.train()
    losses_w = [] # edge weight loss
    iterator=0

    for HC in sample:
        data = HC.to(device)
        if data.edge_index.size()[1]==0: continue
        iterator+=1
        w = model(data.x, data.edge_index, data.edge_attr)
        y, w = data.y.float(), w.squeeze(1)
        #edge weight loss
        loss_w = F.binary_cross_entropy(w, y, reduction='mean')
        # optimize total loss
        if iterator%TrainParams[1]==0: #Update gradients by batch
           optimizer.zero_grad()
           loss_w.backward()
           optimizer.step()

        # store losses
        losses_w.append(loss_w.item())
    loss_w = np.nanmean(losses_w)
    return loss_w,iterator

#Deriving validation metrics. Please note that in this function the optimal acceptance is calculated. This is unique to the tracker module
def validate(model, device, sample):
    model.eval() #Specific feature of pytorch - it has 2 modes eval and train that need to be selected depending on the evaluation.
    opt_thlds, accs, losses = [], [], []
    for HC in sample:
        data = HC.to(device)
        if data.edge_index.size()[1]==0: continue
        try:
            output = model(data.x, data.edge_index, data.edge_attr)
        except:
            continue

        y, output = data.y.float(), output.squeeze(1)
        try:
          loss = F.binary_cross_entropy(output, y, reduction='mean').item()
        except:
            print('Erroneous data set: ',data.x, data.edge_index, data.edge_attr, 'skipping these samples...')
            continue
        diff, opt_thld, opt_acc = 100, 0, 0
        for thld in np.arange(0.01, 0.6, 0.01):
            acc, TPR, TNR = binary_classification_stats(output, y, thld)
            delta = abs(TPR-TNR)
            if (delta.item() < diff):
                diff, opt_thld, opt_acc = delta.item(), thld, acc.item()
        opt_thlds.append(opt_thld)
        accs.append(opt_acc)
        losses.append(loss)
    return np.nanmean(opt_thlds),np.nanmean(losses),np.nanmean(accs)

#Deriving testing metrics
def test(model, device, sample, thld):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for HC in sample:
            data = HC.to(device)
            if data.edge_index.size()[1]==0: continue
            output = model(data.x, data.edge_index, data.edge_attr)
            y, output = data.y.float(), output.squeeze(1)
            acc, TPR, TNR = binary_classification_stats(output, y, thld)
            try:
                loss = F.binary_cross_entropy(output, y,reduction='mean')
            except:
                print('Erroneous data set: ',data.x, data.edge_index, data.edge_attr, 'skipping these samples...')
                continue
            accs.append(acc.item())
            losses.append(loss.item())
    return np.nanmean(losses), np.nanmean(accs)


output_train_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TRAIN_CLUSTERS'+'.pkl'
TrainSamples=UI.PickleOperations(output_train_file_location,'r', 'N/A')[0]
TrainSamples = [obj.Graph for obj in TrainSamples]
output_val_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_VAL_CLUSTERS'+'.pkl'
ValSamples=UI.PickleOperations(output_val_file_location,'r', 'N/A')[0]
ValSamples = [obj.Graph for obj in ValSamples]
output_test_file_location=EOS_DIR+'/ANNDEA/Data/TRAIN_SET/'+TrainSampleID+'_TEST_CLUSTERS'+'.pkl'
TestSamples=UI.PickleOperations(output_test_file_location,'r', 'N/A')[0]
TestSamples = [obj.Graph for obj in TestSamples]



def main(self):
    print(UI.TimeStamp(),'Starting the training process... ')
    State_Save_Path=EOSsubModelDIR+'/'+args.BatchID+'_State'
    Model_Meta_Path=EOSsubModelDIR+'/'+args.BatchID+'_Meta'
    Model_Path=EOSsubModelDIR+'/'+args.BatchID
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
    TrainSampleSize=len(TrainSamples)
    fraction_size=math.ceil(TrainSampleSize/TrainParams[3])
    for epoch in range(0, TrainParams[2]):
       for fraction in range(0, TrainParams[3]):
         sp=fraction*fraction_size
         ep=min((fraction+1)*fraction_size,TrainSampleSize)
         train_loss, itr= train(model, device,TrainSamples[sp:ep], optimizer)
         thld, val_loss,val_acc = validate(model, device, ValSamples)
         test_loss, test_acc = test(model, device,TestSamples, thld)
         scheduler.step()
         print(UI.TimeStamp(),'Epoch ',epoch, ' is completed')
         records.append([epoch,itr,train_loss,thld,val_loss,val_acc,test_loss,test_acc])
         torch.save({    'epoch': epoch,
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(),    # HERE IS THE CHANGE
                      }, State_Save_Path)
    torch.save(model.state_dict(), Model_Path)
    Header=[['Epoch','# Samples','Train Loss','Optimal Threshold','Validation Loss','Validation Accuracy','Test Loss','Test Accuracy']]
    Header+=records
    ModelMeta.CompleteTrainingSession(Header)
    print(UI.PickleOperations(Model_Meta_Path, 'w', ModelMeta)[1])
    exit()
if __name__ == '__main__':
     main(sys.argv[1:])


