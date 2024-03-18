########################################################################################################################
#######################################  This simple script prepares data for CNN  #####################################



########################################    Import libraries    ########################################################
import ast
import argparse
import csv
########################## Visual Formatting #################################################


########################## Setting the parser ################################################
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--ModelParams',help="Please enter the train params: '[<Session No>, <Learning Rate>, <Batch size>, <Epochs>]'", default='[1, 0.0001, 4, 10]')
parser.add_argument('--ModelType',help="What Neural Network type would you like to use: CNN/GNN?", default='CNN')
parser.add_argument('--ModelArchitecture',help="What Type of Image/Graph: CNN, CNN-E", default='CNN')
parser.add_argument('--ModelFramework',help="What Type of Image/Graph: CNN, CNN-E", default='Tensorflow')
########################################     Initialising Variables    #########################################
args = parser.parse_args()
ModelParams=ast.literal_eval(args.ModelParams)
ModelType=args.ModelType
ModelArchitecture=args.ModelArchitecture
ModelFramework=args.ModelFramework

##################################   Loading Directory locations   ##################################################

import sys
sys.path.insert(1, './Utilities/')
csv_reader=open('../config',"r")
config = list(csv.reader(csv_reader))
for c in config:
    if c[0]=='AFS_DIR':
        AFS_DIR=c[1]
    if c[0]=='EOS_DIR':
        EOS_DIR=c[1]
    if c[0]=='PY_DIR':
        PY_DIR=c[1]
csv_reader.close()
import sys
if PY_DIR!='': #Temp solution - the decision was made to move all libraries to EOS drive as AFS get locked during heavy HTCondor submission loads
    sys.path=['',PY_DIR]
    sys.path.append('/usr/lib64/python39.zip')
    sys.path.append('/usr/lib64/python3.9')
    sys.path.append('/usr/lib64/python3.9/lib-dynload')
    sys.path.append('/usr/lib64/python3.9/site-packages')
    sys.path.append('/usr/lib/python3.9/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')
import U_UI as UI
import U_ML as ML


##############################################################################################################################
######################################### Starting the program ################################################################
UI.WelcomeMsg('Initialising ANNDEA model validation unit...','Filips Fedotovs (PhD student at UCL), Wenqing Xie (MSc student at UCL)','Please reach out to filips.fedotovs@cern.ch for any queries')

def zero_divide(a, b):
    if (b==0): return 0
    return a/b
Meta=UI.TrainingSampleMeta('N/A')
Meta.IniTrackSeedMetaData(0,0,0,0,[],0,[],0,0)
ModelMeta=ML.ModelMeta('N/A')
ModelMeta.IniModelMeta(ModelParams,ModelFramework, Meta, ModelArchitecture, ModelType)
if ModelType=='CNN':
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
        model = ML.GenerateModel(ModelMeta,[0.001,0,0,0])
        model.summary()
elif ModelMeta.ModelType=='GNN':
        import torch
        device = torch.device('cpu')
        model = ML.GenerateModel(ModelMeta).to(device)
        print(model)



