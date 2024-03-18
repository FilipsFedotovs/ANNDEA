########################################################################################################################
#######################################  This simple script prepares data for CNN  #####################################



########################################    Import libraries    ########################################################
import ast
import argparse
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
import U_UI as UI
import U_ML as ML



##############################################################################################################################
######################################### Starting the program ################################################################
UI.WelcomeMsg('Initialising ANNDEA model validation unit...','Filips Fedotovs (PhD student at UCL), Wenqing Xie (MSc student at UCL)','Please reach out to filips.fedotovs@cern.ch for any queries')
print(UI.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
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
        model = ML.GenerateModel(ModelMeta,[0,0,0,0])
        model.summary()
elif ModelMeta.ModelType=='GNN':
        import torch
        device = torch.device('cpu')
        model = ML.GenerateModel(ModelMeta).to(device)
        print(model)



