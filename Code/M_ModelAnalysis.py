########################################################################################################################
#######################################  This simple script prepares data for CNN  #####################################



########################################    Import libraries    ########################################################
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
parser.add_argument('--ModelParams',help="Please enter the train params: '[<Session No>, <Learning Rate>, <Batch size>, <Epochs>]'", default='[1, 0.0001, 4, 10]')
parser.add_argument('--ModelType',help="What Neural Network type would you like to use: CNN/GNN?", default='CNN')
parser.add_argument('--ModelArchitecture',help="What Type of Image/Graph: CNN, CNN-E", default='CNN')
########################################     Initialising Variables    #########################################
args = parser.parse_args()
ModelParams=ast.literal_eval(args.ModelParams)
ModelType=args.ModelType
ModelArchitecture=args.ModelArchitecture

##################################   Loading Directory locations   ##################################################

import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import UtilityFunctions as UF



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
Meta=UF.TrainingSampleMeta('N/A')
Meta.IniTrackSeedMetaData(0,0,0,0,[],0,[],0)
ModelMeta=UF.ModelMeta('N/A')
ModelMeta.IniModelMeta(ModelParams, 'PyTorch', Meta, ModelArchitecture, 'GNN')
if ModelType=='CNN':
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
        print(UF.TimeStamp(), bcolors.WARNING+"Model/state data files are missing, skipping this step..." +bcolors.ENDC)
        model = UF.GenerateModel(ModelMeta,[0,0,0,0])
        model.summary()
elif ModelMeta.ModelType=='GNN':
        import torch
        device = torch.device('cpu')
        model = UF.GenerateModel(ModelMeta).to(device)
        print(model)



