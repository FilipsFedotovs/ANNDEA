import os
import subprocess


script_name = "MUTr2_TrainModel.py"


model_parameters_dict_cnn={
    "2T_CNN_ANN_BA_1_1_model": [[1, 4, 1, 3, 2, 2, 2], [], [],[], [], [1, 4, 2], [], [], [], [], [7,2],[5000,5000,3000,50]],
    "2T_CNN_ANN_BA_2_1_model": [[1, 4, 1, 3, 2, 2, 2], [1,4,1,3,2,2,2], [],[], [], [1, 4, 2], [], [], [], [], [7,2],[5000,5000,3000,50]],
    "2T_CNN_ANN_BA_1_2_model": [[1, 4, 1, 3, 2, 2, 2], [], [],[], [], [1, 4, 2], [1,4,2], [], [], [], [7,2],[5000,5000,3000,50]],
    "2T_CNN_ANN_BA_2_2_model": [[1, 4, 1, 3, 2, 2, 2], [1,4,1,3,2,2,2], [],[], [], [1, 4, 2], [1,4,2], [], [], [], [7,2],[5000,5000,3000,50]],
}

fixed_option_dict_cnn = {
    "TrainSampleID": "SND_RTr_MC1_Train_Data_Combined",
    "ModelType": "CNN",
    "ModelArchitecture" : "CNN",
}

model_parameters_dict_gcn={
    "2T_GNN_FC_3_8_model": [[8],[8],[8],[],[],[],[],[],[],[],[7,2],[5000,5000,3000,50]],
    "2T_GNN_FC_3_16_model": [[16],[16],[16],[],[],[],[],[],[],[],[7,2],[5000,5000,3000,50]],
    "2T_GNN_FC_3_32_model": [[32],[32],[32],[],[],[],[],[],[],[],[7,2],[5000,5000,3000,50]],
    "2T_GNN_FC_3_64_model": [[64],[64],[64],[],[],[],[],[],[],[],[7,2],[5000,5000,3000,50]],
}


fixed_option_dict_gcn = {
    "TrainSampleID": "SND_RTr_MC1_Train_Data_Combined",
    "ModelType": "GNN",
    "ModelArchitecture" : "GCN-4N-FC",
}


model_parameters_dict_gmm={
    "2T_GMM_FC_3_8_4_model": [[8,4],[8,4],[8,4],[],[],[],[],[],[],[],[7,2],[5000,5000,3000,50]],
    "2T_GMM_FC_3_16_4_model": [[16,4],[16,4],[16,4],[],[],[],[],[],[],[],[7,2],[5000,5000,3000,50]],
    "2T_GMM_FC_3_8_8_model": [[8,8],[8,8],[8,8],[],[],[],[],[],[],[],[7,2],[5000,5000,3000,50]],
    "2T_GMM_FC_3_16_8_model": [[16,8],[16,8],[16,8],[],[],[],[],[],[],[],[7,2],[5000,5000,3000,50]],
}

fixed_option_dict_gmm = {
    "TrainSampleID": "SND_RTr_MC1_Train_Data_Combined",
    "ModelType": "GNN",
    "ModelArchitecture" : "GMM-5N-FC",
}

option_headers=[
    "ModelName",
    "ModelParams",
]



def get_fixed_options(dict):
    s = ""
    for option_header, option_value in dict.items():
        s+= (" --"+option_header+" ")
        s+= (" " + option_value +" ")
    return s


train_parameter_dict_cnn = {
    "learning rate": 0.0001,
    "Batch size": 4,
    "Epochs": 1,
}

train_parameter_dict_gnn = {
    "learning rate": 0.0001,
    "Batch size": 4,
    "Epochs": 32,
}


model_type = "gmm"
if(model_type=="gcn"):
    fixed_option_dict = fixed_option_dict_gcn
    model_parameters_dict = model_parameters_dict_gcn
    train_parameter_dict = train_parameter_dict_gnn
elif (model_type=="gmm"):
    fixed_option_dict = fixed_option_dict_gmm
    model_parameters_dict = model_parameters_dict_gmm
    train_parameter_dict = train_parameter_dict_gnn
elif (model_type=="cnn"):
    fixed_option_dict = fixed_option_dict_cnn
    model_parameters_dict = model_parameters_dict_cnn
    train_parameter_dict = train_parameter_dict_cnn


for model_name, model_parameters in model_parameters_dict.items():
    variable_options = " --ModelName " + model_name +" --ModelParams " + '"'+str(model_parameters)+'"'
    training_options = " --TrainParams " +'"'+str(list(train_parameter_dict.values()))+'"'
    fixed_options = get_fixed_options(fixed_option_dict)
    command = "python3 "+script_name + variable_options + fixed_options + training_options
    command+= (" --TrainParams " +'"'+str(list(train_parameter_dict.values()))+'"')
    subprocess.Popen(command, shell=True)

    
