import os
import subprocess


script_name = "MUTr2_TrainModel.py"


model_parameters_dict={
    "2T_CNN_ANN_BA_1_1_model": [[1, 4, 1, 3, 2, 2, 2], [], [],[], [], [1, 4, 2], [], [], [], [], [7,2],[5000,5000,3000,50]],
    "2T_CNN_ANN_BA_2_1_model": [[1, 4, 1, 3, 2, 2, 2], [1,4,1,3,2,2,2], [],[], [], [1, 4, 2], [], [], [], [], [7,2],[5000,5000,3000,50]],
    "2T_CNN_ANN_BA_1_2_model": [[1, 4, 1, 3, 2, 2, 2], [], [],[], [], [1, 4, 2], [1,4,2], [], [], [], [7,2],[5000,5000,3000,50]],
    "2T_CNN_ANN_BA_2_2_model": [[1, 4, 1, 3, 2, 2, 2], [1,4,1,3,2,2,2], [],[], [], [1, 4, 2], [1,4,2], [], [], [], [7,2],[5000,5000,3000,50]],
}

fixed_option_dict = {
    "TrainSampleID": "SND_RTr_MC1_Train_Data_Combined",
    "ModelType": "CNN",
    "ModelArchitecture" : "CNN",
}

model_parameters_dict_gnn={
    "2T_GNN_FC_16_16_model": [[16],[16],[],[],[],[],[],[],[],[],[7,2],[5000,5000,3000,50]],
    "2T_GNN_FC_16_16_16_model": [[16],[16],[16],[],[],[],[],[],[],[],[7,2],[5000,5000,3000,50]],
    "2T_GNN_FC_32_32_model": [[32],[32],[],[],[],[],[],[],[],[],[7,2],[5000,5000,3000,50]],
    "2T_GNN_FC_32_32_32_model": [[32],[32],[32],[],[],[],[],[],[],[],[7,2],[5000,5000,3000,50]],
}


fixed_option_dict_gnn = {
    "TrainSampleID": "SND_RTr_MC1_Train_Data_Combined",
    "ModelType": "GNN",
    "ModelArchitecture" : "GNN-4N-FC",
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


gnn = True
if(gnn):
    fixed_option_dict = fixed_option_dict_gnn
    model_parameters_dict = model_parameters_dict_gnn


for model_name, model_parameters in model_parameters_dict.items():
    variable_options = " --ModelName " + model_name +" --ModelParams " + '"'+str(model_parameters)+'"'
    fixed_options = get_fixed_options(fixed_option_dict)
    command = "python3 "+script_name + variable_options + fixed_options
    subprocess.Popen(command, shell=True)

    
