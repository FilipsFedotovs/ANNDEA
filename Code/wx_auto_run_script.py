import os
import subprocess


script_name = "MUTr2_TrainModel.py"


model_parameters_dict={
    "2T_CNN_ANN_BA_1_1_model_wx": [[1, 4, 1, 20, 2, 2, 2], [], [],[], [], [1, 4, 2], [], [], [], [], [7,2],[5000,5000,3000,50]],
    "2T_CNN_ANN_BA_2_1_model_wx": [[1, 4, 1, 20, 2, 2, 2], [1,4,1,20,2,2,2], [],[], [], [1, 4, 2], [], [], [], [], [7,2],[5000,5000,3000,50]],
    "2T_CNN_ANN_BA_1_2_model_wx": [[1, 4, 1, 20, 2, 2, 2], [], [],[], [], [1, 4, 2], [1,4,2], [], [], [], [7,2],[5000,5000,3000,50]],
    "2T_CNN_ANN_BA_2_2_model_wx": [[1, 4, 1, 20, 2, 2, 2], [1,4,1,20,2,2,2], [],[], [], [1, 4, 2], [1,4,2], [], [], [], [7,2],[5000,5000,3000,50]],
}

fixed_option_dict = {
    "TrainSampleID": "SND_RTr_MC1_Train_Data_Combined",
    "ModelType": "CNN",
    "ModelArchitecture" : "CNN",
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



for model_name, model_parameters in model_parameters_dict.items():
    variable_options = " --ModelName " + model_name +" --ModelParams " + str(model_parameters)
    fixed_options = get_fixed_options(fixed_option_dict)
    command = "python3 "+script_name + variable_options + fixed_options
    subprocess.run(command, shell=True)

    
