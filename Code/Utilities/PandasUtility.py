import pandas as pd
import Parameters as PM
import PrintingUtility as Printing
from PrintingUtility import bcolors

def load_data(input_file_location):
    Printing.print_message(
        'Loading raw data from'+
        bcolors.OKBLUE +
        input_file_location +
        bcolors.ENDC
    )
    dataFrame=pd.read_csv(
        input_file_location,
        header=0,
    )
    n_rows=len(dataFrame.axes[0])
    Printing.print_message('The raw data has '+str(n_rows)+' hits')
    return dataFrame

def select_columns(dataFrame, column_name_list):
    Printing.print_message(
        'Selecting the following columns: '+
        '[' +
        ' '. join(column_name_list)+
        ']'
    )
    return dataFrame[column_name_list]

def remove_unreconstructed_hits(dataFrame):
    Printing.print_message('Removing unreconstructed hits...')
    data=dataFrame.dropna()
    n_rows=len(data.axes[0])
    Printing.print_message('The cleaned data has '+str(n_rows)+' hits')
    return data


def rename_hit_columns(dataFrame):
    return dataFrame.rename(columns={
        PM.x: "x",
        PM.y: "y",
        PM.z: "z",
        PM.tx: "tx",
        PM.ty: "ty",
        PM.Hit_ID: "Hit_ID",
        PM.MC_Event_ID: "MC_Event_ID",
        PM.MC_Track_ID: "MC_Track_ID",

    })


def slice_hit_data(dataFrame, Xmin, Xmax, Ymin, Ymax):
    print(UF.TimeStamp(),'Slicing the data...')
    data=dataFrame.drop(data.index[(data["x"] > Xmax) | (data["x"] < Xmin) | (data["y"] > Ymax) | (data["y"] < Ymin)])
    n_rows=len(data.axes[0])
    Printing.print_message('The sliced data has '+str(n_rows)+' hits')
    return data


def convert_ID_to_string(dataFrame):
    data = dataFrame
    data["MC_Event_ID"] = data["MC_Event_ID"].astype(int).astype(str)
    data["MC_Track_ID"] = data["MC_Track_ID"].astype(int).astype(str)
    data["Hit_ID"] = data["Hit_ID"].astype(int).astype(str)

    data['MC_Mother_Track_ID'] = data["MC_Event_ID"] + '-' + data["MC_Track_ID"]
    data=data.drop(["MC_Event_ID"],axis=1)
    data=data.drop(["MC_Track_ID"],axis=1)
    return data

def save_data(dataFrame, output_file_location):
    dataFrame.to_csv(output_file_location,index=False)
    Printing.print_message(
        bcolors.OKGREEN +
        "The data has been created successfully and written to" +
        bcolors.ENDC, bcolors.OKBLUE + 
        output_file_location + 
        bcolors.ENDC
    )


def remove_ill_mc_tracks(dataFrame)
    Printing.print_message(
        'Removing tracks which have less than'+
        PM.MinHitsTrack+
        'hits...'
    )
    hits_per_track = dataFrame.groupby(['MC_Mother_Track_ID'],as_index=False).count()
    
    hits_per_track = hits_per_track.rename(columns={'x': 'hits_per_track'})
    hits_per_track = hits_per_track[["hits_per_track",'MC_Mother_Track_ID']]

    data=pd.merge(dataFrame, hits_per_track, how="left", on=['MC_Mother_Track_ID'])
    data = data[ data['hits_per_track'] >= PM.MinHitsTrack]
    data = data.drop(['hits_per_track'],axis=1)
    data= data.sort_values(['MC_Mother_Track_ID','z'],ascending=[1,1])
    n_rows=len(data.axes[0])
    print(UF.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
    return data