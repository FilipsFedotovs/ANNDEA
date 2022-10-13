import pandas as pd



def renameColumns(dataFrame):
    return dataFrame.rename(columns={
        PM.x: "x",
        PM.y: "y",
        PM.z: "z",
        PM.tx: "tx",
        PM.ty: "ty",
        PM.Hit_ID: "Hit_ID"
    })



def 