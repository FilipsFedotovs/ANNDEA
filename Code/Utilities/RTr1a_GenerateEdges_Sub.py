#Current version 1.1 - add change sys path capability

########################################    Import essential libriries    #############################################
import argparse
import sys
import copy
from statistics import mean
import os
import ast


#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--i',help="Set number", default='1')
parser.add_argument('--j',help="Subset number", default='1')
parser.add_argument('--k',help="SubSubset number", default='1')
parser.add_argument('--l',help="SubSubSubset number", default='1')
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')
parser.add_argument('--PY',help="Python libraries directory location", default='.')
parser.add_argument('--cut_dt',help="Cut on angle difference", default='1.0')
parser.add_argument('--cut_dr',help="Cut on angle difference", default='4000')
parser.add_argument('--cut_dz',help="Cut on z difference", default='3000')
parser.add_argument('--MaxEdgesPerJob',help="Max edges per job", default='0')
parser.add_argument('--BatchID',help="Give name to this train sample", default='')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')

#Working out where are the Py libraries
args = parser.parse_args()
#Loading Directory locations
EOS_DIR=args.EOS
AFS_DIR=args.AFS
PY_DIR=args.PY

if PY_DIR!='': #Temp solution
    sys.path=['',PY_DIR]
    sys.path.append('/usr/lib64/python39.zip')
    sys.path.append('/usr/lib64/python3.9')
    sys.path.append('/usr/lib64/python3.9/lib-dynload')
    sys.path.append('/usr/lib64/python3.9/site-packages')
    sys.path.append('/usr/lib/python3.9/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')
import pandas as pd #We use Panda for a routine data processing
import numpy as np
######################################## Set variables  #############################################################
cut_dt,cut_dr,cut_dz=float(args.cut_dt),float(args.cut_dr),float(args.cut_dz)
p,o,sfx,pfx=args.p,args.o,args.sfx,args.pfx
RecBatchID=args.BatchID
MaxEdgesPerJob=int(args.MaxEdgesPerJob)
i,j,k,l=args.i,args.j,args.k,args.l

import U_UI as UI #This is where we keep routine utility functions
# import U_HC as HC_l
# import U_ML as ML

input_file_location=EOS_DIR+p+'/RTr1_'+RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_clusters.pkl'
HC=UI.PickleOperations(input_file_location,'r','')[0]
print(HC.RawClusterGraph)
print(UI.TimeStamp(),'Generating the edges...')
#print(UI.TimeStamp(),"Hit density of the Cluster",round(X_ID,1),round(Y_ID,1),1, "is  {} hits per cm\u00b3".format(round(len(HC.RawClusterGraph)/(stepX/10000*stepY/10000*stepZ/10000)),2))
GraphStatus = HC.GenerateEdges(cut_dt, cut_dr, cut_dz, [], l, MaxEdgesPerJob)
exit()
print(GraphStatus)
print(HC.RawEdgeGaph)

# if CheckPoint and GraphStatus:
#     print(UI.TimeStamp(),'Saving checkpoint 2...')
#     UI.PickleOperations(CheckPointFile_Edge,'w',HC)
# if GraphStatus:
#     Status = 'ML analysis'
# else:
#     Status = 'Skip tracking'

# output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(args.i)+'_'+str(args.j)+'_'+str(args.k)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(args.i)+'_'+str(args.j)+'_'+str(args.k)+'_'+str(args.l)+sfx
# print(UI.TimeStamp(),'Writing the output...')
# _Rec_Hits_Pool.to_csv(output_file_location,index=False) #Write the final result
# print(UI.TimeStamp(),'Output is written to ',output_file_location)
# exit()


