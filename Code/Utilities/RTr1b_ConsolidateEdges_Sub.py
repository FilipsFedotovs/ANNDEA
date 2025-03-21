#Current version 1.2 - redistribute edge generation

########################################    Import essential libriries    #############################################
import argparse
import sys
import ast
import os

#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--i',help="Set number", default='1')
parser.add_argument('--j',help="Subset number", default='1')
parser.add_argument('--k',help="SubSubset number", default='1')
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')
parser.add_argument('--PY',help="Python libraries directory location", default='.')
parser.add_argument('--GraphProgram',help="Program with the list of files", default='0')
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
######################################## Set variables  #############################################################
p,o,sfx,pfx=args.p,args.o,args.sfx,args.pfx
RecBatchID=args.BatchID
Program=ast.literal_eval(args.GraphProgram)
i,j,k=args.i,args.j,args.k
import U_UI as UI #This is where we keep routine utility functions

master_file=EOS_DIR+p+'/Temp_RTr1a_'+RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'/RTr1a_'+RecBatchID+'_hit_cluster_edges_'+str(i)+'_'+str(j)+'_'+str(k)+'_0.pkl'
if os.path.isfile(master_file):
    master_data=UI.PickleOperations(master_file,'r','')[0]
    for l in range(1,Program[int(i)][int(j)][int(k)]):
        slave_file=EOS_DIR+p+'/Temp_RTr1a_'+RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'/RTr1a_'+RecBatchID+'_hit_cluster_edges_'+str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)+'.pkl'
        slave_data=UI.PickleOperations(slave_file,'r','')[0]
        master_data.Seeds+=slave_data.Seeds
        master_data.SeedFlowValuesAll = [a + b for a, b in zip(master_data.SeedFlowValuesAll, slave_data.SeedFlowValuesAll)]
        master_data.SeedFlowValuesTrue = [a + b for a, b in zip(master_data.SeedFlowValuesTrue, slave_data.SeedFlowValuesTrue)]
else:
    master_file=EOS_DIR+p+'/RTr1_'+RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_clusters.pkl'
    master_data=UI.PickleOperations(master_file,'r','')[0]
    master_data.Seeds=[]
print(UI.TimeStamp(),'Writing the output...')
output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(i)+'_'+str(j)+'_'+str(k)+'.pkl'
print(UI.PickleOperations(output_file_location,'w',master_data)[1])
exit()


