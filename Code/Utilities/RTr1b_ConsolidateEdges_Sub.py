#Current version 1.2 - redistribute edge generation

########################################    Import essential libriries    #############################################
import argparse
import sys
import ast

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
GraphProgram=ast.literal_eval(args.GraphProgram)
print(GraphProgram)
x=input()
exit()

i,j,k=args.i,args.j,args.k
import U_UI as UI #This is where we keep routine utility functions


master_file=EOS_DIR+Program[0][1][3]+'/Temp_'+Program[0][1][5]+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'/'+Program[0][1][5]+'_'+RecBatchID+'_'+Program[0][1][4]+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_0.pkl'
master_data=UI.PickleOperations(master_file,'r','')[0]
for l in range(1,Program[0][1][8][i][j][k]):
    slave_file=EOS_DIR+Program[0][1][3]+'/Temp_'+Program[0][1][5]+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'/'+Program[0][1][5]+'_'+RecBatchID+'_'+Program[0][1][4]+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)+'.pkl'
    slave_data=UI.PickleOperations(slave_file,'r','')[0]
    master_data.RawEdgeGraph+=slave_data.RawEdgeGraph
    master_data.HitPairs+=slave_data.HitPairs
output_file=EOS_DIR+Program[2][1][3]+'/Temp_'+Program[2][1][5]+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'/'+Program[0][1][5]+'_'+RecBatchID+'_'+Program[0][1][4]+'_'+str(i)+'_'+str(j)+'_'+str(k)+'.pkl'
print(UI.PickleOperations(output_file,'w',master_data)[1])

input_file_location=EOS_DIR+p+'/RTr1_'+RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_clusters.pkl'
HC=UI.PickleOperations(input_file_location,'r','')[0]
print(UI.TimeStamp(),'Generating the edges...')
GraphStatus = HC.GenerateEdges(cut_dt, cut_dr, cut_dz, [], int(l), MaxEdgesPerJob)
print(UI.TimeStamp(),'Writing the output...')
output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)+sfx
UI.PickleOperations(output_file_location,'w',HC)
print(UI.TimeStamp(),'Output is written to ',output_file_location)
exit()


