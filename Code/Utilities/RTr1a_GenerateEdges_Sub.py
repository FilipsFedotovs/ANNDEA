#Current version 1.2 - redistribute edge generation

########################################    Import essential libriries    #############################################
import argparse
import sys
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
parser.add_argument('--SeedFlowLog',help="Track the seed cutflow?", default='N')
parser.add_argument('--ModelName',help="Model used to refine seeds", default='N')
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
cut_dt,cut_dr,cut_dz=float(args.cut_dt),float(args.cut_dr),float(args.cut_dz)
p,o,sfx,pfx=args.p,args.o,args.sfx,args.pfx
RecBatchID=args.BatchID
MaxEdgesPerJob=int(args.MaxEdgesPerJob)
SeedFlowLog=args.SeedFlowLog=='Y'
if args.ModelName=='N':
    ModelName=None
else:
    ModelName=args.ModelName
i,j,k,l=args.i,args.j,args.k,args.l
import U_UI as UI #This is where we keep routine utility functions
input_file_location=EOS_DIR+p+'/RTr1_'+RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_clusters.pkl'
HC=UI.PickleOperations(input_file_location,'r','')[0]
print(UI.TimeStamp(),'Generating the edges...')
GraphStatus = HC.GenerateSeeds(cut_dt, cut_dr, cut_dz, int(l), MaxEdgesPerJob, SeedFlowLog, EOS_DIR, ModelName)
print(UI.TimeStamp(),'Writing the output...')
output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)+sfx
UI.PickleOperations(output_file_location,'w',HC)
print(UI.TimeStamp(),'Output is written to ',output_file_location)
exit()


