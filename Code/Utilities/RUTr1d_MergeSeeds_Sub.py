
#This simple script prepares data for CNN
########################################    Import libraries    #############################################
import UtilityFunctions as UF
from UtilityFunctions import EMO

import argparse
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--i',help="Set number", default='1')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
parser.add_argument('--EOS',help="EOS location", default='')
parser.add_argument('--AFS',help="AFS location", default='')
parser.add_argument('--BatchID',help="Give this training sample batch an ID", default='SHIP_UR_v1')
parser.add_argument('--MaxMergeSize',help="A maximum number of track combinations that will be used in a particular HTCondor job for this script", default='20000')
########################################     Main body functions    #########################################
args = parser.parse_args()
i=int(args.i)    #This is just used to name the output file
p=args.p
o=args.o
sfx=args.sfx
pfx=args.pfx
AFS_DIR=args.AFS
EOS_DIR=args.EOS
MaxMergeSize=int(args.MaxMergeSize)
BatchID=args.BatchID

input_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/RUTr1c_'+BatchID+'_Fit_Filtered_Seeds_'+str(i)+'.pkl'

output_file_location=EOS_DIR+'/'+p+'/'+pfx+'_'+BatchID+'_'+o+'_'+str(i)+sfx

print(output_file_location)
exit()
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(UF.TimeStamp(), "Loading fit track seeds from the file",bcolors.OKBLUE+input_file_location+bcolors.ENDC)

base_data=UF.PickleOperations(input_file_location,'r', 'N/A')[0]

print(UF.TimeStamp(), bcolors.OKGREEN+"Loading is successful, there are total of "+str(len(base_data))+" glued tracks..."+bcolors.ENDC)
base_data=base_data[(i*MaxMergeSize):min(((i+1)*MaxMergeSize),len(base_data))]
print(UF.TimeStamp(), bcolors.OKGREEN+"Out of these only "+str(len(base_data))+" fit seeds will be considered here..."+bcolors.ENDC)
print(UF.TimeStamp(), "Initiating the  track merging...")
InitialDataLength=len(base_data)
TrackCounter=0
TrackCounterContinue=True
while TrackCounterContinue:
    if TrackCounter>=len(base_data):
       TrackCounterContinue=False
       break
    progress=round(float(TrackCounter)/float(len(base_data))*100,0)
    print(UF.TimeStamp(),'progress is ',progress,' %', end="\r", flush=True) #Progress display
    SubjectTrack=base_data[TrackCounter]
    for ObjectTrack in base_data[TrackCounter+1:]:
        if SubjectTrack.InjectTrackSeed(ObjectTrack):
           base_data.pop(base_data.index(ObjectTrack))
    TrackCounter+=1
print(str(InitialDataLength), "2-segments track seeds were merged into ", str(len(base_data)), 'tracks...')
print(UF.PickleOperations(output_file_location,'w', base_data)[1])

