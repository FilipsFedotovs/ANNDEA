#This script connects hits in the data to produce tracks
#Tracking Module of the ANNDEA package
#Made by Filips Fedotovs

########################################    Import libraries    #############################################
import csv
csv_reader=open('../config',"r")
config = list(csv.reader(csv_reader))
for c in config:
    if c[0]=='AFS_DIR':
        AFS_DIR=c[1]
    if c[0]=='EOS_DIR':
        EOS_DIR=c[1]
    if c[0]=='PY_DIR':
        PY_DIR=c[1]
csv_reader.close()
import sys
if PY_DIR!='': #Temp solution - the decision was made to move all libraries to EOS drive as AFS get locked during heavy HTCondor submission loads
    sys.path=['',PY_DIR]
    sys.path.append('/usr/lib64/python36.zip')
    sys.path.append('/usr/lib64/python3.6')
    sys.path.append('/usr/lib64/python3.6/lib-dynload')
    sys.path.append('/usr/lib64/python3.6/site-packages')
    sys.path.append('/usr/lib/python3.6/site-packages')
sys.path.append(AFS_DIR+'/Code/Utilities')
import pandas as pd #We use Panda for a routine data processing
import argparse
import numpy as np
from alive_progress import alive_bar
class bcolors:   #We use it for the interface
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print('                                                                                                                                    ')
print('                                                                                                                                    ')
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################        Initialising ANNDEA Hit Tracking module              #####################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script prepares training data for training the tracking model')
parser.add_argument('--RecBatchID',help="Give this reconstruction batch an ID", default='Test_Batch')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/afs/cern.ch/work/f/ffedship/public/SHIP/Source_Data/SHIP_Emulsion_FEDRA_Raw_UR.csv')
######################################## Parsing argument values  #############################################################
args = parser.parse_args()
RecBatchID=args.RecBatchID
input_file_location=args.f

sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import UtilityFunctions as UF #This is where we keep routine utility functions
import Parameters as PM #This is where we keep framework global parameters



#Non standard processes (that don't follow the general pattern) have been coded here
print(bcolors.HEADER+"#############################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(),bcolors.BOLD+'Stage 3:'+bcolors.ENDC+' Using the results from previous steps to map merged trackIDs to the original reconstruction file')
print(UF.TimeStamp(),'Loading the file ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
Data=pd.read_csv(input_file_location,header=0)
#It was discovered that the output is not perfect: while the hit fidelity is achieved we don't have a full plate hit fidelity for a given track. It is still possible for a track to have multiple hits at one plate.
#In order to fix it we need to apply some additional logic to those problematic tracks.
print(UF.TimeStamp(),'Identifying problematic tracks where there is more than one hit per plate...')
Hit_Map=Data[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.Hit_ID]] #Separating the hit map
Data.drop([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID'],axis=1,inplace=True) #Remove the ANNDEA tracking info from the main data
Hit_Map=Hit_Map.dropna() #Remove unreconstructing hits - we are not interested in them atm
Hit_Map_Stats=Hit_Map[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z,PM.Hit_ID]] #Calculating the stats

Hit_Map_Stats=Hit_Map_Stats.groupby([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID']).agg({PM.z:pd.Series.nunique,PM.Hit_ID: pd.Series.nunique}).reset_index() #Calculate the number fo unique plates and hits
Ini_No_Tracks=len(Hit_Map_Stats)
print(UF.TimeStamp(),bcolors.WARNING+'The initial number of tracks is '+ str(Ini_No_Tracks)+bcolors.ENDC)
Hit_Map_Stats=Hit_Map_Stats.rename(columns={PM.z: "No_Plates",PM.Hit_ID:"No_Hits"}) #Renaming the columns so they don't interfere once we join it back to the hit map
Hit_Map_Stats=Hit_Map_Stats[Hit_Map_Stats.No_Plates >= PM.MinHitsTrack]
Prop_No_Tracks=len(Hit_Map_Stats)
print(UF.TimeStamp(),bcolors.WARNING+'After dropping single hit tracks, left '+ str(Prop_No_Tracks)+' tracks...'+bcolors.ENDC)
Hit_Map=pd.merge(Hit_Map,Hit_Map_Stats,how='inner',on = [RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID']) #Join back to the hit map
Good_Tracks=Hit_Map[Hit_Map.No_Plates == Hit_Map.No_Hits] #For all good tracks the number of hits matches the number of plates, we won't touch them
Good_Tracks=Good_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.Hit_ID]] #Just strip off the information that we don't need anymore

Bad_Tracks=Hit_Map[Hit_Map.No_Plates < Hit_Map.No_Hits] #These are the bad guys. We need to remove this extra hits
Bad_Tracks=Bad_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.x,PM.y,PM.z,PM.tx,PM.ty,PM.Hit_ID]]

#Id the problematic plates
Bad_Tracks_Stats=Bad_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z,PM.Hit_ID]]
Bad_Tracks_Stats=Bad_Tracks_Stats.groupby([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z]).agg({PM.Hit_ID: pd.Series.nunique}).reset_index() #Which plates have double hits?
Bad_Tracks_Stats=Bad_Tracks_Stats.rename(columns={PM.Hit_ID: "Problem"}) #Renaming the columns so they don't interfere once we join it back to the hit map
Bad_Tracks=pd.merge(Bad_Tracks,Bad_Tracks_Stats,how='inner',on = [RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z])



Bad_Tracks.sort_values([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z],ascending=[0,0,1],inplace=True)
Bad_Tracks_Head=Bad_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID']]
Bad_Tracks_Head.drop_duplicates(inplace=True)
Bad_Tracks_List=Bad_Tracks.values.tolist() #I find it is much easier to deal with tracks in list format when it comes to fitting
Bad_Tracks_Head=Bad_Tracks_Head.values.tolist()
Bad_Track_Pool=[]

#Bellow we build the track representatation that we can use to fit slopes
with alive_bar(len(Bad_Tracks_Head),force_tty=True, title='Building track representations...') as bar:
            for bth in Bad_Tracks_Head:
               bar()
               bth.append([])
               bt=0
               trigger=False
               while bt<(len(Bad_Tracks_List)):
                   if (bth[0]==Bad_Tracks_List[bt][0] and bth[1]==Bad_Tracks_List[bt][1]):
                      if Bad_Tracks_List[bt][8]==1: #We only build polynomials for hits in a track that do not have duplicates - these are 'trusted hits'
                         bth[2].append(Bad_Tracks_List[bt][2:-2])
                      del Bad_Tracks_List[bt]
                      bt-=1
                      trigger=True
                   elif trigger:
                       break
                   else:
                       continue
                   bt+=1
with alive_bar(len(Bad_Tracks_Head),force_tty=True, title='Fitting the tracks...') as bar:
 for bth in Bad_Tracks_Head:
   bar()
   if len(bth[2])==1: #Only one trusted hit - In these cases whe we take only tx and ty slopes of the single base track. Polynomial of the first degree and the equations of the line are x=ax+tx*z and y=ay+ty*z
       x=bth[2][0][0]
       z=bth[2][0][2]
       tx=bth[2][0][3]
       ax=x-tx*z
       bth.append(ax) #Append x intercept
       bth.append(tx) #Append x slope
       bth.append(0) #Append a placeholder slope (for polynomial case)
       y=bth[2][0][1]
       ty=bth[2][0][4]
       ay=y-ty*z
       bth.append(ay) #Append x intercept
       bth.append(ty) #Append x slope
       bth.append(0) #Append a placeholder slope (for polynomial case)
       del(bth[2])
   elif len(bth[2])==2: #Two trusted hits - In these cases whe we fit a polynomial of the first degree and the equations of the line are x=ax+tx*z and y=ay+ty*z
       x,y,z=[],[],[]
       x=[bth[2][0][0],bth[2][1][0]]
       y=[bth[2][0][1],bth[2][1][1]]
       z=[bth[2][0][2],bth[2][1][2]]
       tx=np.polyfit(z,x,1)[0]
       ax=np.polyfit(z,x,1)[1]
       ty=np.polyfit(z,y,1)[0]
       ay=np.polyfit(z,y,1)[1]
       bth.append(ax) #Append x intercept
       bth.append(tx) #Append x slope
       bth.append(0) #Append a placeholder slope (for polynomial case)
       bth.append(ay) #Append x intercept
       bth.append(ty) #Append x slope
       bth.append(0) #Append a placeholder slope (for polynomial case)
       del(bth[2])
   else: #Three pr more trusted hits - In these cases whe we fit a polynomial of the second degree and the equations of the line are x=ax+(t1x*z)+(t2x*z*z) and y=ay+(t1y*z)+(t2y*z*z)
       x,y,z=[],[],[]
       for i in bth[2]:
           x.append(i[0])
       for j in bth[2]:
           y.append(j[1])
       for k in bth[2]:
           z.append(k[2])

       t2x=np.polyfit(z,x,2)[0]
       t1x=np.polyfit(z,x,2)[1]
       ax=np.polyfit(z,x,2)[2]

       t2y=np.polyfit(z,y,2)[0]
       t1y=np.polyfit(z,y,2)[1]
       ay=np.polyfit(z,y,2)[2]

       bth.append(ax) #Append x intercept
       bth.append(t1x) #Append x slope
       bth.append(t2x) #Append a placeholder slope (for polynomial case)
       bth.append(ay) #Append x intercept
       bth.append(t1y) #Append x slope
       bth.append(t2y) #Append a placeholder slope (for polynomial case)
       del(bth[2])

print(UF.TimeStamp(),'Removing problematic hits...')
#Once we get coefficients for all tracks we convert them back to Pandas dataframe and join back to the data
Bad_Tracks_Head=pd.DataFrame(Bad_Tracks_Head, columns = [RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID','ax','t1x','t2x','ay','t1y','t2y'])
Bad_Tracks=pd.merge(Bad_Tracks,Bad_Tracks_Head,how='inner',on = [RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID'])


#Calculating x and y coordinates of the fitted line for all plates in the track
Bad_Tracks['new_x']=Bad_Tracks['ax']+(Bad_Tracks[PM.z]*Bad_Tracks['t1x'])+((Bad_Tracks[PM.z]**2)*Bad_Tracks['t2x'])
Bad_Tracks['new_y']=Bad_Tracks['ay']+(Bad_Tracks[PM.z]*Bad_Tracks['t1y'])+((Bad_Tracks[PM.z]**2)*Bad_Tracks['t2y'])

#Calculating how far hits deviate from the fit polynomial
Bad_Tracks['d_x']=Bad_Tracks[PM.x]-Bad_Tracks['new_x']
Bad_Tracks['d_y']=Bad_Tracks[PM.y]-Bad_Tracks['new_y']
Bad_Tracks['d_r']=np.sqrt(Bad_Tracks['d_x']**2+Bad_Tracks['d_y']**2) #Absolute distance
Bad_Tracks=Bad_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z,PM.Hit_ID,'d_r']]

#Sort the tracks and their hits by Track ID, Plate and distance to the perfect line
Bad_Tracks.sort_values([RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z,'d_r'],ascending=[0,0,1,1],inplace=True)

#If there are two hits per plate we will keep the one which is closer to the line
Bad_Tracks.drop_duplicates(subset=[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.z],keep='first',inplace=True)
Bad_Tracks=Bad_Tracks[[RecBatchID+'_Brick_ID',RecBatchID+'_Track_ID',PM.Hit_ID]]
Good_Tracks=pd.concat([Good_Tracks,Bad_Tracks]) #Combine all ANNDEA tracks together

Data=pd.merge(Data,Good_Tracks,how='left', on=[PM.Hit_ID]) #Re-map corrected ANNDEA Tracks back to the main data
output_file_location=EOS_DIR+'/ANNDEA/Data/REC_SET/'+RecBatchID+'_RTr_OUTPUT_CLEANED.csv'
Data.to_csv(output_file_location,index=False)
print(UF.TimeStamp(), bcolors.OKGREEN+"The cleaned tracked data has been written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)




