#This is the list of parameters that ANNDEA uses for reconstruction, model training etc. There have been collated here in one place for the user convenience
# Part of ANNDEA package
#Made by Filips Fedotovs
#Current version 1.0

######List of naming conventions
Hit_ID='Hit_ID'
x='x' #Column name x-coordinate of the track hit
y='y' #Column name for y-coordinate of the track hit
tx='tx' #Column name x-coordinate of the track hit
ty='ty' #Column name for y-coordinate of the track hit
z='z' #Column name for z-coordinate of the track hit

Rec_Track_ID='SND_Train_RUTr_GMM_FEDRA_Phase2a_Test_Track_ID' #Column nameActual track id for FEDRA (or other reconstruction software)
Rec_Track_Domain='SND_Train_RUTr_GMM_FEDRA_Phase2a_Test_Brick_ID' #Quarter of the ECC where the track is reconstructed If not present in the data please put the Track ID (the same as above)

MC_Track_ID='MC_Track_ID'  #Column name for Track ID for MC Truth reconstruction data
MC_Event_ID='MC_Event_ID' #Column name for Event id for MC truth reconstruction data (If absent please enter the MCTrack as for above)
MC_VX_ID='MC_Mother_ID' 
MinHitsTrack=2
TST=0.0001
valRatio=0.1

######Tracking Module - list of parameters that are specific to this module
stepX=6000
stepY=6000
stepZ=12000
cut_dt=0.2
cut_dr=60
MaxSeedsPerVxPool=20000
testRatio=0.05
MaxValSampleSize=50000
num_node_features=5
num_edge_features=5
Model_Name='Model_Identity'
ModelArchitecture=[[2], [1], [],[], [], [], [], [], [], [], [5]]


######Track Union Module - list of parameters that are specific to this module

MaxSegments=10000 #This parameter imposes the limit on the number of the tracks form the Start plate when forming the Seeds.
MaxSeeds=50000
# MaxFitTracksPerJob=10000
# MaxTracksPerTrPool=20000

######List of geometrical constain parameters
MinHitsTrack=4
VetoMotherTrack=[]
VetoVertex=['-1', '-2']


def Seed_Bond_Fit_Acceptance(row):
        if row['AntiLink_Strenth']>0:
          return 1.16*(row['Link_Strength']+row['Seed_CNN_Fit'])/row['AntiLink_Strenth']
        else:
          return 100
        
pre_vx_acceptance=0.662
link_acceptance=1.2