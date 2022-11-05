#This is the list of parameters that ANNDEA uses for reconstruction, model training etc. There have been collated here in one place for the user convenience
# Part of ANNDEA package
#Made by Filips Fedotovs
#Current version 1.0

######List of naming conventions
Hit_ID='ID'
x='x' #Column name x-coordinate of the track hit
y='y' #Column name for y-coordinate of the track hit
tx='TX' #Column name x-coordinate of the track hit
ty='TY' #Column name for y-coordinate of the track hit
z='z' #Column name for z-coordinate of the track hit
Rec_Track_ID='FEDRATrackID' #Column nameActual track id for FEDRA (or other reconstruction software)
Rec_Track_Domain='quarter' #Quarter of the ECC where the track is reconstructed If not present in the data please put the Track ID (the same as above)
MC_Track_ID='MCTrack'  #Column name for Track ID for MC Truth reconstruction data
MC_Event_ID='MCEvent' #Column name for Event id for MC truth reconstruction data (If absent please enter the MCTrack as for above)
MinHitsTrack=2
TST=0.0001
valRatio=0.1

######Tracking Module - list of parameters that are specific to this module
stepX=3000
stepY=3000
stepZ=6000
cut_dt=0.2
cut_dr=60
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
MaxSLG=7000
MaxSTG=160#This parameter restricts the maximum length of of the longitudinal and transverse distance between track segments.
MinHitsTrack=2
MaxDOCA=50
MaxAngle=1 #Seed Opening Angle (Magnitude) in radians
VetoMotherTrack=[]

# MaxTracksPerJob=20000
# MaxEvalTracksPerJob=20000
# MaxSeedsPerJob=40000
# MaxVxPerJob=10000
# MaxSeedsPerVxPool=20000
# ##Model parameters
# pre_acceptance=0.5
# post_acceptance=0.5
# bg_acceptance = 1.0
# #pre_vx_acceptance=0.662
# resolution=50
# MaxX=1000.0
# MaxY=1000.0
# MaxZ=3000.0
# GNNMaxX=100.0
# GNNMaxY=100.0
# GNNMaxZ=1315
# GNNMaxTX=0.01
# GNNMaxTY=0.01
# Pre_CNN_Model_Name='1T_50_SHIP_PREFIT_1_model'
# Post_CNN_Model_Name='1T_50_SHIP_POSTFIT_1_model'
# Classifier_Model_Name= 'SND_Reduced_3Class2'
# Post_GNN_Model_Name = 'SND_Glue_Post_GMM3_FullCo_Angle2'
# #ModelArchitecture=[[6, 4, 1, 2, 2, 2, 2], [], [],[], [], [1, 4, 2], [], [], [], [], [7, 1, 1, 4]]
# ModelArchitecture=[[1, 4, 1, 8, 2, 2, 2], [], [],[], [], [1, 4, 2], [], [], [], [], [7, 1, 1, 4]]
# ModelArchitecturePlus=[[1, 4, [2, 2, 2], [8, 8, 8], 2, 2, 2], [], [],[], [], [1, 4, 2], [], [], [], [], [7, 1, 1, 4]]
