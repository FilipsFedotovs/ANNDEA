#This is the list of parameters that ANNADEA uses for reconstruction, model training etc. There have been collated here in one place for the user convenience
# Part of ANNADEA package
#Made by Filips Fedotovs
#Current version 1.0

######List of naming conventions
Hit_ID='ID'
x='X' #Column name x-coordinate of the track hit
y='Y' #Column name for y-coordinate of the track hit
tx='TX' #Column name x-coordinate of the track hit
ty='TY' #Column name for y-coordinate of the track hit
z='Z' #Column name for z-coordinate of the track hit
FEDRA_Track_ID='FEDRA Track ID' #Column nameActual track id for FEDRA (or other reconstruction software)
FEDRA_Track_QUADRANT='Quarter' #Quarter of the ECC where the track is reconstructed If not present in the data please put the Track ID (the same as above)
MC_Track_ID='MC Track'  #Column name for Track ID for MC Truth reconstruction data
MC_Event_ID='MC Event' #Column name for Event id for MC truth reconstruction data (If absent please enter the MCTrack as for above)


######List of geometrical constain parameters
stepX=3000
stepY=3000
stepZ=6000
cut_dt=0.2
cut_dr=60
valRatio=0.1
testRatio=0.05
TST=0.0001

MinHitsTrack=2
num_node_features=5
num_edge_features=5

Model_Name='Model_Identity'
ModelArchitecture=[[2], [1], [],[], [], [], [], [], [], [], [5]]

