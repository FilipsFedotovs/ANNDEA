#This is the list of parameters that EDER-GNN uses for reconstruction, model training etc. There have been collated here in one place for the user convenience
# Part of EDER-GNN package
#Made by Filips Fedotovs
#Current version 1.0

######List of naming conventions
Hit_ID='ID'
x='x' #Column name x-coordinate of the track hit
y='y' #Column name for y-coordinate of the track hit
tx='TX' #Column name x-coordinate of the track hit
ty='TY' #Column name for y-coordinate of the track hit
z='z' #Column name for z-coordinate of the track hit
FEDRA_Track_ID='FEDRATrackID' #Column nameActual track id for FEDRA (or other reconstruction software)
FEDRA_Track_QUADRANT='quarter' #Quarter of the ECC where the track is reconstructed If not present in the data please put the Track ID (the same as above)
MC_Track_ID='MCTrack'  #Column name for Track ID for MC Truth reconstruction data
MC_Event_ID='MCEvent' #Column name for Event id for MC truth reconstruction data (If absent please enter the MCTrack as for above)


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
num_node_features=3
num_edge_features=3

Model_Name='Model_Identity'
ModelArchitecture=[[2], [1], [],[], [], [], [], [], [], [], [5]]

