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
# parser.add_argument('--TrackFitCutRes',help="Track Fit cut Residual", default=1000,type=int)
# parser.add_argument('--TrackFitCutSTD',help="Track Fit cut", default=10,type=int)
# parser.add_argument('--TrackFitCutMRes',help="Track Fit cut", default=200,type=int)
# parser.add_argument('--stepX',help="Enter X step size", default='0')
# parser.add_argument('--stepY',help="Enter Y step size", default='0')
# parser.add_argument('--stepZ',help="Enter Z step size", default='0')
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')
parser.add_argument('--PY',help="Python libraries directory location", default='.')
parser.add_argument('--ModelName',help="Name of the model to use?", default='0')
parser.add_argument('--BatchID',help="Give name to this train sample", default='')
parser.add_argument('--p',help="Path to the output file", default='')
parser.add_argument('--o',help="Path to the output file name", default='')
parser.add_argument('--pfx',help="Path to the output file name", default='')
parser.add_argument('--sfx',help="Path to the output file name", default='')
parser.add_argument('--CheckPoint',help="Save cluster sets during individual cluster tracking.", default='N')
parser.add_argument('--SeedFlowLog',help="Track the seed cutflow?", default='N')

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
######################################## Set variables  #############################################################
ModelName=args.ModelName
CheckPoint=args.CheckPoint.upper()=='Y'
RecBatchID=args.BatchID
SeedFlowLog=args.SeedFlowLog=='Y'
# TrackFitCutRes=args.TrackFitCutRes
# TrackFitCutSTD=args.TrackFitCutSTD
# TrackFitCutMRes=args.TrackFitCutMRes
p,o,sfx,pfx=args.p,args.o,args.sfx,args.pfx
i,j,k=args.i,args.j,args.k
#stepX,stepY,stepZ=float(args.stepX),float(args.stepY),float(args.stepZ)
import U_UI as UI #This is where we keep routine utility functions
import U_ML as ML
#This function combines two segment object. Example: Segment 1 is [[a, _ ,b ,_ ,_ ][0.9,0.0,0.9,0.0,0.0]];  Segment 2 is [[a, _ ,c ,_ ,_ ][0.9,0.0,0.8,0.0,0.0]]; Segment 3 is [[_, d ,b ,_ ,_ ][0.0,0.8,0.8,0.0,0.0]]
#In order to combine segments we have to have at least one common hit and no clashes. Segment 1 and 2 have a common hit a, but their third plates clash. Segment 1 can be combined with segment 3 which yields: [[a, d ,b ,_ ,_ ][0.8,0.0,1.7,0.0,0.0]]
#Please note that if combination occurs then the hit weights combine together too
def InjectHit(Predator,Prey, Soft):
          if Soft==False:
             OverlapDetected=False
             _intersection=list(set(Predator[0]) & set(Prey[0])) #Do we have any common hits between two track segment skeletons?
             if len(_intersection)<2: #In order for hits to be combined there should always be at least two hits overlapping. Usually it will be '_' and a hit. It can be two hits overlap (Enforcing the existing connection)
                return(Predator,False)
             elif len(_intersection)==3: #This is the case when we have two overlapping hits (reinforcement) and an overlapping hole
                _intersection=[value for value in _intersection if value != "_"] #Remove the hole since we are not interested
                _index1 = Predator[0].index(_intersection[0]) #Find the index of the first overlapping hit
                _index2 = Predator[0].index(_intersection[1]) #Find the index of the second overlapping hit
                New_Predator=copy.deepcopy(Predator)
                New_Predator[1][_index1]+=Prey[1][_index1] #Add additional weights (Reinforcement of the hit connection)
                New_Predator[1][_index2]+=Prey[1][_index2] #Add additional weifght (Reinforcement of the hit connection)
                return(New_Predator,True) #Return the enriched track segment
             New_Predator=copy.deepcopy(Predator)
             _prey_trigger_count=0
             for el in range (len(Prey[0])): #If we have one overlapping hit and one overlapping hole
                 if Prey[0][el]!='_' and Predator[0][el]!='_': #Comparing hits not holes
                     if Prey[0][el]!=Predator[0][el]: #Reject if there is a clash
                        return(Predator,False)
                     elif Prey[0][el]==Predator[0][el]: #We found that there there is an overlapping hit
                        OverlapDetected=True
                        New_Predator[1][el]+=Prey[1][el] #Add new weight for the overlapping hit
                        _prey_trigger_count+=1
                 elif Predator[0][el]=='_' and Prey[0][el]!=Predator[0][el]: #If there is an overlap we fill the hole with a new hit from the subject pair
                     New_Predator[0][el]=Prey[0][el]
                     New_Predator[1][el]+=Prey[1][el] #Add new weigth for the new hit
                 if _prey_trigger_count>=2: #Max overlap can be 2 only, break the process to save time
                     break

             if OverlapDetected:
                return(New_Predator,True) #If overalp detected we return the updated track segment with incorporated hit
             else:
                return(Predator,False) #Return the same segment unchanged
          if Soft==True: #This is jsut a loop to check the overlap between two track segments (used later)
             for el1 in Prey[0]:
                 for el2 in Predator:
                  if el1==el2:
                     return True
             return False

Status='Edge graph generation'
CheckPointFile_ML=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(i)+'_'+str(j)+'_'+str(k) + '_CP_ML.csv'
# CheckPointFile_Prep_1=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(i)+'_'+str(j) +'_'+str(k) +'_CP_Prep_1.csv'
# CheckPointFile_Prep_2=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(i)+'_'+str(j) +'_'+str(k) +'_CP_Prep_2.csv'
# CheckPointFile_Tracking_TH=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(i)+'_'+str(j)+'_'+str(k) +'_CP_Tracking_TH.csv'
# CheckPointFile_Tracking_RP=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(i)+'_'+str(j)+'_'+str(k) +'_CP_Tracking_RP.csv'

# if os.path.isfile(CheckPointFile_Tracking_TH) and os.path.isfile(CheckPointFile_Tracking_RP):
#         UI.Msg('location','Loading checkpoint file ',CheckPointFile_Tracking_TH)
#         _Tot_Hits = UI.LogOperations(CheckPointFile_Tracking_TH,'r','N/A')
#         UI.Msg('location','Loading checkpoint file ',CheckPointFile_Tracking_RP)
#         _Rec_Hits_Pool = UI.LogOperations(CheckPointFile_Tracking_RP,'r','N/A')
#         UI.Msg('location','Loading checkpoint file ',CheckPointFile_Prep_2)
#         _z_map=pd.read_csv(CheckPointFile_Prep_2)
#         for i in range(len(_Tot_Hits)):
#             for j in range(len(_Tot_Hits[i])):
#                 _Tot_Hits[i][j]=ast.literal_eval(_Tot_Hits[i][j])
#             for k in range(len(_Tot_Hits[i][0])):
#                 if type(_Tot_Hits[i][0][k]) is float:
#                     _Tot_Hits[i][0][k]=str(int(_Tot_Hits[i][0][k]))
#                 if type(_Tot_Hits[i][0][k]) is int:
#                     _Tot_Hits[i][0][k]=str(_Tot_Hits[i][0][k])
#         Status = 'Tracking continuation'
# elif os.path.isfile(CheckPointFile_Prep_1) and os.path.isfile(CheckPointFile_Prep_2):
#         UI.Msg('location','Loading checkpoint file ',CheckPointFile_Prep_1)
#         _Tot_Hits = UI.LogOperations(CheckPointFile_Prep_1,'r','N/A')
#         for i in range(len(_Tot_Hits)):
#             for j in range(len(_Tot_Hits[i])):
#                 _Tot_Hits[i][j]=ast.literal_eval(_Tot_Hits[i][j])
#             for k in range(len(_Tot_Hits[i][0])):
#                 if type(_Tot_Hits[i][0][k]) is float:
#                     _Tot_Hits[i][0][k]=str(int(_Tot_Hits[i][0][k]))
#                 if type(_Tot_Hits[i][0][k]) is int:
#                     _Tot_Hits[i][0][k]=str(_Tot_Hits[i][0][k])
#         UI.Msg('location','Loading checkpoint file ',CheckPointFile_Prep_2)
#         _z_map=pd.read_csv(CheckPointFile_Prep_2)
#         Status = 'Tracking'
if os.path.isfile(CheckPointFile_ML):
        UI.Msg('location','Loading checkpoint file ',CheckPointFile_ML)
        _Tot_Hits = pd.read_csv(CheckPointFile_ML)
        Status = 'Tracking'

#Specifying the full path to input/output files

torch_import=True
input_file_location=EOS_DIR+p+'/Temp_RTr1b_'+RecBatchID+'_'+str(i)+'_'+str(j)+'/RTr1b_'+RecBatchID+'_hit_cluster_edges_consolidated_'+str(i)+'_'+str(j)+'_'+str(k)+'.pkl'
#if os.path.isfile(input_file_location) and Status=='Edge graph generation':
if Status=='Edge graph generation':
    HC=UI.PickleOperations(input_file_location,'r',' ')[0]
    if HC.ClusterSize<=1 or len(HC.Seeds)<1:
        Status = 'Skip tracking'
else:
    Status = 'Skip tracking'

if Status=='Edge graph generation':
    print(UI.TimeStamp(),'Reconstructing graph...')
    GraphStatus = HC.GenerateSeedGraph()
    if GraphStatus:
        Status = 'ML analysis'
    else:
        Status = 'Skip tracking'

if Status == 'ML analysis':
    print(UI.TimeStamp(),'Classifying the edges...')
    if args.ModelName!='blank':
        print(UI.TimeStamp(),'Preparing the model')
        import torch
        EOSsubDIR=EOS_DIR+'/'+'ANNDEA'
        EOSsubModelDIR=EOSsubDIR+'/'+'Models'
        #Load the model meta file
        Model_Meta_Path=EOSsubModelDIR+'/'+args.ModelName+'_Meta'
        #Specify the model path
        Model_Path=EOSsubModelDIR+'/'+args.ModelName
        ModelMeta=UI.PickleOperations(Model_Meta_Path, 'r', 'N/A')[0]
        #Meta file contatins training session stats. They also record the optimal acceptance.
        Acceptance=ModelMeta.TrainSessionsData[-1][-1][3]
        device = torch.device('cpu')
        #In PyTorch we don't save the actual model like in Tensorflow. We just save the weights, so we must regenerate the model again. The recipe is in the Model Meta file
        model = ML.GenerateModel(ModelMeta).to(device)
        model.load_state_dict(torch.load(Model_Path))
        model.eval() #In Pytorch this function sets the model into the evaluation mode.
        w = model(HC.Graph.x, HC.Graph.edge_index, HC.Graph.edge_attr) #Here we use the model to assign the weights between Hit edges
        weights=w.tolist()
        _Tot_Hits=[]
        for sd,w in zip(HC.Seeds, weights):
            _Tot_Hits.append(sd[:3]+w) #Join the Hit Pair classification back to the hit pairs

        for cwl in _Tot_Hits:
            for lh in HC.Hits:
                if lh[0]==cwl[0]:
                    cwl.append(lh[3])
                    break

        for cwl in _Tot_Hits:
            for rh in HC.Hits:
                if rh[0]==cwl[1]:
                    cwl.append(rh[3])
                    break

        _Tot_Hits=pd.DataFrame(_Tot_Hits, columns = ['l_HitID','r_HitID','label','link_strength', 'l_z', 'r_z'])
        _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['link_strength'] <= Acceptance], inplace = True) #Remove all hit pairs that fail GNN classification

        if SeedFlowLog:
            HC.SeedFlowValuesAll[10]=len(_Tot_Hits)
            _truth_only=_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['label'] == 0]) #Remove all hit pairs that fail GNN classification
            HC.SeedFlowValuesTrue[10]=len(_truth_only)

    else:
        _Tot_Hits=[]
        for sd in HC.Seeds:
            _Tot_Hits.append(sd[:3]+1) #Join the Hit Pair classification back to the hit pairs

        for cwl in _Tot_Hits:
            for lh in HC.Hits:
                if lh[0]==cwl[0]:
                    cwl.append(lh[3])
                    break

        for cwl in _Tot_Hits:
            for rh in HC.Hits:
                if rh[0]==cwl[1]:
                    cwl.append(rh[3])
                    break

        _Tot_Hits=pd.DataFrame(_Tot_Hits, columns = ['l_HitID','r_HitID','label','link_strength', 'l_z', 'r_z'])

        if SeedFlowLog:
            HC.SeedFlowValuesAll[10]=len(_Tot_Hits)
            _truth_only=_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['label'] == 0]) #Remove all hit pairs that fail GNN classification
            HC.SeedFlowValuesTrue[10]=len(_truth_only)

    print(UI.TimeStamp(),'Number of all  hit combinations passing GNN selection:',len(_Tot_Hits))
    if CheckPoint:
             print(UI.TimeStamp(),'Saving the checkpoint...')
             _Tot_Hits.to_csv(CheckPointFile_ML,index=False)
    Status='Track preparation'

if Status=='Track preparation':
        _Tot_Hits=_Tot_Hits[['r_HitID','l_HitID','r_z','l_z','link_strength']]
        print(UI.TimeStamp(),'Preparing the weighted hits for tracking...')
        #Bellow just some data prep before the tracking procedure
        _Tot_Hits.sort_values(by = ['r_HitID', 'l_z','link_strength'], ascending=[True,True, False],inplace=True)
        _Loc_Hits_r=_Tot_Hits[['r_z']].rename(columns={'r_z': 'z'})
        _Loc_Hits_l=_Tot_Hits[['l_z']].rename(columns={'l_z': 'z'})
        _Loc_Hits=pd.concat([_Loc_Hits_r,_Loc_Hits_l])
        _Loc_Hits.sort_values(by = ['z'], ascending=[True],inplace=True)
        _Loc_Hits.drop_duplicates(subset=['z'], keep='first', inplace=True)
        _Loc_Hits=_Loc_Hits.reset_index(drop=True)
        _Loc_Hits=_Loc_Hits.reset_index()
        _z_map_r=_Tot_Hits[['r_HitID','r_z']].rename(columns={'r_z': 'z','r_HitID': 'HitID'})
        _z_map_l=_Tot_Hits[['l_HitID','l_z']].rename(columns={'l_z': 'z','l_HitID': 'HitID'})
        _z_map=pd.concat([_z_map_r,_z_map_l])
        _z_map.drop_duplicates(subset=['HitID','z'], keep='first', inplace=True)
        _Loc_Hits_r=_Loc_Hits.rename(columns={'index': 'r_index', 'z': 'r_z'})
        _Loc_Hits_l=_Loc_Hits.rename(columns={'index': 'l_index', 'z': 'l_z'})
        _Tot_Hits=pd.merge(_Tot_Hits,_Loc_Hits_r, how='inner', on=['r_z'])
        _Tot_Hits=pd.merge(_Tot_Hits,_Loc_Hits_l, how='inner', on=['l_z'])
        _Tot_Hits=_Tot_Hits[['r_HitID','l_HitID','r_index','l_index','link_strength']]
        _Tot_Hits.sort_values(by = ['r_HitID', 'l_index','link_strength'], ascending=[True,True, False],inplace=True)
        _Tot_Hits.drop_duplicates(subset=['r_HitID', 'l_index','link_strength'], keep='first', inplace=True)
        _Tot_Hits.sort_values(by = ['l_HitID', 'r_index','link_strength'], ascending=[True,True, False],inplace=True)
        _Tot_Hits.drop_duplicates(subset=['l_HitID', 'r_index','link_strength'], keep='first', inplace=True)
        _Tot_Hits=_Tot_Hits.values.tolist()
        _Temp_Tot_Hits=[]

        #Bellow we build the track segment structure object that is made of two arrays:
        #First array contains a list of hits arranged in z-order
        #Second array contains the weights assighned to the hit combination

        #Example if we have a cluster with 5 plates and a hit pair a and b at first and third plate and the weight is 0.9 then the object will look like [[a, _ ,b ,_ ,_ ][0.9,0.0,0.9,0.0,0.0]]. '_' represents a missing hit at a plate
        for el in _Tot_Hits:
                        _Temp_Tot_Hit_El = [[],[]]
                        for pos in range(len(_Loc_Hits)):
                            if pos==el[2]:
                                _Temp_Tot_Hit_El[0].append(el[0])
                                _Temp_Tot_Hit_El[1].append(el[4])
                            elif pos==el[3]:
                                _Temp_Tot_Hit_El[0].append(el[1])
                                _Temp_Tot_Hit_El[1].append(el[4])
                            else:
                                _Temp_Tot_Hit_El[0].append('_')
                                _Temp_Tot_Hit_El[1].append(0.0)
                        _Temp_Tot_Hits.append(_Temp_Tot_Hit_El)
        _Tot_Hits=_Temp_Tot_Hits
        # if CheckPoint:
        #     print(UI.TimeStamp(),'Saving checkpoint 4...')
        #     UI.LogOperations(CheckPointFile_Prep_1,'w',_Tot_Hits)
        #     _z_map.to_csv(CheckPointFile_Prep_2,index=False)
        Status='Tracking'

if Status=='Tracking' or Status=='Tracking continuation':
    print(UI.TimeStamp(),'Tracking the cluster...')
    if Status=='Tracking':
        _Rec_Hits_Pool=[]
    _intital_size=len(_Tot_Hits)
    KeepTracking=True
    while len(_Tot_Hits)>0 and KeepTracking:
                    _Tot_Hits_PCopy=copy.deepcopy(_Tot_Hits)
                    _Tot_Hits_Predator=[]
                    #Bellow we build all possible hit combinations that can occur in the data
                    print(UI.TimeStamp(),'Building all possible track combinations...')
                    for prd in range(len(_Tot_Hits_PCopy)):
                        Predator=_Tot_Hits_PCopy[prd]
                        for pry in range(prd+1,len(_Tot_Hits_PCopy)):
                               #This function combines two segment object. Example: Segment 1 is [[a, _ ,b ,_ ,_ ][0.9,0.0,0.9,0.0,0.0]];  Segment 2 is [[a, _ ,c ,_ ,_ ][0.9,0.0,0.8,0.0,0.0]]; Segment 3 is [[_, d ,b ,_ ,_ ][0.0,0.8,0.8,0.0,0.0]]
                               #In order to combine segments we have to have at least one common hit and no clashes. Segment 1 and 2 have a common hit a, but their third plates clash. Segment 1 can be combined with segment 3 which yields: [[a, d ,b ,_ ,_ ][0.8,0.0,1.7,0.0,0.0]]
                               #Please note that if combination occurs then the hit weights combine together too
                               Predator=InjectHit(Predator,_Tot_Hits_PCopy[pry],False)[0]
                        _Tot_Hits_Predator.append(Predator)
                    #We calculate the average value of the segment weight
                    for s in _Tot_Hits_Predator:
                        s=s[0].append(mean(s.pop(1)))
                    _Tot_Hits_Predator = [item for l in _Tot_Hits_Predator for item in l]
                    for s in range(len(_Tot_Hits_Predator)):
                        for h in range(len(_Tot_Hits_Predator[s])):
                            if _Tot_Hits_Predator[s][h] =='_':
                                _Tot_Hits_Predator[s][h]='H_'+str(s) #Giving holes a unique name to avoid problems later
                    column_no=len(_Tot_Hits_Predator[0])-1
                    columns=[]

                    #Residual_Cut=(TrackFitCutRes>min(stepX,stepY) and TrackFitCutSTD>min(stepX,stepY) and TrackFitCutMRes>min(stepX,stepY))
                    # if Residual_Cut==False:
                    #     print(UI.TimeStamp(),'Applying physical assumptions...')
                    #     #Here we making sure that the tracks satisfy minimum fit requirements
                    #     for thp in _Tot_Hits_Predator:
                    #         fit_data_x=[]
                    #         fit_data_y=[]
                    #         fit_data_z=[]
                    #         for cc in range(column_no):
                    #             for td in temp_data_list:
                    #                 if td[0][0]=='H':
                    #                     break
                    #                 elif td[0]==thp[cc]:
                    #                     fit_data_x.append(td[1])
                    #                     fit_data_y.append(td[2])
                    #                     fit_data_z.append(td[3])
                    #                     break
                    #
                    #         line_x=np.polyfit(fit_data_z,fit_data_x,1)
                    #         line_y=np.polyfit(fit_data_z,fit_data_y,1)
                    #         x_residual=[x * line_x[0] for x in fit_data_z]
                    #         x_residual=[x + line_x[1] for x in x_residual]
                    #         x_residual=(np.array(x_residual)-np.array(fit_data_x))
                    #         x_residual=[x ** 2 for x in x_residual]
                    #
                    #         y_residual=[y * line_y[0] for y in fit_data_z]
                    #         y_residual=[y + line_y[1] for y in y_residual]
                    #         y_residual=(np.array(y_residual)-np.array(fit_data_y))
                    #         y_residual=[y ** 2 for y in y_residual]
                    #         residual=np.array(y_residual)+np.array(x_residual)
                    #         residual=np.sqrt(residual)
                    #         RES=sum(residual)/len(fit_data_x)
                    #         STD=np.std(residual)
                    #         MRES=max(residual)
                    #         _Tot_Hits_Predator[_Tot_Hits_Predator.index(thp)]+=[RES,STD,MRES]

                    #converting the list objects into Pandas dataframe
                    for c in range(column_no):
                        columns.append(str(c))
                    columns+=['average_link_strength']
                    # if Residual_Cut==False:
                    #     columns+=['RES','STD','MRES']
                    _Tot_Hits_Predator=pd.DataFrame(_Tot_Hits_Predator, columns = columns)
                    # if Residual_Cut==False:
                    #     _Tot_Hits_Predator=_Tot_Hits_Predator.drop(_Tot_Hits_Predator.index[(_Tot_Hits_Predator['RES'] > TrackFitCutRes) | (_Tot_Hits_Predator['STD'] > TrackFitCutSTD) | (_Tot_Hits_Predator['MRES'] > TrackFitCutMRes)]) #Remove tracks with a bad fit

                    KeepTracking=len(_Tot_Hits_Predator)>0
                    _Tot_Hits_Predator.sort_values(by = ['average_link_strength'], ascending=[False],inplace=True) #Keep all the best hit combinations at the top
                    _Tot_Hits_Predator=_Tot_Hits_Predator.drop(['average_link_strength'],axis=1) #We don't need the segment fit anymore
                    # if Residual_Cut==False:
                    #     _Tot_Hits_Predator=_Tot_Hits_Predator.drop(['RES','STD','MRES'],axis=1) #We don't need the segment fit anymore
                    for c in range(column_no):
                        _Tot_Hits_Predator.drop_duplicates(subset=[str(c)], keep='first', inplace=True) #Iterating over hits, make sure that they belong to the best-fit track
                    _Tot_Hits_Predator=_Tot_Hits_Predator.values.tolist()
                    for seg in range(len(_Tot_Hits_Predator)):
                        _Tot_Hits_Predator[seg]=[s for s in _Tot_Hits_Predator[seg] if ('H' in s)==False] #Remove holes from the track representation
                    _Rec_Hits_Pool+=_Tot_Hits_Predator
                    for seg in _Tot_Hits_Predator:
                        _itr=0
                        while _itr<len(_Tot_Hits):
                            if InjectHit(seg,_Tot_Hits[_itr],True): #We remove all the hits that become part of successful segments from the initial pool so we can rerun the process again with leftover hits
                                del _Tot_Hits[_itr]
                            else:
                                _itr+=1
                    # if CheckPoint:
                    #     print(UI.TimeStamp(),'(Re-)Saving checkpoint 5...')
                    #     UI.LogOperations(CheckPointFile_Tracking_TH,'w',_Tot_Hits)
                    #     UI.LogOperations(CheckPointFile_Tracking_RP,'w',_Rec_Hits_Pool)
                    Status='Tracking continuation'
    #Transpose the rows
    _track_list=[]
    _segment_id=RecBatchID+'_'+str(i)+'_'+str(j)+'_'+str(k) #Each segment name will have a relevant prefix (since numeration is only unique within an isolated cluster)
    _no_tracks=len(_Rec_Hits_Pool)
    for t in range(len(_Rec_Hits_Pool)):
                  for h in _Rec_Hits_Pool[t]:
                         _track_list.append([_segment_id+'-'+str(t+1),h])
    _Rec_Hits_Pool=pd.DataFrame(_track_list, columns = ['Segment_ID','HitID'])
    _z_map['HitID']=_z_map['HitID'].astype(str)
    _Rec_Hits_Pool=pd.merge(_z_map, _Rec_Hits_Pool, how="right", on=['HitID'])
    _Rec_Hits_Pool=_Rec_Hits_Pool.rename(columns={"z": "Master_z" })
    _Rec_Hits_Pool=_Rec_Hits_Pool.rename(columns={"Segment_ID": "Master_Segment_ID" })
    print(UI.TimeStamp(),_no_tracks, 'track segments have been reconstructed in this cluster set ...')

print(_Rec_Hits_Pool)
print(HC.Hits)
exit()
#If Cluster tracking yielded no segments we just create an empty array for consistency
if Status=='Skip tracking':
    _Rec_Hits_Pool=pd.DataFrame([], columns = ['HitID','Master_z','Master_Segment_ID'])

output_file_location=EOS_DIR+p+'/Temp_'+pfx+'_'+RecBatchID+'_'+str(i)+'_'+str(j)+'/'+pfx+'_'+RecBatchID+'_'+o+'_'+str(i)+'_'+str(j)+'_'+str(k)+sfx
print(UI.TimeStamp(),'Writing the output...')
_Rec_Hits_Pool.to_csv(output_file_location,index=False) #Write the final result
print(UI.TimeStamp(),'Output is written to ',output_file_location)
exit()


