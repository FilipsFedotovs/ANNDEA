###This file contains the utility functions that are commonly used in ANNDEA packages
import numpy as np
def GetEquationOfLine(Data):
          x=[]
          for i in range(0,len(Data)):
              x.append(i)
          line=np.polyfit(x,Data,1)
          return line
class ModelMeta:
      def __init__(self,ModelID):
          self.ModelID=ModelID
      def __eq__(self, other):
        return (self.ModelID) == (other.ModelID)
      def __hash__(self):
        return hash(self.ModelID)
      def IniModelMeta(self, ModelParams, framework, DataMeta, architecture, type):
          self.ModelParameters=ModelParams
          self.ModelFramework=framework
          self.ModelArchitecture=architecture
          self.ModelType=type
          self.TrainSessionsDataID=[]
          self.TrainSessionsDateTime=[]
          self.TrainSessionsParameters=[]
          self.TrainSessionsData=[]
          if hasattr(DataMeta,'ClassHeaders'):
              self.ClassHeaders=DataMeta.ClassHeaders
          if hasattr(DataMeta,'ClassNames'):
              self.ClassNames=DataMeta.ClassNames
          if hasattr(DataMeta,'ClassValues'):
              self.ClassValues=DataMeta.ClassValues

          if (self.ModelFramework=='PyTorch') and (self.ModelArchitecture=='TCN'):
              self.num_node_features=DataMeta.num_node_features
              self.num_edge_features=DataMeta.num_edge_features
              self.stepX=DataMeta.stepX
              self.stepY=DataMeta.stepY
              self.stepZ=DataMeta.stepZ
              self.cut_dt=DataMeta.cut_dt
              self.cut_dr=DataMeta.cut_dr
              self.cut_dz=DataMeta.cut_dz
          else:
              if hasattr(DataMeta,'MaxSLG'):
                  self.MaxSLG=DataMeta.MaxSLG
              if hasattr(DataMeta,'MaxSTG'):
                  self.MaxSTG=DataMeta.MaxSTG
              if hasattr(DataMeta,'MaxDST'):
                  self.MaxDST=DataMeta.MaxDST
              if hasattr(DataMeta,'MaxVXT'):
                  self.MaxVXT=DataMeta.MaxVXT
              if hasattr(DataMeta,'MaxDOCA'):
                  self.MaxDOCA=DataMeta.MaxDOCA
              if hasattr(DataMeta,'MaxAngle'):
                  self.MaxAngle=DataMeta.MaxAngle
              if hasattr(DataMeta,'MinHitsTrack'):
                  self.MinHitsTrack=DataMeta.MinHitsTrack
      def IniTrainingSession(self, TrainDataID, DateTime, TrainParameters):
          self.TrainSessionsDataID.append(TrainDataID)
          self.TrainSessionsDateTime.append(DateTime)
          self.TrainSessionsParameters.append(TrainParameters)
      def CompleteTrainingSession(self, TrainData):
          if len(self.TrainSessionsData)>=len(self.TrainSessionsDataID):
             self.TrainSessionsData=self.TrainSessionsData[:len(self.TrainSessionsDataID)-1]
          elif len(self.TrainSessionsData)<(len(self.TrainSessionsDataID)-1):
             self.TrainSessionsDataID=self.TrainSessionsDataID[:len(self.TrainSessionsData)+1]
          self.TrainSessionsData.append(TrainData)
      def ModelTrainStatus(self,TST):
            if len(self.TrainSessionsDataID)==len(self.TrainSessionsData):
                if len(self.TrainSessionsData)>=3:
                    test_input=[self.TrainSessionsData[-3][1],self.TrainSessionsData[-2][1],self.TrainSessionsData[-1][1]]
                    LossDataForChecking=[]
                    AccDataForChecking=[]
                    for i in test_input:
                               LossDataForChecking.append(i[6])
                               AccDataForChecking.append(i[7])
                    LossGrad=GetEquationOfLine(LossDataForChecking)[0]
                    AccGrad=GetEquationOfLine(AccDataForChecking)[0]
                    if LossGrad>=-TST and AccGrad<=TST:
                        return 1
                    else:
                        return 2
                else:
                    return 2
            else:
                return 0
def GenerateModel(ModelMeta,TrainParams=None):
      if ModelMeta.ModelFramework=='PyTorch':
         import torch
         import torch.nn as nn

         from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid,Softmax
         import torch.nn.functional as F

         from torch import Tensor
         import torch_geometric
         from torch_geometric.nn import MessagePassing
         from torch_geometric.nn import global_mean_pool
         if ModelMeta.ModelArchitecture=='TCN':
            from MTr_IN import InteractionNetwork as IN
            class MLP(nn.Module):
                  def __init__(self, input_size, output_size, hidden_size):
                     super(MLP, self).__init__()

                     if ModelMeta.ModelParameters[0]==3:
                         self.layers = nn.Sequential(
                             nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, output_size),
                         )
                     elif ModelMeta.ModelParameters[0]==4:
                         self.layers = nn.Sequential(
                             nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, output_size),
                         )

                     elif ModelMeta.ModelParameters[0]==5:
                         self.layers = nn.Sequential(
                             nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, output_size),
                         )

                     elif ModelMeta.ModelParameters[0]==6:
                         self.layers = nn.Sequential(
                             nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, output_size),
                         )

                     elif ModelMeta.ModelParameters[0]==7:
                         self.layers = nn.Sequential(
                             nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, output_size),
                         )

                     elif ModelMeta.ModelParameters[0]==8:
                         self.layers = nn.Sequential(
                             nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, output_size),
                         )

                     elif ModelMeta.ModelParameters[0]==2:
                         self.layers = nn.Sequential(
                             nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, output_size),
                         )
                     elif ModelMeta.ModelParameters[0]==1:
                         self.layers = nn.Sequential(
                             nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, output_size),
                         )
                  def forward(self, C):
                     return self.layers(C)

            class TCN(nn.Module):
                 def __init__(self, node_indim, edge_indim, hs):
                     super(TCN, self).__init__()
                     if ModelMeta.ModelParameters[2]==2:
                         self.in_w1 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w2 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)

                     elif ModelMeta.ModelParameters[2]==1:
                         self.in_w1 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)

                     elif ModelMeta.ModelParameters[2]==3:
                         self.in_w1 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w2 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w3 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)

                     elif ModelMeta.ModelParameters[2]==4:
                         self.in_w1 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w2 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w3 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w4 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                     elif ModelMeta.ModelParameters[2]==5:
                         self.in_w1 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w2 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w3 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w4 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w5 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                     elif ModelMeta.ModelParameters[2]==6:
                         self.in_w1 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w2 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w3 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w4 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w5 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w6 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                     elif ModelMeta.ModelParameters[2]==7:
                         self.in_w1 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w2 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w3 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w4 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w5 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w6 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w7 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                     elif ModelMeta.ModelParameters[2]==8:
                         self.in_w1 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w2 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w3 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w4 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w5 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w6 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w7 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)
                         self.in_w8 = IN(node_indim, edge_indim,
                                         node_outdim=node_indim, edge_outdim=edge_indim,
                                         hidden_size=hs)

                     self.W = MLP(edge_indim*(ModelMeta.ModelParameters[2]+1), 1, ModelMeta.ModelParameters[1])

                 def forward(self, x: Tensor, edge_index: Tensor,
                             edge_attr: Tensor) -> Tensor:

                     # re-embed the graph twice with add aggregation
                     if ModelMeta.ModelParameters[2]==2:
                         x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)

                         x2, edge_attr_2 = self.in_w2(x1, edge_index, edge_attr_1)

                         # combine all edge features, use to predict edge weights
                         initial_edge_attr = torch.cat([edge_attr, edge_attr_1,
                                                        edge_attr_2], dim=1)
                     if ModelMeta.ModelParameters[2]==3:
                         x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)

                         x2, edge_attr_2 = self.in_w2(x1, edge_index, edge_attr_1)

                         x3, edge_attr_3 = self.in_w3(x2, edge_index, edge_attr_2)

                         # combine all edge features, use to predict edge weights
                         initial_edge_attr = torch.cat([edge_attr, edge_attr_1,
                                                        edge_attr_2, edge_attr_3], dim=1)
                     if ModelMeta.ModelParameters[2]==4:
                         x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)

                         x2, edge_attr_2 = self.in_w2(x1, edge_index, edge_attr_1)

                         x3, edge_attr_3 = self.in_w3(x2, edge_index, edge_attr_2)

                         x4, edge_attr_4 = self.in_w4(x3, edge_index, edge_attr_3)

                         # combine all edge features, use to predict edge weights
                         initial_edge_attr = torch.cat([edge_attr, edge_attr_1,
                                                        edge_attr_2, edge_attr_3, edge_attr_4], dim=1)

                     if ModelMeta.ModelParameters[2]==5:
                         x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)

                         x2, edge_attr_2 = self.in_w2(x1, edge_index, edge_attr_1)

                         x3, edge_attr_3 = self.in_w3(x2, edge_index, edge_attr_2)

                         x4, edge_attr_4 = self.in_w4(x3, edge_index, edge_attr_3)

                         x5, edge_attr_5 = self.in_w5(x4, edge_index, edge_attr_4)

                         # combine all edge features, use to predict edge weights
                         initial_edge_attr = torch.cat([edge_attr, edge_attr_1,
                                                        edge_attr_2, edge_attr_3, edge_attr_4,edge_attr_5], dim=1)

                     if ModelMeta.ModelParameters[2]==6:
                         x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)

                         x2, edge_attr_2 = self.in_w2(x1, edge_index, edge_attr_1)

                         x3, edge_attr_3 = self.in_w3(x2, edge_index, edge_attr_2)

                         x4, edge_attr_4 = self.in_w4(x3, edge_index, edge_attr_3)

                         x5, edge_attr_5 = self.in_w5(x4, edge_index, edge_attr_4)

                         x6, edge_attr_6 = self.in_w6(x5, edge_index, edge_attr_5)

                         # combine all edge features, use to predict edge weights
                         initial_edge_attr = torch.cat([edge_attr, edge_attr_1,
                                                        edge_attr_2, edge_attr_3, edge_attr_4,edge_attr_5,edge_attr_6], dim=1)
                     if ModelMeta.ModelParameters[2]==7:
                         x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)

                         x2, edge_attr_2 = self.in_w2(x1, edge_index, edge_attr_1)

                         x3, edge_attr_3 = self.in_w3(x2, edge_index, edge_attr_2)

                         x4, edge_attr_4 = self.in_w4(x3, edge_index, edge_attr_3)

                         x5, edge_attr_5 = self.in_w5(x4, edge_index, edge_attr_4)

                         x6, edge_attr_6 = self.in_w6(x5, edge_index, edge_attr_5)

                         x7, edge_attr_7 = self.in_w7(x6, edge_index, edge_attr_6)

                         # combine all edge features, use to predict edge weights
                         initial_edge_attr = torch.cat([edge_attr, edge_attr_1,
                                                        edge_attr_2, edge_attr_3, edge_attr_4,edge_attr_5,edge_attr_6,edge_attr_7], dim=1)
                     if ModelMeta.ModelParameters[2]==8:
                         x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)

                         x2, edge_attr_2 = self.in_w2(x1, edge_index, edge_attr_1)

                         x3, edge_attr_3 = self.in_w3(x2, edge_index, edge_attr_2)

                         x4, edge_attr_4 = self.in_w4(x3, edge_index, edge_attr_3)

                         x5, edge_attr_5 = self.in_w5(x4, edge_index, edge_attr_4)

                         x6, edge_attr_6 = self.in_w6(x5, edge_index, edge_attr_5)

                         x7, edge_attr_7 = self.in_w7(x6, edge_index, edge_attr_6)

                         x8, edge_attr_8 = self.in_w8(x7, edge_index, edge_attr_7)

                         # combine all edge features, use to predict edge weights
                         initial_edge_attr = torch.cat([edge_attr, edge_attr_1,
                                                        edge_attr_2, edge_attr_3, edge_attr_4,edge_attr_5,edge_attr_6,edge_attr_7,edge_attr_8], dim=1)

                     if ModelMeta.ModelParameters[2]==1:
                         x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)
                         # combine all edge features, use to predict edge weights
                         initial_edge_attr = torch.cat([edge_attr, edge_attr_1], dim=1)
                     edge_weights = torch.sigmoid(self.W(initial_edge_attr))
                     return edge_weights
            model = TCN(ModelMeta.num_node_features, ModelMeta.num_edge_features, ModelMeta.ModelParameters[3])
            return model
         elif ModelMeta.ModelArchitecture=='MLP':

         #Example [Input Sizes, Dropout, Output Size]

             class MLP(nn.Module):
                def __init__(self, input_size, hidden_sizes, dropout_rate=0.5):
                    super(MLP, self).__init__()
                    layers = []
                    prev_size = input_size  # Start with input size
                    for hidden_size in hidden_sizes:
                        layers.append(nn.Linear(prev_size, hidden_size))  # Add Linear layer
                        layers.append(nn.ReLU())  # Activation function
                        layers.append(nn.Dropout(dropout_rate))  # Dropout
                        prev_size = hidden_size  # Update previous layer size
                    layers.append(nn.Linear(prev_size, 1))  # Output layer (1 neuron for binary classification)
                    self.model = nn.Sequential(*layers)  # Convert list to nn.Sequential
                def forward(self, x):
                    return self.model(x)  # Return raw logits for BCEWithLogitsLoss

             model = MLP(ModelMeta.ModelParameters[0], ModelMeta.ModelParameters[1], ModelMeta.ModelParameters[2])
             return model

         elif ModelMeta.ModelArchitecture=='GCN-4N-IC':
            from torch_geometric.nn import GCNConv
            HiddenLayer=[]
            OutputLayer=[]
            for el in ModelMeta.ModelParameters:
              if ModelMeta.ModelParameters.index(el)<=4 and len(el)>0:
                 HiddenLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)==10:
                 OutputLayer=el
            class GCN(torch.nn.Module):
                def __init__(self):
                    super(GCN, self).__init__()
                    torch.manual_seed(12345)
                    if len(HiddenLayer)==3:
                        self.conv1 = GCNConv(4 , HiddenLayer[0][0])
                        self.conv2 = GCNConv(HiddenLayer[0][0],HiddenLayer[1][0])
                        self.conv3 = GCNConv(HiddenLayer[1][0],HiddenLayer[2][0])
                        self.lin = Linear(HiddenLayer[2][0],OutputLayer[1])
                    if len(HiddenLayer)==1:
                        self.conv1 = GCNConv(4 , HiddenLayer[0][0])
                        self.lin = Linear(HiddenLayer[0][0],OutputLayer[1])
                    if OutputLayer[1]>1:
                        self.softmax = Softmax(dim=-1)

                def forward(self, x, edge_index, edge_attr, batch):
                    # 1. Obtain node embeddings
                    if len(HiddenLayer)==1:
                        x = self.conv1(x, edge_index)
                        x = x.relu()
                    if len(HiddenLayer)==3:
                        x = self.conv1(x, edge_index)
                        x = x.relu()
                        x = self.conv2(x, edge_index)
                        x = x.relu()
                        x = self.conv3(x, edge_index)
                        x = x.relu()
                    # 2. Readout layer
                    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
                    # 3. Apply a final classifier
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = self.lin(x)
                    if OutputLayer[1]>1:
                        x = self.softmax(x)
                    return x
            return GCN()
         elif ModelMeta.ModelArchitecture=='GCN-6N-IC':
            from torch_geometric.nn import GCNConv
            HiddenLayer=[]
            OutputLayer=[]
            for el in ModelMeta.ModelParameters:
              if ModelMeta.ModelParameters.index(el)<=4 and len(el)>0:
                 HiddenLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)==10:
                 OutputLayer=el

            class GCN(torch.nn.Module):
                def __init__(self):
                    super(GCN, self).__init__()
                    torch.manual_seed(12345)
                    if len(HiddenLayer)==3:
                        self.conv1 = GCNConv(6 , HiddenLayer[0][0])
                        self.conv2 = GCNConv(HiddenLayer[0][0],HiddenLayer[1][0])
                        self.conv3 = GCNConv(HiddenLayer[1][0],HiddenLayer[2][0])
                        self.lin = Linear(HiddenLayer[2][0],OutputLayer[1])
                    if len(HiddenLayer)==3:
                        self.conv1 = GCNConv(6 , HiddenLayer[0][0])
                        self.lin = Linear(HiddenLayer[0][0],OutputLayer[1])
                    if OutputLayer[1]>1:
                        self.softmax = Softmax(dim=-1)

                def forward(self, x, edge_index, edge_attr, batch):
                    # 1. Obtain node embeddings
                    if len(HiddenLayer)==3:
                        x = self.conv1(x, edge_index)
                        x = x.relu()
                        x = self.conv2(x, edge_index)
                        x = x.relu()
                        x = self.conv3(x, edge_index)
                        x = x.relu()
                    if len(HiddenLayer)==1:
                        x = self.conv1(x, edge_index)
                        x = x.relu()
                    # 2. Readout layer
                    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
                    # 3. Apply a final classifier
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = self.lin(x)
                    if OutputLayer[1]>1:
                        x = self.softmax(x)
                    return x
            return GCN()
         elif ModelMeta.ModelArchitecture=='GCN-5N-FC':
            from torch_geometric.nn import GCNConv
            HiddenLayer=[]
            OutputLayer=[]
            for el in ModelMeta.ModelParameters:
              if ModelMeta.ModelParameters.index(el)<=4 and len(el)>0:
                 HiddenLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)==10:
                 OutputLayer=el

            class GCN(torch.nn.Module):
                def __init__(self):
                    super(GCN, self).__init__()
                    torch.manual_seed(12345)
                    if len(HiddenLayer)==3:
                        self.conv1 = GCNConv(5 , HiddenLayer[0][0])
                        self.conv2 = GCNConv(HiddenLayer[0][0],HiddenLayer[1][0])
                        self.conv3 = GCNConv(HiddenLayer[1][0],HiddenLayer[2][0])
                        self.lin = Linear(HiddenLayer[2][0],OutputLayer[1])
                    elif len(HiddenLayer)==4:
                        self.conv1 = GCNConv(5 , HiddenLayer[0][0])
                        self.conv2 = GCNConv(HiddenLayer[0][0],HiddenLayer[1][0])
                        self.conv3 = GCNConv(HiddenLayer[1][0],HiddenLayer[2][0])
                        self.conv4 = GCNConv(HiddenLayer[2][0],HiddenLayer[3][0])
                        self.lin = Linear(HiddenLayer[3][0],OutputLayer[1])
                    if OutputLayer[1]>1:
                        self.softmax = Softmax(dim=-1)
                def forward(self, x, edge_index, edge_attr, batch):
                    # 1. Obtain node embeddings
                    if len(HiddenLayer)==3:
                        x = self.conv1(x, edge_index)
                        x = x.relu()
                        x = self.conv2(x, edge_index)
                        x = x.relu()
                        x = self.conv3(x, edge_index)
                        x = x.relu()
                    elif len(HiddenLayer)==4:
                        x = self.conv1(x, edge_index)
                        x = x.relu()
                        x = self.conv2(x, edge_index)
                        x = x.relu()
                        x = self.conv3(x, edge_index)
                        x = x.relu()
                        x = self.conv4(x, edge_index)
                        x = x.relu()

                    # 2. Readout layer
                    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

                    # 3. Apply a final classifier
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = self.lin(x)
                    if OutputLayer[1]>1:
                        x = self.softmax(x)
                    return x
            return GCN()
         elif ModelMeta.ModelArchitecture=='TAG-4N-IC':
            from torch_geometric.nn import TAGConv
            HiddenLayer=[]
            OutputLayer=[]
            for el in ModelMeta.ModelParameters:
              if ModelMeta.ModelParameters.index(el)<=4 and len(el)>0:
                 HiddenLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)==10:
                 OutputLayer=el

            class TAG(torch.nn.Module):
                def __init__(self):
                    super(TAG, self).__init__()
                    torch.manual_seed(12345)
                    if len(HiddenLayer)==3:
                        self.conv1 = TAGConv(4 , HiddenLayer[0][0])
                        self.conv2 = TAGConv(HiddenLayer[0][0],HiddenLayer[1][0])
                        self.conv3 = TAGConv(HiddenLayer[1][0],HiddenLayer[2][0])
                        self.lin = Linear(HiddenLayer[2][0],OutputLayer[1])
                    elif len(HiddenLayer)==4:
                        self.conv1 = TAGConv(4 , HiddenLayer[0][0])
                        self.conv2 = TAGConv(HiddenLayer[0][0],HiddenLayer[1][0])
                        self.conv3 = TAGConv(HiddenLayer[1][0],HiddenLayer[2][0])
                        self.conv4 = TAGConv(HiddenLayer[2][0],HiddenLayer[3][0])
                        self.lin = Linear(HiddenLayer[3][0],OutputLayer[1])
                    if OutputLayer[1]>1:
                        self.softmax = Softmax(dim=-1)

                def forward(self, x, edge_index, edge_attr, batch):
                    # 1. Obtain node embeddings
                    if len(HiddenLayer)==3:
                        x = self.conv1(x, edge_index)
                        x = x.relu()
                        x = self.conv2(x, edge_index)
                        x = x.relu()
                        x = self.conv3(x, edge_index)
                        x = x.relu()
                    elif len(HiddenLayer)==4:

                        x = self.conv1(x, edge_index)
                        x = x.relu()
                        x = self.conv2(x, edge_index)
                        x = x.relu()
                        x = self.conv3(x, edge_index)
                        x = x.relu()
                        x = self.conv4(x, edge_index)
                        x = x.relu()
                    # 2. Readout layer
                    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

                    # 3. Apply a final classifier
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = self.lin(x)
                    if OutputLayer[1]>1:
                        x = self.softmax(x)
                    return x
            return TAG()
         elif ModelMeta.ModelArchitecture=='TAG-5N-FC':
            from torch_geometric.nn import TAGConv
            HiddenLayer=[]
            OutputLayer=[]
            for el in ModelMeta.ModelParameters:
              if ModelMeta.ModelParameters.index(el)<=4 and len(el)>0:
                 HiddenLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)==10:
                 OutputLayer=el

            class TAG(torch.nn.Module):
                def __init__(self):
                    super(TAG, self).__init__()
                    torch.manual_seed(12345)
                    if len(HiddenLayer)==3:
                        self.conv1 = TAGConv(5 , HiddenLayer[0][0])
                        self.conv2 = TAGConv(HiddenLayer[0][0],HiddenLayer[1][0])
                        self.conv3 = TAGConv(HiddenLayer[1][0],HiddenLayer[2][0])
                        self.lin = Linear(HiddenLayer[2][0],OutputLayer[1])
                    elif len(HiddenLayer)==4:
                        self.conv1 = TAGConv(5 , HiddenLayer[0][0])
                        self.conv2 = TAGConv(HiddenLayer[0][0],HiddenLayer[1][0])
                        self.conv3 = TAGConv(HiddenLayer[1][0],HiddenLayer[2][0])
                        self.conv4 = TAGConv(HiddenLayer[2][0],HiddenLayer[3][0])
                        self.lin = Linear(HiddenLayer[3][0],OutputLayer[1])
                    if OutputLayer[1]>1:
                        self.softmax = Softmax(dim=-1)

                def forward(self, x, edge_index, edge_attr, batch):
                    # 1. Obtain node embeddings
                    if len(HiddenLayer)==3:
                        x = self.conv1(x, edge_index)
                        x = x.relu()
                        x = self.conv2(x, edge_index)
                        x = x.relu()
                        x = self.conv3(x, edge_index)
                        x = x.relu()
                    elif len(HiddenLayer)==4:
                        x = self.conv1(x, edge_index)
                        x = x.relu()
                        x = self.conv2(x, edge_index)
                        x = x.relu()
                        x = self.conv3(x, edge_index)
                        x = x.relu()
                        x = self.conv4(x, edge_index)
                        x = x.relu()
                    # 2. Readout layer
                    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
                    # 3. Apply a final classifier
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = self.lin(x)
                    if OutputLayer[1]>1:
                        x = self.softmax(x)
                    return x
            return TAG()
         elif ModelMeta.ModelArchitecture=='GMM-5N-FC':
            from torch_geometric.nn import GMMConv
            HiddenLayer=[]
            OutputLayer=[]
            for el in ModelMeta.ModelParameters:
              if ModelMeta.ModelParameters.index(el)<=4 and len(el)>0:
                 HiddenLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)==10:
                 OutputLayer=el

            class GMM(torch.nn.Module):
                def __init__(self):
                    super(GMM, self).__init__()
                    torch.manual_seed(12345)
                    if len(HiddenLayer)==3:
                        self.conv1 = GMMConv(5 , HiddenLayer[0][0],dim=3,kernel_size=HiddenLayer[0][1])
                        self.conv2 = GMMConv(HiddenLayer[0][0],HiddenLayer[1][0],dim=3,kernel_size=HiddenLayer[1][1])
                        self.conv3 = GMMConv(HiddenLayer[1][0],HiddenLayer[2][0],dim=3,kernel_size=HiddenLayer[2][1])
                        self.lin = Linear(HiddenLayer[2][0],OutputLayer[1])
                    elif len(HiddenLayer)==4:
                        self.conv1 = GMMConv(5 , HiddenLayer[0][0],dim=3,kernel_size=HiddenLayer[0][1])
                        self.conv2 = GMMConv(HiddenLayer[0][0],HiddenLayer[1][0],dim=3,kernel_size=HiddenLayer[1][1])
                        self.conv3 = GMMConv(HiddenLayer[1][0],HiddenLayer[2][0],dim=3,kernel_size=HiddenLayer[2][1])
                        self.conv4 = GMMConv(HiddenLayer[2][0],HiddenLayer[3][0],dim=3,kernel_size=HiddenLayer[3][1])
                        self.lin = Linear(HiddenLayer[3][0],OutputLayer[1])
                    if OutputLayer[1]>1:
                        self.softmax = Softmax(dim=-1)

                def forward(self, x, edge_index, edge_attr, batch):
                    # 1. Obtain node embeddings
                    if len(HiddenLayer)==3:
                        x = self.conv1(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv2(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv3(x, edge_index,edge_attr)
                        x = x.relu()
                    elif len(HiddenLayer)==4:
                        x = self.conv1(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv2(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv3(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv4(x, edge_index,edge_attr)
                        x = x.relu()
                    # 2. Readout layer
                    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
                    # 3. Apply a final classifier
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = self.lin(x)
                    if OutputLayer[1]>1:
                        x = self.softmax(x)
                    return x
            return GMM()
         elif ModelMeta.ModelArchitecture=='GMM-4N-IC':
            from torch_geometric.nn import GMMConv
            HiddenLayer=[]
            OutputLayer=[]
            for el in ModelMeta.ModelParameters:
              if ModelMeta.ModelParameters.index(el)<=4 and len(el)>0:
                 HiddenLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)==10:
                 OutputLayer=el

            class GMM(torch.nn.Module):
                def __init__(self):
                    super(GMM, self).__init__()
                    torch.manual_seed(12345)
                    if len(HiddenLayer)==3:
                        self.conv1 = GMMConv(4 , HiddenLayer[0][0],dim=4,kernel_size=HiddenLayer[0][1])
                        self.conv2 = GMMConv(HiddenLayer[0][0],HiddenLayer[1][0],dim=4,kernel_size=HiddenLayer[1][1])
                        self.conv3 = GMMConv(HiddenLayer[1][0],HiddenLayer[2][0],dim=4,kernel_size=HiddenLayer[2][1])
                        self.lin = Linear(HiddenLayer[2][0],OutputLayer[1])
                    elif len(HiddenLayer)==4:
                        self.conv1 = GMMConv(4 , HiddenLayer[0][0],dim=4,kernel_size=HiddenLayer[0][1])
                        self.conv2 = GMMConv(HiddenLayer[0][0],HiddenLayer[1][0],dim=4,kernel_size=HiddenLayer[1][1])
                        self.conv3 = GMMConv(HiddenLayer[1][0],HiddenLayer[2][0],dim=4,kernel_size=HiddenLayer[2][1])
                        self.conv4 = GMMConv(HiddenLayer[2][0],HiddenLayer[3][0],dim=4,kernel_size=HiddenLayer[3][1])
                        self.lin = Linear(HiddenLayer[3][0],OutputLayer[1])
                    if OutputLayer[1]>1:
                        self.softmax = Softmax(dim=-1)

                def forward(self, x, edge_index, edge_attr, batch):
                    # 1. Obtain node embeddings
                    if len(HiddenLayer)==3:
                        x = self.conv1(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv2(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv3(x, edge_index,edge_attr)
                        x = x.relu()
                    elif len(HiddenLayer)==4:
                        x = self.conv1(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv2(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv3(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv4(x, edge_index,edge_attr)
                        x = x.relu()

                    # 2. Readout layer
                    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
                    # 3. Apply a final classifier
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = self.lin(x)
                    if OutputLayer[1]>1:
                        x = self.softmax(x)
                    return x
            return GMM()
         elif ModelMeta.ModelArchitecture=='GMM-6N-IC':
            from torch_geometric.nn import GMMConv
            HiddenLayer=[]
            OutputLayer=[]
            for el in ModelMeta.ModelParameters:
              if ModelMeta.ModelParameters.index(el)<=4 and len(el)>0:
                 HiddenLayer.append(el)
              elif ModelMeta.ModelParameters.index(el)==10:
                 OutputLayer=el

            class GMM(torch.nn.Module):
                def __init__(self):
                    super(GMM, self).__init__()
                    torch.manual_seed(12345)
                    if len(HiddenLayer)==3:
                        self.conv1 = GMMConv(6 , HiddenLayer[0][0],dim=4,kernel_size=HiddenLayer[0][1])
                        self.conv2 = GMMConv(HiddenLayer[0][0],HiddenLayer[1][0],dim=4,kernel_size=HiddenLayer[1][1])
                        self.conv3 = GMMConv(HiddenLayer[1][0],HiddenLayer[2][0],dim=4,kernel_size=HiddenLayer[2][1])
                        self.lin = Linear(HiddenLayer[2][0],OutputLayer[1])
                    if len(HiddenLayer)==2:
                        self.conv1 = GMMConv(6 , HiddenLayer[0][0],dim=4,kernel_size=HiddenLayer[0][1])
                        self.conv2 = GMMConv(HiddenLayer[0][0],HiddenLayer[1][0],dim=4,kernel_size=HiddenLayer[1][1])
                        self.lin = Linear(HiddenLayer[1][0],OutputLayer[1])
                    elif len(HiddenLayer)==1:
                        self.conv1 = GMMConv(6 , HiddenLayer[0][0],dim=4,kernel_size=HiddenLayer[0][1])
                        self.lin = Linear(HiddenLayer[0][0],OutputLayer[1])
                    elif len(HiddenLayer)==4:
                        self.conv1 = GMMConv(6 , HiddenLayer[0][0],dim=4,kernel_size=HiddenLayer[0][1])
                        self.conv2 = GMMConv(HiddenLayer[0][0],HiddenLayer[1][0],dim=4,kernel_size=HiddenLayer[1][1])
                        self.conv3 = GMMConv(HiddenLayer[1][0],HiddenLayer[2][0],dim=4,kernel_size=HiddenLayer[2][1])
                        self.conv4 = GMMConv(HiddenLayer[2][0],HiddenLayer[3][0],dim=4,kernel_size=HiddenLayer[3][1])
                        self.lin = Linear(HiddenLayer[3][0],OutputLayer[1])
                    if OutputLayer[1]>1:
                        self.softmax = Softmax(dim=-1)

                def forward(self, x, edge_index, edge_attr, batch):
                    # 1. Obtain node embeddings
                    if len(HiddenLayer)==3:
                        x = self.conv1(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv2(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv3(x, edge_index,edge_attr)
                        x = x.relu()
                    if len(HiddenLayer)==2:
                        x = self.conv1(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv2(x, edge_index,edge_attr)
                        x = x.relu()
                    elif len(HiddenLayer)==1:
                        x = self.conv1(x, edge_index,edge_attr)
                        x = x.relu()
                    elif len(HiddenLayer)==4:
                        x = self.conv1(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv2(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv3(x, edge_index,edge_attr)
                        x = x.relu()
                        x = self.conv4(x, edge_index,edge_attr)
                        x = x.relu()
                    # 2. Readout layer
                    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
                    # 3. Apply a final classifier
                    x = F.dropout(x, p=0.5, training=self.training)
                    x = self.lin(x)
                    if OutputLayer[1]>1:
                        x = self.softmax(x)
                    return x
            return GMM()
      elif ModelMeta.ModelFramework=='Tensorflow':
          if ModelMeta.ModelType=='CNN':
            act_fun_list=['N/A','linear','exponential','elu','relu', 'selu','sigmoid','softmax','softplus','softsign','tanh']
            import tensorflow as tf
            from tensorflow import keras
            from keras.models import Sequential
            from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
            from keras.optimizers import Adam
            from keras import callbacks
            from keras import backend as K
            HiddenLayer=[]
            FullyConnectedLayer=[]
            OutputLayer=[]
            ImageLayer=[]
            for el in ModelMeta.ModelParameters:
              if len(el)==7:
                 HiddenLayer.append(el)
              elif len(el)==3:
                 FullyConnectedLayer.append(el)
              elif len(el)==2:
                 OutputLayer=el
              elif len(el)==4:
                 ImageLayer=el
            H=int(round(ImageLayer[0]/ImageLayer[3],0))*2
            W=int(round(ImageLayer[1]/ImageLayer[3],0))*2
            L=int(round(ImageLayer[2]/ImageLayer[3],0))
            model = Sequential()
            for HL in HiddenLayer:
                     Nodes=HL[0]*16
                     KS=HL[1]
                     PS=HL[3]
                     DR=float(HL[6]-1)/10.0
                     if HiddenLayer.index(HL)==0:
                        model.add(Conv3D(Nodes, activation=act_fun_list[HL[2]],kernel_size=(KS[0],KS[1],KS[2]),kernel_initializer='he_uniform', input_shape=(H,W,L,1)))
                     else:
                        model.add(Conv3D(Nodes, activation=act_fun_list[HL[2]],kernel_size=(KS[0],KS[1],KS[2]),kernel_initializer='he_uniform'))
                     if PS[0]>1 and PS[1]>1 and PS[2]>1:
                        model.add(MaxPooling3D(pool_size=(PS[0], PS[1], PS[2])))
                     model.add(BatchNormalization(center=HL[4]>1, scale=HL[5]>1))
                     model.add(Dropout(DR))
            model.add(Flatten())
            for FC in FullyConnectedLayer:
                         Nodes=4**FC[0]
                         DR=float(FC[2]-1)/10.0
                         model.add(Dense(Nodes, activation=act_fun_list[FC[1]], kernel_initializer='he_uniform'))
                         model.add(Dropout(DR))
            if OutputLayer[1]==1:
                model.add(Dense(OutputLayer[1]))
                opt = Adam(learning_rate=TrainParams[0])
                model.compile(loss='mean_squared_error',optimizer=opt,metrics=['mse'])
            else:
                model.add(Dense(OutputLayer[1], activation=act_fun_list[OutputLayer[0]]))
                opt = Adam(learning_rate=TrainParams[0])
                model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
     # Compile the model
            return model
def LoadRenderImages(Seeds,StartSeed,EndSeed,num_classes=2):
    import tensorflow as tf
    NewSeeds=Seeds[StartSeed-1:min(EndSeed,len(Seeds))]
    ImagesY=np.empty([len(NewSeeds),1])

    ImagesX=np.empty([len(NewSeeds),NewSeeds[0].H,NewSeeds[0].W,NewSeeds[0].L],dtype=np.uint8)
    for im in range(len(NewSeeds)):
        if num_classes>1:
            if hasattr(NewSeeds[im],'Label'):
               ImagesY[im]=int(float(NewSeeds[im].Label))
            else:
               ImagesY[im]=0
        else:
            if hasattr(NewSeeds[im],'Label'):
               ImagesY[im]=float(NewSeeds[im].Label)
            else:
               ImagesY[im]=0.0
        BlankRenderedImage=[]
        for x in range(-NewSeeds[im].bX,NewSeeds[im].bX):
          for y in range(-NewSeeds[im].bY,NewSeeds[im].bY):
            for z in range(0,NewSeeds[im].bZ):
             BlankRenderedImage.append(0)
        RenderedImage = np.array(BlankRenderedImage)
        RenderedImage = np.reshape(RenderedImage,(NewSeeds[im].H,NewSeeds[im].W,NewSeeds[im].L))
        for Hits in NewSeeds[im].TrackPrint:
                   RenderedImage[Hits[0]+NewSeeds[im].bX][Hits[1]+NewSeeds[im].bY][Hits[2]]=1
        ImagesX[im]=RenderedImage
    ImagesX= ImagesX[..., np.newaxis]
    if num_classes>1:
        ImagesY=tf.keras.utils.to_categorical(ImagesY,num_classes)
    return (ImagesX,ImagesY)
def zero_divide(a, b):
    if (b==0): return 0
    return a/b
def CNNtrain(model, Sample, Batches,num_classes, BatchSize):
    loss_accumulative = 0
    acc_accumulative = 0
    for ib in range(Batches):
# Calling psutil.cpu_precent() for 4 seconds
        Subsample=[]
        SampleSize=len(Sample)
        for s in range(min(BatchSize,SampleSize)):
            Subsample.append(Sample.pop(0))
        BatchImages=LoadRenderImages(Subsample,1,BatchSize,num_classes)
        t=model.train_on_batch(BatchImages[0],BatchImages[1])
        loss_accumulative+=t[0].item()
        acc_accumulative+=t[1].item()
    loss=loss_accumulative/Batches
    acc=acc_accumulative/Batches
    return loss,acc
def GNNtrain(model, Sample, optimizer,criterion):
    model.train()
    for data in Sample:
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()
    return loss
def GNNvalidate(model, Sample,criterion):
    model.eval()
    correct = 0
    loss_accumulative = 0
    batch_size=0
    for data in Sample:
         out = model(data.x, data.edge_index, data.edge_attr, data.batch)
         if len(data.y)>batch_size:
             batch_size=len(data.y)      
         pred = out.argmax(dim=1)  # Use the class with the highest probability.
         y_index = data.y.argmax(dim=1)
         correct += int((pred == y_index).sum())  # Check against ground-truth labels.
         loss = criterion(out, data.y)
         loss_accumulative += float(loss)
    return (correct / len(Sample.dataset), (loss_accumulative*batch_size)/len(Sample.dataset))
def CNNvalidate(model, Sample, Batches,num_classes, BatchSize):
    loss_accumulative = 0
    acc_accumulative = 0
    for ib in range(Batches):
        Subsample=[]
        SampleSize=len(Sample)  
        for s in range(min(BatchSize,SampleSize)):
            Subsample.append(Sample.pop(0))
        BatchImages=LoadRenderImages(Subsample,1,BatchSize,num_classes)
        v=model.test_on_batch(BatchImages[0],BatchImages[1])
        loss_accumulative+=v[0].item()
        acc_accumulative+=v[1].item()
    loss=loss_accumulative/Batches
    acc=acc_accumulative/Batches

    return loss,acc


