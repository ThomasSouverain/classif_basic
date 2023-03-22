import os

import torch
os.environ['TORCH'] = torch.__version__

from torch.nn import Parameter, Module, Sigmoid
import torch_scatter
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import ReLU

from torch_geometric.nn import GINConv
from torch_geometric.nn import MessagePassing

from classif_basic.graph.train import activate_gpu

torch.manual_seed(42)


class AbstractLAFLayer(Module):
    def __init__(self, **kwargs):
        super(AbstractLAFLayer, self).__init__()
        assert 'units' in kwargs or 'weights' in kwargs
        if 'device' in kwargs.keys():
            self.device = kwargs['device']
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ngpus = torch.cuda.device_count()
        
        if 'kernel_initializer' in kwargs.keys():
            assert kwargs['kernel_initializer'] in [
                'random_normal',
                'glorot_normal',
                'he_normal',
                'random_uniform',
                'glorot_uniform',
                'he_uniform']
            self.kernel_initializer = kwargs['kernel_initializer']
        else:
            self.kernel_initializer = 'random_normal'

        if 'weights' in kwargs.keys():
            self.weights = Parameter(kwargs['weights'].to(self.device), \
                                     requires_grad=True)
            self.units = self.weights.shape[1]
        else:
            self.units = kwargs['units']
            params = torch.empty(12, self.units, device=self.device)
            if self.kernel_initializer == 'random_normal':
                torch.nn.init.normal_(params)
            elif self.kernel_initializer == 'glorot_normal':
                torch.nn.init.xavier_normal_(params)
            elif self.kernel_initializer == 'he_normal':
                torch.nn.init.kaiming_normal_(params)
            elif self.kernel_initializer == 'random_uniform':
                torch.nn.init.uniform_(params)
            elif self.kernel_initializer == 'glorot_uniform':
                torch.nn.init.xavier_uniform_(params)
            elif self.kernel_initializer == 'he_uniform':
                torch.nn.init.kaiming_uniform_(params)
            self.weights = Parameter(params, \
                                     requires_grad=True)
        e = torch.tensor([1,-1,1,-1], dtype=torch.float32, device=self.device)
        self.e = Parameter(e, requires_grad=False)
        num_idx = torch.tensor([1,1,0,0], dtype=torch.float32, device=self.device).\
                                view(1,1,-1,1)
        self.num_idx = Parameter(num_idx, requires_grad=False)
        den_idx = torch.tensor([0,0,1,1], dtype=torch.float32, device=self.device).\
                                view(1,1,-1,1)
        self.den_idx = Parameter(den_idx, requires_grad=False)
        

class LAFLayer(AbstractLAFLayer):
    def __init__(self, eps=1e-7, **kwargs):
        super(LAFLayer, self).__init__(**kwargs)
        self.eps = eps
    
    def forward(self, data, index, dim=0, **kwargs):
        eps = self.eps
        sup = 1.0 - eps 
        e = self.e

        x = torch.clamp(data, eps, sup)
        x = torch.unsqueeze(x, -1)
        e = e.view(1,1,-1)        

        exps = (1. - e)/2. + x*e 
        exps = torch.unsqueeze(exps, -1)
        exps = torch.pow(exps, torch.relu(self.weights[0:4]))

        scatter = torch_scatter.scatter_add(exps, index.view(-1), dim=dim)
        scatter = torch.clamp(scatter, eps)

        sqrt = torch.pow(scatter, torch.relu(self.weights[4:8]))
        alpha_beta = self.weights[8:12].view(1,1,4,-1)
        terms = sqrt * alpha_beta

        num = torch.sum(terms * self.num_idx, dim=2)
        den = torch.sum(terms * self.den_idx, dim=2)
        
        multiplier = 2.0*torch.clamp(torch.sign(den), min=0.0) - 1.0

        den = torch.where((den < eps) & (den > -eps), multiplier*eps, den)

        res = num / den
        return res

class GINLAFConv(GINConv):
    def __init__(self, nn, units=1, node_dim=9, **kwargs): # TODO change dim of node embeddings? But how? 
        super(GINLAFConv, self).__init__(nn, **kwargs)
        self.laf = LAFLayer(units=units, kernel_initializer='random_uniform')
        self.mlp = torch.nn.Linear(node_dim*units, node_dim)
        self.dim = node_dim
        self.units = units
    
    def aggregate(self, inputs, index):
        x = torch.sigmoid(inputs)
        x = self.laf(x, index)
        x = x.view((-1, self.dim * self.units))
        x = self.mlp(x)
        return x
    
    # we add the forward, to control for the batch size (vs 945/946 error of model prediction) # TODO delete if not necessary 
    # def forward(self, x, edge_index):
    #     device = activate_gpu() # TODO delete temporary activation of GPU
    #     x = x.to(device)
    #     edge_index = edge_index.to(device)
    #     nb_indivs_x = x.shape[0] # TODO improve propagate? Or not necessary, as it is only to initialize message passing 
    #     x = (x, x) # propagate_type: (x: OptPairTensor)
    #     # here we control TODO requires self.batch to be initialized
    #     out = self.propagate(edge_index, x=x)#, size=(nb_indivs_x, nb_indivs_x))
        
    #     x_r = x[1]

    #     # alternative to "block" wrong dimensions (in the previous function)
    #     # null signal if not the good shape
    #     if x_r.shape != out.shape:
    #         #print("null signal")
    #         out = (1 + self.eps) * x_r
        
    #     elif x_r is not None:
    #         #print("no_null")
    #         out = out + (1 + self.eps) * x_r

    #     return self.nn(out)

class LAFNet(torch.nn.Module):
    def __init__(self, list_data):
        super().__init__()
        #super(LAFNet, self).__init__()

        data_end_childs = list_data[-1]

        num_features = data_end_childs.num_features #TODO pass data as input
        dim = 9
        units = 5
        
        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINLAFConv(nn1, units=units, node_dim=num_features)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINLAFConv(nn2, units=units, node_dim=dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINLAFConv(nn3, units=units, node_dim=dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINLAFConv(nn4, units=units, node_dim=dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINLAFConv(nn5, units=units, node_dim=dim)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, data_end_childs.num_classes)

    def forward(self, list_data, device, skip_connection=False):

        data=list_data[-1]
        # TODO apply LAF to weight ancestors, then childs... layers with edges ; for the moment 1 edge
            # for i, data in enumerate(list_data[:-1]):
        # print(f"\n Layer with causal ascendance {i+1} : ")
        x = data.x.float().to(device)
        edge_index=data.edge_index.to(device)
                
        x = F.relu(self.conv1(x, edge_index)) # we add batch in the BN (hmm, des Chocapic...)
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        #print(f"x just after bn1, ..., 5: {x.shape[0]}")
        # here, we test without global pooling -> useless batch indicator?
        #x = global_add_pool(x, batch, size=len(batch)) # control for the passed size of batches
        #print(f"x just after pool (with batch): {x.shape[0]}")
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)
