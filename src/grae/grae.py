import sys
from .numpy_dataset import FromNumpyDataset

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from phate import PHATE
from .base_model import BaseModel
from .torch_models import AETorchModule

import numpy as np

# Defaults
n_components = 2
batch_size = 128
lr = 0.001
weight_decay = 0
epochs = 100
hidden_dims = [80, 40, 10]
device = 'cpu'
lam = 1

class GRAE(BaseModel):
    """
    Baseline GRAE model with PHATE
    """

    def __init__(self,
                 n_components = n_components,
                 lr = lr,
                 batch_size = batch_size,
                 weight_decay = weight_decay,
                 random_state = None,
                 device = device,
                 optimizer = None,
                 torch_module = None,
                 epochs = epochs,
                 scheduler = None,
                 criterion = None,
                 hidden_dims = hidden_dims,
                 embedder_params = None,
                 lam = lam
                 ):

        self.n_components = n_components
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device
        self.torch_module = torch_module
        self.optimizer = optimizer
        self.epochs = epochs
        self.scheduler = scheduler
        self.criterion = criterion
        self.data_shape = None
        self.hidden_dims = hidden_dims
        self.lam = lam

        if embedder_params is None:
            self.embedder = PHATE(random_state = random_state, n_components = self.n_components)
        else: 
            self.embedder = PHATE(random_state = random_state, n_components = self.n_components, **embedder_params)

        # super().__init__()

    def init_torch_module(self, data_shape):

        input_size = data_shape

        self.torch_module = AETorchModule(input_dim   = input_size,
                                          hidden_dims = self.hidden_dims,
                                          z_dim       = self.n_components)

    def fit(self, x):

        self.data_shape = x.shape[1]

        z_target  = self.embedder.fit_transform(x)

        ################ add row normalized ################ 
        # # Calculate the sum of each row
        # row_sums = x.sum(axis=1, keepdims=True)
        # # Normalize each row by dividing by its sum
        # x = x / row_sums
        ################ 

        # x = TensorDataset(torch.tensor(x, dtype = torch.float))
        tensor_dataset = TensorDataset(torch.tensor(x, dtype = torch.float, device = self.device),
                                       torch.tensor(z_target, dtype = torch.float, device = self.device))

        if self.random_state is not None:

            torch.manual_seed(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


        if self.torch_module is None:
            self.init_torch_module(self.data_shape)

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.torch_module.parameters(),
                                              lr=self.lr,
                                              weight_decay=self.weight_decay)

        if self.criterion is None:
            self.criterion = nn.MSELoss()

        self.loader = self.get_loader(tensor_dataset)

        self.train_loop(self.torch_module, self.epochs, self.loader, self.optimizer, self.device) 

        
    def get_loader(self, x):
        return torch.utils.data.DataLoader(x, batch_size = self.batch_size, shuffle = True)

    def compute_loss(self, x, x_hat, z_target, z):

        loss_recon = self.criterion(x, x_hat)
        loss_emb = self.criterion(z_target, z)

        loss = loss_recon + self.lam * loss_emb

        self.recon_loss_temp = loss_recon.item()
        self.emb_loss_temp = loss_emb.item()

        loss.backward()


    def train_loop(self, model, epochs, train_loader, optimizer, device = 'cpu'):

        self.epoch_losses_recon = []
        self.epoch_losses_emb  = []
        
        for _, epoch in enumerate(range(epochs)):

            model = model.train()
            model = model.to(device)

            running_recon_loss = 0
            running_emb_loss = 0

            for _, (x, z_target) in enumerate(train_loader, 0):

                x = x.to(device)
                z_target = z_target.to(device)

                optimizer.zero_grad()

                recon, z = model(x)

                self.compute_loss(x, recon, z, z_target)

                running_recon_loss += self.recon_loss_temp
                running_emb_loss += self.emb_loss_temp

                optimizer.step()

            self.epoch_losses_recon.append(running_recon_loss / len(train_loader))
            self.epoch_losses_emb.append(running_emb_loss / len(train_loader))
            if epoch%10 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Recon Loss: {self.epoch_losses_recon[-1]:.7f}, Geo Loss: {self.epoch_losses_emb[-1]}") 



    def transform(self, x):
        self.torch_module.eval()

        x = TensorDataset(torch.tensor(x, dtype = torch.float, device = self.device))

        loader = torch.utils.data.DataLoader(x, batch_size=self.batch_size,
                                             shuffle=False)
 
        z = [self.torch_module.encoder(batch[0].to(self.device)).cpu().detach().numpy() for batch in loader]
        return np.concatenate(z)


    def inverse_transform(self, x):
        self.torch_module.eval()
        x = FromNumpyDataset(x)
        loader = torch.utils.data.DataLoader(x, batch_size=self.batch_size,
                                             shuffle=False)
        x_hat = [self.torch_module.decoder(batch.to(self.device)).cpu().detach().numpy()
                 for batch in loader]

        return np.concatenate(x_hat)