from telnetlib import XASCII
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from jax import random
import matplotlib
import jax

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from utils.density import continuous_energy_from_image, prepare_image, sample_from_image_density

def loss_function(log_det_jacobian, z, target_distribution):
    log_likelihood = target_distribution.log_prob(z).sum(1) + log_det_jacobian.sum(1)
    return -log_likelihood.mean()

class MLP(nn.Module):
  def __init__(self,in_size,hid_size,out_size,n_layers):
    super(MLP, self).__init__()
    self.layers = nn.ModuleList()
    self.layers.append(nn.Linear(in_size,hid_size))
    for i in range(n_layers-1):
      self.layers.append(nn.Linear(hid_size,hid_size))
    self.layers.append(nn.Linear(hid_size,out_size))
  def forward(self,x):
    x = self.layers[0](x)
    for i in range(len(self.layers)-2):
      x = torch.relu(self.layers[i+1](x))
    return self.layers[-1](x)

class CouplingTransformation2D(nn.Module):
    def __init__(self, left, hidden_size=64, num_hidden_layers=2):
        super(CouplingTransformation2D,self).__init__()
        self.mlp = MLP(2, hidden_size, 2, num_hidden_layers)
        self.mask = torch.FloatTensor([1,0]) if left else torch.FloatTensor([0,1])
        self.mask = self.mask.view(1,-1)
        self.scale = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.shift = nn.Parameter(torch.zeros(1),requires_grad=True)

    def forward(self, x, reverse=False):
        # x.size() is (B,2)
        x_masked = x * self.mask
        # log_scale and shift have size (B,1)
        log_scale, shift = self.mlp(x_masked).chunk(2, dim=1)
        log_scale = log_scale.tanh() * self.scale + self.shift
        # log_scale and shift have size (B,2)
        log_scale = log_scale * (1-self.mask)
        shift = shift * (1-self.mask)
        if reverse:
            x = (x - shift) * torch.exp(-log_scale)
        else:
            x =  x * torch.exp(log_scale) + shift
        return x, log_scale

class RealNVP(nn.Module):
    def __init__(self, n_paired_transforms, n_mlp_layers):
        super(RealNVP,self).__init__()
        self.transforms = nn.ModuleList()
        for i in range(n_paired_transforms):
            self.transforms.append(CouplingTransformation2D(left=False, num_hidden_layers=n_mlp_layers))
            self.transforms.append(CouplingTransformation2D(left=True, num_hidden_layers=n_mlp_layers))

    def forward(self, x):
        z, log_det_jacobian = x, torch.zeros_like(x)
        for transform in self.transforms:
            z, log_scale = transform(z)
            log_det_jacobian += log_scale
        return z, log_det_jacobian

    def backward(self, z):
        for transform in self.transforms[::-1]:
            z, _ = transform(z,reverse=True)
        return z

def train_epoch(model,optimizer,loader,target_distribution):
    model.train()
    train_loss = 0
    for b in loader:
        z, log_det_jacobian = model(b)
        loss = loss_function(log_det_jacobian,z,target_distribution)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    return train_loss/ len(loader) 

def eval_epoch(model,loader,target_distribution):
    model.eval()
    eval_loss = 0
    for b in loader:
        z, log_det_jacobian = model(b)
        loss = loss_function(log_det_jacobian,z,target_distribution)
        eval_loss+=loss.item()
    return eval_loss/ len(loader)

class my_dataset(Dataset):
    def __init__(self,samples):
        self.samples = samples
    def __len__(self):
        return self.samples.shape[0]
    def __getitem__(self,idx):
        return self.samples[idx]

def create_two_spirals_data(n,noise=0):
    """
    Modified from https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py
    """
    m = np.sqrt(np.random.rand(n // 2, 1)) * 900 * (2 * np.pi) / 360
    a = 0.7
    d1x = -a * np.cos(m) * m + np.random.rand(n // 2, 1) * a / 2
    d1y = a * np.sin(m) * m + np.random.rand(n // 2, 1) * a / 2
    x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
    x += np.random.randn(*x.shape) * noise
    data = x.astype(np.float32)
    return data

if __name__ == '__main__':
    
    data = 'moon' #{'moon','spiral', 'labrador', 'zebra'}
    n_paired_transforms = 2
    n_mlp_layers = 2
    n_samples = int(2e3)
    noise_std = 0
    epochs = int(5e3)
    lr = 1e-3

    if data == 'spiral':
        data+=f'{noise_std:.0e}'
        X = create_two_spirals_data(n_samples,noise=noise_std)
        X = torch.FloatTensor(StandardScaler().fit_transform(X))
    elif data == 'moon':
        X, _ = datasets.make_moons(n_samples=n_samples,noise=noise_std)
        X = torch.FloatTensor(StandardScaler().fit_transform(X))
    elif data == 'zebra':
        img = matplotlib.image.imread(f'./template/data/{data}.jpeg')
        density, energy = prepare_image(
            img, crop=[80, 400, 0, 500], white_cutoff=225, gauss_sigma=3, background=0.01
        )
        key = jax.random.PRNGKey(0)
        X = torch.FloatTensor(np.array(sample_from_image_density(n_samples, density, key)))
    
    rng = random.PRNGKey(0)

    train_size = int(0.8*X.shape[0])
    train_samples = X[:train_size]
    test_samples = X[train_size:]

    train_dataset = my_dataset(train_samples)
    test_dataset = my_dataset(test_samples)

    train_loader = DataLoader(train_dataset,batch_size=128,shuffle=True)
    eval_loader = DataLoader(test_dataset,batch_size=128,shuffle=True)

    mu = torch.FloatTensor([0.])
    sigma = torch.FloatTensor([1.])
    target_distribution = Normal(mu, sigma)

    flow_model = RealNVP(n_paired_transforms, n_mlp_layers)
    optimizer = torch.optim.Adam(flow_model.parameters(),lr=lr)
    result_dir = f'template/models/{data}_ntransform_{n_paired_transforms}_samples_{n_samples}_epochs_{epochs}_lr_{lr:.0e}'
    
    train_loss_list = []; eval_loss_list = []; checkpoints = [int(epochs/4),int(epochs/2),int(3*epochs/4)]
    os.makedirs(result_dir,exist_ok=True); chkpnt = 1
    for epoch in tqdm(range(epochs)):
        train_loss = train_epoch(flow_model,optimizer,train_loader,target_distribution)
        eval_loss = eval_epoch(flow_model,eval_loader,target_distribution)
        train_loss_list.append(train_loss); eval_loss_list.append(eval_loss)
        if epoch in checkpoints:
            torch.save(flow_model.state_dict(),result_dir + f'/model_chkpnt_{chkpnt}')
            chkpnt+=1
        
    torch.save(flow_model.state_dict(),result_dir + f'/model_chkpnt_{chkpnt}')
    torch.save(train_loss_list,result_dir + '/train_loss'); torch.save(eval_loss_list,result_dir + '/eval_loss')
    
    
    