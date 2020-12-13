# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 20:53:49 2020

@author: cb425
"""

# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image


bs = 100
# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

class VAE(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim+y_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        
        # decoder part
        self.fc4 = nn.Linear(z_dim+y_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x, label):
        c = torch.nn.functional.one_hot(label,10)
        input_x = torch.cat((x, c), dim=-1)
        h = F.relu(self.fc1(input_x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z, label):
        c = torch.nn.functional.one_hot(label,10)
        input_z = torch.cat((z, c), dim=-1)
        h = F.relu(self.fc4(input_z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x, label):
        mu, log_var = self.encoder(x.view(-1, 784),label)
        z = self.sampling(mu, log_var)
        return self.decoder(z,label), mu, log_var

# build model
vae = VAE(x_dim=784, y_dim=10, h_dim1= 512, h_dim2=256, z_dim=100)
if torch.cuda.is_available():
    vae.cuda()
    
    
optimizer = optim.Adam(vae.parameters())
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data,label)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    
def test():
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, label in test_loader:
            recon, mu, log_var = vae(data,label)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
for epoch in range(1, 51):
    train(epoch)
    test()

with torch.no_grad():
    z = torch.randn(64, 100)
    
    labels = (torch.ones(8, 1) * 0).type(torch.LongTensor)
    for ii in range(7):
            gen_labels_temp = (torch.ones(8, 1) * (ii+1)).type(torch.LongTensor)
            labels = torch.cat([labels,gen_labels_temp],0).squeeze()
    sample = vae.decoder(z,labels)
    save_image(sample.view(64, 1, 28, 28), 'cVAE_generative_sample' + '.png')
    train_dataset.train_data = train_dataset.train_data.type(torch.DoubleTensor)
    save_image(1/255*train_dataset.train_data[:64].view(64, 1, 28, 28), 'VAE_original_sample' + '.png')

torch.save(vae, 'cvae.pth')
