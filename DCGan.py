# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 19:42:26 2020

@author: cb425
"""

import torch
from torch import nn

import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
    
        return self.model(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        
        return self.model(x)
    
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

if __name__ == '__main__':
    torch.manual_seed(123)
    
    device = ""
    if torch.cuda.is_available():              # check if a GPU is available
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    transforming = transforms.Compose([           # Converts the data to a PyTorch tensor range from 0 to 1. 
        transforms.Resize(size=(64,64)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),                 # Since the image backgrounds are black, most of the coefficients are equal to 0 when they’re represented using this range.
        transforms.Normalize((0.5,), (0.5,)),   # changes the range of the coefficients to -1 to 1
        ])
    
    train_set = torchvision.datasets.MNIST(
    root=".", train=True, download=True, transform=transforming
    )
    
    # shuffle the data from train_set and return batches of 32 samples that would be used to train the neural networks.
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    
    discriminator = Discriminator().to(device=device)
    generator = Generator().to(device=device)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Learning rate for optimizers
    lr = 0.0002
    
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    
    num_epochs = 20
    loss_function = nn.BCELoss()
    
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr,  betas=(beta1, 0.999))
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr,  betas=(beta1, 0.999))
        
    for epoch in range(num_epochs):
        for n, (real_samples, mnist_labels) in enumerate(train_loader,0):
            # Data for training the discriminator
            real_samples = real_samples.to(device=device)
            b_size = real_samples.size(0)
            real_samples_labels = torch.ones((b_size, 1)).to(
                device=device
            )
            latent_space_samples = torch.randn((b_size, 100, 1, 1)).to(
                device=device
            )
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((b_size, 1)).to(
                device=device
            )
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat(
                (real_samples_labels, generated_samples_labels)
            )
    
    
            # Training the discriminator with real samples
            discriminator.zero_grad()
            output_discriminator = discriminator(real_samples).view(-1)
            loss_discriminator_real = loss_function(
                output_discriminator, real_samples_labels
            )
            loss_discriminator_real.backward()
            
            # Training the discriminator with fake samples
            output_discriminator = discriminator(generated_samples).view(-1)
            loss_discriminator_fake = loss_function(
                output_discriminator, generated_samples_labels
            )
            loss_discriminator_fake.backward()
            optimizer_discriminator.step()
    
            
            # Data for training the generator
            latent_space_samples = torch.randn((b_size, 100, 1, 1)).to(
                device=device
            )
    
            # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples).view(-1)
            loss_generator = loss_function(
                output_discriminator_generated, real_samples_labels
            )
            loss_generator.backward()
            optimizer_generator.step()

            # Show loss
            if n == batch_size - 1:
                print(f"Epoch: {epoch} Loss D.: {loss_discriminator_real+loss_discriminator_fake}")
                print(f"Epoch: {epoch} Loss G.: {loss_generator}")
                
    with torch.no_grad():
        z = torch.randn((64, 100, 1, 1)).cuda()
        sample = generator(z).cuda()
        
        save_image(sample.view(64, 1, 64, 64), 'DCGAN_generative_sample' + '.png')
    
    
    torch.save(generator.state_dict(), 'DCGAN_generator.pth')
    torch.save(discriminator.state_dict(), 'DCGAN_discriminator.pth')