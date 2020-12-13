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
        self.conv1_1 = nn.Conv2d(nc, ndf//2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(10, ndf//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(ndf*2)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(ndf*4)
        self.conv4 = nn.Conv2d(ndf * 4, 1, 4, 1, 0)

    def forward(self, input_x, label):
        x = F.leaky_relu(self.conv1_1(input_x), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.sigmoid(self.conv4(x))
        return x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, ngf*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(ngf*2)
        self.deconv1_2 = nn.ConvTranspose2d(10, ngf*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(ngf*2)
        self.deconv2 = nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(ngf*2)
        self.deconv3 = nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(ngf)
        self.deconv4 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1)

    def forward(self, input_x, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input_x)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        return x
    
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
ngf = 128

# Size of feature maps in discriminator
ndf = 128

if __name__ == '__main__':
    torch.manual_seed(123)
    
    device = "cpu"
    
        
    transforming = transforms.Compose([           # Converts the data to a PyTorch tensor range from 0 to 1. 
        transforms.Resize(size=(32,32)),
        transforms.CenterCrop(32),
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
    
    num_epochs = 5
    N_Class = 10
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
            

            gen_labels = (torch.rand(b_size, 1) * N_Class).type(torch.LongTensor)
            gen_y = torch.zeros(b_size, N_Class)
            gen_y = Variable(gen_y.scatter_(1, gen_labels.view(b_size, 1), 1).view(b_size, N_Class,1,1).to(device=device))
            gen_y_for_D = gen_y.view(b_size, N_Class, 1, 1).contiguous().expand(-1, -1, 32, 32)
            
            real_y = torch.zeros(b_size, N_Class)
            real_y = real_y.scatter_(1, mnist_labels.view(b_size, 1), 1).view(b_size, N_Class, 1, 1).contiguous()
            real_y = Variable(real_y.expand(-1, -1, 32, 32).to(device=device))
        
            generated_samples = generator(latent_space_samples,gen_y)
            generated_samples_labels = torch.zeros((b_size, 1)).to(
                device=device
            )
            
            # Training the discriminator with real samples
            discriminator.zero_grad()
            output_discriminator = discriminator(real_samples,real_y).squeeze()
            loss_discriminator_real = loss_function(
                output_discriminator, real_samples_labels
            )
            loss_discriminator_real.backward()
            
            # Training the discriminator with fake samples
            output_discriminator = discriminator(generated_samples,gen_y_for_D).view(-1)
            loss_discriminator_fake = loss_function(
                output_discriminator, generated_samples_labels
            )
            loss_discriminator_fake.backward()
            optimizer_discriminator.step()
    
            
            # Data for training the generator
            latent_space_samples = torch.randn((b_size, 100, 1, 1)).to(
                device=device
            )
            gen_labels = (torch.rand(b_size, 1) * N_Class).type(torch.LongTensor)
            gen_y = torch.zeros(b_size, N_Class)
            gen_y = Variable(gen_y.scatter_(1, gen_labels.view(b_size, 1), 1).view(b_size, N_Class,1,1).to(device=device))
            gen_y_for_D = gen_y.view(b_size, N_Class, 1, 1).contiguous().expand(-1, -1, 32, 32)
    
            # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples,gen_y)
            output_discriminator_generated = discriminator(generated_samples,gen_y_for_D).view(-1)
            loss_generator = loss_function(
                output_discriminator_generated, real_samples_labels
            )
            loss_generator.backward()
            optimizer_generator.step()

            # Show loss
            if n == b_size - 1:
                print(f"Epoch: {epoch} Loss D.: {loss_discriminator_real+loss_discriminator_fake}")
                print(f"Epoch: {epoch} Loss G.: {loss_generator}")
                
    with torch.no_grad():
        z = torch.randn((64, 100, 1, 1))
        
        gen_labels = (torch.ones(8, 1) * 0).type(torch.LongTensor)
        
        for ii in range(7):
            gen_labels_temp = (torch.ones(8, 1) * (ii+1)).type(torch.LongTensor)
            gen_labels = torch.cat([gen_labels,gen_labels_temp],0)
            
        gen_y = torch.zeros(64, N_Class)
        gen_y = Variable(gen_y.scatter_(1, gen_labels.view(64, 1), 1).view(64, N_Class,1,1).to(device=device))
            
        sample = generator(z,gen_y)
        save_image(sample.view(64, 1, 32, 32), 'cDCGAN_generative_sample' + '.png')
    
    
    torch.save(generator.state_dict(), 'cDCGAN_generator.pth')
    torch.save(discriminator.state_dict(), 'cDCGAN_discriminator.pth')