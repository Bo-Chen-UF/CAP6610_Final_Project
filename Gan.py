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

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)   # reshape, avoids explicit data copy
        output = self.model(x)
        return output

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output
    


if __name__ == '__main__':
    torch.manual_seed(111)
    
    device = ""
    if torch.cuda.is_available():              # check if a GPU is available
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    transform = transforms.Compose([           # Converts the data to a PyTorch tensor range from 0 to 1. 
        transforms.ToTensor(),                 # Since the image backgrounds are black, most of the coefficients are equal to 0 when they’re represented using this range.
        transforms.Normalize((0.5,), (0.5,))   # changes the range of the coefficients to -1 to 1
        ])
    
    train_set = torchvision.datasets.MNIST(
    root=".", train=True, download=True, transform=transform
    )
    
    # shuffle the data from train_set and return batches of 32 samples that would be used to train the neural networks.
    batch_size = 100
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    
    discriminator = Discriminator().to(device=device)
    generator = Generator().to(device=device)
    
    lr = 0.0001
    num_epochs = 50
    k = 2
    loss_function = nn.BCELoss()
    
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
        
    for epoch in range(num_epochs):
        for n, (real_samples, mnist_labels) in enumerate(train_loader):
            # Data for training the discriminator
            real_samples = real_samples.to(device=device)
            real_samples_labels = torch.ones((batch_size, 1)).to(
                device=device
            )
            latent_space_samples = torch.randn((batch_size, 100)).to(
                device=device
            )
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((batch_size, 1)).to(
                device=device
            )
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat(
                (real_samples_labels, generated_samples_labels)
            )
    
            # Training the discriminator
            for ii in range(k):
                discriminator.zero_grad()
                output_discriminator = discriminator(all_samples)
                loss_discriminator = loss_function(
                    output_discriminator, all_samples_labels
                )
                loss_discriminator.backward(retain_graph=True)
                optimizer_discriminator.step()
    
            # Data for training the generator
            latent_space_samples = torch.randn((batch_size, 100)).to(
                device=device
            )
    
            # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(
                output_discriminator_generated, real_samples_labels
            )
            loss_generator.backward()
            optimizer_generator.step()

            # Show loss
            if n == batch_size - 1:
                print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                print(f"Epoch: {epoch} Loss G.: {loss_generator}")
                
    with torch.no_grad():
        z = torch.randn(64, 100).cuda()
        sample = generator(z).cuda()
        
        save_image(sample.view(64, 1, 28, 28), './samples/GAN_generative_sample_' + '.png')
        train_set.train_data = train_set.train_data.type(torch.DoubleTensor)
        save_image(1/255*train_set.train_data[:64].view(64, 1, 28, 28), './samples/GAN_original_sample_' + '.png')
    
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')