import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

from read_celeba import *
from padding_same_conv import Conv2d
import numpy as np
import matplotlib.pyplot as plt

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = Conv2d(3, 64, 5, stride=2)
        
        self.conv2 = Conv2d(64, 128, 5, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = Conv2d(128, 256, 5, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = Conv2d(256, 512, 3, stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.fc5_mu = nn.Linear(4*4*512, z_dim)
        self.fc5_logvar = nn.Linear(4*4*512, z_dim)
    
    def forward(self, x):
        self.h_conv1 = F.relu(self.conv1(x))
        # (64, 32, 32)
        self.h_conv2 = self.conv2(self.h_conv1)
        self.h_conv2 = F.relu(self.bn2(self.h_conv2))
        # (128, 16, 16)
        self.h_conv3 = self.conv3(self.h_conv2)
        self.h_conv3 = F.relu(self.bn3(self.h_conv3))
        # (256, 8, 8)
        self.h_conv4 = self.conv4(self.h_conv3)
        self.h_conv4 = F.relu(self.bn4(self.h_conv4))
        # (512, 4, 4)
        self.mu = self.fc5_mu(self.h_conv4.view(-1,512*4*4))
        self.logvar = self.fc5_logvar(self.h_conv4.view(-1,512*4*4))
        return self.mu, self.logvar

def sample_z(mu, logvar):
    eps = torch.randn(mu.size()).to(device)
    return mu + torch.exp(logvar / 2) * eps

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, 8*8*512)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = Conv2d(512, 256, 3)
        self.bn2 = nn.BatchNorm2d(256)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = Conv2d(256, 128, 5)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = Conv2d(128, 3, 5)

    def forward(self, z):
        self.h_conv1 = F.relu(self.fc1(z).view(-1,512,8,8)) 
        # (512, 8, 8)
        self.h_conv2 = self.up2(self.h_conv1)
        self.h_conv2 = self.conv2(self.h_conv2)
        self.h_conv2 = F.relu(self.bn2(self.h_conv2))
        # (256, 16, 16)
        self.h_conv3 = self.up3(self.h_conv2)
        self.h_conv3 = self.conv3(self.h_conv3)
        self.h_conv3 = F.relu(self.bn3(self.h_conv3))
        # (128, 32, 32)
        self.h_conv4 = self.up4(self.h_conv3)
        self.h_conv4 = self.conv4(self.h_conv4)
        self.x_samp = torch.sigmoid(self.h_conv4)
        return self.x_samp

z_dim = 100
num_epochs = 5

netEnc = Encoder(z_dim).to(device)
netEnc.apply(weights_init)
netDec = Decoder(z_dim).to(device)
netDec.apply(weights_init)

criterion = nn.BCELoss()
params = list(netEnc.parameters()) + list(netDec.parameters())
opt = optim.Adam(params, lr=2e-4, betas=(0.5, 0.999))


z_fixed = torch.randn(64, z_dim, device=device)

out_folder = "out/out_vae/"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
print("Starting Training ...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Initialize
        netEnc.zero_grad()
        netDec.zero_grad()

        x_real = data[0].to(device)
        z_mu, z_var = netEnc(x_real)
        z_code = sample_z(z_mu, z_var)
        x_rec = netDec(z_code)
        
        rec_loss = criterion(x_rec, x_real)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
        loss = rec_loss + 0.001*kl_loss
        loss.backward()
        opt.step()

        # Results
        if i % 50 == 0:
            print("[%d/%d][%d/%d]\tRec_loss: %.4f,\tKL_loss: %.4f"%(epoch+1, num_epochs, i, len(dataloader), rec_loss.item(), kl_loss.mean().item()))
        
        if i%200 == 0:
            x_fixed = netDec(z_fixed).cpu().detach()
            plt.figure(figsize=(8,8))
            plt.imshow(np.transpose(vutils.make_grid(x_fixed, padding=2, normalize=True).cpu(),(1,2,0)))
            plt.axis("off")
            plt.savefig(out_folder+str(epoch).zfill(2)+"_"+str(i).zfill(4)+".jpg", bbox_inches="tight")
            plt.close()