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

def sample_z(mu, logvar):
    eps = torch.randn(mu.size()).to(device)
    return mu + torch.exp(logvar / 2) * eps

def NormalNLLLoss(x, mu, logvar):
    NormalNLL =  -1 * (-0.5 * (2*np.log(np.pi) + logvar) - (x - mu)**2 / (2*torch.exp(logvar/2)))
    return NormalNLL

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = Conv2d(3, 128, 5, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.conv2 = Conv2d(128, 256, 5, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv3 = Conv2d(256, 512, 5, stride=2)
        self.bn3 = nn.BatchNorm2d(512)
        
        self.conv4 = Conv2d(512, 1024, 3, stride=2)
        self.bn4 = nn.BatchNorm2d(1024)
        
        self.fc5_mu = nn.Linear(4*4*1024, z_dim)
        self.fc5_logvar = nn.Linear(4*4*1024, z_dim)
    
    def forward(self, x):
        self.h_conv1 = self.conv1(x)
        self.h_conv2 = F.relu(self.bn1(self.h_conv1))
        # (128, 32, 32)
        self.h_conv2 = self.conv2(self.h_conv1)
        self.h_conv2 = F.relu(self.bn2(self.h_conv2))
        # (256, 16, 16)
        self.h_conv3 = self.conv3(self.h_conv2)
        self.h_conv3 = F.relu(self.bn3(self.h_conv3))
        # (512, 8, 8)
        self.h_conv4 = self.conv4(self.h_conv3)
        self.h_conv4 = F.relu(self.bn4(self.h_conv4))
        # (1024, 4, 4)
        self.z_mu = self.fc5_mu(self.h_conv4.view(-1,1024*4*4))
        self.z_logvar = self.fc5_logvar(self.h_conv4.view(-1,1024*4*4))
        self.z_samp = sample_z(self.z_mu, self.z_logvar)
        # discriminator
        self.disc = (self.z_samp**2).mean(1, keepdim=True)
        self.disc_sq = torch.sigmoid(self.disc)
        return self.z_samp, self.z_mu, self.z_logvar, self.disc_sq

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 8*8*1024)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = Conv2d(1024, 512, 3)
        self.bn2 = nn.BatchNorm2d(512)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = Conv2d(512, 256, 5)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = Conv2d(256, 3, 5)

    def forward(self, z):
        self.h_conv1 = F.relu(self.fc1(z).view(-1,1024,8,8)) 
        # (1024, 8, 8)
        self.h_conv2 = self.up2(self.h_conv1)
        self.h_conv2 = self.conv2(self.h_conv2)
        self.h_conv2 = F.relu(self.bn2(self.h_conv2))
        # (512, 16, 16)
        self.h_conv3 = self.up3(self.h_conv2)
        self.h_conv3 = self.conv3(self.h_conv3)
        self.h_conv3 = F.relu(self.bn3(self.h_conv3))
        # (256, 32, 32)
        self.h_conv4 = self.up4(self.h_conv3)
        self.h_conv4 = self.conv4(self.h_conv4)
        self.x_samp = torch.sigmoid(self.h_conv4)
        return self.x_samp

z_dim = 64
num_epochs = 20
netE = Encoder(z_dim).to(device)
netE.apply(weights_init)
netG = Generator(z_dim).to(device)
netG.apply(weights_init)

def zero_grad_list(net_list):
    for net in net_list:
        net.zero_grad()

paramsR = list(netE.parameters()) + list(netG.parameters())
optR = optim.Adam(paramsR, lr=2e-5, betas=(0.5, 0.999))
optD = optim.Adam(netE.parameters(), lr=1e-4, betas=(0.5, 0.999))
optG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))

for i, data in enumerate(dataloader, 0):
    x_fixed = data[0].to(device)
    break
z_fixed = torch.randn(32, z_dim, device=device)

model_name = "test_model"
out_folder = "out/" + model_name + "/"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
save_folder =  "save/" + model_name + "/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

print("Starting Training ...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Prepare Batch
        x_real = data[0].to(device)
        b_size = x_real.size(0)
        z_real = torch.randn(b_size, z_dim, device=device)
        ones = torch.full((b_size, 1), 1.0, device=device)
        zeros = torch.full((b_size, 1), 0.0, device=device)

        # =============== Train Discriminator ===============
        zero_grad_list([netE, netG])
        z_samp, z_mu, z_logvar, d_real = netE(x_real)
        d_real_loss = nn.BCELoss()(d_real, zeros)
        x_gen = netG(z_real)
        z_samp, z_mu, z_logvar, d_fake = netE(x_gen)
        d_fake_loss = nn.BCELoss()(d_fake, ones)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optD.step()

        # =============== Train Generator ===============
        # Adversarial Loss
        zero_grad_list([netE, netG])
        x_gen = netG(z_real)
        z_samp, z_mu, z_logvar, d_fake = netE(x_gen)
        g_loss = nn.BCELoss()(d_fake, zeros)
        g_loss.backward()
        optG.step()

        # =============== Mutual Information ===============
        zero_grad_list([netE, netG])
        x_gen = netG(z_real)
        z_samp, z_mu, z_logvar, _ = netE(x_gen)
        r_loss = NormalNLLLoss(z_real, z_mu, z_logvar).mean()
        r_loss.backward()
        optR.step()

        # =============== Result ===============
        if i % 50 == 0:
            print("[%d/%d][%s/%d] R_loss: %.4f | D_loss: %.4f | G_loss: %.4f"\
            %(epoch+1, num_epochs, str(i).zfill(4), len(dataloader),\
            r_loss.item(), d_loss.mean().item(), g_loss.mean().item()))

        if i%200 == 0:
            # Output Images
            z_samp, z_mu, z_logvar, _ = netE(x_fixed)
            x_rec = netG(z_mu).detach()
            x_samp = netG(z_fixed).detach()
            x_fig = torch.cat([x_fixed[0:8], x_rec[0:8], x_fixed[8:16], x_rec[8:16], x_samp], 0)
            x_fig = x_fig.cpu()
            plt.figure(figsize=(8,8))
            plt.imshow(np.transpose(vutils.make_grid(x_fig, padding=2, normalize=True).cpu(),(1,2,0)))
            plt.axis("off")
            plt.savefig(out_folder+str(epoch).zfill(2)+"_"+str(i).zfill(4)+".jpg", bbox_inches="tight")
            plt.close()
            # Save Model
            torch.save(netE.state_dict(), save_folder+"netE.pt")
            torch.save(netG.state_dict(), save_folder+"netG.pt")