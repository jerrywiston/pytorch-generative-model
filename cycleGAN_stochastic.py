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
    NormalNLL =  -0.5 * (2*np.log(np.pi) + logvar) - (x - mu)**2 / (2*torch.exp(logvar/2))
    return -1 * NormalNLL

class GeneratorZ(nn.Module):
    def __init__(self, z_dim):
        super(GeneratorZ, self).__init__()
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
        return self.z_samp, self.z_mu, self.z_logvar

class GeneratorX(nn.Module):
    def __init__(self, z_dim):
        super(GeneratorX, self).__init__()
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

class DiscriminatorZ(nn.Module):
    def __init__(self, z_dim):
        super(DiscriminatorZ, self).__init__()
        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, z):
        self.h1 = F.relu(self.fc1(z))
        self.h2 = F.relu(self.fc2(self.h1))
        self.h3 = F.relu(self.fc3(self.h2))
        self.d_logit = self.fc4(self.h3)
        self.d_prob = torch.sigmoid(self.d_logit)
        return self.d_prob, self.d_logit

class DiscriminatorX(nn.Module):
    def __init__(self):
        super(DiscriminatorX, self).__init__()
        self.conv1 = Conv2d(3, 128, 5, stride=2)
        
        self.conv2 = Conv2d(128, 256, 5, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv3 = Conv2d(256, 512, 5, stride=2)
        self.bn3 = nn.BatchNorm2d(512)
        
        self.conv4 = Conv2d(512, 1024, 3, stride=2)
        self.bn4 = nn.BatchNorm2d(1024)
        
        self.fc5 = nn.Linear(4*4*1024, 1)

    def forward(self, x):
        self.h_conv1 = F.relu(self.conv1(x))
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
        self.d_logit = self.fc5(self.h_conv4.view(-1,1024*4*4))
        self.d_prob = torch.sigmoid(self.d_logit)
        return self.d_prob, self.d_logit

z_dim = 64
num_epochs = 20

netGz = GeneratorZ(z_dim).to(device)
netGz.apply(weights_init)
netGx = GeneratorX(z_dim).to(device)
netGx.apply(weights_init)
netDz = DiscriminatorZ(z_dim).to(device)
netDz.apply(weights_init)
netDx = DiscriminatorX().to(device)
netDx.apply(weights_init)

def zero_grad_list(net_list):
    for net in net_list:
        net.zero_grad()

paramsG = list(netGz.parameters()) + list(netGx.parameters())
optG = optim.Adam(paramsG, lr=2e-4, betas=(0.5, 0.999))
paramsD = list(netDz.parameters()) + list(netDx.parameters())
optD = optim.Adam(paramsD, lr=1e-4, betas=(0.5, 0.999))

for i, data in enumerate(dataloader, 0):
    x_fixed = data[0].to(device)
    break
z_fixed = torch.randn(32, z_dim, device=device)

model_name = "cycleGAN_stochastic"
out_folder = "out/" + model_name + "/"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
save_folder =  "save/" + model_name + "/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

print("Starting Training ...")
lamda_gz = 0.01
lamda_rx = 1.0
lamda_gx = 1.0
lamda_rz = 0.01

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Prepare Batch
        x_real = data[0].to(device)
        b_size = x_real.size(0)
        z_real = torch.randn(b_size, z_dim, device=device)
        ones = torch.full((b_size, 1), 1.0, device=device)
        zeros = torch.full((b_size, 1), 0.0, device=device)

        # =============== Train Discriminator ===============
        zero_grad_list([netGz, netGx, netDz, netDx])

        # D(Z)
        dz_real, _ = netDz(z_real)
        dz_real_loss = nn.BCELoss()(dz_real, ones)
        z_samp, z_mu, z_logvar = netGz(x_real)
        dz_fake, _ = netDz(z_samp)
        dz_fake_loss = nn.BCELoss()(dz_fake, zeros)
        dz_loss = dz_real_loss + dz_fake_loss
        dz_loss.backward()
        
        # D(X)
        dx_real, _ = netDx(x_real)
        dx_real_loss = nn.BCELoss()(dx_real, ones)
        x_gen = netGx(z_real)
        dx_fake, _ = netDx(x_gen)
        dx_fake_loss = nn.BCELoss()(dx_fake, zeros)
        dx_loss = dx_real_loss + dx_fake_loss
        dx_loss.backward()

        # Step
        optD.step()

        # =============== Train Generator ===============
        zero_grad_list([netGz, netGx, netDz, netDx])

        # (X -> Z -> X)
        # Reconstruction Loss
        z_samp, z_mu, z_logvar = netGz(x_real)
        x_rec = netGx(z_samp)
        rx_loss = nn.MSELoss()(x_rec, x_real)
        # Distribution Loss
        dz_fake, _ = netDz(z_samp)
        gz_loss = nn.BCELoss()(dz_fake, ones)
        # Backward
        xzx_loss = lamda_rx*rx_loss + lamda_gz*gz_loss
        xzx_loss.backward()

        # (Z -> X -> Z)
        # Reconstruction Loss
        x_gen = netGx(z_real)
        z_samp, z_mu, z_logvar = netGz(x_gen)
        rz_loss = NormalNLLLoss(z_real, z_mu, z_logvar).mean()
        # Distribution Loss
        dx_fake, _ = netDx(x_gen)
        gx_loss = nn.BCELoss()(dx_fake, ones)
        # Backward
        zxz_loss = lamda_rz*rz_loss + lamda_gx*gx_loss
        zxz_loss.backward()

        # Step
        optG.step()

        # =============== Result ===============
        if i % 50 == 0:
            print("[%d/%d][%s/%d] Rz_loss: %.4f | Dx_loss: %.4f | Gx_loss: %.4f || Rx_loss: %.4f | Dz_loss: %.4f | Gz_loss: %.4f"\
            %(epoch+1, num_epochs, str(i).zfill(4), len(dataloader),\
            rz_loss.item(), dx_loss.mean().item(), gx_loss.mean().item(),\
            rx_loss.item(), dz_loss.mean().item(), gz_loss.mean().item()))

        if i%200 == 0:
            # Output Images
            _, z_code, _ = netGz(x_fixed)
            x_rec = netGx(z_code).detach()
            x_samp = netGx(z_fixed).detach()
            x_fig = torch.cat([x_fixed[0:8], x_rec[0:8], x_fixed[8:16], x_rec[8:16], x_samp], 0)
            x_fig = x_fig.cpu()
            plt.figure(figsize=(8,8))
            plt.imshow(np.transpose(vutils.make_grid(x_fig, padding=2, normalize=True).cpu(),(1,2,0)))
            plt.axis("off")
            plt.savefig(out_folder+str(epoch).zfill(2)+"_"+str(i).zfill(4)+".jpg", bbox_inches="tight")
            plt.close()
            # Save Model
            torch.save(netGx.state_dict(), save_folder+"netGx.pt")
            torch.save(netDx.state_dict(), save_folder+"netDx.pt")
            torch.save(netGz.state_dict(), save_folder+"netGz.pt")
            torch.save(netDz.state_dict(), save_folder+"netDz.pt")