import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

from read_celeba import *
from padding_same_conv import Conv2d
from blurPooling import BlurPool2d
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

def NormalNLLLoss(x, mu, logvar):
    #NormalNLL =  0.5 * (np.log(2*np.pi) + logvar) + (x - mu)**2 / (2*torch.exp(logvar/2))
    NormalNLL =  0.5 * (np.log(2*np.pi) + logvar) + (x - mu)**2 / (2*torch.exp(logvar))
    return NormalNLL

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sn=True, bn=True):
        super(ResBlock, self).__init__()
        self.bn = bn
        self.sn = sn
        if sn:
            self.conv0 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0))
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
            self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        else:
            self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        if bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        h_conv0 = self.conv0(x)
        if self.bn:
            h_conv1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
            h_conv2 = self.bn2(self.conv2(h_conv1))
        else:
            h_conv1 = F.relu(self.conv1(x), inplace=True)
            h_conv2 = self.conv2(h_conv1)
        out = F.relu(h_conv0 + h_conv2, inplace=True)
        return out

class Encoder(nn.Module):
    def __init__(self, z_dim, ndf=128):
        super(Encoder, self).__init__()
        self.ndf = ndf
        # (128,128,3) -> (64,64,64)
        self.res1 = ResBlock(3, ndf, sn=False)
        self.pool1 = BlurPool2d(filt_size=3, channels=ndf, stride=2)
        # (64,64,64) -> (32,32,128)
        self.res2 = ResBlock(ndf, ndf*2, sn=False)
        self.pool2 = BlurPool2d(filt_size=3, channels=ndf*2, stride=2)
        # (32,32,128) -> (16,16,256)
        self.res3 = ResBlock(ndf*2, ndf*4, sn=False)
        self.pool3 = BlurPool2d(filt_size=3, channels=ndf*4, stride=2)
        # (16,16,256) -> (8,8,512)
        self.res4 = ResBlock(ndf*4, ndf*4, sn=False)
        self.pool4 = BlurPool2d(filt_size=3, channels=ndf*4, stride=2)
        # (8,8,512) -> (4,4,1024)
        self.res5 = ResBlock(ndf*4, ndf*8, sn=False)
        self.pool5 = BlurPool2d(filt_size=3, channels=ndf*8, stride=2)
        # (4*4*1024 -> z_dim)
        self.fc6_mu = nn.Linear(4*4*ndf*8, z_dim)
        self.fc6_logvar = nn.Linear(4*4*ndf*8, z_dim)
    
    def forward(self, x):
        # Res Block
        h_res1 = self.res1(x)
        h_pool1 = self.pool1(h_res1)
        h_res2 = self.res2(h_pool1)
        h_pool2 = self.pool2(h_res2)
        h_res3 = self.res3(h_pool2)
        h_pool3 = self.pool3(h_res3)
        h_res4 = self.res4(h_pool3)
        h_pool4 = self.pool4(h_res4)
        h_res5 = self.res5(h_pool4)
        h_pool5 = self.pool5(h_res5)
        # Fully Connected
        z_mu = self.fc6_mu(h_pool5.view(-1,self.ndf*8*4*4))
        z_logvar = self.fc6_logvar(h_pool5.view(-1,self.ndf*8*4*4))
        z_samp = self.sample_z(z_mu, z_logvar)
        return z_samp, z_mu, z_logvar
    
    def sample_z(self, mu, logvar):
        eps = torch.randn(mu.size()).to(device)
        return mu + torch.exp(logvar / 2) * eps

class Generator(nn.Module):
    def __init__(self, z_dim, ndf=64):
        super(Generator, self).__init__()
        self.ndf = ndf
        # (8,8,512) -> (8,8,512)
        self.fc1 = nn.Linear(z_dim, 8*8*ndf*8)
        self.res1 = ResBlock(ndf*8,ndf*8, sn=False)
        # (8,8,512) -> (16,16,256)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.res2 = ResBlock(ndf*8,ndf*4, sn=False)
        # (16,16,256) -> (32,32,256)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.res3 = ResBlock(ndf*4,ndf*4, sn=False)
        # (32,32,256) -> (64,64,128)
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.res4 = ResBlock(ndf*4,ndf*2, sn=False)
        # (64,64,128) -> (128,128,64)
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.res5 = ResBlock(ndf*2,ndf*1, sn=False)
        # (128,128,64) -> (128,128,3)
        self.conv6 = nn.Conv2d(in_channels=ndf, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h_fc1 = F.relu(self.fc1(z).view(-1,self.ndf*8,8,8)) 
        h_res1 = self.res1(h_fc1)
        h_up2 = self.up2(h_res1)
        h_res2 = self.res2(h_up2)
        h_up3 = self.up3(h_res2)
        h_res3 = self.res3(h_up3)
        h_up4 = self.up4(h_res3)
        h_res4 = self.res4(h_up4)
        h_up5 = self.up5(h_res4)
        h_res5 = self.res5(h_up5)
        x_samp = torch.sigmoid(self.conv6(h_res5))
        return x_samp

class DiscriminatorZ(nn.Module):
    def __init__(self, z_dim, ndf=512):
        super(DiscriminatorZ, self).__init__()
        self.fc1 = nn.Linear(z_dim, ndf)
        self.fc2 = nn.utils.spectral_norm(nn.Linear(ndf, ndf))
        self.fc3 = nn.utils.spectral_norm(nn.Linear(ndf, ndf))
        self.fc4 = nn.Linear(ndf, 1)

    def forward(self, z):
        h1 = F.leaky_relu(self.fc1(z))
        h2 = F.leaky_relu(self.fc2(h1))
        h3 = F.leaky_relu(self.fc3(h2))
        d_logit = self.fc4(h3)
        d_prob = torch.sigmoid(d_logit)
        return d_prob, d_logit

class DiscriminatorX(nn.Module):
    def __init__(self, ndf=128):
        super(DiscriminatorX, self).__init__()
        self.ndf = ndf
        # (128,128,3) -> (64,64,64)
        self.res1 = ResBlock(3, ndf, bn=False)
        self.pool1 = BlurPool2d(filt_size=3, channels=ndf, stride=2)
        # (64,64,64) -> (32,32,128)
        self.res2 = ResBlock(ndf, ndf*2, bn=False)
        self.pool2 = BlurPool2d(filt_size=3, channels=ndf*2, stride=2)
        # (32,32,128) -> (16,16,256)
        self.res3 = ResBlock(ndf*2, ndf*4, bn=False)
        self.pool3 = BlurPool2d(filt_size=3, channels=ndf*4, stride=2)
        # (16,16,256) -> (8,8,512)
        self.res4 = ResBlock(ndf*4, ndf*4, bn=False)
        self.pool4 = BlurPool2d(filt_size=3, channels=ndf*4, stride=2)
        # (8,8,512) -> (4,4,1024)
        self.res5 = ResBlock(ndf*4, ndf*8, bn=False)
        self.pool5 = BlurPool2d(filt_size=3, channels=ndf*8, stride=2)
        # (4*4*1024 -> 1)
        self.fc6 = nn.Linear(4*4*ndf*8, 1)

    def forward(self, x):
        # Res Block x6
        h_res1 = self.res1(x)
        h_pool1 = self.pool1(h_res1)
        h_res2 = self.res2(h_pool1)
        h_pool2 = self.pool2(h_res2)
        h_res3 = self.res3(h_pool2)
        h_pool3 = self.pool3(h_res3)
        h_res4 = self.res4(h_pool3)
        h_pool4 = self.pool4(h_res4)
        h_res5 = self.res5(h_pool4)
        h_pool5 = self.pool5(h_res5)
        # Fully Connected
        d_logit = self.fc6(h_pool5.view(-1,self.ndf*8*4*4))
        d_prob = torch.sigmoid(d_logit)
        return d_prob, d_logit

z_dim = 128
num_epochs = 20

netE = Encoder(z_dim).to(device)
netE.apply(weights_init)
netG = Generator(z_dim).to(device)
netG.apply(weights_init)
netDz = DiscriminatorZ(z_dim).to(device)
netDz.apply(weights_init)
netDx = DiscriminatorX().to(device)
netDx.apply(weights_init)

def zero_grad_list(net_list):
    for net in net_list:
        net.zero_grad()

paramsG = list(netE.parameters()) + list(netG.parameters())
optG = optim.Adam(paramsG, lr=2e-4, betas=(0.5, 0.999))
paramsD = list(netDz.parameters()) + list(netDx.parameters())
optD = optim.Adam(paramsD, lr=1e-4, betas=(0.5, 0.999))

for i, data in enumerate(dataloader, 0):
    x_fixed = data[0].to(device)
    break
z_fixed = torch.randn(32, z_dim, device=device)

model_name = "cycleGAN_stochastic_sn_res128_3"
out_folder = "out/" + model_name + "/"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
save_folder =  "save/" + model_name + "/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

print("Starting Training ...")
lamda_gz = 1.0
lamda_rx = 1.0
lamda_gx = 1.0
lamda_rz = 1.0

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Prepare Batch
        x_real = data[0].to(device)
        b_size = x_real.size(0)
        z_real = torch.randn(b_size, z_dim, device=device)
        ones = torch.full((b_size, 1), 1.0, device=device)
        zeros = torch.full((b_size, 1), 0.0, device=device)

        # =============== Train Discriminator ===============
        for _ in range(2):
            zero_grad_list([netE, netG, netDz, netDx])
            # D(Z)
            dz_real, _ = netDz(z_real)
            dz_real_loss = nn.BCELoss()(dz_real, ones)
            z_samp, z_mu, z_logvar = netE(x_real)
            dz_fake, _ = netDz(z_samp)
            dz_fake_loss = nn.BCELoss()(dz_fake, zeros)
            dz_loss = dz_real_loss + dz_fake_loss
            dz_loss.backward()
            # D(X)
            dx_real, _ = netDx(x_real)
            dx_real_loss = nn.BCELoss()(dx_real, ones)
            x_gen = netG(z_real)
            dx_fake, _ = netDx(x_gen)
            dx_fake_loss = nn.BCELoss()(dx_fake, zeros)
            dx_loss = dx_real_loss + dx_fake_loss
            dx_loss.backward()
            # Step
            optD.step()

        # =============== Train Generator ===============
        zero_grad_list([netE, netG, netDz, netDx])

        # (X -> Z -> X)
        # Reconstruction Loss
        z_samp, z_mu, z_logvar = netE(x_real)
        x_rec = netG(z_samp)
        rx_loss = nn.MSELoss()(x_rec, x_real)
        rx_loss = torch.clamp(rx_loss, min=0.01) # Clip the Reconstruction Loss
        # Distribution Loss
        dz_fake, _ = netDz(z_samp)
        gz_loss = nn.BCELoss()(dz_fake, ones)
        # Backward
        xzx_loss = lamda_rx*rx_loss + lamda_gz*gz_loss
        xzx_loss.backward()

        # (Z -> X -> Z)
        # Reconstruction Loss
        x_gen = netG(z_real)
        z_samp, z_mu, z_logvar = netE(x_gen)
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
            z_samp, z_mu, z_logvar = netE(x_fixed)
            x_rec = netG(z_mu).detach()
            x_samp = netG(z_fixed).detach()
            x_fig = torch.cat([x_fixed[0:8], x_rec[0:8], x_fixed[8:16], x_rec[8:16], x_samp], 0)
            x_fig = x_fig.cpu()
            path = out_folder+str(epoch).zfill(2)+"_"+str(i).zfill(4)+".jpg"
            vutils.save_image(x_fig, path, padding=2, normalize=True)
            # Save Model
            torch.save(netE.state_dict(), save_folder+"netE.pt")
            torch.save(netG.state_dict(), save_folder+"netG.pt")
            torch.save(netDx.state_dict(), save_folder+"netDx.pt")
            torch.save(netDz.state_dict(), save_folder+"netDz.pt")