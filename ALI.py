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
        self.z_code = self.fc5_mu(self.h_conv4.view(-1,1024*4*4))
        return self.z_code

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

class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        # X
        self.conv1 = Conv2d(3, 128, 5, stride=2)
        self.conv2 = Conv2d(128, 256, 5, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = Conv2d(256, 512, 5, stride=2)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = Conv2d(512, 1024, 3, stride=2)
        self.bn4 = nn.BatchNorm2d(1024)
        self.fc5 = nn.Linear(4*4*1024, 512)
        self.fc6 = nn.Linear(512, 1)

        # Z
        self.zfc1 = nn.Linear(z_dim, 512)
        self.zfc2 = nn.Linear(512, 512)
        self.zfc3 = nn.Linear(512, 1)

        # Cat
        self.fc1_cat = nn.Linear(1024, 512)
        self.fc2_cat = nn.Linear(512, 1)

    def forward(self, x, z):
        # X Path
        self.hx_conv1 = F.relu(self.conv1(x))
        self.hx_conv2 = self.conv2(self.hx_conv1)
        self.hx_conv2 = F.relu(self.bn2(self.hx_conv2))
        self.hx_conv3 = self.conv3(self.hx_conv2)
        self.hx_conv3 = F.relu(self.bn3(self.hx_conv3))
        self.hx_conv4 = self.conv4(self.hx_conv3)
        self.hx_conv4 = F.relu(self.bn4(self.hx_conv4))
        self.hx5 = self.fc5(self.hx_conv4.view(-1,1024*4*4))
        self.dx_logit = self.fc6(self.hx5)
        self.dx_prob = torch.sigmoid(self.dx_logit)

        # Z Path
        self.hz1 = F.relu(self.zfc1(z))
        self.hz2 = F.relu(self.zfc2(self.hz1))
        self.dz_logit = self.zfc3(self.hz2)
        self.dz_prob = torch.sigmoid(self.dz_logit)

        # Cat Path
        self.h_cat = torch.cat([self.hx5, self.hz2], 1)
        self.h1_cat = self.fc1_cat(self.h_cat)
        self.dc_logit = self.fc2_cat(self.h1_cat)
        self.dc_prob = torch.sigmoid(self.dc_logit)
        return self.dx_prob, self.dz_prob, self.dc_prob

z_dim = 64
num_epochs = 20

netEnc = Encoder(z_dim).to(device)
netEnc.apply(weights_init)
netGen = Generator(z_dim).to(device)
netGen.apply(weights_init)
netDis = Discriminator(z_dim).to(device)
netDis.apply(weights_init)

optEnc = optim.Adam(netEnc.parameters(), lr=2e-4, betas=(0.5, 0.999))
optGen = optim.Adam(netGen.parameters(), lr=2e-4, betas=(0.5, 0.999))
optDis = optim.Adam(netDis.parameters(), lr=1e-4, betas=(0.5, 0.999))

for i, data in enumerate(dataloader, 0):
    x_fixed = data[0].to(device)
    break
z_fixed = torch.randn(32, z_dim, device=device)

model_name = "ali2"
out_folder = "out/" + model_name + "/"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
save_folder =  "save/" + model_name + "/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

def zero_grad_list(net_list):
    for net in net_list:
        net.zero_grad()
pre_epoch = 2
print("Starting Training ...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Initialize
        x_real = data[0].to(device)
        b_size = x_real.size(0)
        z_real = torch.randn(b_size, z_dim, device=device)
        zeros = torch.full((b_size, 1), 0.0, device=device)
        ones = torch.full((b_size, 1), 1.0, device=device)
        
        # Discriminator
        zero_grad_list([netEnc, netGen, netDis])
        # X -> Z
        z_enc = netEnc(x_real)
        dx0_prob, dz0_prob, dc0_prob = netDis(x_real, z_enc)
        dx0_loss = nn.BCELoss()(dx0_prob, ones) # Real
        dz0_loss = nn.BCELoss()(dz0_prob, zeros) # Fake
        dc0_loss = nn.BCELoss()(dc0_prob, zeros) # Mode 0
        # Z -> X
        x_gen = netGen(z_real)
        dx1_prob, dz1_prob, dc1_prob = netDis(x_gen, z_real)
        dx1_loss = nn.BCELoss()(dx1_prob, zeros) # Fake
        dz1_loss = nn.BCELoss()(dz1_prob, ones) # Real
        dc1_loss = nn.BCELoss()(dc1_prob, ones) # Mode 1
        # step
        if epoch < pre_epoch:
            d_loss = dz0_loss + dz1_loss
        else:
            d_loss = dx0_loss + dz0_loss + dc0_loss + dx1_loss + dz1_loss + dc1_loss
        d_loss.backward()
        optDis.step()

        # Encoder
        zero_grad_list([netEnc, netGen, netDis])
        z_enc = netEnc(x_real)
        _, ez_prob, ec_prob = netDis(x_real, z_enc)
        ez_loss = nn.BCELoss()(ez_prob, ones)
        ec_loss = nn.BCELoss()(ec_prob, ones)
        if epoch < pre_epoch:
            e_loss = ez_loss
        else:
            e_loss = ez_loss + ec_loss
        e_loss.backward()
        optEnc.step()

        # Generator
        zero_grad_list([netEnc, netGen, netDis])
        x_gen = netGen(z_real)
        gx_prob, _, gc_prob = netDis(x_gen, z_real)
        gx_loss = nn.BCELoss()(gx_prob, ones)
        gc_loss = nn.BCELoss()(gc_prob, zeros)
        if epoch < pre_epoch:
            pass
        else:
            g_loss = gx_loss + gc_loss
            g_loss.backward()
            optGen.step()
        
        # Results
        if i % 50 == 0:
            if epoch < pre_epoch:
                print("[%d/%d][%s/%d] D_loss: %.4f | G_loss: %.4f | E_loss: %.4f"\
                %(epoch+1, num_epochs, str(i).zfill(4), len(dataloader), \
                d_loss.item(), 0, e_loss.mean().item()))
            else:
                print("[%d/%d][%s/%d] D_loss: %.4f | G_loss: %.4f | E_loss: %.4f"\
                %(epoch+1, num_epochs, str(i).zfill(4), len(dataloader), \
                d_loss.item(), g_loss.mean().item(), e_loss.mean().item()))
        
        if i%200 == 0:
            # Output Images
            x_rec = netGen(netEnc(x_fixed)).detach()
            x_samp = netGen(z_fixed).detach()
            x_fig = torch.cat([x_fixed[0:8], x_rec[0:8], x_fixed[8:16], x_rec[8:16], x_samp], 0)
            x_fig = x_fig.cpu()
            plt.figure(figsize=(8,8))
            plt.imshow(np.transpose(vutils.make_grid(x_fig, padding=2, normalize=True).cpu(),(1,2,0)))
            plt.axis("off")
            plt.savefig(out_folder+str(epoch).zfill(2)+"_"+str(i).zfill(4)+".jpg", bbox_inches="tight")
            plt.close()
            # Save Model
            torch.save(netEnc.state_dict(), save_folder+"netEnc.pt")
            torch.save(netGen.state_dict(), save_folder+"netGen.pt")
            torch.save(netDis.state_dict(), save_folder+"netDis.pt")