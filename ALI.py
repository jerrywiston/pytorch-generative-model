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
        self.conv1 = Conv2d(3, 128, 5, stride=2)

        self.conv2 = Conv2d(128, 256, 5, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv3 = Conv2d(256, 512, 5, stride=2)
        self.bn3 = nn.BatchNorm2d(512)
        
        self.conv4 = Conv2d(512, 1024, 3, stride=2)
        self.bn4 = nn.BatchNorm2d(1024)
        
        self.fc5 = nn.Linear(4*4*1024, 512)
        self.fc6 = nn.Linear(1024, 1)

        self.zfc1 = nn.Linear(z_dim, 512)
        self.zfc2 = nn.Linear(512, 512)
        self.zfc3 = nn.Linear(512, 512)

    def forward(self, x, z):
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
        self.hx5 = self.fc5(self.h_conv4.view(-1,1024*4*4))

        self.hz1 = F.relu(self.fc1(z))
        self.hz2 = F.relu(self.fc2(self.h1))
        self.hz3 = F.relu(self.fc3(self.h2))

        self.h_concat = torch.cat([self.hx5, self.hz3])
        self.d_logit = self.fc6(self.h_concat)
        self.d_prob = torch.sigmoid(self.d_logit)
        return self.d_prob, self.d_logit

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

z_fixed = torch.randn(64, z_dim, device=device)

model_name = "ali"
out_folder = "out/" + model_name + "/"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
save_folder =  "save/" + model_name + "/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

def zero_grad_list(net_list):
    for net in net_list:
        net.zero_grad()

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
        d0_prob, d0_logit = netDis(x_real, z_enc)
        d0_loss = nn.BCELoss(d0_prob, zeros)
        # Z -> X
        x_gen = netGen(z_real)
        d1_prob, d1_logit = netDis(x_gen, z_real)
        d1_loss = nn.BCELoss(d1_prob, ones)
        # step
        d_loss = d1_loss + d2_loss
        d_loss.backward()
        optDis.step()

        # Encoder
        zero_grad_list([netEnc, netGen, netDis])
        z_enc = netEnc(x_real)
        e_prob, e_logit = netDis(x_real, z_enc)
        e_loss = nn.BCELoss(e_prob, ones)
        e_loss.backward()
        optEnc.step()

        # Generator
        zero_grad_list([netEnc, netGen, netDis])
        x_gen = netGen(z_real)
        g_prob, g_logit = netDis(x_real, z_enc)
        g_loss = nn.BCELoss(g_prob, zeros)
        g_loss.backward()
        optGen.step()

        # Results
        if i % 50 == 0:
            print("[%d/%d][%s/%d] D_loss: %.4f | G_loss: %.4f | E_loss: %.4f"\
            %(epoch+1, num_epochs, str(i).zfill(4), len(dataloader), d_loss.item(), g_loss.mean().item(), e_loss.mean().item()))
        
        if i%200 == 0:
            # Output Images
            x_fixed = netDec(z_fixed).cpu().detach()
            plt.figure(figsize=(8,8))
            plt.imshow(np.transpose(vutils.make_grid(x_fixed, padding=2, normalize=True).cpu(),(1,2,0)))
            plt.axis("off")
            plt.savefig(out_folder+str(epoch).zfill(2)+"_"+str(i).zfill(4)+".jpg", bbox_inches="tight")
            plt.close()
            # Save Model
            torch.save(netEnc.state_dict(), save_folder+"netEnc.pt")
            torch.save(netGen.state_dict(), save_folder+"netGen.pt")
            torch.save(netDis.state_dict(), save_folder+"netDis.pt")