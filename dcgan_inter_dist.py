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

def distance_loss(x1, x2, xm):
    d12 = ((x1 - x2)**2).sum([1,2,3])
    d1m = ((x1 - xm)**2).sum([1,2,3])
    d2m = ((x2 - xm)**2).sum([1,2,3])
    dist_loss1 = (d1m - d12).mean()
    dist_loss2 = (d2m - d12).mean()
    dist_loss = dist_loss1 + dist_loss2
    return dist_loss

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

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
    def __init__(self):
        super(Discriminator, self).__init__()
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

z_dim = 128
num_epochs = 20

netG = Generator(z_dim).to(device)
netG.apply(weights_init)
netD = Discriminator().to(device)
netD.apply(weights_init)

criterion = nn.BCELoss()
optD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))
optG = optim.Adam(netG.parameters(), lr=4e-4, betas=(0.5, 0.999))

z_fixed = torch.randn(64, z_dim, device=device)

model_name = "gan_"
out_folder = "out/" + model_name + "/"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
save_folder =  "save/" + model_name + "/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
print("Starting Training ...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Initialize
        netD.zero_grad()
        netG.zero_grad()

        # Real Batch
        x_real = data[0].to(device)
        d_real, _ = netD(x_real)
        b_size = x_real.size(0)
        d_label = torch.full((b_size, 1), 1.0, device=device)
        d_real_loss = criterion(d_real, d_label)
        d_real_loss.backward()

        # Fake Batch
        z_samp = torch.randn(b_size, z_dim, device=device)
        x_samp = netG(z_samp)
        d_fake, _ = netD(x_samp)
        d_label.fill_(0.0)
        d_fake_loss = criterion(d_fake, d_label)
        d_fake_loss.backward()
        
        # Update D
        d_loss = d_real_loss + d_fake_loss
        optD.step()

        # Update G
        netG.zero_grad()
        d_label.fill_(1.0)
        z_samp = torch.randn(b_size, z_dim, device=device)
        x_samp = netG(z_samp)
        d_fake, _ = netD(x_samp)
        g_loss = criterion(d_fake, d_label)
        g_loss.backward()
        #optG.step()

        z1_samp = torch.randn(b_size, z_dim, device=device)
        z2_samp = torch.randn(b_size, z_dim, device=device)
        zm_samp = 0.5*z1_samp + 0.5*z2_samp
        x1_samp = netG(z1_samp)
        x2_samp = netG(z2_samp)
        xm_samp = netG(zm_samp)
        dist_loss = 1e-3*distance_loss(x1_samp, x2_samp, xm_samp)
        dist_loss.backward()
        optG.step()

        # Results
        if i % 50 == 0:
            print("[%d/%d][%s/%d] D_loss: %.4f | G_loss: %.4f | Dist_loss: %.4f"\
            %(epoch+1, num_epochs, str(i).zfill(4), len(dataloader), d_loss.item(), g_loss.item(), dist_loss.item()))
        
        if i%200 == 0:
            # Output Images
            x_fixed = netG(z_fixed).cpu().detach()
            plt.figure(figsize=(8,8))
            plt.imshow(np.transpose(vutils.make_grid(x_fixed, padding=2, normalize=True).cpu(),(1,2,0)))
            plt.axis("off")
            plt.savefig(out_folder+str(epoch).zfill(2)+"_"+str(i).zfill(4)+".jpg", bbox_inches="tight")
            plt.close()
            # Save Model
            torch.save(netG.state_dict(), save_folder+"netGen.pt")
            torch.save(netD.state_dict(), save_folder+"netDis.pt")
