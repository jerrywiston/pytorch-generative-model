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

def adaIN_Layer(h_conv, w, aff_s, aff_b):
    channel = h_conv.size(1)   
    h_s = aff_s(w).view(-1,channel,1,1)
    h_b = aff_b(w).view(-1,channel,1,1)
    #noise = torch.randn(h_conv.size()).to(device)
    #h_conv = h_conv + noise
    h_norm = torch.nn.InstanceNorm2d(channel)(h_conv)
    h_style = h_s * h_norm + h_b
    
    return h_style

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        # Trainable Constant Input
        self.const = nn.Parameter(torch.randn(512, 4, 4, device='cuda:0'))
        
        # Mapping Network
        self.fc1 = nn.Linear(z_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, z_dim)
        
        # adaIN Affine Transform
        # 8x8
        self.aff1_s = nn.Linear(z_dim, 256, bias=False)
        self.aff1_b = nn.Linear(z_dim, 256, bias=False)
        # 16x16
        self.aff2_s = nn.Linear(z_dim, 128, bias=False)
        self.aff2_b = nn.Linear(z_dim, 128, bias=False)
        # 32x32
        self.aff3_s = nn.Linear(z_dim, 64, bias=False)
        self.aff3_b = nn.Linear(z_dim, 64, bias=False)
        # 64x64
        self.aff4_s = nn.Linear(z_dim, 32, bias=False)
        self.aff4_b = nn.Linear(z_dim, 32, bias=False)

        # Upsample Convolution Layer
        # 4x4x512 -> 8x8x256
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = Conv2d(512, 256, 3)
        self.bn1 = nn.BatchNorm2d(256)
        # 8x8x256 -> 16x16x128
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = Conv2d(256, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
        # 16x16x128 -> 32x32x64
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = Conv2d(128, 64, 5)
        self.bn3 = nn.BatchNorm2d(64)
        # 32x32x64 -> 64x64x32
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = Conv2d(64, 32, 5)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Color Transform Layer
        self.conv1_rgb = Conv2d(256, 3, 3)
        self.conv2_rgb = Conv2d(128, 3, 3)
        self.conv3_rgb = Conv2d(64, 3, 3)
        self.conv4_rgb = Conv2d(32, 3, 1)

    def forward(self, z):
        self.h_map1 = F.relu(self.fc1(z))
        self.w = F.relu(self.fc2(self.h_map1))
        # 4x4
        self.const_feat = self.const.view(-1,512,4,4).repeat(z.size(0),1,1,1)
        self.h_conv1 = self.up1(self.const_feat)
        self.h_conv1 = self.conv1(self.h_conv1)
        self.h_conv1 = adaIN_Layer(self.h_conv1, self.w, self.aff1_s, self.aff1_b)
        self.h_rgb1 = torch.sigmoid(self.conv1_rgb(self.h_conv1))
        self.h_conv1 = F.relu(self.h_conv1)
        # 8x8
        self.h_conv2 = self.up2(self.h_conv1)
        self.h_conv2 = self.conv2(self.h_conv2)
        self.h_conv2 = adaIN_Layer(self.h_conv2, self.w, self.aff2_s, self.aff2_b)
        self.h_rgb2 = torch.sigmoid(self.conv2_rgb(self.h_conv2))
        self.h_conv2 = F.relu(self.h_conv2)
        # 16x16
        self.h_conv3 = self.up3(self.h_conv2)
        self.h_conv3 = self.conv3(self.h_conv3)
        self.h_conv3 = adaIN_Layer(self.h_conv3, self.w, self.aff3_s, self.aff3_b)
        self.h_rgb3 = torch.sigmoid(self.conv3_rgb(self.h_conv3))
        self.h_conv3 = F.relu(self.h_conv3)
        # 32x32
        self.h_conv4 = self.up4(self.h_conv3)
        self.h_conv4 = self.conv4(self.h_conv4)
        self.h_conv4 = adaIN_Layer(self.h_conv4, self.w, self.aff4_s, self.aff4_b)
        self.h_rgb4 = torch.sigmoid(self.conv4_rgb(self.h_conv4))
        # 64x64
        #return h_rgb1, h_rgb2, h_rgb3, h_rgb4 
        return self.h_rgb4

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = Conv2d(3, 128, 5, stride=2, bias=False)
        self.conv2 = nn.utils.spectral_norm(Conv2d(128, 256, 5, stride=2, bias=False))       
        self.conv3 = nn.utils.spectral_norm(Conv2d(256, 512, 5, stride=2, bias=False))
        self.conv4 = nn.utils.spectral_norm(Conv2d(512, 1024, 3, stride=2, bias=False))
        self.fc5 = nn.Linear(4*4*1024, 1)

    def forward(self, x):
        self.h_conv1 = F.leaky_relu(self.conv1(x))
        # (128, 32, 32)
        self.h_conv2 = self.conv2(self.h_conv1)
        self.h_conv2 = F.leaky_relu(self.h_conv2)
        # (256, 16, 16)
        self.h_conv3 = self.conv3(self.h_conv2)
        self.h_conv3 = F.leaky_relu(self.h_conv3)
        # (512, 8, 8)
        self.h_conv4 = self.conv4(self.h_conv3)
        self.h_conv4 = F.leaky_relu(self.h_conv4)
        # (1024, 4, 4)
        self.d_logit = self.fc5(self.h_conv4.view(-1,1024*4*4))
        self.d_prob = torch.sigmoid(self.d_logit)
        return self.d_prob, self.d_logit

z_dim = 64
num_epochs = 20

netG = Generator(z_dim).to(device)
netG.apply(weights_init)
netD = Discriminator().to(device)
netD.apply(weights_init)

criterion = nn.BCELoss()
optD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))
optG = optim.Adam(netG.parameters(), lr=4e-4, betas=(0.5, 0.999))

z_fixed = torch.randn(64, z_dim, device=device)

model_name = "styleGAN"
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
        optG.step()

        # Results
        if i % 50 == 0:
            print("[%d/%d][%s/%d] D_loss: %.4f | G_loss: %.4f"\
            %(epoch+1, num_epochs, str(i).zfill(4), len(dataloader), d_loss.item(), g_loss.item()))
        
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
