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

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = Conv2d(3, 128, 5, stride=2)
        
        self.conv2 = Conv2d(128, 256, 5, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv3 = Conv2d(256, 512, 5, stride=2)
        self.bn3 = nn.BatchNorm2d(512)
        
        self.conv4 = Conv2d(512, 1024, 3, stride=2)
        self.bn4 = nn.BatchNorm2d(1024)
        
        self.fc5_mu = nn.Linear(4*4*1024, z_dim)
        self.fc5_logvar = nn.Linear(4*4*1024, z_dim)
    
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
        self.z_mu = self.fc5_mu(self.h_conv4.view(-1,1024*4*4))
        self.z_logvar = self.fc5_logvar(self.h_conv4.view(-1,1024*4*4))
        self.z_code = sample_z(self.z_mu, self.z_logvar)
        return self.z_code, self.z_mu

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
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

class Projector():
    def __init__(self, z_dim):
        super(Projector, self).__init__()
        self.conv1 = Conv2d(3, 16, 5)
        self.conv2 = Conv2d(16, 32, 5)
        self.conv3 = Conv2d(32, 3, 3)

    def forward(self, x):
        self.h_conv1 = F.relu(self.conv1(x))
        self.h_conv2 = F.relu(self.conv2(self.h_conv1))
        self.xp = torch.sigmoid(self.conv3(self.h_conv2))
        return self.xp

z_dim = 64
num_epochs = 20

netEnc = Encoder(z_dim).to(device)
netEnc.apply(weights_init)
netDec = Decoder(z_dim).to(device)
netDec.apply(weights_init)
netDis = Discriminator(z_dim).to(device)
netDis.apply(weights_init)

params = list(netEnc.parameters()) + list(netDec.parameters())
optRec = optim.Adam(params, lr=2e-4, betas=(0.5, 0.999))
optDis = optim.Adam(netDis.parameters(), lr=1e-4, betas=(0.5, 0.999))

for i, data in enumerate(dataloader, 0):
    x_fixed = data[0].to(device)
    break
z_fixed = torch.randn(32, z_dim, device=device)

model_name = "aae_random"
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
        netEnc.zero_grad()
        netDec.zero_grad()
        netDis.zero_grad()
        x_real = data[0].to(device)
        b_size = x_real.size(0)

        # Adversarial
        # Fake Batch
        z_code, _ = netEnc(x_real)
        d_fake, _ = netDis(z_code)
        d_label = torch.full((b_size, 1), 0.0, device=device)
        d_fake_loss = nn.BCELoss()(d_fake, d_label)
        d_fake_loss.backward()
        # Real Batch
        z_samp = torch.randn(b_size, z_dim, device=device)
        d_real, _ = netDis(z_samp)
        d_label.fill_(1.0)
        d_real_loss = nn.BCELoss()(d_real, d_label)
        d_real_loss.backward()
        d_loss = d_real_loss + d_fake_loss
        optDis.step()

        # Reconstruction
        netEnc.zero_grad()
        netDec.zero_grad()
        netDis.zero_grad()
        z_code, _ = netEnc(x_real)
        x_rec = netDec(z_code)
        d_fake, _ = netDis(z_code)
        d_label.fill_(1.0)
        code_loss = nn.BCELoss()(d_fake, d_label)
        rec_loss = nn.MSELoss()(x_rec, x_real)
        aae_loss = rec_loss + 0.001*code_loss
        aae_loss.backward()
        optRec.step()

        # Results
        if i % 50 == 0:
            print("[%d/%d][%s/%d] R_loss: %.4f | C_loss: %.4f | D_loss: %.4f"\
            %(epoch+1, num_epochs, str(i).zfill(4), len(dataloader), rec_loss.item(), code_loss.mean().item(), d_loss.mean().item()))
        
        if i%200 == 0:
            # Output Images
            _, z_code = netEnc(x_fixed)
            x_rec = netDec(z_code).detach()
            x_samp = netDec(z_fixed).detach()
            x_fig = torch.cat([x_fixed[0:8], x_rec[0:8], x_fixed[8:16], x_rec[8:16], x_samp], 0)
            x_fig = x_fig.cpu()
            plt.figure(figsize=(8,8))
            plt.imshow(np.transpose(vutils.make_grid(x_fig, padding=2, normalize=True).cpu(),(1,2,0)))
            plt.axis("off")
            plt.savefig(out_folder+str(epoch).zfill(2)+"_"+str(i).zfill(4)+".jpg", bbox_inches="tight")
            plt.close()
            # Save Model
            torch.save(netEnc.state_dict(), save_folder+"netEnc.pt")
            torch.save(netDec.state_dict(), save_folder+"netDec.pt")
            torch.save(netDis.state_dict(), save_folder+"netDis.pt")