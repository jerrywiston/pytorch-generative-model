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

def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    p2_norm_x = X.pow(2).sum(1).unsqueeze(0)
    norms_x = X.sum(1).unsqueeze(0)
    prods_x = torch.mm(norms_x, norms_x.t())
    dists_x = p2_norm_x + p2_norm_x.t() - 2 * prods_x

    p2_norm_y = Y.pow(2).sum(1).unsqueeze(0)
    norms_y = X.sum(1).unsqueeze(0)
    prods_y = torch.mm(norms_y, norms_y.t())
    dists_y = p2_norm_y + p2_norm_y.t() - 2 * prods_y

    dot_prd = torch.mm(norms_x, norms_y.t())
    dists_c = p2_norm_x + p2_norm_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats

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

z_dim = 64
num_epochs = 20

netEnc = Encoder(z_dim).to(device)
netEnc.apply(weights_init)
netDec = Decoder(z_dim).to(device)
netDec.apply(weights_init)

params = list(netEnc.parameters()) + list(netDec.parameters())
optRec = optim.Adam(params, lr=2e-4, betas=(0.5, 0.999))

z_fixed = torch.randn(64, z_dim, device=device)

model_name = "wae"
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
        x_real = data[0].to(device)
        b_size = x_real.size(0)

        # Reconstruction Loss
        z_code = netEnc(x_real)
        x_rec = netDec(z_code)
        rec_loss = nn.MSELoss()(x_rec, x_real)

        # MMD Loss
        z_real = torch.randn(b_size, z_dim, device=device)
        mmd_loss = imq_kernel(z_real, z_code, h_dim=z_dim)
        mmd_loss = mmd_loss.mean()

        aae_loss = rec_loss - 100*mmd_loss
        aae_loss.backward()
        optRec.step()

        # Results
        if i % 50 == 0:
            print("[%d/%d][%s/%d] R_loss: %.4f | MMD_loss: %.4f"\
            %(epoch+1, num_epochs, str(i).zfill(4), len(dataloader), rec_loss.item(), mmd_loss.mean().item()))
        
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
            torch.save(netDec.state_dict(), save_folder+"netDec.pt")