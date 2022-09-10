import torch
from torch import nn
import torch.nn.functional as F
from unet_model import UNet
from einops.layers.torch import Rearrange, Reduce

class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):

        '''
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''

        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]
        
        # torch.sum --->  Conv2D(in=3, out=1, k=3, s=1, p=1)

        return torch.cat([R, G, B], dim=1)
 
 
def MLP_A_real(deep):

    return nn.Sequential(*[nn.Sequential(
            nn.Linear(64, 64),
            nn.PReLU(),
            Rearrange('b c w h -> b c h w'),
            nn.Linear(64, 64),
            nn.PReLU(),
            Rearrange('b c h w -> b w h c'),
            nn.Linear(3, 3),
            nn.PReLU(),
            Rearrange('b w h c -> b c w h'),
            nn.Linear(64, 64),
            nn.PReLU(),      
        ) for _ in range(deep)])
def MLP_A_imag(deep):
    
    return nn.Sequential(*[nn.Sequential(
            nn.Linear(64, 64),
            nn.PReLU(),
            Rearrange('b c w h -> b c h w'),
            nn.Linear(64, 64),
            nn.PReLU(),
            Rearrange('b c h w -> b w h c'),
            nn.Linear(3, 3),
            nn.PReLU(),
            Rearrange('b w h c -> b c w h'),
            nn.Linear(64, 64),
            nn.PReLU(),      
        ) for _ in range(deep)])
def MLP_B_real(deep):
    
    return nn.Sequential(*[nn.Sequential(
            nn.Linear(128, 128),
            nn.PReLU(),
            Rearrange('b c w h -> b c h w'),
            nn.Linear(128, 128),
            nn.PReLU(),
            Rearrange('b c h w -> b w h c'),
            nn.Linear(3, 3),
            nn.PReLU(),
            Rearrange('b w h c -> b c w h'),
            nn.Linear(128, 128),
            nn.PReLU(),      
        ) for _ in range(deep)])
def MLP_B_imag(deep):
    
    return nn.Sequential(*[nn.Sequential(
            nn.Linear(128, 128),
            nn.PReLU(),
            Rearrange('b c w h -> b c h w'),
            nn.Linear(128, 128),
            nn.PReLU(),
            Rearrange('b c h w -> b w h c'),
            nn.Linear(3, 3),
            nn.PReLU(),
            Rearrange('b w h c -> b c w h'),
            nn.Linear(128, 128),
            nn.PReLU(),      
        ) for _ in range(deep)])
    

def MLP_C_real(deep):
    
    return nn.Sequential(*[nn.Sequential(
            nn.Linear(256, 256),
            nn.PReLU(),
            Rearrange('b c w h -> b c h w'),
            nn.Linear(256, 256),
            nn.PReLU(),
            Rearrange('b c h w -> b w h c'),
            nn.Linear(3, 3),
            nn.PReLU(),
            Rearrange('b w h c -> b c w h'),
            nn.Linear(256, 256),
            nn.PReLU(),      
        ) for _ in range(deep)])
    
def MLP_C_imag(deep):
    
    return nn.Sequential(*[nn.Sequential(
            nn.Linear(256, 256),
            nn.PReLU(),
            Rearrange('b c w h -> b c h w'),
            nn.Linear(256, 256),
            nn.PReLU(),
            Rearrange('b c h w -> b w h c'),
            nn.Linear(3, 3),
            nn.PReLU(),
            Rearrange('b w h c -> b c w h'),
            nn.Linear(256, 256),
            nn.PReLU(),      
        ) for _ in range(deep)])


class Zero_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = UNet(n_channels=12)
        self.app = ApplyCoeffs()
        self.a_path_real = MLP_A_real(4)
        self.b_path_real = MLP_B_real(4)
        self.c_path_real = MLP_C_real(4)
        
        self.a_path_imag = MLP_A_imag(4)
        self.b_path_imag = MLP_B_imag(4)
        self.c_path_imag = MLP_C_imag(4)
       
        
        
    def forward(self, x):
        input_a = F.interpolate(x, size=[64, 64],   mode='bilinear', align_corners=True)
        input_b = F.interpolate(x, size=[128, 128],   mode='bilinear', align_corners=True)
        input_c = F.interpolate(x, size=[256, 256],   mode='bilinear', align_corners=True)

        a_f = torch.fft.fft2(input_a)
        a_f_real = a_f.real
        a_f_imag = a_f.imag
        
        b_f = torch.fft.fft2(input_b)
        b_f_real = b_f.real
        b_f_imag = b_f.imag
        
        c_f = torch.fft.fft2(input_c)
        c_f_real = c_f.real
        c_f_imag = c_f.imag

        
        
        coeff_a = torch.real(torch.fft.ifft2(torch.complex(self.a_path_real(a_f_real), self.a_path_imag(a_f_imag))))
        coeff_b = torch.real(torch.fft.ifft2(torch.complex(self.b_path_real(b_f_real), self.b_path_imag(b_f_imag))))
        coeff_c = torch.real(torch.fft.ifft2(torch.complex(self.c_path_real(c_f_real), self.c_path_imag(c_f_imag))))
        

        
        
        
        
        
        a = F.interpolate(coeff_a, [input_c.shape[2], input_c.shape[3]],   mode='bilinear', align_corners=True)
        b = F.interpolate(coeff_b, [input_c.shape[2], input_c.shape[3]],   mode='bilinear', align_corners=True)
        c = F.interpolate(coeff_c, [input_c.shape[2], input_c.shape[3]],   mode='bilinear', align_corners=True)
        
        
        coeff = torch.cat((input_c, a, b, c), dim=1)
        coeff = self.net(coeff)
        
        coeff = F.interpolate(coeff, [x.shape[2], x.shape[3]],   mode='bilinear', align_corners=True)
        out = self.app(coeff, x)

        return  out     



net = Zero_Net()

data = torch.zeros(1, 3, 3840, 2160)
print(net(data).shape)
