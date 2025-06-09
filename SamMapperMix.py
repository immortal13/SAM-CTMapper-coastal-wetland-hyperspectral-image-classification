import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from einops import rearrange, repeat
import math
from Pixel2SP import Pixel2SP
from utils import FeatureConverter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, padding=0, bias=False),
        )
    def forward(self, x):
        return self.net(x)

class sub_Attention(nn.Module):
    def __init__(self, h, w, scale, dim, num_heads):
        super(sub_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
        )
        ## Pixel2SP module
        #### 仅需设置3个超参数：superpixel_scale、ETA_POS、n_iters，但基本只需调整superpixel_scale 
        n_iters = 5  # iteration of DSLIC
        ETA_POS = 1.8 # scale coefficient of positional pixel features I^xy, it is very important
        global mask
        mask = 0

        n_spixels = int(h*w/scale)
        self.SP_assign = Pixel2SP(FeatureConverter(ETA_POS), n_iters, n_spixels)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
    
    def get_A(self, segments_map, n_spixels):
        A = np.zeros([n_spixels, n_spixels], dtype=np.float32)
        (h, w) = segments_map.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = segments_map[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue
                    A[idx1, idx2] = A[idx2, idx1] = 1   
        np.fill_diagonal(A,0)
        ## 由[M,M] → [B,H,M,M]
        A = np.expand_dims(A,0).repeat(int(self.num_heads), axis=0) ##这里h改成1了
        A = np.expand_dims(A,0) #(1, 4, 1144, 1144)
        A_cuda=torch.from_numpy(A).to(device)#.cuda()
        return A_cuda

    def forward(self, x):
        Q, ops, f, spf, pf = self.SP_assign(x.contiguous()) ## 这里是x_[i]，分了h个头
        Q_d = Q.detach()
        segments_map_cuda = ops['map_idx'](torch.argmax(Q_d, 1, True).int()) 
        segments_map = segments_map_cuda[0,0].cpu().numpy() #[1, 1, 145, 145]
        n_spixels = np.max(segments_map) + 1
        global mask
        if isinstance(mask, int):
            mask = self.get_A(segments_map, n_spixels)
        x = rearrange(spf[0], '(b c) m -> b m c', b=1) # [B, M, C]

        ###
        b, n, _, h = *x.shape, self.num_heads  
        qkv = self.to_qkv(x).chunk(3, dim = -1) # b, n, inner_dim * 3   
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv) ### 这里h设为1了

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        ### the core code block: SNN mask
        if mask is not None: 
            zero_vec = -9e15 * torch.ones_like(attn) 
            attn = torch.where(mask > 0, attn, zero_vec)
            attn = attn[0][0].cpu().detach().numpy()
            np.fill_diagonal(attn,1)
            attn = torch.from_numpy(attn).to(device)#.cuda()
            s1,s2 = attn.shape[0], attn.shape[1]
            attn = attn.reshape(1,1,s1,s2)

        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b h n d -> b n (h d)') # b, n, h*inner_dim 把8个头concate成一个头
        out =  self.to_out(out)

        out = ops['map_sp2p'](torch.unsqueeze(out[0].t(),0).contiguous(), Q_d) #[1,C,H,W]
        attn_out = out #self.proj(out)
        return attn_out

class sub_Conv(nn.Module):
    def __init__(self, dim, num_heads):
        super(sub_Conv, self).__init__()
        expand_ratio = 1
        self.dim = dim 
        self.num_heads = num_heads
        self.split_groups = dim//self.num_heads

        for i in range(self.num_heads):
            local_conv = nn.Sequential(
            nn.BatchNorm2d(dim//self.num_heads),    
            nn.Conv2d(dim//self.num_heads, dim//self.num_heads, 1, padding=0, bias=False), 
            nn.Conv2d(dim//self.num_heads, dim//self.num_heads, 
                kernel_size=(3+i*2), padding=(1+i), stride=1, groups=dim//self.num_heads),
            nn.LeakyReLU()  
            )
            setattr(self, f"local_conv_{i + 1}", local_conv)  
        self.proj0 = nn.Conv2d(dim, dim*expand_ratio, kernel_size=1, padding=0, stride=1, groups=dim//self.num_heads)
        self.bn = nn.BatchNorm2d(dim*expand_ratio)
        self.act = nn.LeakyReLU()
        self.proj1 = nn.Conv2d(dim*expand_ratio, dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        s = x.permute(0,2,3,1).reshape(B, H, W, self.num_heads, self.dim//self.num_heads).permute(3, 0, 4, 1, 2)
        for i in range(self.num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            s_i = local_conv(s[i]).reshape(B, self.split_groups, -1, H, W) #[1, 32, 1, 512, 217]
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out,s_i],2)
        s_out = s_out.reshape(B, self.dim, H, W)

        s_out = self.proj0(s_out)
        s_out = self.act(self.bn(s_out))
        s_out = self.proj1(s_out)
        return s_out 


class Attention(nn.Module):
    def __init__(self, h, w, ratio, scale, dim, num_heads1, num_heads2):
        super(Attention, self).__init__()
        self.dim = dim
        self.conv_dim = int(dim - (dim/(ratio+1)))
        self.trans_dim = int(dim/(ratio+1))
        # print(self.conv_dim, self.trans_dim, "self.conv_dim, self.trans_dim")
        self.num_heads1 = num_heads1
        self.num_heads2 = num_heads2

        #### conv branch
        self.MSC = sub_Conv(self.conv_dim, self.num_heads1)

        #### attn branch       
        self.MSA = sub_Attention(h, w, scale, self.trans_dim, self.num_heads2) 

        ### fuse
        ## way 2
        d = max(int(dim/4), 64)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            # nn.BatchNorm2d(dim//2),
            nn.Conv2d(dim, d, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU())
        
        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Conv2d(d, self.conv_dim, kernel_size=1, stride=1))
        self.fcs.append(nn.Conv2d(d, self.trans_dim, kernel_size=1, stride=1))

        self.softmax = nn.Softmax(dim=1)

        # ### way 3 CS
        # self.channel_interaction = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(self.conv_dim, self.conv_dim//4, kernel_size=1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(self.conv_dim//4, self.trans_dim, kernel_size=1),
        # )
        # self.spatial_interaction = nn.Sequential(
        #     nn.BatchNorm2d(self.trans_dim),
        #     nn.Conv2d(self.trans_dim, self.trans_dim//8, kernel_size=1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(self.trans_dim//8, 1, kernel_size=1)
        # )

        # ### way 4 SC
        # self.channel_interaction = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     # nn.BatchNorm2d(self.trans_dim),
        #     nn.Conv2d(self.trans_dim, self.trans_dim//4, kernel_size=1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(self.trans_dim//4, self.conv_dim, kernel_size=1),
        # )
        # self.spatial_interaction = nn.Sequential(
        #     nn.BatchNorm2d(self.conv_dim),
        #     nn.Conv2d(self.conv_dim, self.conv_dim//8, kernel_size=1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(self.conv_dim//8, 1, kernel_size=1)
        # )

        # ### way 5 CC
        # self.channel_interaction1 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     # nn.BatchNorm2d(self.trans_dim),
        #     nn.Conv2d(self.trans_dim, self.trans_dim//4, kernel_size=1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(self.trans_dim//4, self.conv_dim, kernel_size=1),
        # )
        # self.channel_interaction2 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     # nn.BatchNorm2d(self.conv_dim),
        #     nn.Conv2d(self.conv_dim, self.conv_dim//4, kernel_size=1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(self.conv_dim//4, self.trans_dim, kernel_size=1),
        # )

        # ### way 6 SS
        # self.spatial_interaction1 = nn.Sequential(
        #     nn.BatchNorm2d(self.trans_dim),
        #     nn.Conv2d(self.trans_dim, self.trans_dim//8, kernel_size=1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(self.trans_dim//8, 1, kernel_size=1)
        # )
        # self.spatial_interaction2 = nn.Sequential(
        #     nn.BatchNorm2d(self.conv_dim),
        #     nn.Conv2d(self.conv_dim, self.conv_dim//8, kernel_size=1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(self.conv_dim//8, 1, kernel_size=1)
        # )
       
    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
        B, C, H, W = x.size()
        ### split
        conv_x, trans_x = torch.split(x, (self.conv_dim, self.trans_dim), dim=1)

        ### conv layer
        conv_x = self.MSC(conv_x)
        ### attention layer 
        trans_x = self.MSA(trans_x) 

        # ### fuse
        ## way 1
        # feats = torch.cat((conv_x, trans_x), dim=1)

        # way 2
        feats_u = torch.cat((conv_x, trans_x), dim=1)
        feats_s = self.gap(feats_u) #(1,128,1,1)
        feats_z = self.fc(feats_s)
        # # way 22
        feats_a = torch.sigmoid(self.fcs[0](feats_z))
        feats_b = torch.sigmoid(self.fcs[1](feats_z))
        feats = feats_u + torch.cat((feats_a*conv_x, feats_b*trans_x), dim=1)
        ## way 23
        # feats_a = self.fcs[0](feats_z)
        # feats_b = self.fcs[1](feats_z)
        # attention_vectors = torch.cat((feats_a.view(B, 1, self.conv_dim, 1, 1), feats_b.view(B, 1, self.conv_dim, 1, 1)), dim=1)
        # attention_vectors = self.softmax(attention_vectors)
        # feats = torch.cat(((conv_x*attention_vectors)[0][0].unsqueeze(0), (trans_x*attention_vectors)[0][1].unsqueeze(0)), dim=1)
        
        # ## way 3
        # # Adaptive Interaction Module (AIM)
        # # C-Map (before sigmoid)
        # channel_map = self.channel_interaction(conv_x) ## [1,64,1,1]
        # # S-Map (before sigmoid) 
        # spatial_map = self.spatial_interaction(trans_x) ## [1,1,800,600]
        # # C-I
        # trans_x = torch.sigmoid(channel_map) * trans_x 
        # # S-I
        # conv_x = torch.sigmoid(spatial_map) * conv_x
        # #
        # feats = torch.cat((conv_x, trans_x), dim=1)

        # ## way 4
        # # Adaptive Interaction Module (AIM)
        # feats_orig = torch.cat((conv_x, trans_x), dim=1)
        # # C-Map (before sigmoid)
        # channel_map = self.channel_interaction(trans_x) ## [1,64,1,1]
        # # S-Map (before sigmoid) 
        # spatial_map = self.spatial_interaction(conv_x) ## [1,1,800,600]
        # # C-I
        # conv_x = torch.sigmoid(channel_map) * conv_x 
        # # S-I
        # trans_x = torch.sigmoid(spatial_map) * trans_x
        # #
        # feats = feats_orig + torch.cat((conv_x, trans_x), dim=1)

        # ## way 5
        # # Adaptive Interaction Module (AIM)
        # # C-Map (before sigmoid)
        # channel_map = self.channel_interaction1(trans_x) ## [1,64,1,1]
        # # S-Map (before sigmoid) 
        # spatial_map = self.channel_interaction2(conv_x) ## [1,1,800,600]
        # # C-I
        # conv_x = torch.sigmoid(channel_map) * conv_x 
        # # S-I
        # trans_x = torch.sigmoid(spatial_map) * trans_x
        # #
        # feats = torch.cat((conv_x, trans_x), dim=1)

        # ## way 6
        # # Adaptive Interaction Module (AIM)
        # # C-Map (before sigmoid)
        # channel_map = self.spatial_interaction1(trans_x) ## [1,64,1,1]
        # # S-Map (before sigmoid) 
        # spatial_map = self.spatial_interaction2(conv_x) ## [1,1,800,600]
        # # C-I
        # conv_x = torch.sigmoid(channel_map) * conv_x 
        # # S-I
        # trans_x = torch.sigmoid(spatial_map) * trans_x
        # #
        # feats = torch.cat((conv_x, trans_x), dim=1)

        return feats

class Encoder_layer(nn.Module):
    def __init__(self, h, w, ratio, scale, dim, num_heads1, num_heads2, mlp_ratio=1):#, S2P=None, P2S=None, mask=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(h, w, ratio, scale, dim, num_heads1, num_heads2)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio))
        
    def forward(self, x): 
        B,C,H,W = x.size()
        x = x + self.attn(rearrange(self.norm1(rearrange(x, 'b c h w -> b (h w) c')), 'b (h w) c -> b c h w', h = H, w = W))
        x = x + self.mlp(rearrange(self.norm2(rearrange(x, 'b c h w -> b (h w) c')), 'b (h w) c -> b c h w', h = H, w = W))
        return x 

def bn_conv_lrelu(in_c, out_c, kernel_size):
    return nn.Sequential(
        nn.BatchNorm2d(in_c),
        nn.Conv2d(in_c, out_c, kernel_size, padding=(kernel_size-1)//2, bias=False),
        nn.LeakyReLU()
    )

class SamMapperMix(nn.Module):
    def __init__(self, h, w, channel, class_count, L1, L2, H1, H2, ratio, scale):
        super(SamMapperMix, self).__init__()
        self.channel = 128
        self.ratio = ratio ## C_cnn/C_trans, uneven number
        self.scale = scale ## superpixel segmentation scale
        self.L1 = L1 ## number of stage
        self.L2 = L2 ## number of mix block
        self.H1 = H1 ## number of head of MSC layer 
        self.H2 = H2 ## number of head of MSA layer

        ## 1.stem module
        self.stem = nn.Sequential(bn_conv_lrelu(channel, self.channel, 3))
        
        ## 2.transformer backbone
        self.patch_embed = nn.Sequential() 
        self.block = nn.Sequential()        
        for i in range(self.L1):
            #### patch embed layer
            if i==0:
                self.patch_embed.add_module('patchembed_'+str(i), 
                    bn_conv_lrelu(self.channel, self.channel, 3))
            else:
                self.patch_embed.add_module('patchembed_'+str(i), 
                    bn_conv_lrelu(self.channel, self.channel, 3))
            #### transformer encoder layers
            for j in range(self.L2):
                self.block.add_module('encoder_'+str(i)+str(j), 
                    Encoder_layer(h, w, self.ratio, self.scale, dim=self.channel, num_heads1=self.H1, num_heads2=self.H2))

        ## 3.classification head
        self.fc_add = nn.Linear(self.channel, class_count)

    def forward(self, x):
        (h, w, c) = x.shape
        x = torch.unsqueeze(x.permute([2, 0, 1]), 0)
        ## stem
        x = self.stem(x)
        # ## stage ##
        count = 0
        for i in range(self.L1):
            x = self.patch_embed[i](x)
            for j in range(self.L2):
                # print(i,j,count)
                x = self.block[count](x) #[B,L,C]
                count = count + 1
        vis = x
        ## classification
        x = x[0].permute(1,2,0).reshape([h * w, -1])
        x = self.fc_add(x)
        x = F.softmax(x, -1)
        return x, vis