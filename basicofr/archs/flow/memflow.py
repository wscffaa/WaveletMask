"""MemFlow: Memory-based Optical Flow Completion
Integrated single-file implementation for inference.
P model is designed for optical flow completion.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

# ============================================================================
# Utility Functions
# ============================================================================

def coords_grid(batch, ht, wd, device=None):
    """Generate coordinate grids"""
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    grid = coords[None].repeat(batch, 1, 1, 1)
    if device is not None:
        grid = grid.to(device)
    return grid


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1
    
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    
    return img


def upflow8(flow, mode='bilinear'):
    """Upsample flow by 8x"""
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


# ============================================================================
# Correlation Block
# ============================================================================

class CorrBlock:
    """Correlation pyramid for efficient matching"""
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        
        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)
        
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)
    
    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)
            
            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
        
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()
    
    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)
        
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


# ============================================================================
# Feature Encoder
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        num_groups = planes // 8
        
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if stride != 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if stride != 1:
                self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if stride != 1:
                self.norm3 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if stride != 1:
                self.norm3 = nn.Sequential()
        
        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)
    
    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return self.relu(x+y)


class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        mul = input_dim // 3
        self.mul = mul
        
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64 * mul)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64 * mul)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64 * mul)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
        
        self.conv1 = nn.Conv2d(input_dim, 64 * mul, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.in_planes = 64 * mul
        self.layer1 = self._make_layer(64 * mul, stride=1)
        self.layer2 = self._make_layer(96 * mul, stride=2)
        self.layer3 = self._make_layer(128 * mul, stride=2)
        
        # output convolution
        self.conv2 = nn.Conv2d(128 * mul, output_dim, kernel_size=1)
        
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.conv2(x)
        
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        
        return x


# ============================================================================
# Attention Modules
# ============================================================================

class RelPosEmb(nn.Module):
    def __init__(self, max_pos_size, dim_head):
        super().__init__()
        self.rel_height = nn.Embedding(2 * max_pos_size - 1, dim_head)
        self.rel_width = nn.Embedding(2 * max_pos_size - 1, dim_head)
        
        deltas = torch.arange(max_pos_size).view(1, -1) - torch.arange(max_pos_size).view(-1, 1)
        rel_ind = deltas + max_pos_size - 1
        self.register_buffer('rel_ind', rel_ind)
    
    def forward(self, q):
        batch, heads, h, w, c = q.shape
        height_emb = self.rel_height(self.rel_ind[:h, :h].reshape(-1))
        width_emb = self.rel_width(self.rel_ind[:w, :w].reshape(-1))
        
        height_emb = rearrange(height_emb, '(x u) d -> x u () d', x=h)
        width_emb = rearrange(width_emb, '(y v) d -> y () v d', y=w)
        
        height_score = einsum('b h x y d, x u v d -> b h x y u v', q, height_emb)
        width_score = einsum('b h x y d, y u v d -> b h x y u v', q, width_emb)
        
        return height_score + width_score


class Attention(nn.Module):
    def __init__(self, *, args, dim, max_pos_size=100, heads=4, dim_head=128):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        
        self.to_qk = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)
        self.pos_emb = RelPosEmb(max_pos_size, dim_head)
    
    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape
        
        q, k = self.to_qk(fmap).chunk(2, dim=1)
        
        q, k = map(lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=heads), (q, k))
        q = self.scale * q
        
        sim = einsum('b h x y d, b h u v d -> b h x y u v', q, k)
        sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')
        attn = sim.softmax(dim=-1)
        
        return attn


class Aggregate(nn.Module):
    def __init__(self, args, dim, heads=4, dim_head=128):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        
        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None
    
    def forward(self, attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape
        
        v = self.to_v(fmap)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        
        if self.project is not None:
            out = self.project(out)
        
        out = fmap + self.gamma * out
        
        return out


# ============================================================================
# Motion Encoder and Update Block
# ============================================================================

class PCBlock4_Deep_nopool_res(nn.Module):
    """Partial Convolution Block"""
    def __init__(self, C_in, C_out, k_conv):
        super().__init__()
        self.conv_list = nn.ModuleList([
            nn.Conv2d(C_in, C_in, kernel, stride=1, padding=kernel//2, groups=C_in) for kernel in k_conv])
        
        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_out, 1, padding=0),
        )
    
    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = self.ffn2(x)
        return x


class SKMotionEncoder6_Deep_nopool_res(nn.Module):
    """Motion feature encoder for MemFlow basic"""
    def __init__(self, args):
        super().__init__()
        cor_planes = 81*4*args.cost_heads_num*2
        self.convc1 = PCBlock4_Deep_nopool_res(cor_planes, 256, k_conv=args.k_conv)
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, k_conv=args.k_conv)
        
        self.convf1_ = nn.Conv2d(4, 128, 1, 1, 0)
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64, k_conv=args.k_conv)
        
        self.conv = PCBlock4_Deep_nopool_res(64+192, 128-4, k_conv=args.k_conv)
    
    def forward(self, flow, corr):
        cor = F.gelu(self.convc1(corr))
        cor = self.convc2(cor)
        
        flo = self.convf1_(flow)
        flo = self.convf2(flo)
        
        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)
        
        return torch.cat([out, flow], dim=1)


class SKUpdateBlock6_Deep_nopoolres_AllDecoder(nn.Module):
    """Update block for iterative refinement"""
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args
        
        args.k_conv = [1, 15]
        args.PCUpdater_conv = [1, 7]
        
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        self.gru = PCBlock4_Deep_nopool_res(128+hidden_dim+hidden_dim+128, 128, k_conv=args.PCUpdater_conv)
        self.flow_head = PCBlock4_Deep_nopool_res(128, 4, k_conv=args.k_conv)
        
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9*2, 1, padding=0))
        
        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=1)
    
    def forward(self, net, inp, corr, flow, attention):
        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)
        
        # Attentional update
        net = self.gru(torch.cat([net, inp_cat], dim=1))
        delta_flow = self.flow_head(net)
        
        # scale mask to balance gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow


class SKUpdateBlock6_Deep_nopoolres_AllDecoder_SingleHead(nn.Module):
    """Update block for iterative refinement (single-head, 2-channel output)"""
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args
        
        args.k_conv = [1, 15]
        args.PCUpdater_conv = [1, 7]
        
        # Single-head encoder (324 channels input)
        self.encoder = SKMotionEncoder6_Deep_nopool_res_SingleHead(args)
        self.gru = PCBlock4_Deep_nopool_res(128+hidden_dim+hidden_dim+128, 128, k_conv=args.PCUpdater_conv)
        self.flow_head = PCBlock4_Deep_nopool_res(128, 2, k_conv=args.k_conv)  # 2-channel output
        
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))  # Single set of masks
        
        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=1)
    
    def forward(self, net, inp, corr, flow, attention):
        motion_features = self.encoder(flow, corr)
        
        # For single-head model without memory, skip aggregator when attention is None
        if attention is not None:
            motion_features_global = self.aggregator(attention, motion_features)
        else:
            motion_features_global = motion_features  # Use motion features directly
        
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)
        
        # Attentional update
        net = self.gru(torch.cat([net, inp_cat], dim=1))
        delta_flow = self.flow_head(net)
        
        # scale mask to balance gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow


class SKMotionEncoder6_Deep_nopool_res_SingleHead(nn.Module):
    """Motion feature encoder for MemFlow basic (single-head)"""
    def __init__(self, args):
        super().__init__()
        # Single-head: 81*4*1 = 324 channels
        cor_planes = 81*4*args.cost_heads_num
        self.convc1 = PCBlock4_Deep_nopool_res(cor_planes, 256, k_conv=args.k_conv)
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, k_conv=args.k_conv)
        
        self.convf1_ = nn.Conv2d(2, 128, 1, 1, 0)  # 2-channel flow input
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64, k_conv=args.k_conv)
        
        self.conv = PCBlock4_Deep_nopool_res(64+192, 128-2, k_conv=args.k_conv)  # Output 126 + 2 = 128
    
    def forward(self, flow, corr):
        cor = F.gelu(self.convc1(corr))
        cor = self.convc2(cor)
        
        flo = self.convf1_(flow)
        flo = self.convf2(flo)
        
        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)
        
        return torch.cat([out, flow], dim=1)


class SKMotionEncoder6_Deep_nopool_res_Mem_skflow(nn.Module):
    """Motion feature encoder for MemFlow"""
    def __init__(self, args):
        super().__init__()
        self.cor_planes = cor_planes = (args.corr_radius * 2 + 1) ** 2 * args.cost_heads_num * args.corr_levels
        self.convc1 = PCBlock4_Deep_nopool_res(cor_planes, 256, k_conv=args.k_conv)
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, k_conv=args.k_conv)
        
        self.convf1 = nn.Conv2d(2, 128, 1, 1, 0)
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64, k_conv=args.k_conv)
        
        self.conv = PCBlock4_Deep_nopool_res(64+192, 128-2, k_conv=args.k_conv)
    
    def forward(self, flow, corr):
        cor = F.gelu(self.convc1(corr))
        cor = self.convc2(cor)
        
        flo = self.convf1(flow)
        flo = self.convf2(flo)
        
        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)
        
        return torch.cat([out, flow], dim=1)


class SKUpdateBlock6_Deep_nopoolres_AllDecoder2_Mem_predict(nn.Module):
    """Update block for flow prediction"""
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args
        
        args.k_conv = [1, 15]
        args.PCUpdater_conv = [1, 7]
        
        self.encoder = SKMotionEncoder6_Deep_nopool_res_Mem_skflow(args)
        if hasattr(self.args, 'concat_flow') and self.args.concat_flow:
            print('input forward warped flow to GRU')
            self.gru_new = PCBlock4_Deep_nopool_res(128 + hidden_dim + 2, 128, k_conv=args.PCUpdater_conv)
        else:
            self.gru_new = PCBlock4_Deep_nopool_res(128 + hidden_dim, 128, k_conv=args.PCUpdater_conv)
        self.flow_head = PCBlock4_Deep_nopool_res(128, 2, k_conv=args.k_conv)
        
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        
        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=1)
    
    def get_motion_and_value(self, flow, corr):
        motion_features = self.encoder(flow, corr)
        value = self.aggregator.to_v(motion_features)
        return motion_features, value
    
    def forward(self, inp, motion_features_global):
        inp_cat = torch.cat([inp, motion_features_global], dim=1)
        
        # Attentional update
        net = self.gru_new(inp_cat)
        delta_flow = self.flow_head(net)
        
        # scale mask to balance gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow


# ============================================================================
# MemFlowNet Main Model
# ============================================================================

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except:
    HAS_FLASH_ATTN = False
    print('[MemFlowNet] Flash attention not available, using standard attention')


class MemFlowNet_P(nn.Module):
    """MemFlow P model for optical flow completion"""
    def __init__(self, pretrained_path=None):
        super().__init__()
        
        # Create config object
        class Config:
            def __init__(self):
                self.corr_radius = 4
                self.corr_levels = 4
                self.cost_heads_num = 1
                self.k_conv = [1, 15]
                self.PCUpdater_conv = [1, 7]
                self.concat_flow = True  # P model uses concat_flow=True
                self.cnet = 'basicencoder'
                self.fnet = 'basicencoder'
                self.gma = 'GMA-SK2'
                self.corr_fn = 'default'
                self.train_avg_length = 10
        
        self.cfg = Config()
        self.hidden_dim = 128
        self.context_dim = 128
        
        # Feature network and context network
        print("[Using basicencoder as feature encoder]")
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')
        
        print("[Using basicencoder as context encoder]")
        self.cnet = BasicEncoder(output_dim=128, norm_fn='batch')  # Output 128 for P model
        
        # Update block
        print("[Using GMA-SK2 for flow completion]")
        self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder2_Mem_predict(
            args=self.cfg, hidden_dim=128)
        
        # Attention module
        self.att = Attention(
            args=self.cfg, dim=self.context_dim, heads=1, 
            max_pos_size=160, dim_head=self.context_dim)
        
        self.train_avg_length = self.cfg.train_avg_length
        self.motion_prompt = nn.Parameter(torch.randn(1, 128, 1, 1))
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)
    
    def load_pretrained(self, path):
        """Load pretrained weights"""
        print(f"[MemFlowNet] Loading pretrained weights from {path}")
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        
        # Handle different state dict formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"[MemFlowNet] Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"[MemFlowNet] Unexpected keys: {unexpected_keys}")
        print("[MemFlowNet] Pretrained weights loaded successfully")
    
    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)
        return coords0, coords1
    
    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        
        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)
    
    def encode_features(self, frame, flow_init=None):
        """Encode image features"""
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError
        
        fmaps = self.fnet(frame).float()
        if need_reshape:
            # B*T*C*H*W
            fmaps = fmaps.view(b, t, *fmaps.shape[-3:])
            frame = frame.view(b, t, *frame.shape[-3:])
            coords0, coords1 = self.initialize_flow(frame[:, 0, ...])
        else:
            coords0, coords1 = self.initialize_flow(frame)
        if flow_init is not None:
            coords1 = coords1 + flow_init
        
        return coords0, coords1, fmaps
    
    def encode_context(self, frame):
        """Encode context features"""
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError
        
        # shape is b*c*h*w
        inp = self.cnet(frame)
        inp = torch.relu(inp)
        query, key = self.att.to_qk(inp).chunk(2, dim=1)
        
        if need_reshape:
            # B*C*T*H*W
            query = query.view(b, t, *query.shape[-3:]).transpose(1, 2).contiguous()
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
            # B*T*C*H*W
            inp = inp.view(b, t, *inp.shape[-3:])
        
        return query, key, inp
    
    def get_motion_feature(self, flow, coords1, fmaps):
        """Extract motion features from correlation volume"""
        corr_fn = CorrBlock(fmaps[:, 0, ...], fmaps[:, 1, ...],
                            num_levels=self.cfg.corr_levels, radius=self.cfg.corr_radius)
        corr = corr_fn(coords1 + flow)  # index correlation volume
        _, current_value = self.update_block.get_motion_and_value(flow, corr)
        return current_value
    
    def predict_flow(self, inp, query, ref_keys, value, forward_warp_flow=None, test_mode=False):
        """Predict flow from context and motion features"""
        B, _, H, W = inp.shape
        
        if ref_keys is not None and value is not None and HAS_FLASH_ATTN and inp.is_cuda:
            query = query.flatten(start_dim=2).permute(0, 2, 1).unsqueeze(2)
            ref_keys = ref_keys.flatten(start_dim=2).permute(0, 2, 1).unsqueeze(2)
            # get global motion
            # B, L, N, C
            value = value.flatten(start_dim=2).permute(0, 2, 1).unsqueeze(2)
            scale = self.att.scale * math.log(max(ref_keys.shape[1], 2), max(self.train_avg_length, 2))
            # FlashAttention requires fp16 or bf16
            orig_dtype = query.dtype
            query_h = query.half()
            ref_keys_h = ref_keys.half()
            value_h = value.half()
            hidden_states = flash_attn_func(query_h, ref_keys_h, value_h, dropout_p=0.0, 
                                           softmax_scale=scale, causal=False)
            hidden_states = hidden_states.to(orig_dtype).squeeze(2).permute(0, 2, 1).reshape(B, -1, H, W)
            
            motion_features_global = (self.motion_prompt.repeat(B, 1, H, W) + 
                                     self.update_block.aggregator.gamma * hidden_states)
        elif ref_keys is not None and value is not None:
            # Standard attention fallback (CPU or no FlashAttention)
            # ref_keys/value may already be in (B, L, 1, C) flash format from FlowEstimator
            if ref_keys.dim() == 4 and ref_keys.shape[2] == 1:
                # Already in flash format: (B, L, 1, C) → (B, L, C)
                ref_k = ref_keys.squeeze(2)
                ref_v = value.squeeze(2)
            else:
                ref_k = ref_keys.flatten(start_dim=2).permute(0, 2, 1)
                ref_v = value.flatten(start_dim=2).permute(0, 2, 1)
            q = query.flatten(start_dim=2).permute(0, 2, 1)  # (B, L_q, C)
            scale = self.att.scale * math.log(max(ref_k.shape[1], 2), max(self.train_avg_length, 2))
            attn = torch.matmul(q * scale, ref_k.transpose(-1, -2))  # (B, L_q, L_k)
            attn = attn.softmax(dim=-1)
            hidden_states = torch.matmul(attn, ref_v)  # (B, L_q, C)
            hidden_states = hidden_states.permute(0, 2, 1).reshape(B, -1, H, W)
            
            motion_features_global = (self.motion_prompt.repeat(B, 1, H, W) + 
                                     self.update_block.aggregator.gamma * hidden_states)
        else:
            motion_features_global = self.motion_prompt.repeat(B, 1, H, W)
        
        if hasattr(self.cfg, 'concat_flow') and self.cfg.concat_flow:
            # If forward_warp_flow is not provided, use zero flow
            if forward_warp_flow is None:
                forward_warp_flow = torch.zeros(B, 2, H, W, device=inp.device, dtype=inp.dtype)
            motion_features_global = torch.cat([motion_features_global, forward_warp_flow], dim=1)
        
        _, up_mask, delta_flow = self.update_block(inp, motion_features_global)
        
        if hasattr(self.cfg, 'concat_flow') and self.cfg.concat_flow:
            delta_flow = delta_flow + forward_warp_flow
        
        # upsample predictions
        flow_up = self.upsample_flow(delta_flow, up_mask)
        
        return flow_up
    
    def forward(self, image1, image2, ref_keys=None, ref_values=None, test_mode=True):
        """
        Forward pass for flow completion
        Args:
            image1: First image (B, 3, H, W) in range [0, 255]
            image2: Second image (B, 3, H, W) in range [0, 255]
            ref_keys: Optional reference keys for memory
            ref_values: Optional reference values for memory
            test_mode: Whether in test mode
        Returns:
            flow: Optical flow (B, 2, H, W)
        """
        # Normalize images to [-1, 1]
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        
        # Encode features
        b = image1.shape[0]
        frames = torch.stack([image1, image2], dim=1)  # B, 2, C, H, W
        
        coords0, coords1, fmaps = self.encode_features(frames)
        query, key, inp = self.encode_context(frames)
        
        # Use only current frame context for prediction
        inp = inp[:, 0, ...]  # B, C, H, W
        query = query[:, :, 0, ...]  # B, C, H, W
        
        # Predict flow
        flow = self.predict_flow(inp, query, ref_keys, ref_values, test_mode=test_mode)
        
        return flow


def create_memflow(pretrained_path=None):
    """Helper function to create MemFlowNet_P model"""
    return MemFlowNet_P(pretrained_path=pretrained_path)


# ============================================================================
# MemFlowNet Basic Model (with iterative refinement)
# ============================================================================

class MemFlowNet(nn.Module):
    """MemFlow basic model for optical flow estimation with iterative refinement"""
    def __init__(self, pretrained_path=None, decoder_depth=12, model_type='single'):
        """
        Args:
            pretrained_path: Path to pretrained weights
            decoder_depth: Number of refinement iterations
            model_type: 'single' for single-head (2-channel flow output) or 
                       'double' for double-head (4-channel flow+mask output)
        """
        super().__init__()
        
        # Create config object based on model type
        class Config:
            def __init__(self, model_type):
                self.corr_radius = 4
                self.corr_levels = 4
                self.k_conv = [1, 15]
                self.PCUpdater_conv = [1, 7]
                self.cnet = 'basicencoder'
                self.fnet = 'basicencoder'
                self.corr_fn = 'default'
                self.train_avg_length = 10
                self.decoder_depth = decoder_depth
                self.model_type = model_type
                
                # Model-specific configurations
                if model_type == 'single':
                    # Single-head: 1 correlation head, 2-channel flow output
                    self.cost_heads_num = 1
                    self.gma = 'GMA-SK'
                    print("[Config] Single-head model: 1 corr head, 2-channel output")
                elif model_type == 'double':
                    # Double-head: 2 correlation heads, 4-channel flow+mask output  
                    self.cost_heads_num = 2
                    self.gma = 'GMA-SK2'
                    print("[Config] Double-head model: 2 corr heads, 4-channel output")
                else:
                    raise ValueError(f"Unknown model_type: {model_type}, must be 'single' or 'double'")
        
        self.cfg = Config(model_type)
        self.model_type = model_type
        self.hidden_dim = 128
        self.context_dim = 128
        
        # Feature network and context network
        print("[Using basicencoder as feature encoder]")
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')
        
        print("[Using basicencoder as context encoder]")
        self.cnet = BasicEncoder(output_dim=256, norm_fn='batch')
        
        # Update block based on model type
        print(f"[Using {self.cfg.gma} with {decoder_depth} iterations]")
        
        if model_type == 'single':
            # Use single-head update block
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder_SingleHead(
                args=self.cfg, hidden_dim=128)
        else:
            # Use double-head update block  
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder(
                args=self.cfg, hidden_dim=128)
        
        # Attention module
        self.att = Attention(
            args=self.cfg, dim=self.context_dim, heads=1, 
            max_pos_size=160, dim_head=self.context_dim)
        
        self.train_avg_length = self.cfg.train_avg_length
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)
    
    def load_pretrained(self, path):
        """Load pretrained weights"""
        print(f"[MemFlowNet] Loading pretrained weights from {path}")
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        
        # Handle different state dict formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Remove 'module.' prefix and fix key names
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove module prefix
            if k.startswith('module.'):
                k = k[7:]
            
            # Fix key name mismatch: convf1 -> convf1_ (for single-head model)
            if 'update_block.encoder.convf1.weight' in k or 'update_block.encoder.convf1.bias' in k:
                k = k.replace('.convf1.', '.convf1_.')
            
            new_state_dict[k] = v
        
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"[MemFlowNet] Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"[MemFlowNet] Unexpected keys: {unexpected_keys}")
        print("[MemFlowNet] Pretrained weights loaded successfully")
    
    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)
        return coords0, coords1
    
    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        
        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)
    
    def encode_features(self, frame):
        """Encode image features"""
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError
        
        fmaps = self.fnet(frame).float()
        if need_reshape:
            # B*T*C*H*W
            fmaps = fmaps.view(b, t, *fmaps.shape[-3:])
            frame = frame.view(b, t, *frame.shape[-3:])
            coords0, coords1 = self.initialize_flow(frame[:, 0, ...])
        else:
            coords0, coords1 = self.initialize_flow(frame)
        
        return coords0, coords1, fmaps
    
    def encode_context(self, frame):
        """Encode context features"""
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError
        
        # shape is b*c*h*w
        cnet = self.cnet(frame)
        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        query, key = self.att.to_qk(inp).chunk(2, dim=1)
        
        if need_reshape:
            # B*C*T*H*W
            query = query.view(b, t, *query.shape[-3:]).transpose(1, 2).contiguous()
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
            # B*T*C*H*W
            net = net. view(b, t, *net.shape[-3:])
            inp = inp.view(b, t, *inp.shape[-3:])
        
        return query, key, net, inp
    
    def predict_flow(self, net, inp, coords0, coords1, fmaps, test_mode=False):
        """Predict flow with iterative refinement"""
        corr_fn = CorrBlock(fmaps[:, 0, ...], fmaps[:, 1, ...],
                            num_levels=self.cfg.corr_levels, radius=self.cfg.corr_radius)
        
        flow_predictions = []
        
        for itr in range(self.cfg.decoder_depth):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            
            # Standard GMA update (without memory)
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, None)
            
            # F(t+1) = F(t) + Δ(t)
            coords1 = coords1 + delta_flow
            
            # upsample predictions
            if test_mode and itr < self.cfg.decoder_depth - 1:
                continue
            
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)
        
        if test_mode:
            return coords1 - coords0, flow_up
        else:
            return flow_predictions
    
    def forward(self, image1, image2, iters=None, test_mode=True):
        """
        Forward pass for flow estimation
        Args:
            image1: First image (B, 3, H, W) in range [0, 255]
            image2: Second image (B, 3, H, W) in range [0, 255]
            iters: Number of iterations (overrides decoder_depth if provided)
            test_mode: Whether in test mode
        Returns:
            flow: Optical flow (B, 2, H, W)
        """
        # Override decoder depth if iters is provided
        if iters is not None:
            original_depth = self.cfg.decoder_depth
            self.cfg.decoder_depth = iters
        
        # Normalize images to [-1, 1]
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        
        # Encode features
        b = image1.shape[0]
        frames = torch.stack([image1, image2], dim=1)  # B, 2, C, H, W
        
        coords0, coords1, fmaps = self.encode_features(frames)
        query, key, net, inp = self.encode_context(frames)
        
        # Use only current frame context for prediction
        net = net[:, 0, ...]  # B, C, H, W
        inp = inp[:, 0, ...]  # B, C, H, W
        
        # Predict flow
        if test_mode:
            _, flow = self.predict_flow(net, inp, coords0, coords1, fmaps, test_mode=True)
        else:
            flow_predictions = self.predict_flow(net, inp, coords0, coords1, fmaps, test_mode=False)
            flow = flow_predictions[-1]
        
        # Restore original decoder depth if it was overridden
        if iters is not None:
            self.cfg.decoder_depth = original_depth
        
        return flow


def create_memflow_basic(pretrained_path=None, decoder_depth=12, model_type='single'):
    """
    Helper function to create MemFlowNet basic model
    
    Args:
        pretrained_path: Path to pretrained weights
        decoder_depth: Number of refinement iterations (default: 12)
        model_type: 'single' for single-head or 'double' for double-head (default: 'single')
    """
    return MemFlowNet(pretrained_path=pretrained_path, decoder_depth=decoder_depth, model_type=model_type)


