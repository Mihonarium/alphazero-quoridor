import sys
sys.path.append('..')
from utils import *

import argparse
from torch import nn
import torch as t
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.autograd import Variable
from typing import Optional, Union
from einops import rearrange, repeat
from fancy_einsum import einsum
import math
import numpy as np

def layer_init(layer: nn.Linear, row_norm=np.sqrt(2), bias_const=0.0) -> nn.Linear:
    """Initialize the provided linear layer.

    - Each row of the weight has the specified norm
    - Each element of the bias is bias_const.
    """
    t.nn.init.orthogonal_(layer.weight, row_norm)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        assert embedding_size % 2 == 0
        self.embedding_size = embedding_size
        s = t.linspace(0, 1, embedding_size // 2)
        self.div = t.pow(t.tensor(10000), s)
        self.selector = t.linspace(0, embedding_size-1, embedding_size) % 2 == 0

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, ) - for each batch element, the number of noise steps
        Out: shape (batch, embedding_size)
        """
        (b,) = x.shape
        div = repeat(self.div, "d -> b (d f)", b=b, f=2).to(x.device)
        x = repeat(x, "b -> b (d f)", f=2, d=self.embedding_size//2)
        m = x / div
        x = t.where(self.selector.to(x.device), t.sin(m), t.cos(m))
        return x
    
class SinusoidalPositionEmbeddings2D(nn.Module):
    def __init__(self, embedding_size: int, c: int, height: int, width: int):
        super().__init__()
        assert embedding_size % c == 0
        self.embedding_size = embedding_size // c
        self.emb_2d = nn.Parameter(positionalencoding2d(self.embedding_size, height * c, width))
        

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, h, w) - for each batch element, the number of noise steps
        Out: shape (batch, embedding_size, h, w)
        """
        (b, c, h, w) = x.shape
        x = repeat(x, "b c h w -> b d (c h) w", d = self.embedding_size)
        emb_2d = repeat(self.emb_2d.to(x.device), "d (c h) w -> b d (c h) w", b=b, c=c, h=h)        
        return rearrange(x * emb_2d, "b d (c h) w -> b (d c) h w", c=c, h=h)
    
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = t.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = t.exp(t.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = t.arange(0., width).unsqueeze(1)
    pos_h = t.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = t.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = t.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = t.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = t.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

def swish(x: t.Tensor) -> t.Tensor:
    return x * nn.Sigmoid()(x)


class SiLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return swish(x)


class SelfAttention(nn.Module):
    qkv_proj: nn.Linear
    output_proj: nn.Linear
    attn_dropout: nn.Dropout
    resid_dropout: nn.Dropout
    def __init__(self, channels: int, num_heads: int = 4, dropout=0):
        """Self-Attention with two spatial dimensions.

        channels: the number of channels. Should be divisible by the number of heads.
        """
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        head_size = channels // num_heads
        self.qkv_proj = nn.Linear(channels, num_heads * head_size * 3)
        self.output_proj = nn.Linear(num_heads * head_size, channels)
        self.num_heads = num_heads
        self.head_size = head_size
        self.head_size_sqrt = t.tensor(head_size).sqrt()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def split_into_heads(self, x: t.Tensor) -> t.Tensor:
        return rearrange(x, "batch seq (head head_size) -> batch head seq head_size", 
                         head=self.num_heads, head_size=self.head_size)
    
    def attention_pattern_pre_softmax(self, Q: t.Tensor, K: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)
        Return the attention pattern after scaling but before softmax.

        pattern[batch, head, q, k] should be the match between a query at sequence position q and a key at sequence position k.
        """
        Q, K = self.split_into_heads(Q), self.split_into_heads(K)
        pattern = Q @ K.transpose(-1, -2)
        return pattern / self.head_size_sqrt
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        out: shape (batch, channels, height, width)
        """
        B, C, H, W = x.shape
        x = rearrange(x, "batch ch h w -> batch (h w) ch")
        qkv = self.qkv_proj(x)
        Q, K, V = t.split(qkv, dim=-1, split_size_or_sections = self.num_heads * self.head_size)
        attn = self.attention_pattern_pre_softmax(Q, K)
        attn = t.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        V = self.split_into_heads(V)
        #combined_values = t.einsum("bhks, bhqk -> bhqs", V, attn) # attn @ V
        combined_values = attn @ V
        return rearrange(self.output_proj(self.resid_dropout(rearrange(combined_values, 
                    "batch heads seq head_size -> batch seq (heads head_size)"))),
                         "batch (h w) ch -> batch ch h w", h=H, w=W)

class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, dropout=0):
        super().__init__()
        self.group_norm = nn.GroupNorm(1, channels)
        self.attention = SelfAttention(channels, num_heads, dropout)

    def forward(self, x: t.Tensor) -> t.Tensor:
        y = self.group_norm(x)
        y = self.attention(y)
        return x + y
    
class AttnWithLinearBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, dropout=0):
        super().__init__()
        self.attn_block = AttentionBlock(channels, num_heads, dropout)
        self.linear1 = layer_init(nn.Linear(channels, channels))
        self.silu = SiLU()
        self.linear2 = layer_init(nn.Linear(channels, channels))
        self.batchnorm = nn.BatchNorm2d(channels)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: t.Tensor) -> t.Tensor:
        y = self.attn_block(x).transpose(-3, -1)
        y = self.linear1(y)
        y = self.silu(y)
        y = self.linear2(y).transpose(-1, -3)
        y = self.batchnorm(y)
        return x + self.resid_dropout(y)

class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, groups: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            SiLU()
        )
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.conv_block(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, step_dim: int, groups: int):
        """
        input_channels: number of channels in the input to foward
        output_channels: number of channels in the returned output
        step_dim: embedding dimension size for the number of steps
        groups: number of groups in the GroupNorms

        Note that the conv in the left branch is needed if c_in != c_out.
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        if input_channels != output_channels:
            self.res_conv = nn.Conv2d(input_channels, output_channels, 1)
        
        self.conv1 = ConvBlock(input_channels, output_channels, groups)
        
        self.embed_processing = nn.Sequential(
            SiLU(),
            layer_init(nn.Linear(step_dim, output_channels))
        )
        
        self.conv2 = ConvBlock(output_channels, output_channels, groups)

    def forward(self, x: t.Tensor, time_emb: t.Tensor) -> t.Tensor:
        """
        Note that the output of the (silu, linear) block should be of shape (batch, c_out). Since we would like to add this to the output of the first (conv, norm, silu) block, which will have a different shape, we need to first add extra dimensions to the output of the (silu, linear) block.
        """
        conv1_result = self.conv1(x)
        time_emb = time_emb.to(x.device)
        time_emb = self.embed_processing(time_emb).unsqueeze(-1).unsqueeze(-1)
        conv2_result = self.conv2(conv1_result + time_emb)
        if self.input_channels != self.output_channels:
            x = self.res_conv(x)
        return x + conv2_result
    
class DownBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, time_emb_dim: int, groups: int, downsample: bool, num_heads: int = 4, dropout: float = 0):
        super().__init__()
        self.downsample = downsample
        self.res_block1 = ResidualBlock(channels_in, channels_out, time_emb_dim, groups)
        self.res_block2 = ResidualBlock(channels_out, channels_out, time_emb_dim, groups)
        self.attention_block = AttentionBlock(channels_out, num_heads, dropout)
        if downsample:
            self.final_conv = nn.Conv2d(channels_out, channels_out, 4, stride=2, padding=1)

    def forward(self, x: t.Tensor, step_emb: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        """
        x: shape (batch, channels, height, width)
        step_emb: shape (batch, emb)
        Return: (downsampled output, full size output to skip to matching UpBlock)
        """
        res1_result = self.res_block1(x, step_emb)
        res2_result = self.res_block2(res1_result, step_emb)
        attn_result = self.attention_block(res2_result)
        if not self.downsample:
            return attn_result, attn_result
        final_conv_result = self.final_conv(attn_result)
        return final_conv_result, attn_result
    
    

class QuoridorNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        
        print(self.action_size, args, self.board_x, self.board_y)

        super(QuoridorNNet, self).__init__()
        
        # image_shape: tuple[int, int, int],
        #dim_mults = (1, 2, 4, 8)
        dim_mults = (1, 2, 4, 8)
        groups = 1        
        time_emb_dim = 4
        num_heads = 8
        self.time_embedding = t.zeros(time_emb_dim)
        self.pos_emb = SinusoidalPositionEmbeddings2D(args.num_channels, 4, self.board_x, self.board_y)
        channels = args.num_channels
        self.initial_conv = nn.Conv2d(4, channels, 7, padding=3)
        attn_blocks = 2
        self.attn_blocks = nn.ModuleList([
            AttnWithLinearBlock(channels, num_heads, args.dropout) for i in range(attn_blocks)
        ])
        previous_dim_mults = [1] + list(dim_mults)
        self.down_blocks = nn.ModuleList([
            DownBlock(previous_dim_mults[i] * channels, dim_mult * channels, time_emb_dim, groups, i != len(dim_mults)-1, num_heads, args.dropout) for i, dim_mult in enumerate(dim_mults)
            # DownBlock(previous_dim_mults[i] * channels, dim_mult * channels, time_emb_dim, groups, True) for i, dim_mult in enumerate(dim_mults)
        ])
        # self.final_conv = nn.Conv2d(dim_mults[-1] * channels, channels * 8, 4, stride=2, padding=1)
        num_policy_head_layers = 2
        self.policy_head = nn.ModuleList([
            module for i in range(num_policy_head_layers-1) for module in (layer_init(nn.Linear(channels * dim_mults[-1], channels * dim_mults[-1])), nn.Dropout(args.dropout), nn.Tanh())
        ] + [layer_init(nn.Linear(channels * dim_mults[-1], self.action_size), row_norm=0.01)])
        num_value_head_layers = 2
        self.value_head = nn.ModuleList([
            module for i in range(num_value_head_layers-1) for module in (layer_init(nn.Linear(channels * dim_mults[-1], channels * dim_mults[-1])), nn.Dropout(args.dropout), nn.Tanh())
        ] + [layer_init(nn.Linear(channels * dim_mults[-1], 1))])
        
        

    def forward(self, s, logits=False):
        #                                                           s: batch_size*4 x board_x x board_y
        s = s.view(-1, 4, self.board_x, self.board_y)                # batch_size x 4 x board_x x board_y
        x = self.pos_emb(s) + self.initial_conv(s)
        b, d, h, w = x.shape
        for block in self.attn_blocks:
            x = block(x)
        time_emb = repeat(self.time_embedding, "x -> b x", b=b)
        # print("shape before blocks", x.shape)
        # skips = []
        for block in self.down_blocks:
            x, _ = block(x, time_emb)
        # x = self.final_conv(x)
        x = x.squeeze(-1).squeeze(-1)
        pi = x
        v = x
        for policy_layer in self.policy_head: # batch_size x action_size
            pi = policy_layer(pi)
        for value_layer in self.value_head: # batch_size x 1
            v = value_layer(v)
        
        assert (pi.isnan() == 0).all()
        assert (v.isnan() == 0).all()
        if (pi == 0.0).all():
            print(pi, v)
            print("x:", x)
            print("s:", s)
            print(self.pos_emb(s) + self.initial_conv(s))
        if logits:
            return pi, t.tanh(v)
        return F.softmax(pi, dim=1), t.tanh(v)
