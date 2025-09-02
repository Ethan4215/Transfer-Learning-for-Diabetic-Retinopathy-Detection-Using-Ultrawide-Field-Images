from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed

# utils/transformer_block_with_attn.py
from timm.models.vision_transformer import Mlp

class BlockWithReturnAttention(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionWithReturn(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, return_attention=False):
        if return_attention:
            x_norm = self.norm1(x)
            attn_output, attn_weights = self.attn(x_norm, return_attention=True)
            x = x + attn_output
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x
        
class AttentionWithReturn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  

        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn_weights = attn.softmax(dim=-1)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        if return_attention:
            return out, attn_weights
        else:
            return out

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone (encoder-only version) """
    def __init__(self, img_size=1024, patch_size=16, in_chans=3,
                 embed_dim=512, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()


        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False) 

        self.blocks = nn.ModuleList([
            BlockWithReturnAttention(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialise pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialise patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialise cls_token
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialise nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add positional embedding
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, imgs):
        latent = self.forward_encoder(imgs)
        return latent
    
    def forward_encoder_with_attention(self, x):
        # Embed patches
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        # Add CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        attn_weights_all = []
        for blk in self.blocks:
            x, attn_weights = blk(x, return_attention=True)
            attn_weights_all.append(attn_weights)
        x = self.norm(x)
        return x, attn_weights_all
 


# Create model function
def mae_vit_encoder2(img_size=1024, patch_size=64, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, **kwargs):
    
    model = MaskedAutoencoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            **kwargs
        )

    return model


# Set recommended architecture
mae_vit_encoder = mae_vit_encoder2