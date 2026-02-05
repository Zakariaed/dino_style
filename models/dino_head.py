import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOHead(nn.Module):
    """DINO projection head with prototypes"""
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class iBOTHead(nn.Module):
    """iBOT prediction head for masked patch prediction"""
    def __init__(self, in_dim, out_dim, patch_out_dim, norm=None, 
                 act='gelu', last_norm=None, nlayers=3, hidden_dim=2048, 
                 bottleneck_dim=256, norm_last_layer=True, shared_head=False):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.shared_head = shared_head
        
        if nlayers == 1:
            if shared_head:
                self.mlp = nn.Linear(in_dim, bottleneck_dim)
            else:
                self.mlp = nn.ModuleList([
                    nn.Linear(in_dim, bottleneck_dim),
                    nn.Linear(in_dim, bottleneck_dim)
                ])
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if norm is not None:
                layers.append(norm(hidden_dim))
            layers.append(nn.GELU() if act == 'gelu' else nn.ReLU())
            
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm is not None:
                    layers.append(norm(hidden_dim))
                layers.append(nn.GELU() if act == 'gelu' else nn.ReLU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            
            if shared_head:
                self.mlp = nn.Sequential(*layers)
            else:
                self.mlp = nn.ModuleList([
                    nn.Sequential(*layers),
                    nn.Sequential(*[layer for layer in layers])
                ])
        
        self.apply(self._init_weights)
        
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
            
        self.last_layer2 = nn.utils.weight_norm(nn.Linear(bottleneck_dim, patch_out_dim, bias=False))
        self.last_layer2.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer2.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        if self.shared_head:
            x = self.mlp(x)
            x = F.normalize(x, dim=-1, p=2)
            x1 = self.last_layer(x)
            x2 = self.last_layer2(x)
        else:
            x1 = self.mlp[0](x)
            x1 = F.normalize(x1, dim=-1, p=2)
            x1 = self.last_layer(x1)
            
            x2 = self.mlp[1](x)
            x2 = F.normalize(x2, dim=-1, p=2)
            x2 = self.last_layer2(x2)
        
        return x1, x2
