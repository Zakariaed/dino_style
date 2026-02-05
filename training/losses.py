import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # Teacher temp schedule (though teacher is frozen, we still use temp for softmax)
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # Teacher temp from schedule
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class iBOTLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ncrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp_epochs, nepochs, 
                 student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, patch_out_dim))
        
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, student_mask, epoch):
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output
        
        temp = self.teacher_temp_schedule[epoch]
        
        # CLS token loss
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        
        teacher_cls = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls = teacher_cls.detach().chunk(2)
        
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_cls):
            for v in range(len(student_cls_c)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        # Patch loss
        student_patch = student_patch / self.student_temp
        teacher_patch = F.softmax((teacher_patch - self.center2) / temp, dim=-1)
        
        # Only compute loss on masked patches
        mask_bool = student_mask.bool()
        student_patch_masked = student_patch[mask_bool]
        teacher_patch_masked = teacher_patch[:teacher_patch.size(0)//self.ncrops][mask_bool[:teacher_patch.size(0)//self.ncrops]]
        
        patch_loss = torch.sum(-teacher_patch_masked * F.log_softmax(student_patch_masked, dim=-1), dim=-1)
        total_loss += patch_loss.mean()
        n_loss_terms += 1
        
        total_loss /= n_loss_terms
        self.update_center(teacher_cls, teacher_patch)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        batch_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        
        batch_center2 = torch.sum(teacher_patch, dim=0, keepdim=True)
        dist.all_reduce(batch_center2)
        batch_center2 = batch_center2 / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum + batch_center2 * (1 - self.center_momentum)

class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropy loss for uniform distribution"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (batch_size, embed_dim)
        x = F.normalize(x, p=2, dim=-1)
        
        # Compute pairwise distances
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            dist_mat = torch.cdist(x, x, p=2)
            # For each sample, find distance to nearest neighbor (excluding itself)
            dist_mat = dist_mat + torch.eye(dist_mat.size(0), device=dist_mat.device) * 1e6
            min_dist = dist_mat.min(dim=-1)[0]
            # KL divergence approximation
            loss = -torch.log(min_dist + self.eps).mean()
        return loss

class SPDMatrixLoss(nn.Module):
    """SPD Matrix distance loss using Log-Euclidean metric"""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def compute_spd_matrix(self, features):
        """Compute SPD matrix from features (covariance-like)"""
        # features: (batch, dim)
        features = F.normalize(features, p=2, dim=-1)
        # Compute Gram matrix (F @ F^T)
        spd = torch.matmul(features, features.transpose(-2, -1))
        # Add small value to diagonal for numerical stability
        spd = spd + torch.eye(spd.size(-1), device=spd.device) * self.eps
        return spd

    def log_euclidean_distance(self, spd1, spd2):
        """Compute Log-Euclidean distance between two SPD matrices"""
        # Log-Euclidean: ||log(A) - log(B)||_F
        try:
            # Compute matrix logarithm using eigendecomposition
            eigval1, eigvec1 = torch.linalg.eigh(spd1)
            eigval2, eigvec2 = torch.linalg.eigh(spd2)
            
            # Ensure positive eigenvalues
            eigval1 = torch.clamp(eigval1, min=self.eps)
            eigval2 = torch.clamp(eigval2, min=self.eps)
            
            # Compute log of matrices
            log_spd1 = eigvec1 @ torch.diag_embed(torch.log(eigval1)) @ eigvec1.transpose(-2, -1)
            log_spd2 = eigvec2 @ torch.diag_embed(torch.log(eigval2)) @ eigvec2.transpose(-2, -1)
            
            # Frobenius norm of difference
            diff = log_spd1 - log_spd2
            dist = torch.norm(diff, p='fro')
            
        except RuntimeError:
            # Fallback to simpler distance if eigendecomposition fails
            dist = torch.norm(spd1 - spd2, p='fro')
            
        return dist

    def forward(self, student_features, teacher_features):
        """
        Compute SPD matrix distance loss
        Args:
            student_features: (batch, dim)
            teacher_features: (batch, dim)
        """
        student_spd = self.compute_spd_matrix(student_features)
        teacher_spd = self.compute_spd_matrix(teacher_features)
        
        # Compute Log-Euclidean distance
        distance = self.log_euclidean_distance(student_spd, teacher_spd)
        
        # MSE-style loss
        loss = distance ** 2
        return loss

class CLSTokenLoss(nn.Module):
    """Specific loss for CLS token distillation"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_cls, teacher_cls):
        """
        Args:
            student_cls: (batch, dim) - student CLS token embeddings
            teacher_cls: (batch, dim) - teacher CLS token embeddings
        """
        # Normalize
        student_cls = F.normalize(student_cls, p=2, dim=-1)
        teacher_cls = F.normalize(teacher_cls, p=2, dim=-1)
        
        # Cosine similarity loss
        loss = 1 - F.cosine_similarity(student_cls, teacher_cls, dim=-1).mean()
        
        return loss
