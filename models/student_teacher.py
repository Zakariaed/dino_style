import torch
import torch.nn as nn
from models.vision_transformer import vit_small, vit_large
from models.dino_head import DINOHead, iBOTHead

class MultiCropWrapper(nn.Module):
    """Wrapper to handle multi-crop forward pass"""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, mask=None, return_backbone_feat=False, return_all_tokens=False):
        if not isinstance(x, list):
            x = [x]
        
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        
        start_idx = 0
        output = []
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx:end_idx]), return_all_tokens=return_all_tokens)
            if return_all_tokens:
                output.append(_out)
            else:
                output.append(_out)
            start_idx = end_idx
        
        if return_all_tokens:
            output = torch.cat(output)
            if return_backbone_feat:
                return output
            return self.head(output)
        
        output = torch.cat(output)
        if return_backbone_feat:
            return output
        return self.head(output)

class StudentTeacherModel(nn.Module):
    """
    Student-Teacher model for DINO-style distillation
    Teacher is frozen and initialized with pretrained weights
    Student learns from frozen teacher
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Build student (ViT-Small/16)
        student_backbone = vit_small(patch_size=args.patch_size)
        
        # Build teacher (ViT-Large/16) - FROZEN
        teacher_backbone = vit_large(patch_size=args.patch_size)
        
        # Get embedding dimensions
        student_embed_dim = student_backbone.embed_dim
        teacher_embed_dim = teacher_backbone.embed_dim
        
        # DINO heads
        student_head = DINOHead(
            in_dim=student_embed_dim,
            out_dim=args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
        
        teacher_head = DINOHead(
            in_dim=teacher_embed_dim,
            out_dim=args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
        
        # iBOT heads
        student_ibot_head = iBOTHead(
            in_dim=student_embed_dim,
            out_dim=args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm_last_layer=args.norm_last_layer,
            shared_head=args.shared_head_teacher,
        )
        
        teacher_ibot_head = iBOTHead(
            in_dim=teacher_embed_dim,
            out_dim=args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm_last_layer=args.norm_last_layer,
            shared_head=args.shared_head_teacher,
        )
        
        # Wrap with multi-crop
        self.student = MultiCropWrapper(student_backbone, student_head)
        self.teacher = MultiCropWrapper(teacher_backbone, teacher_head)
        
        self.student_ibot = MultiCropWrapper(student_backbone, student_ibot_head)
        self.teacher_ibot = MultiCropWrapper(teacher_backbone, teacher_ibot_head)
        
        # Load pretrained weights for teacher
        if args.teacher_pretrained_weights:
            self.load_teacher_weights(args.teacher_pretrained_weights)
        
        # Freeze teacher completely (no EMA, just frozen)
        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.teacher_ibot.parameters():
            p.requires_grad = False
        
        print(f"Student: {sum(p.numel() for p in self.student.parameters() if p.requires_grad):,} trainable params")
        print(f"Teacher: {sum(p.numel() for p in self.teacher.parameters()):,} total params (all frozen)")

    def load_teacher_weights(self, pretrained_path):
        """Load pretrained DINOv3 weights into teacher"""
        print(f"Loading teacher weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'teacher' in checkpoint:
            state_dict = checkpoint['teacher']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Load backbone weights
        backbone_state = {k.replace('backbone.', ''): v for k, v in state_dict.items() if 'backbone' in k}
        if len(backbone_state) > 0:
            msg = self.teacher.backbone.load_state_dict(backbone_state, strict=False)
            print(f"Loaded teacher backbone: {msg}")
        
        # Load head weights if available
        head_state = {k.replace('head.', ''): v for k, v in state_dict.items() if 'head' in k}
        if len(head_state) > 0:
            msg = self.teacher.head.load_state_dict(head_state, strict=False)
            print(f"Loaded teacher head: {msg}")

    def forward(self, images, masks=None, return_features=False):
        """
        Forward pass through student and teacher
        Args:
            images: list of crops [global_crop1, global_crop2, local_crop1, ...]
            masks: masks for iBOT (only for global crops)
            return_features: whether to return backbone features
        """
        # Student forward on all crops
        student_output = self.student(images)
        
        # Student iBOT forward (with all tokens for masked prediction)
        student_ibot_output = self.student_ibot(images, return_all_tokens=True)
        
        # Teacher forward only on global crops (first 2)
        with torch.no_grad():
            teacher_output = self.teacher(images[:2])
            teacher_ibot_output = self.teacher_ibot(images[:2], return_all_tokens=True)
        
        if return_features:
            student_feat = self.student(images, return_backbone_feat=True)
            teacher_feat = self.teacher(images[:2], return_backbone_feat=True)
            return student_output, teacher_output, student_ibot_output, teacher_ibot_output, student_feat, teacher_feat
        
        return student_output, teacher_output, student_ibot_output, teacher_ibot_output
