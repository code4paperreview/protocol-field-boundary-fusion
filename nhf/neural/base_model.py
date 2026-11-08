# -*- coding: utf-8 -*-
import math
from typing import List, Optional, Tuple, Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.register_buffer("alpha_t", torch.tensor([1 - alpha, alpha], dtype=torch.float))

    def forward(self, predictions, targets, mask=None):
        if self.alpha_t.device != predictions.device:
            self.alpha_t = self.alpha_t.to(predictions.device)
        ce_loss = F.cross_entropy(
            predictions.view(-1, self.num_classes),
            targets.view(-1),
            reduction="none"
        )
        p_t = torch.exp(-ce_loss)
        alpha_t = self.alpha_t[targets.view(-1)]
        focal = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        if mask is not None:
            focal = focal.view_as(targets) * mask
            return focal.sum() / mask.sum().clamp(min=1.0)
        return focal.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets, mask=None):
        probs = F.softmax(predictions, dim=-1)[..., 1]
        targets_f = targets.float()
        if mask is not None:
            probs = probs * mask
            targets_f = targets_f * mask
        intersection = (probs * targets_f).sum()
        union = probs.sum() + targets_f.sum()
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class BoundaryConsistencyLoss(nn.Module):
    def __init__(self, window_size=3, consistency_weight=1.0):
        super().__init__()
        self.window_size = window_size
        self.consistency_weight = consistency_weight

    def forward(self, predictions, targets, mask=None):
        B, L, _ = predictions.shape
        probs = F.softmax(predictions, dim=-1)[..., 1]
        tot = 0.0
        cnt = 0
        for i in range(L - self.window_size + 1):
            p_win = probs[:, i:i + self.window_size]
            t_win = targets[:, i:i + self.window_size].float()
            if mask is not None:
                m_win = mask[:, i:i + self.window_size].float()
                if m_win.sum() == 0: continue
                p_mean = (p_win * m_win).sum(dim=1) / m_win.sum(dim=1).clamp(min=1.0)
                t_mean = (t_win * m_win).sum(dim=1) / m_win.sum(dim=1).clamp(min=1.0)
                p_var = (((p_win - p_mean.unsqueeze(1)) ** 2) * m_win).sum(dim=1) / m_win.sum(dim=1).clamp(min=1.0)
                t_var = (((t_win - t_mean.unsqueeze(1)) ** 2) * m_win).sum(dim=1) / m_win.sum(dim=1).clamp(min=1.0)
            else:
                p_var = p_win.var(dim=1)
                t_var = t_win.var(dim=1)
            tot += F.mse_loss(p_var, t_var, reduction="mean")
            cnt += 1
        if cnt == 0: return torch.tensor(0.0, device=predictions.device)
        return (tot / cnt) * self.consistency_weight


class UnsupervisedSetLoss(nn.Module):
    def __init__(self, w_cons=1.0, w_smooth=0.15, w_sparse=0.05):
        super().__init__()
        self.w_cons = w_cons
        self.w_smooth = w_smooth
        self.w_sparse = w_sparse

    def forward(self, emissions_per_msg, feats, mask):
        B, N, L, _ = emissions_per_msg.shape
        p = torch.softmax(emissions_per_msg, dim=-1)[..., 1]
        m = mask.float()
        denom = m.sum(dim=1).clamp(min=1.0)
        bar_p = (p * m).sum(dim=1) / denom
        eps = 1e-6
        bar = bar_p.unsqueeze(1).expand_as(p)
        p_clamped = (p * m + eps) / (m + eps)
        bar_clamped = (bar * m + eps) / (m + eps)
        kl1 = (p_clamped * (p_clamped.add(eps).log() - bar_clamped.add(eps).log())) * m
        kl2 = (bar_clamped * (bar_clamped.add(eps).log() - p_clamped.add(eps).log())) * m
        L_cons = (kl1 + kl2).sum() / m.sum().clamp(min=1.0)
        f = (feats * m.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)
        df = (f[:, 1:, :] - f[:, :-1, :]).pow(2).sum(dim=-1)
        w = (1.0 - bar_p)[:, 1:] * (1.0 - bar_p)[:, :-1]
        L_smooth = (w * df).sum() / w.sum().clamp(min=1.0)
        L_sparse = bar_p.mean()
        return self.w_cons * L_cons + self.w_smooth * L_smooth + self.w_sparse * L_sparse


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        L = x.size(0)
        return x + self.pe[:L] # type: ignore


class AlignmentEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.emb2 = nn.Embedding(2, d_model)
        self.emb4 = nn.Embedding(4, d_model)
        self.emb8 = nn.Embedding(8, d_model)
        nn.init.normal_(self.emb2.weight, std=0.02)
        nn.init.normal_(self.emb4.weight, std=0.02)
        nn.init.normal_(self.emb8.weight, std=0.02)

    def forward(self, x):
        L, Bx, D = x.shape
        idx = torch.arange(L, device=x.device)
        add = self.emb2(idx % 2) + self.emb4(idx % 4) + self.emb8(idx % 8)
        return x + add.unsqueeze(1)


class MultiScaleLocalWindowEncoder(nn.Module):
    def __init__(self, d_model, window_sizes=[3, 5, 7], hidden_dim=128):
        super().__init__()
        self.num_scales = len(window_sizes)
        self.encoders = nn.ModuleList()
        out_each = d_model // self.num_scales
        self.proj_in_dim = out_each * self.num_scales
        for k in window_sizes:
            pad = (k - 1) // 2
            self.encoders.append(
                nn.Sequential(
                    nn.Conv1d(d_model, hidden_dim, kernel_size=k, padding=pad),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Conv1d(hidden_dim, out_each, kernel_size=1),
                )
            )
        self.proj = nn.Linear(self.proj_in_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x1 = x.permute(1, 2, 0)
        outs = [enc(x1) for enc in self.encoders]
        o = torch.cat(outs, dim=1).permute(2, 0, 1)
        o = self.proj(o)
        o = self.norm(o + residual)
        return o

class SetTemplateFiLM(nn.Module):
    def __init__(self, d_model, hidden=256, window_sizes=[3, 5, 9], hist_compress_dim=32):
        super().__init__()
        self.d_model = d_model
        self.window_sizes = window_sizes
        self.hist_compress_dim = hist_compress_dim
        self.eps = 1e-9

        template_dim = (2 * d_model) + 4 + len(self.window_sizes) + self.hist_compress_dim
        dynamic_mlp_input_dim = template_dim + d_model
        self.mlp = nn.Sequential(
            nn.Linear(dynamic_mlp_input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2 * d_model),
        )
        
        self.histogram_compressor = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, self.hist_compress_dim, kernel_size=1)
        )
        
        self.norm = nn.LayerNorm(d_model)

    def _compute_vertical_entropy(self, X_BNL, mask_BNL):
        B, N, L = X_BNL.shape
        X_BLN = X_BNL.permute(0, 2, 1)
        mask_BLN = mask_BNL.permute(0, 2, 1).float()
        
        X_flat = X_BLN.reshape(B * L, N)
        mask_flat = mask_BLN.reshape(B * L, N)
        
        num_valid_bytes = mask_flat.sum(dim=1).clamp(min=self.eps)
        
        one_hot = F.one_hot(X_flat, num_classes=256).float()
        one_hot_masked = one_hot * mask_flat.unsqueeze(-1)
        counts = one_hot_masked.sum(dim=1)
        
        probs = counts / num_valid_bytes.unsqueeze(-1)
        entropy = -torch.sum(probs * torch.log2(probs.clamp(min=self.eps)), dim=-1)
        
        fully_masked_positions = (mask_flat.sum(dim=1) < 1)
        entropy[fully_masked_positions] = 0.0
        
        return entropy.view(B, L)

    def _compute_entropy_features(self, X_BNL, mask_BNL):
        current_entropy = self._compute_vertical_entropy(X_BNL, mask_BNL)
        B, L = current_entropy.shape
        
        entropy_features = torch.zeros(B, L, 4, device=X_BNL.device)
        entropy_features[:, :, 0] = current_entropy
        entropy_features[:, 1:, 1] = current_entropy[:, :-1]
        entropy_features[:, :-1, 2] = current_entropy[:, 1:]
        entropy_features[:, :, 3] = entropy_features[:, :, 0] - entropy_features[:, :, 1]
        return entropy_features

    def _compute_window_entropy_features(self, X_BNL, mask_BNL):
        B, N, L = X_BNL.shape
        all_window_entropies = []

        for k in self.window_sizes:
            padding = (k - 1) // 2
            X_padded = F.pad(X_BNL, (padding, padding), 'constant', 0)
            mask_padded = F.pad(mask_BNL.float(), (padding, padding), 'constant', 0)
            
            X_windows = X_padded.unfold(2, k, 1)
            mask_windows = mask_padded.unfold(2, k, 1)

            X_windows_flat = X_windows.permute(0, 2, 1, 3).reshape(B * L, N * k)
            mask_windows_flat = mask_windows.permute(0, 2, 1, 3).reshape(B * L, N * k)

            num_valid_bytes = mask_windows_flat.sum(dim=1).clamp(min=self.eps)
            
            one_hot = F.one_hot(X_windows_flat, num_classes=256).float()
            one_hot_masked = one_hot * mask_windows_flat.unsqueeze(-1)
            counts = one_hot_masked.sum(dim=1)
            
            probs = counts / num_valid_bytes.unsqueeze(-1)
            entropy = -torch.sum(probs * torch.log2(probs.clamp(min=self.eps)), dim=-1)
            
            fully_masked_windows = (mask_windows_flat.sum(dim=1) < 1)
            entropy[fully_masked_windows] = 0.0

            all_window_entropies.append(entropy.view(B, L))
            
        return torch.stack(all_window_entropies, dim=-1)

    def _compute_histogram_features(self, X_BNL, mask_BNL):
        B, N, L = X_BNL.shape
        X_BLN = X_BNL.permute(0, 2, 1)
        mask_BLN = mask_BNL.permute(0, 2, 1).float()
        
        X_flat = X_BLN.reshape(B * L, N)
        mask_flat = mask_BLN.reshape(B * L, N)
        
        num_valid_bytes = mask_flat.sum(dim=1).clamp(min=self.eps)
        
        one_hot = F.one_hot(X_flat, num_classes=256).float()
        one_hot_masked = one_hot * mask_flat.unsqueeze(-1)
        counts = one_hot_masked.sum(dim=1)
        
        hist = counts / num_valid_bytes.unsqueeze(-1)
        
        fully_masked_positions = (mask_flat.sum(dim=1) < 1)
        hist[fully_masked_positions] = 0.0
        
        hist = hist.view(B, L, 256)
        
        hist_compressed = self.histogram_compressor(hist.permute(0, 2, 1)).permute(0, 2, 1)
        return hist_compressed

    def forward(self, feat_LBND: torch.Tensor, mask_BNL: torch.Tensor, X_BNL: torch.Tensor) -> torch.Tensor:
        feat_BNLD = feat_LBND.permute(1, 2, 0, 3).contiguous()
        m = mask_BNL.float()
        denom_N = m.sum(dim=1, keepdim=True).clamp(min=self.eps)
        
        mean_expanded = (feat_BNLD * m.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom_N.unsqueeze(-1)
        var = ((feat_BNLD - mean_expanded).pow(2) * m.unsqueeze(-1)).sum(dim=1) / denom_N.squeeze(1).unsqueeze(-1)
        mean = mean_expanded.squeeze(1)

        with torch.no_grad():
            entropy_feat = self._compute_entropy_features(X_BNL, mask_BNL)
            window_entropy_feat = self._compute_window_entropy_features(X_BNL, mask_BNL)
            hist_feat_compressed = self._compute_histogram_features(X_BNL, mask_BNL)

        tmpl = torch.cat([mean, var, entropy_feat, window_entropy_feat, hist_feat_compressed], dim=-1)
        B, N, L, D = feat_BNLD.shape
        tmpl_expanded = tmpl.unsqueeze(1).expand(-1, N, -1, -1)
        dynamic_input = torch.cat([tmpl_expanded, feat_BNLD], dim=-1)
        gamma, beta = self.mlp(dynamic_input).chunk(2, dim=-1)
        out_BNLD = feat_BNLD * gamma + beta
        out_BNLD = out_BNLD * m.unsqueeze(-1)
        
        out = out_BNLD.permute(2, 0, 1, 3).contiguous()
        out = self.norm(out)

        if torch.isnan(out).any() or torch.isinf(out).any():
            warnings.warn("NaN/Inf detected in SetTemplateFiLM output. Clamping to zero.", UserWarning)
            out = torch.nan_to_num(out)

        return out


class VerticalAttention(nn.Module):
    def __init__(self, d_model, nhead=2, dropout=0.1, num_layers=1, max_len=4096, window_sizes=[3, 5, 7]):
        super().__init__()
        self.local_window = MultiScaleLocalWindowEncoder(d_model, window_sizes=window_sizes)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=dropout, activation="gelu", batch_first=False)
        self.l_transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.norm = nn.LayerNorm(d_model)
        self.pos_h = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_LBN: torch.Tensor, N: int, mask_BNL: torch.Tensor) -> torch.Tensor:
        L, BN, D = x_LBN.shape
        B = BN // N
        x = self.local_window(x_LBN)
        x = self.pos_h(x)
        x = x + self.l_transformer(x)
        x4 = x.view(L, B, N, D)
        x4r = x4.permute(2, 0, 1, 3).contiguous().view(N, L * B, D)
        valid_N = mask_BNL.permute(0, 2, 1).contiguous().view(B * L, N)
        key_padding_mask = ~valid_N.bool()
        all_pad_rows = ~valid_N.any(dim=1)
        if all_pad_rows.any():
            safe_kpm = key_padding_mask.clone()
            safe_kpm[all_pad_rows] = False
        else:
            safe_kpm = key_padding_mask
        attn_out, _ = self.mha(x4r, x4r, x4r, key_padding_mask=safe_kpm)
        if all_pad_rows.any():
            attn_out[:, all_pad_rows, :] = 0.0
        attn_out = attn_out.view(N, L, B, D).permute(1, 2, 0, 3)
        out = self.norm(x4 + self.dropout(attn_out))
        return out


class HorizontalFlow(nn.Module):
    def __init__(self, d_model, nhead=2, num_layers=1, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=False)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)

    def forward(self, vert_feat: torch.Tensor, mask_BNL: torch.Tensor) -> torch.Tensor:
        L, B, N, D = vert_feat.shape
        x = vert_feat.view(L, B * N, D)
        key_padding = ~mask_BNL.view(B * N, L)
        x = self.pos(x)
        x = self.transformer(x, src_key_padding_mask=key_padding)
        return x.view(L, B, N, D)


class DynamicFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_model * 2, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
        self.msg_attn = nn.Sequential(nn.Linear(d_model, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, vert_feat: torch.Tensor, horiz_feat: torch.Tensor, mask_BNL: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        L, B, N, D = vert_feat.shape
        comb = torch.cat([vert_feat, horiz_feat], dim=-1)
        gate = self.gate(comb)
        per_msg = gate * vert_feat + (1 - gate) * horiz_feat
        logits = self.msg_attn(per_msg).squeeze(-1)
        logits = logits.permute(1, 0, 2).contiguous()
        mask_BLN = mask_BNL.permute(0, 2, 1).contiguous()
        logits = logits.masked_fill(~mask_BLN.bool(), float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        per_msg_BLND = per_msg.permute(1, 0, 2, 3).contiguous()
        fused = torch.einsum("bln,blnd->bld", attn, per_msg_BLND).permute(1, 0, 2).contiguous()
        return fused, per_msg, attn

class ProtocolBoundaryModel(nn.Module):
    def __init__(self,
                 vocab_size=256, d_model=128, nhead=4, num_layers=2, dropout=0.3,
                 max_len=4096,
                 window_sizes=[3, 5, 7],
                 film_window_sizes=[3, 5, 9], hist_compress_dim=32,
                 use_focal_loss=True, focal_alpha=0.25, focal_gamma=2.0,
                 crf_weight=0.6, focal_weight=0.3,
                 use_dice_loss=True, dice_weight=0.2,
                 use_consistency_loss=False, consistency_weight=0.0,
                 use_unsup_loss=False, unsup_weight=0.0,
                 **kwargs: Any):
        super().__init__()

        self.use_focal_loss = bool(use_focal_loss)
        self.use_dice_loss = bool(use_dice_loss)
        self.use_sup_consistency = bool(use_consistency_loss)
        self.use_unsup_loss = bool(use_unsup_loss)
        self.crf_weight, self.focal_weight, self.dice_weight = float(crf_weight), float(focal_weight), float(dice_weight)
        self.sup_consistency_weight, self.unsup_weight = float(consistency_weight), float(unsup_weight)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.align_enc = AlignmentEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=False)
        self.h_transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.vertical = VerticalAttention(d_model=d_model, nhead=nhead, dropout=dropout, num_layers=1, max_len=max_len, window_sizes=window_sizes)
        self.h_flow = HorizontalFlow(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)

        self.film_vert = SetTemplateFiLM(d_model=d_model, hidden=d_model * 2, window_sizes=film_window_sizes, hist_compress_dim=hist_compress_dim)
        self.film_horiz = SetTemplateFiLM(d_model=d_model, hidden=d_model * 2, window_sizes=film_window_sizes, hist_compress_dim=hist_compress_dim)
        
        self.fusion = DynamicFusion(d_model=d_model)
        self.classifier = nn.Linear(d_model, 2)
        self.aux_classifier = nn.Linear(d_model, 2)
        self.crf = CRF(2, batch_first=True)

        if self.use_focal_loss: self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        if self.use_dice_loss: self.dice = DiceLoss()
        if self.use_sup_consistency: self.sup_consistency = BoundaryConsistencyLoss()
        if self.use_unsup_loss: self.unsup = UnsupervisedSetLoss()
        
        self.final_dropout = nn.Dropout(dropout)
        p_prior = 0.15
        with torch.no_grad():
            if self.classifier.bias is not None:
                b = math.log(p_prior / (1.0 - p_prior))
                self.classifier.bias.data[0] = -b
                self.classifier.bias.data[1] = b

    def _encode(self, X_BNL: torch.Tensor, mask_BNL: torch.Tensor):
        B, N, L = X_BNL.shape
        x = self.embedding(X_BNL.view(-1, L)).permute(1, 0, 2)
        x = self.pos(x)
        key_padding = ~mask_BNL.view(B * N, L)
        x = self.h_transformer(x, src_key_padding_mask=key_padding)
        vert = self.vertical(x, N, mask_BNL)
        horiz = self.h_flow(vert, mask_BNL)
        
        vert_filmed = self.film_vert(vert, mask_BNL, X_BNL)
        horiz_filmed = self.film_horiz(horiz, mask_BNL, X_BNL)

        fused_LBD, per_msg_LBND, attn_BLN = self.fusion(vert_filmed, horiz_filmed, mask_BNL)
        return fused_LBD, per_msg_LBND, attn_BLN, (vert, horiz)

    def forward(self, X: torch.Tensor, mask: torch.Tensor, labels: Optional[torch.Tensor] = None):
        B, N, L = X.shape
        fused_LBD, per_msg_LBND, attn_BLN, (vert_orig, horiz_orig) = self._encode(X, mask)
        fused_BLD = fused_LBD.permute(1, 0, 2).contiguous()
        emissions = self.classifier(self.final_dropout(fused_BLD))

        if not torch.isfinite(emissions).all():
            emissions = torch.nan_to_num(emissions, nan=0.0, posinf=1e4, neginf=-1e4)

        sample_mask = mask.any(dim=1).clone()
        has_any = sample_mask.any(dim=1)
        if (~has_any).any():
            fallback = mask[:, 0, :]
            sample_mask = torch.where((~has_any).unsqueeze(-1), fallback, sample_mask)
            still_empty = ~sample_mask.any(dim=1)
            if still_empty.any():
                sample_mask[still_empty, 0] = True
                if labels is not None:
                    labels = labels.clone()
                    labels[still_empty, 0] = 0

        if labels is not None:
            loss = 0.0 * emissions.sum()
            if self.crf_weight > 0: loss += self.crf_weight * -self.crf(emissions, labels.long(), mask=sample_mask)
            if self.use_focal_loss and self.focal_weight > 0: loss += self.focal_weight * self.focal(emissions, labels.long(), mask=sample_mask)
            if self.use_dice_loss and self.dice_weight > 0: loss += self.dice_weight * self.dice(emissions, labels.long(), mask=sample_mask)
            if self.use_sup_consistency and self.sup_consistency_weight > 0: loss += self.sup_consistency_weight * self.sup_consistency(emissions, labels.long(), mask=sample_mask)
            if self.use_unsup_loss and self.unsup_weight > 0:
                feats_BNLD = per_msg_LBND.permute(1, 2, 0, 3).contiguous()
                aux_logits = self.aux_classifier(per_msg_LBND).permute(1, 2, 0, 3).contiguous()
                loss += self.unsup_weight * self.unsup(aux_logits, feats_BNLD, mask)
            return loss
        else:
            with torch.no_grad():
                return self.crf.decode(emissions, mask=sample_mask) # type: ignore

    def forward_per_message(self, X: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """Decode one boundary sequence per message."""
        B, N, L = X.shape
        _, per_msg_LBND, _, _ = self._encode(X, mask)

        # per_msg_LBND: (L, B, N, D) → (B*N, L, D)
        D = per_msg_LBND.shape[-1]
        per_msg_BNL_D = per_msg_LBND.permute(1, 2, 0, 3).reshape(B * N, L, D)
        emissions = self.classifier(per_msg_BNL_D)  # (B*N, L, 2)

        if not torch.isfinite(emissions).all():
            emissions = torch.nan_to_num(emissions, nan=0.0, posinf=1e4, neginf=-1e4)

        # Per-message mask: (B, N, L) → (B*N, L)
        pm_mask = mask.reshape(B * N, L).clone()
        empty = ~pm_mask.any(dim=1)
        if empty.any():
            pm_mask[empty, 0] = True

        with torch.no_grad():
            return self.crf.decode(emissions, mask=pm_mask)

    def predict_boundaries(self, X: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        self.eval()
        return self(X, mask)

