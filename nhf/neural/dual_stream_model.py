# -*- coding: utf-8 -*-
"""Dual-stream boundary predictor."""

import torch
from .base_model import ProtocolBoundaryModel as BaseProtocolBoundaryModel


class ProtocolBoundaryModel(BaseProtocolBoundaryModel):
    """Dual-stream boundary model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = "dual_stream"

    def _encode(self, X_BNL, mask_BNL):
        B, N, L = X_BNL.shape
        x = self.embedding(X_BNL.view(-1, L)).permute(1, 0, 2)
        x = self.pos(x)

        key_padding = ~mask_BNL.view(B * N, L)
        h_first = self.h_transformer(x, src_key_padding_mask=key_padding)
        vert = self.vertical(h_first, N, mask_BNL)

        L_dim, BN, D = h_first.shape
        horiz = h_first.view(L_dim, B, N, D)

        vert_filmed = self.film_vert(vert, mask_BNL, X_BNL)
        horiz_filmed = self.film_horiz(horiz, mask_BNL, X_BNL)
        fused_LBD, per_msg_LBND, attn_BLN = self.fusion(vert_filmed, horiz_filmed, mask_BNL)
        return fused_LBD, per_msg_LBND, attn_BLN, (vert, horiz)
