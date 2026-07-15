"""
Gated SAE (Rajamanoharan et al.): separate a latent's on/off decision from its
magnitude, so the L1 penalty shrinks magnitudes without pushing the gate closed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from overcomplete.sae import SAE


class GatedSAE(SAE):
    def __init__(self, input_shape, nb_concepts, encoder_module=None,
                 dictionary_params=None, device="cpu"):
        super().__init__(input_shape=input_shape, nb_concepts=nb_concepts,
                         encoder_module=encoder_module,
                         dictionary_params=dictionary_params, device=device)
        self.r_mag = nn.Parameter(torch.zeros(nb_concepts, device=device))
        self.b_mag = nn.Parameter(torch.zeros(nb_concepts, device=device))
        self.b_gate = nn.Parameter(torch.zeros(nb_concepts, device=device))

    def encode(self, x):
        pre_codes, _ = self.encoder(x)
        pi_gate = pre_codes + self.b_gate
        codes_gate = (pi_gate > 0).float()
        pi_mag = self.r_mag.exp() * pre_codes + self.b_mag
        codes = codes_gate * torch.relu(pi_mag)
        return pi_gate, codes
