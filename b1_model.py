"""B1 extends B without changing the preserved B implementation or its keys."""
import torch
from torch import nn
from torch.nn import functional as F
import abc_fusion_experiment as abc
from b1_history import FEATURE_NAMES


class B1Model(abc.FusionModel):
    def __init__(self, d=64, history_hidden=32):
        super().__init__('B', d=d)
        self.history_mlp = nn.Sequential(nn.Linear(len(FEATURE_NAMES), history_hidden),
                                         nn.ReLU(), nn.Linear(history_hidden, d))
        nn.init.zeros_(self.history_mlp[-1].weight)
        nn.init.zeros_(self.history_mlp[-1].bias)

    def load_b_weights(self, state):
        expected = {k for k in self.state_dict() if not k.startswith('history_mlp.')}
        if set(state) != expected:
            raise ValueError(f'B state keys differ: missing={expected-set(state)}, extra={set(state)-expected}')
        incompatible = self.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys or any(not key.startswith('history_mlp.') for key in incompatible.missing_keys):
            raise ValueError('Unexpected incompatible B checkpoint')
        # Also guarantees exact B behavior when reinitializing an already-used B1 instance.
        nn.init.zeros_(self.history_mlp[-1].weight)
        nn.init.zeros_(self.history_mlp[-1].bias)

    def forward(self, image, image_mask, q_input, gas_input, history):
        q = self.q_encoder(q_input) + self.history_mlp(history)
        identity = self.gas_id(self.gas_indices).unsqueeze(0).expand(len(q), -1, -1)
        gas = self.gas_encoder(torch.cat([gas_input, identity], -1))
        gas_valid = gas_input[..., 1] < .5
        features = self.cnn(image)
        h, w = features.shape[-2:]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=image.device),
                                torch.linspace(-1, 1, w, device=image.device), indexing='ij')
        xy = torch.stack([xx, yy], -1).reshape(h*w, 2)
        pixels = features.flatten(2).transpose(1, 2) + self.position(xy).unsqueeze(0)
        pixel_valid = F.adaptive_max_pool2d(image_mask.float(), (h, w)).flatten(1) > 0
        combined = torch.cat([q, abc.common.masked_mean(gas, gas_valid),
                              abc.common.masked_mean(pixels, pixel_valid),
                              self.gas_readout(gas, q, gas_valid, 'B'),
                              self.image_readout(pixels, q, pixel_valid, 'B')], -1)
        return self.head(combined)
