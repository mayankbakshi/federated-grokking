import torch
import torch.nn as nn


class ModularTransformer(nn.Module):
    """Small pre-norm transformer for modular arithmetic grokking.

    Vocab: {0, ..., p-1, '='}. The '=' token has id p.
    Input sequence: [x, y, =]. Predicts z at the '=' position.
    """

    def __init__(
        self,
        p: int = 97,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        init_scale: float = 1.0,
        logit_scale: float = 1.0,
    ):
        """Construct the model; both scale factors are 1.0 in the saved study."""
        super().__init__()
        self.p = p
        self.logit_scale = logit_scale
        self.tok_emb = nn.Embedding(p + 1, d_model)
        self.pos_emb = nn.Embedding(3, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, p)
        if init_scale != 1.0:
            with torch.no_grad():
                for param in self.parameters():
                    param.mul_(init_scale)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        eq = torch.full((B,), self.p, dtype=torch.long, device=x.device)
        seq = torch.stack([x, y, eq], dim=1)
        pos = torch.arange(3, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.tok_emb(seq) + self.pos_emb(pos)
        h = self.transformer(h)
        logits = self.head(h[:, -1, :])
        if self.logit_scale != 1.0:
            logits = logits / self.logit_scale
        return logits
