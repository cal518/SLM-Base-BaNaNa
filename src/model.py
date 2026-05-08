# ── RoPE ────────────────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        t     = torch.arange(MAX_SEQ).float()
        freqs = torch.outer(t, inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos()[None, None], persistent=False)
        self.register_buffer('sin', emb.sin()[None, None], persistent=False)

    def forward(self, T, device):
        return self.cos[:, :, :T].to(device), self.sin[:, :, :T].to(device)

def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

def apply_rope(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# ── RMSNorm ─────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(HIDDEN))

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(RMS_EPS).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.w


# ── SwiGLU ──────────────────────────────────────────────────────
class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(HIDDEN, INTER, bias=False)
        self.up   = nn.Linear(HIDDEN, INTER, bias=False)
        self.down = nn.Linear(INTER, HIDDEN, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ── GQA ─────────────────────────────────────────────────────────
class GQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.groups = N_HEADS // N_KV_HEADS
        self.q_proj = nn.Linear(HIDDEN, N_HEADS    * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, N_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, N_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, HIDDEN,    bias=False)
        self.rope   = RotaryEmbedding()

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, N_HEADS,    HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, T, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, T, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        cos, sin = self.rope(T, x.device)
        q, k = apply_rope(q, k, cos, sin)
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


# ── Block ────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1   = RMSNorm()
        self.attn = GQA()
        self.n2   = RMSNorm()
        self.ffn  = SwiGLU()

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x


# ── Modelo ──────────────────────────────────────────────────────
class SLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed   = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.blocks  = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.norm    = RMSNorm()
        self.lm_head = nn.Linear(HIDDEN, VOCAB_SIZE, bias=False)
        if TIE_EMBED:
            self.lm_head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        # Embeddings: escala 1/sqrt(hidden) → logits iniciam perto de zero
        nn.init.normal_(self.embed.weight, std=HIDDEN ** -0.5)
        # Todas as projecoes lineares: N(0, 0.02)
        for name, p in self.named_parameters():
            if p.dim() < 2 or 'embed' in name or 'lm_head' in name:
                continue
            nn.init.normal_(p, std=0.02)
        # Residual scaling em o_proj e down_proj: 1/sqrt(2*L)
        # evita explosao de variancia no residual stream
        scale = (2 * N_LAYERS) ** -0.5
        for b in self.blocks:
            nn.init.normal_(b.attn.o_proj.weight, std=0.02 * scale)
            nn.init.normal_(b.ffn.down.weight,    std=0.02 * scale)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        for b in self.blocks:
            x = b(x)
        logits = self.lm_head(self.norm(x))
        if labels is None:
            return logits
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB_SIZE),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


model = SLM().to(DEVICE).to(DTYPE)
print(f'Parametros: {model.n_params()/1e6:.2f}M')

# Sanidade: loss inicial deve ser ln(32000) ≈ 10.37
with torch.no_grad(), torch.autocast(DEVICE, DTYPE):
    _x = torch.randint(0, VOCAB_SIZE, (2, 64), device=DEVICE)
    _loss = model(_x, _x).item()
    del _x
print(f'Loss inicial: {_loss:.4f}  (esperado ≈ {math.log(VOCAB_SIZE):.4f})')
assert _loss < math.log(VOCAB_SIZE) * 1.2, f'Init ruim: {_loss:.2f}'
