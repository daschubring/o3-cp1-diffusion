import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Generic fully connected multilayer perceptron. Used in single sphere smiley example."""
    def __init__(
        self,
        inDim=4,
        hiddenDims=(256, 256, 256, 256),
        outDim=3,
        activation=nn.SiLU
    ):
        super().__init__()

        dims = (inDim,) + tuple(hiddenDims) + (outDim,)

        layers = []

        for i in range(len(dims) - 2):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                activation()
            ]

        layers += [
            nn.Linear(dims[-2], dims[-1])
        ]

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        return self.net(x)
        

class ScalarTimeEmbedding(nn.Module):
    """Shared embedding of the scalar diffusion time."""

    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        # t has shape [B]
        return self.net(t[:, None])       # [B, dim]


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dim):
        super().__init__()

        Conv = nn.Conv1d if dim == 1 else nn.Conv2d

        self.conv1 = Conv(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.conv2 = Conv(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = Conv(
                in_channels,
                out_channels,
                kernel_size=1,
            )

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(x))

        t = self.time_proj(t_emb)

        while t.ndim < h.ndim:
            t = t.unsqueeze(-1)

        h = h + t
        h = self.conv2(F.silu(h))

        return h + self.skip(x)


class ResStack(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_dim,
        num_blocks,
        dim,
    ):
        super().__init__()

        blocks = [
            ResBlock(
                in_channels,
                out_channels,
                time_dim,
                dim,
            )
        ]

        blocks += [
            ResBlock(
                out_channels,
                out_channels,
                time_dim,
                dim,
            )
            for _ in range(num_blocks - 1)
        ]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, t_emb):
        for block in self.blocks:
            x = block(x, t_emb)

        return x

class UNet(nn.Module):
    def __init__(
        self,
        channels=(16, 32, 64),
        time_dim=32,       
        num_blocks=(2, 2, 2),
        dim=1,
    ):
        super().__init__()

        if dim not in (1, 2):
            raise ValueError("dim must be 1 or 2")

        self.dim = dim
        Conv = nn.Conv1d if dim == 1 else nn.Conv2d

        if len(channels) < 2:
            raise ValueError(
                "channels must contain at least two resolutions"
            )

        if isinstance(num_blocks, int):
            if num_blocks < 1:
                raise ValueError("num_blocks must be at least 1")
        
            num_blocks = (num_blocks,) * len(channels)
        
        else:
            num_blocks = tuple(num_blocks)
        
            if len(num_blocks) != len(channels):
                raise ValueError(
                    "num_blocks must have one entry per resolution"
                )
        
            if any(n < 1 for n in num_blocks):
                raise ValueError(
                    "all num_blocks entries must be at least 1"
                )

        self.channels = tuple(channels)
        self.num_down = len(channels) - 1

        # One shared nonlinear embedding of diffusion time.
        self.time_embedding = ScalarTimeEmbedding(time_dim)

        # [B, 3, L] -> [B, channels[0], L]
        self.input_conv = Conv(
            3,
            channels[0],
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )

        # ----------------
        # Encoder
        # ----------------

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for c_in, c_out, n_blocks in zip(
            channels[:-1],
            channels[1:],
            num_blocks[:-1],
        ):
            self.encoder_blocks.append(
                ResStack(
                    c_in,
                    c_in,
                    time_dim,
                    n_blocks,
                    dim,
                )
            )

            self.downsamples.append(
                Conv(
                    c_in,
                    c_out,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    padding_mode="circular",
                )
            )

        # ----------------
        # Bottleneck
        # ----------------

        self.bottleneck = ResStack(
            channels[-1],
            channels[-1],
            time_dim,
            num_blocks[-1],
            dim,
        )

        # ----------------
        # Decoder
        # ----------------

        self.upsample_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for c_coarse, c_fine, n_blocks in zip(
            reversed(channels[1:]),
            reversed(channels[:-1]),
            reversed(num_blocks[:-1]),
        ):
            self.upsample_convs.append(
                Conv(
                    c_coarse,
                    c_fine,
                    kernel_size=3,
                    padding=1,
                    padding_mode="circular",
                )
            )

            # After upsampling we concatenate the encoder skip:
            #
            # c_fine + c_fine = 2*c_fine
            self.decoder_blocks.append(
                ResStack(
                    2 * c_fine,
                    c_fine,
                    time_dim,
                    n_blocks,
                    dim,
                )
            )

        # Return to three ambient components per spin.
        self.output_conv = Conv(
            channels[0],
            3,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )

    def _prepare_time(self, t, batch_size, device, dtype):
        t = torch.as_tensor(t, device=device, dtype=dtype)

        if t.ndim == 0:
            return t.expand(batch_size)

        if t.shape[0] != batch_size:
            raise ValueError(
            f"Expected first dimension of t to be {batch_size}, "
            f"got {t.shape[0]}"
        )
    
        if any(dim != 1 for dim in t.shape[1:]):
            raise ValueError(
                "All dimensions of t after the batch axis must be singleton"
            )
    
        return t.reshape(batch_size)

    def forward(self, x, t):
        expected_ndim = self.dim + 2

        if x.ndim != expected_ndim or x.shape[-1] != 3:
            if self.dim == 1:
                raise ValueError("x must have shape [B, L, 3]")
            else:
                raise ValueError("x must have shape [B, H, W, 3]")
        
        batch_size = x.shape[0]
        spatial_shape = x.shape[1:-1]
        
        factor = 2 ** self.num_down
        
        if any(n % factor != 0 for n in spatial_shape):
            raise ValueError(
                f"All spatial dimensions {spatial_shape} "
                f"must be divisible by {factor}"
            )

        # ----------------
        # Time embedding
        # ----------------

        t = self._prepare_time(
            t,
            batch_size,
            x.device,
            x.dtype,
        )

        t_emb = self.time_embedding(t)

        # Conv1d uses [batch, channels, length].
        h = x.movedim(-1, 1)           # [B, 3, L]

        h = self.input_conv(h)

        # ----------------
        # Encoder
        # ----------------

        skips = []

        for blocks, down in zip(
            self.encoder_blocks,
            self.downsamples,
        ):
            h = blocks(h, t_emb)

            skips.append(h)

            h = down(h)

        # ----------------
        # Bottleneck
        # ----------------

        h = self.bottleneck(h, t_emb)

        # ----------------
        # Decoder
        # ----------------

        for up_conv, blocks, skip in zip(
            self.upsample_convs,
            self.decoder_blocks,
            reversed(skips),
        ):
            h = F.interpolate(
                h,
                scale_factor=2,
                mode="nearest",
            )

            h = up_conv(h)

            h = torch.cat(
                (h, skip),
                dim=1,
            )

            h = blocks(h, t_emb)

        # ----------------
        # Score
        # ----------------

        score = self.output_conv(F.silu(h))
        score = score.movedim(1, -1)
        
        return score