import torch
import torch.nn as nn
import hparams as hp
import utils

from transformer.Models import Encoder, Decoder
from transformer.Layers import Linear, PostNet
from modules import LengthRegulator, CBHG


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Sine(nn.Module):

    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d
        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]
        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)
        for freq in freq_bands:
            for p_fn in self.kwargs[
                "periodic_fns"
            ]:  # p_fn is sin, cos, to define the periodic function
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, disable_pos_enc=False):
    if i == -1:
        return nn.Identity(), 3

    if disable_pos_enc:
        func = lambda x: x
        embed_kwargs = {
            "include_input": True,
            "input_dims": 3,
            "max_freq_log2": multires - 1,
            "num_freqs": multires,
            "log_sampling": True,
            "periodic_fns": [func, func],
        }

        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj: eo.embed(x)
    else:
        embed_kwargs = {
            "include_input": True,
            "input_dims": 3,
            "max_freq_log2": multires - 1,
            "num_freqs": multires,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj: eo.embed(x)

    return embed, embedder_obj.out_dim


class FastSpeech(nn.Module):
    """FastSpeech"""

    def __init__(self):
        super(FastSpeech, self).__init__()
        # TODO This is to put the positional encoding after the decoder
        self.multires = 5
        self.pos_enc, _ = get_embedder(multires=self.multires)
        self.pos_up_dim = nn.Linear((2 * self.multires + 1) * hp.decoder_dim, hp.decoder_dim)
        self.relu_after_pos_up_dim = nn.ReLU()
    
        self.pos_fc = nn.Linear(hp.decoder_dim, hp.decoder_dim)
        self.siren = Sine(1.)

        self.encoder = Encoder()
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder()

        self.mel_linear = Linear(hp.decoder_dim, hp.num_mels)
        self.postnet = CBHG(hp.num_mels, K=8, projections=[256, hp.num_mels])
        self.last_linear = Linear(hp.num_mels * 2, hp.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~utils.get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.0)

    def forward(
        self,
        src_seq,
        src_pos,
        mel_pos=None,
        mel_max_length=None,
        length_target=None,
        alpha=1.0,
    ):
        # src_seq = self.pos_enc(src_seq)
        encoder_output, _ = self.encoder(src_seq, src_pos)

        if self.training:
            length_regulator_output, duration_predictor_output = self.length_regulator(
                encoder_output,
                target=length_target,
                alpha=alpha,
                mel_max_length=mel_max_length,
            )

            decoder_output = self.decoder(length_regulator_output, mel_pos)

            # NOTE SPE STARTS
            decoder_output = self.pos_enc(decoder_output)
            decoder_output = self.pos_fc(decoder_output)
            decoder_output = self.siren(decoder_output)
            # NOTE SPE ENDS

            mel_output = self.mel_linear(decoder_output)
            mel_output = self.mask_tensor(mel_output, mel_pos, mel_max_length)
            residual = self.postnet(mel_output)
            residual = self.last_linear(residual)
            mel_postnet_output = mel_output + residual
            mel_postnet_output = self.mask_tensor( 
                mel_postnet_output, mel_pos, mel_max_length
            )

            return mel_output, mel_postnet_output, duration_predictor_output
        else:
            length_regulator_output, decoder_pos = self.length_regulator(
                encoder_output, alpha=alpha
            )

            decoder_output = self.decoder(length_regulator_output, decoder_pos)

            # NOTE SPE STARTS
            decoder_output = self.pos_enc(decoder_output)
            decoder_output = self.pos_fc(decoder_output)
            decoder_output = self.siren(decoder_output)
            # NOTE SPE ENDS

            mel_output = self.mel_linear(decoder_output)
            residual = self.postnet(mel_output)
            residual = self.last_linear(residual)
            mel_postnet_output = mel_output + residual

            return mel_output, mel_postnet_output


if __name__ == "__main__":
    # Test
    model = FastSpeech()
    print(sum(param.numel() for param in model.parameters()))
