import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

from vq import STE, EMA, RT, GumbelSoftmax, NSVQ

from diveq import DIVEQ
from sf_diveq import SFDIVEQ
from diveq_detach import DIVEQDetach
from sf_diveq_detach import SFDIVEQDetach

class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()

        self.num_embeddings = int(2 ** args.codebook_bits)
        self.embedding_dim = args.embedding_dim
        self.codebook_optimization = args.codebook_optimization
        self.device = args.device

        self.encoder = Encoder(double_z=False, z_channels=256, resolution=256, in_channels=3, out_ch=3, ch=128,
                               ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0).to(device=self.device)
        self.decoder = Decoder(double_z=False, z_channels=256, resolution=256, in_channels=3, out_ch=3, ch=128,
                               ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0).to(device=self.device)

        if self.codebook_optimization == 'diveq':
            self.vq = DIVEQ(self.num_embeddings, self.embedding_dim)
        elif self.codebook_optimization == 'sfdiveq':
            self.vq = SFDIVEQ(self.num_embeddings, self.embedding_dim)
        elif self.codebook_optimization == 'diveq_detach':
            self.vq = DIVEQDetach(self.num_embeddings, self.embedding_dim)
        elif self.codebook_optimization == 'sfdiveq_detach':
            self.vq = SFDIVEQDetach(self.num_embeddings, self.embedding_dim)
        elif self.codebook_optimization == 'ste':
            self.vq = STE(self.num_embeddings, self.embedding_dim)
        elif self.codebook_optimization == 'rt':
            self.vq = RT(self.num_embeddings, self.embedding_dim)
        elif self.codebook_optimization == 'nsvq':
            self.vq = NSVQ(self.num_embeddings, self.embedding_dim)
        elif self.codebook_optimization == 'gumbel_softmax':
            self.vq = GumbelSoftmax(self.num_embeddings, self.embedding_dim)
        elif self.codebook_optimization == 'ema':
            self.vq = EMA(self.num_embeddings, self.embedding_dim)

        if self.codebook_optimization == 'gumbel_softmax':
            self.quant_conv = nn.Conv2d(self.embedding_dim, self.num_embeddings, 1).to(device=self.device)
        else:
            self.quant_conv = nn.Conv2d(self.embedding_dim, self.embedding_dim, 1).to(device=self.device)

        self.post_quant_conv = nn.Conv2d(self.embedding_dim, self.embedding_dim, 1).to(device=self.device)

        self.latent_list = []
        self.counter = 1

        print(f"VQ method: {self.vq}")

    def forward(self, imgs):
        z = self.encoder(imgs)
        z = self.quant_conv(z)

        z_permute = z.permute(0, 2, 3, 1).contiguous()  # Convert BCHW -> BHWC
        z_shape = z_permute.shape
        z_flat = z_permute.view(-1, self.embedding_dim)  # Flatten the input

        if self.codebook_optimization in ['ste', 'ema', 'gumbel_softmax', 'rt']:
            if self.codebook_optimization == 'gumbel_softmax':
                quantized, indices, perplexity, vq_loss = self.vq(z)
            else:
                quantized, indices, perplexity, vq_loss = self.vq(z_flat)
                quantized = (quantized.view(z_shape)).permute(0, 3, 1, 2).contiguous()  # Convert BHWC -> BCHW
        else:
            quantized, indices, perplexity = self.vq(z_flat)
            quantized = (quantized.view(z_shape)).permute(0, 3, 1, 2).contiguous()  # Convert BHWC -> BCHW
            vq_loss = torch.tensor(0.0, dtype=torch.float32, device=imgs.device)  # as a placeholder

        post_quant_conv_mapping = self.post_quant_conv(quantized)
        decoded_images = self.decoder(post_quant_conv_mapping)

        return decoded_images, indices, perplexity, vq_loss

    def encode(self, imgs):
        z = self.encoder(imgs)
        z = self.quant_conv(z)

        if self.codebook_optimization == 'gumbel_softmax':
            quantized, indices, perplexity = self.vq.inference(z)
            return quantized, indices
        else:
            z_permute = z.permute(0, 2, 3, 1).contiguous()  # Convert BCHW -> BHWC
            z_shape = z_permute.shape
            z_flat = z_permute.view(-1, self.embedding_dim)  # Flatten the input
            if self.codebook_optimization in ['sfdiveq', 'sfdiveq_detach']:
                quantized, indices, perplexity, indices_for_transformer = self.vq.inference(z_flat)
                quantized = (quantized.view(z_shape)).permute(0, 3, 1,2).contiguous()  # Convert BHWC -> BCHW
                return quantized, indices, indices_for_transformer
            else:
                quantized, indices, perplexity = self.vq.inference(z_flat)
                quantized = (quantized.view(z_shape)).permute(0, 3, 1, 2).contiguous()  # Convert BHWC -> BCHW
                return quantized, indices

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.conv_out
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        lda = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        lda = torch.clamp(lda, 0, 1e4).detach()
        return 0.8 * lda

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))