import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import DatasetNew
from vq import STE, EMA, RT, Gumbel_Softmax, NSVQ, DIVEQ, SFDIVEQ, DIVEQ_DETACH, SFDIVEQ_DETACH
import argparse
from tqdm import tqdm
import os
from lpips import LPIPS
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from util_funcs import set_temperature

parser = argparse.ArgumentParser(description="VQVAE")
parser.add_argument("--size", type=int, default=256, help='Image height and width')
parser.add_argument('--embedding_dim', type=int, default=512, help='Latent dimension')
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=5.5e-05)
parser.add_argument('--bitrate', type=int, default=10, help='VQ bitrate')
parser.add_argument("--codebook_optimization", type=str, default='sfdiveq', help='method to optimize VQ codebook: "ste", "ema", "rt", "gumbel_softmax", "nsvq", "diveq", "sfdiveq", "diveq_detach", "sfdiveq_detach" ')
parser.add_argument('--codebook-replacement', type=bool, default=True, help='Whether to apply codebook replacement')
parser.add_argument('--replacement_num_batches', type=int, default=500, help='Number of batches after which to apply codebook replacement')
parser.add_argument('--discard_threshold', type=float, default=0.01, help='Threshold (percentage) for discarding unused codebook vectors')
parser.add_argument('--eps', type=float, default=1e-12, help='Epsilon')
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument("--path", type=str, help="path to training set directory")
parser.add_argument("--run_idx", type=int, default=1, help="No. of the experiment")
parser.add_argument("--start_sfvq", type=int, default=2, help="The epoch to start quantizing the latent with SFDIVEQ (only used when using SFDIVEQ or SFDIVEQ_DETACH for training)")
parser.add_argument("--variance", type=float, default=0.001, help="variance of the DIVEQ and SFDIVEQ")
parser.add_argument("--cbr_method", type=str, default='new', help="Codebook replacement technique to use")

args = parser.parse_args()

# path to dataset directory
args.path = r"path/to/dataset/directory"
###############################################################################################
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(False),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(False),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels, num_hiddens // 8, kernel_size=4, stride=2, padding=1)  # 256 -> 128
        self._conv_2 = nn.Conv2d(num_hiddens // 8, num_hiddens // 4, kernel_size=4, stride=2, padding=1)  # 128 -> 64
        self._conv_3 = nn.Conv2d(num_hiddens // 4, num_hiddens // 2, kernel_size=4, stride=2, padding=1)  # 64 -> 32
        if latent_size == 16:
            self._conv_4 = nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1)  # 32 -> 16
        elif latent_size == 32:
            self._conv_4 = nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, x):
        x = F.relu(self._conv_1(x))
        x = F.relu(self._conv_2(x))
        x = F.relu(self._conv_3(x))
        x = F.relu(self._conv_4(x))
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels, num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        if latent_size == 16:
            self._conv_trans_1 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2,padding=1)  # 16 -> 32
        elif latent_size == 32:
            self._conv_trans_1 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=3, stride=1, padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(num_hiddens // 2, num_hiddens // 4, kernel_size=4, stride=2,padding=1)  # 32 -> 64
        self._conv_trans_3 = nn.ConvTranspose2d(num_hiddens // 4, num_hiddens // 8, kernel_size=4, stride=2,padding=1)  # 64 -> 128
        self._conv_trans_4 = nn.ConvTranspose2d(num_hiddens // 8, 3, kernel_size=4, stride=2, padding=1)  # 128 -> 256

    def forward(self, x):
        x = self._conv_1(x)
        x = self._residual_stack(x)
        x = F.relu(self._conv_trans_1(x))
        x = F.relu(self._conv_trans_2(x))
        x = F.relu(self._conv_trans_3(x))
        return self._conv_trans_4(x)

##################################################################

class Model(nn.Module):
    def __init__(self, args, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim):
        super(Model, self).__init__()

        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        if args.codebook_optimization == 'gumbel_softmax':
            self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=num_embeddings, kernel_size=1,stride=1)
        else:
            self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1,stride=1)

        if args.codebook_optimization == 'diveq':
            self._vq_vae = DIVEQ(args, num_embeddings, embedding_dim)
        elif args.codebook_optimization == 'nsvq':
            self._vq_vae = NSVQ(args, num_embeddings, embedding_dim)
        elif args.codebook_optimization == 'diveq_detach':
            self._vq_vae = DIVEQ_DETACH(args, num_embeddings, embedding_dim)
        elif args.codebook_optimization == 'sfdiveq_detach':
            self._vq_vae = SFDIVEQ_DETACH(args, num_embeddings, embedding_dim)
        elif args.codebook_optimization == 'sfdiveq':
            self._vq_vae = SFDIVEQ(args, num_embeddings, embedding_dim)
        elif args.codebook_optimization == 'ste':
            self._vq_vae = STE(args, num_embeddings, embedding_dim)
        elif args.codebook_optimization == 'rt':
            self._vq_vae = RT(args, num_embeddings, embedding_dim)
        elif args.codebook_optimization == 'gumbel_softmax':
            self._vq_vae = Gumbel_Softmax(args, num_embeddings, embedding_dim)
        elif args.codebook_optimization == 'ema':
            self._vq_vae = EMA(args, num_embeddings, embedding_dim, 0.25, 0.99)


        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.latent_list = []
        self.counter = 1
        self.num_embeddings = num_embeddings

    def forward_sfvq(self, x, epoch, batch_idx, num_batches, args):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        if epoch >= args.start_sfvq:
            loss, quantized, perplexity, min_indices = self._vq_vae(z)
            x_recon = self._decoder(quantized)
            return loss, x_recon, perplexity, min_indices
        else:
            if (epoch == args.start_sfvq - 1) and (batch_idx >= num_batches - 50):
                z_temp = (z.permute(0, 2, 3, 1).contiguous()).view(-1, args.embedding_dim)
                self.latent_list.append(z_temp.cpu().detach())

                if batch_idx == num_batches:
                    stacked = torch.stack(self.latent_list[0:-1], dim=0).reshape(-1, args.embedding_dim)
                    latents = torch.cat((stacked, self.latent_list[-1]), dim=0)

                    initial_cb = torch.zeros((self.num_embeddings, args.embedding_dim), dtype=torch.float32)
                    hop_size = int(np.floor(latents.shape[0] / self.num_embeddings))

                    for jj in range(self.num_embeddings):
                        initial_cb[jj] = torch.mean(latents[jj * hop_size:(jj + 1) * hop_size], dim=0)

                    self._vq_vae.codebooks.data = initial_cb.to(args.device)

            x_recon = self._decoder(z)
            return x_recon

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity

    def inference(self,x):
        with torch.no_grad():
            z = self._encoder(x)
            z = self._pre_vq_conv(z)
            quantized = self._vq_vae.inference(z)
            x_recon = self._decoder(quantized)
            return x_recon

#############################################################
dataset_name = 'afhq'
num_hiddens = 256
num_residual_hiddens = 128
num_residual_layers = 6
latent_size = 16

os.makedirs("checkpoints", exist_ok=True)

data_variance = np.load(f'datavar_{dataset_name}.npy')[0]
perceptual_loss = LPIPS().eval().to(device=args.device)

num_embeddings = int(2**args.bitrate)
milestones = [int(args.epochs*0.4), int(args.epochs*0.7)]

dataset = DatasetNew(args)
training_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

model = Model(args, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, args.embedding_dim).to(args.device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=False)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

model.train()

steps_per_epoch = len(training_loader)
total_num_batches = args.epochs * steps_per_epoch
batch_counter = 0
cb_hist_list = []

if args.codebook_optimization in ['sfdiveq_detach', 'sfdiveq']:
    os.makedirs(f"hist_{args.codebook_optimization}", exist_ok=True)
    pdf_file = PdfPages(f'./hist_{args.codebook_optimization}/hist_{args.bitrate}bit_bs{args.batch_size}_lr{args.lr}_var{args.variance}_r{args.run_idx}.pdf')

for epoch in range(args.epochs):
    cb_hist = np.zeros((num_embeddings - 1,))

    if args.codebook_optimization == 'gumbel_softmax':
        current_temp = set_temperature(epoch, max_epoch=args.epochs)
        model._vq_vae.temperature = current_temp

    with tqdm(range(len(training_loader))) as pbar:
        for i, data in zip(pbar, training_loader):
            batch_counter += 1
            data = data.to(args.device)
            optimizer.zero_grad()

            if args.codebook_optimization in ['sfdiveq_detach', 'sfdiveq']:
                if epoch >= args.start_sfvq:
                    vq_loss, data_recon, perplexity, min_indices = model.forward_sfvq(data, epoch, i + 1, steps_per_epoch,args)
                else:
                    data_recon = model.forward_sfvq(data, epoch, i + 1, steps_per_epoch, args)
            else:
                vq_loss, data_recon, perplexity = model.forward(data)

            recon_error = F.mse_loss(data_recon, data) / data_variance
            perceptual_error = torch.mean(perceptual_loss(data, data_recon.clamp(-1,1)))

            if args.codebook_optimization in ['diveq', 'nsvq', 'diveq_detach', 'sfdiveq_detach', 'sfdiveq']:
                loss = recon_error + (1 * perceptual_error)
            elif args.codebook_optimization in ['ste', 'ema', 'gumbel_softmax', 'rt']:
                loss = recon_error + (1 * perceptual_error) + vq_loss

            loss.backward()

            if args.codebook_optimization in ['sfdiveq_detach', 'sfdiveq']:
                if epoch >= args.start_sfvq:
                    cb_hist[torch.unique(min_indices).cpu().detach().numpy()] += 1
                    pbar.set_postfix(Epoch=epoch + 1, Recon_Loss=f"{recon_error.item():.6f}",Perc_Loss=f"{perceptual_error.item():.6f}", Perplexity=f"{perplexity.item():.2f}")
                else:
                    zero_tensor = torch.tensor(0.0, dtype=torch.float32)
                    pbar.set_postfix(Epoch=epoch + 1, Recon_Loss=f"{recon_error.item():.6f}",Perc_Loss=f"{perceptual_error.item():.6f}")
            elif args.codebook_optimization in ['ema', 'rt']:
                pbar.set_postfix(Epoch=epoch + 1, Recon_Loss=f"{recon_error.item():.6f}",Perc_Loss=f"{perceptual_error.item():.6f}",Perplexity=f"{perplexity.item():.2f}")
            else:
                pbar.set_postfix(Epoch=epoch + 1, Recon_Loss=f"{recon_error.item():.6f}", Perc_Loss=f"{perceptual_error.item():.6f}", Perplexity=f"{perplexity.item():.2f}")

            optimizer.step()
            pbar.update(0)

            if args.codebook_optimization in ['diveq', 'nsvq', 'diveq_detach', 'ste', 'ema', 'gumbel_softmax', 'rt']:
                if args.codebook_replacement:
                    if (batch_counter % 100 == 0) & (batch_counter <= 2000):
                        model._vq_vae.codebook_replacement(100)
                    if (batch_counter % args.replacement_num_batches == 0) & (2000 < batch_counter <= total_num_batches - 1000):
                        model._vq_vae.codebook_replacement(args.replacement_num_batches)


    scheduler.step()
    cb_hist_list.append(cb_hist)

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(),f"checkpoints/vqvae_{args.codebook_optimization}_epoch{str(epoch + 1)}_{args.bitrate}bit_bs{args.batch_size}_lr{args.lr}_var{args.variance}_cbr{args.cbr_method}_r{args.run_idx}.pt")


if args.codebook_optimization in ['sfdiveq_detach', 'sfdiveq']:
    for i in range(len(cb_hist_list)):
        num_bars = num_embeddings - 1
        histogram = np.log10(cb_hist_list[i] + 1)
        fig = plt.figure(figsize=(10, 6))
        plt.bar(np.arange(1, num_bars + 1), height=histogram, width=1)
        plt.title(f'SFVQ Codebook Usage Histogram | Epoch={i + 1}')
        pdf_file.savefig(fig, bbox_inches='tight')
    pdf_file.close()
