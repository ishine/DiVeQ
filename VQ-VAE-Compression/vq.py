import  torch
import torch.nn as nn
from torch.distributions import normal, uniform
import torch.nn.functional as F
from util_funcs import cbr_new, cbr_old
from torch import einsum

class STE(nn.Module):
    def __init__(self, args, num_embeddings, embedding_dim):
        super(STE, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        bitrate = args.bitrate
        self.eps = args.eps
        self.discarding_threshold = args.discard_threshold
        self.device = args.device
        self.cbr_method = args.cbr_method

        codebooks = torch.load(f'initial_cb_{bitrate}bit.pt', weights_only=True)
        # codebooks = uniform.Uniform(-1.0 / self._num_embeddings, 1.0 / self._num_embeddings).sample([self._num_embeddings, self._embedding_dim ])
        self.codebooks = torch.nn.Parameter(codebooks, requires_grad=True)
        self.codebooks_used = torch.zeros(self._num_embeddings, dtype=torch.int32)

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebooks ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codebooks.t()))

        min_indices = torch.argmin(distances, dim=1)

        hard_quantized_input = self.codebooks[min_indices]

        quantized = hard_quantized_input.view(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # Commitment Loss
        q_latent_loss = F.mse_loss(quantized, inputs.detach())  # Codebook Loss
        loss = q_latent_loss + (0.25 * e_latent_loss)

        quantized_final = inputs + (quantized - inputs).detach()

        # Perplexity Computation
        encoding_indices = min_indices.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=self.device)
        encodings.scatter_(1, encoding_indices, 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        with torch.no_grad():
            self.codebooks_used[min_indices.cpu()] += 1

        # convert quantized from BHWC -> BCHW
        return loss, quantized_final.permute(0, 3, 1, 2).contiguous(), perplexity

    def inference(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebooks ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codebooks.t()))
        min_indices = torch.argmin(distances, dim=1)
        hard_quantized_input = self.codebooks[min_indices]
        quantized = hard_quantized_input.view(input_shape)
        return quantized.permute(0, 3, 1, 2).contiguous()


    def codebook_replacement(self, num_batches):
        if self.cbr_method == 'new':
            self.codebooks_used, self.codebooks = cbr_new(self.codebooks_used, self.codebooks, num_batches, self.discarding_threshold, self.eps, self._embedding_dim)
        elif self.cbr_method == 'old':
            self.codebooks_used, self.codebooks = cbr_old(self.codebooks_used, self.codebooks, num_batches,self.discarding_threshold, self.eps, self._embedding_dim)


class RT(nn.Module):
    def __init__(self, args, num_embeddings, embedding_dim):
        super(RT, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        bitrate = args.bitrate
        self.eps = args.eps
        self.discarding_threshold = args.discard_threshold
        self.device = args.device
        self.cbr_method = args.cbr_method

        codebooks = torch.load(f'initial_cb_{bitrate}bit.pt', weights_only=True)
        # codebooks = uniform.Uniform(-1.0 / self._num_embeddings, 1.0 / self._num_embeddings).sample([self._num_embeddings, self._embedding_dim ])
        self.codebooks = torch.nn.Parameter(codebooks, requires_grad=True)
        self.codebooks_used = torch.zeros(self._num_embeddings, dtype=torch.int32)

    @staticmethod
    def get_very_efficient_rotation(u, q, e):
        w = ((u + q) / torch.norm(u + q, dim=1, keepdim=True)).detach()
        e = e - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)), w.unsqueeze(1)) + 2 * torch.bmm(
            torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
        return e

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        flat_input = inputs.view(-1, self._embedding_dim)

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebooks ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codebooks.t()))

        min_indices = torch.argmin(distances, dim=1)

        quantized = self.codebooks[min_indices]
        quantized_reshaped = quantized.view(input_shape)
        pre_norm_q = self.get_very_efficient_rotation(flat_input / (torch.norm(flat_input, dim=1, keepdim=True) + 1e-6),
                                                      quantized / (torch.norm(quantized, dim=1, keepdim=True) + 1e-6),
                                                      flat_input.unsqueeze(1)).squeeze()
        q_tilde = pre_norm_q * (torch.norm(quantized, dim=1, keepdim=True) / (torch.norm(flat_input, dim=1, keepdim=True) + 1e-6)).detach()

        quantized_final = q_tilde.view(input_shape)

        # Perplexity Computation
        encoding_indices = min_indices.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=self.device)
        encodings.scatter_(1, encoding_indices, 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        e_latent_loss = F.mse_loss(quantized_reshaped.detach(), inputs)  # Commitment Loss
        q_latent_loss = F.mse_loss(quantized_reshaped, inputs.detach())  # Codebook Loss
        loss = q_latent_loss + (0.25 * e_latent_loss)

        with torch.no_grad():
            self.codebooks_used[min_indices.cpu()] += 1

        return loss, quantized_final.permute(0, 3, 1, 2).contiguous(), perplexity

    def codebook_replacement(self, num_batches):
        if self.cbr_method == 'new':
            self.codebooks_used, self.codebooks = cbr_new(self.codebooks_used, self.codebooks, num_batches, self.discarding_threshold, self.eps, self._embedding_dim)
        elif self.cbr_method == 'old':
            self.codebooks_used, self.codebooks = cbr_old(self.codebooks_used, self.codebooks, num_batches,self.discarding_threshold, self.eps, self._embedding_dim)

    def inference(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebooks ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codebooks.t()))
        min_indices = torch.argmin(distances, dim=1)
        hard_quantized_input = self.codebooks[min_indices]
        quantized = hard_quantized_input.view(input_shape)
        return quantized.permute(0, 3, 1, 2).contiguous()


class DIVEQ(nn.Module):
    def __init__(self, args, num_embeddings, embedding_dim):
        super(DIVEQ, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        bitrate = args.bitrate
        self.eps = args.eps
        self.discarding_threshold = args.discard_threshold
        self.device = args.device
        self.noise_var = args.variance
        self.cbr_method = args.cbr_method

        codebooks = torch.load(f'initial_cb_{bitrate}bit.pt', weights_only=True)
        # codebooks = uniform.Uniform(-1.0 / self._num_embeddings, 1.0 / self._num_embeddings).sample([self._num_embeddings, self._embedding_dim ])
        self.codebooks = torch.nn.Parameter(codebooks, requires_grad=True)
        self.codebooks_used = torch.zeros(self._num_embeddings, dtype=torch.int32)

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebooks ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codebooks.t()))

        min_indices = torch.argmin(distances, dim=1)

        hard_quantized_input = self.codebooks[min_indices]
        direction = hard_quantized_input - flat_input

        random_vectors = normal.Normal(0, self.noise_var).sample(flat_input.shape).to(self.device) + direction
        normalized_random_vectors = random_vectors / torch.linalg.norm(random_vectors, dim=1, keepdim=True)

        error_magnitude = torch.linalg.norm(hard_quantized_input - flat_input, dim=1, keepdim=True)

        vq_error = error_magnitude * normalized_random_vectors.detach()

        quantized_input = flat_input + vq_error
        quantized_final = quantized_input.view(input_shape)

        loss = torch.tensor(0.0, dtype=torch.float32)

        # Perplexity Computation
        encoding_indices = min_indices.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=self.device)
        encodings.scatter_(1, encoding_indices, 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        with torch.no_grad():
            self.codebooks_used[min_indices.cpu()] += 1

        # convert quantized from BHWC -> BCHW
        return loss, quantized_final.permute(0, 3, 1, 2).contiguous(), perplexity

    def inference(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebooks ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codebooks.t()))
        min_indices = torch.argmin(distances, dim=1)
        hard_quantized_input = self.codebooks[min_indices]
        quantized = hard_quantized_input.view(input_shape)
        return quantized.permute(0, 3, 1, 2).contiguous()

    def codebook_replacement(self, num_batches):
        if self.cbr_method == 'new':
            self.codebooks_used, self.codebooks = cbr_new(self.codebooks_used, self.codebooks, num_batches,
                                                          self.discarding_threshold, self.eps, self._embedding_dim)
        elif self.cbr_method == 'old':
            self.codebooks_used, self.codebooks = cbr_old(self.codebooks_used, self.codebooks, num_batches,
                                                          self.discarding_threshold, self.eps, self._embedding_dim)

class NSVQ(nn.Module):
    def __init__(self, args, num_embeddings, embedding_dim):
        super(NSVQ, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        bitrate = args.bitrate
        self.eps = args.eps
        self.discarding_threshold = args.discard_threshold
        self.device = args.device
        self.cbr_method = args.cbr_method

        codebooks = torch.load(f'initial_cb_{bitrate}bit.pt', weights_only=True)
        # codebooks = uniform.Uniform(-1.0 / self._num_embeddings, 1.0 / self._num_embeddings).sample([self._num_embeddings, self._embedding_dim ])
        self.codebooks = torch.nn.Parameter(codebooks, requires_grad=True)
        self.codebooks_used = torch.zeros(self._num_embeddings, dtype=torch.int32)

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebooks ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codebooks.t()))

        min_indices = torch.argmin(distances, dim=1)

        hard_quantized_input = self.codebooks[min_indices]
        random_vector = normal.Normal(0, 1).sample(flat_input.shape).to(inputs.device)
        norm_quantization_residual = torch.linalg.norm(flat_input - hard_quantized_input, dim=1, keepdim=True)
        norm_random_vector = torch.linalg.norm(random_vector, dim=1, keepdim=True)
        vq_error = (norm_quantization_residual / (norm_random_vector + self.eps)) * random_vector

        quantized_input = flat_input + vq_error
        quantized_final = quantized_input.view(input_shape)

        loss = torch.tensor(0.0, dtype=torch.float32)

        # Perplexity Computation
        encoding_indices = min_indices.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=self.device)
        encodings.scatter_(1, encoding_indices, 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        with torch.no_grad():
            self.codebooks_used[min_indices.cpu()] += 1

        # convert quantized from BHWC -> BCHW
        return loss, quantized_final.permute(0, 3, 1, 2).contiguous(), perplexity

    def inference(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebooks ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codebooks.t()))
        min_indices = torch.argmin(distances, dim=1)
        hard_quantized_input = self.codebooks[min_indices]
        quantized = hard_quantized_input.view(input_shape)
        return quantized.permute(0, 3, 1, 2).contiguous()

    def codebook_replacement(self, num_batches):
        if self.cbr_method == 'new':
            self.codebooks_used, self.codebooks = cbr_new(self.codebooks_used, self.codebooks, num_batches,
                                                          self.discarding_threshold, self.eps, self._embedding_dim)
        elif self.cbr_method == 'old':
            self.codebooks_used, self.codebooks = cbr_old(self.codebooks_used, self.codebooks, num_batches,
                                                          self.discarding_threshold, self.eps, self._embedding_dim)


class DIVEQ_DETACH(nn.Module):
    def __init__(self, args, num_embeddings, embedding_dim):
        super(DIVEQ_DETACH, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        bitrate = args.bitrate
        self.eps = args.eps
        self.discarding_threshold = args.discard_threshold
        self.device = args.device
        self.cbr_method = args.cbr_method

        codebooks = torch.load(f'initial_cb_{bitrate}bit.pt', weights_only=True)
        # codebooks = uniform.Uniform(-1.0 / self._num_embeddings, 1.0 / self._num_embeddings).sample([self._num_embeddings, self._embedding_dim ])
        self.codebooks = torch.nn.Parameter(codebooks, requires_grad=True)
        self.codebooks_used = torch.zeros(self._num_embeddings, dtype=torch.int32)

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebooks ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codebooks.t()))

        min_indices = torch.argmin(distances, dim=1)

        hard_quantized_input = self.codebooks[min_indices]
        error_magnitude = torch.linalg.norm(hard_quantized_input - flat_input, dim=1, keepdim=True)
        vq_error = error_magnitude * ((hard_quantized_input - flat_input) / (error_magnitude + 1e-15)).detach()

        quantized_input = flat_input + vq_error
        quantized_final = quantized_input.view(input_shape)

        loss = torch.tensor(0.0, dtype=torch.float32)

        # Perplexity Computation
        encoding_indices = min_indices.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=self.device)
        encodings.scatter_(1, encoding_indices, 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        with torch.no_grad():
            self.codebooks_used[min_indices.cpu()] += 1

        # convert quantized from BHWC -> BCHW
        return loss, quantized_final.permute(0, 3, 1, 2).contiguous(), perplexity

    def inference(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebooks ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codebooks.t()))
        min_indices = torch.argmin(distances, dim=1)
        hard_quantized_input = self.codebooks[min_indices]
        quantized = hard_quantized_input.view(input_shape)
        return quantized.permute(0, 3, 1, 2).contiguous()

    def codebook_replacement(self, num_batches):
        if self.cbr_method == 'new':
            self.codebooks_used, self.codebooks = cbr_new(self.codebooks_used, self.codebooks, num_batches,
                                                          self.discarding_threshold, self.eps, self._embedding_dim)
        elif self.cbr_method == 'old':
            self.codebooks_used, self.codebooks = cbr_old(self.codebooks_used, self.codebooks, num_batches,
                                                          self.discarding_threshold, self.eps, self._embedding_dim)

class SFDIVEQ_DETACH(nn.Module):
    def __init__(self, args, num_embeddings, embedding_dim):
        super(SFDIVEQ_DETACH, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        bitrate = args.bitrate
        self.eps = args.eps
        self.discarding_threshold = args.discard_threshold
        self.device = args.device

        codebooks = uniform.Uniform(-1.0 / self._num_embeddings, 1.0 / self._num_embeddings).sample([self._num_embeddings, self._embedding_dim ])
        self.codebooks = torch.nn.Parameter(codebooks, requires_grad=True)

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        dither = torch.rand((self._num_embeddings - 1, 1), device=inputs.device)
        integer_index = torch.linspace(0, self._num_embeddings - 2, self._num_embeddings - 1, device=inputs.device).to(torch.int64)
        c0 = self.codebooks[integer_index]
        c1 = self.codebooks[integer_index + 1]
        dithered_codebook = ((1 - dither) * c0) + (dither * c1)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(dithered_codebook ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, dithered_codebook.t()))

        min_indices = torch.argmin(distances, dim=1)
        cb_first = self.codebooks[min_indices]
        cb_second = self.codebooks[min_indices + 1]
        s = dither[min_indices]

        error_magnitude1 = torch.linalg.norm(cb_first - flat_input, dim=1, keepdim=True)
        error_magnitude2 = torch.linalg.norm(cb_second - flat_input, dim=1, keepdim=True)
        vq_error1 = error_magnitude1 * (((1 - s) * (cb_first - flat_input)) / (error_magnitude1 + 1e-15)).detach()
        vq_error2 = error_magnitude2 * ((s * (cb_second - flat_input)) / (error_magnitude2 + 1e-15)).detach()

        quantized_input = flat_input + vq_error1 + vq_error2
        quantized_final = quantized_input.view(input_shape)

        loss = torch.tensor(0.0, dtype=torch.float32)

        # Perplexity Computation
        encoding_indices = min_indices.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=self.device)
        encodings.scatter_(1, encoding_indices, 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized_final.permute(0, 3, 1, 2).contiguous(), perplexity, min_indices

    def inference(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebooks ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codebooks.t()))
        integer_index = torch.argmin(distances, dim=1).clamp(min=1, max=self._num_embeddings - 2)

        cm = self.codebooks[integer_index - 1]
        cc = self.codebooks[integer_index]
        cp = self.codebooks[integer_index + 1]

        remainder_index_m = (((cc - cm) * (flat_input - cm)).sum(dim=1) / (cc - cm).square().sum(dim=1)).unsqueeze(-1).clamp(min=0., max=1.)
        remainder_index_p = (((cp - cc) * (flat_input - cc)).sum(dim=1) / (cp - cc).square().sum(dim=1)).unsqueeze(-1).clamp(min=0., max=1.)

        xhat_m = ((1 - remainder_index_m) * cm) + (remainder_index_m * cc)
        xhat_p = ((1 - remainder_index_p) * cc) + (remainder_index_p * cp)
        distance_m = (flat_input - xhat_m).square().sum(dim=1)
        distance_p = (flat_input - xhat_p).square().sum(dim=1)

        offset = (distance_p < distance_m).to(torch.int64).squeeze() - 1

        c0 = self.codebooks[integer_index + offset]
        c1 = self.codebooks[integer_index + offset + 1]

        remainder_index = (((c1 - c0) * (flat_input - c0)).sum(dim=1) / (c1 - c0).square().sum(dim=1)).clamp(min=0.,max=1.)
        quantized = (c0 + (remainder_index.reshape(-1, 1) * (c1 - c0))).view(input_shape)

        return quantized.permute(0, 3, 1, 2).contiguous()


class SFDIVEQ(nn.Module):
    def __init__(self, args, num_embeddings, embedding_dim):
        super(SFDIVEQ, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        bitrate = args.bitrate
        self.eps = args.eps
        self.discarding_threshold = args.discard_threshold
        self.device = args.device
        self.noise_var = args.variance

        codebooks = uniform.Uniform(-1.0 / self._num_embeddings, 1.0 / self._num_embeddings).sample([self._num_embeddings, self._embedding_dim ])
        self.codebooks = torch.nn.Parameter(codebooks, requires_grad=True)

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        dither = torch.rand((self._num_embeddings - 1, 1), device=inputs.device)
        integer_index = torch.linspace(0, self._num_embeddings - 2, self._num_embeddings - 1, device=inputs.device).to(torch.int64)
        c0 = self.codebooks[integer_index]
        c1 = self.codebooks[integer_index + 1]
        dithered_codebook = ((1 - dither) * c0) + (dither * c1)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(dithered_codebook ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, dithered_codebook.t()))

        min_indices = torch.argmin(distances, dim=1)
        cb_first = self.codebooks[min_indices]
        cb_second = self.codebooks[min_indices + 1]
        s = dither[min_indices]

        direction_first = cb_first - flat_input
        random_vectors1 = normal.Normal(0, self.noise_var).sample(flat_input.shape).to(self.device) + direction_first
        normalized_random_vectors1 = random_vectors1 / torch.linalg.norm(random_vectors1, dim=1, keepdim=True)

        direction_second = cb_second - flat_input
        random_vectors2 = normal.Normal(0, self.noise_var).sample(flat_input.shape).to(self.device) + direction_second
        normalized_random_vectors2 = random_vectors2 / torch.linalg.norm(random_vectors2, dim=1, keepdim=True)

        error_magnitude1 = torch.linalg.norm(cb_first - flat_input, dim=1, keepdim=True)
        error_magnitude2 = torch.linalg.norm(cb_second - flat_input, dim=1, keepdim=True)
        vq_error1 = error_magnitude1 * ((1 - s) * normalized_random_vectors1).detach()
        vq_error2 = error_magnitude2 * (s * normalized_random_vectors2).detach()

        quantized_input = flat_input + vq_error1 + vq_error2
        quantized_final = quantized_input.view(input_shape)

        loss = torch.tensor(0.0, dtype=torch.float32)

        # Perplexity Computation
        encoding_indices = min_indices.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=self.device)
        encodings.scatter_(1, encoding_indices, 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized_final.permute(0, 3, 1, 2).contiguous(), perplexity, min_indices

    def inference(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebooks ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codebooks.t()))
        integer_index = torch.argmin(distances, dim=1).clamp(min=1, max=self._num_embeddings - 2)

        cm = self.codebooks[integer_index - 1]
        cc = self.codebooks[integer_index]
        cp = self.codebooks[integer_index + 1]

        remainder_index_m = (((cc - cm) * (flat_input - cm)).sum(dim=1) / (cc - cm).square().sum(dim=1)).unsqueeze(-1).clamp(min=0., max=1.)
        remainder_index_p = (((cp - cc) * (flat_input - cc)).sum(dim=1) / (cp - cc).square().sum(dim=1)).unsqueeze(-1).clamp(min=0., max=1.)

        xhat_m = ((1 - remainder_index_m) * cm) + (remainder_index_m * cc)
        xhat_p = ((1 - remainder_index_p) * cc) + (remainder_index_p * cp)
        distance_m = (flat_input - xhat_m).square().sum(dim=1)
        distance_p = (flat_input - xhat_p).square().sum(dim=1)

        offset = (distance_p < distance_m).to(torch.int64).squeeze() - 1

        c0 = self.codebooks[integer_index + offset]
        c1 = self.codebooks[integer_index + offset + 1]

        remainder_index = (((c1 - c0) * (flat_input - c0)).sum(dim=1) / (c1 - c0).square().sum(dim=1)).clamp(min=0.,max=1.)
        quantized = (c0 + (remainder_index.reshape(-1, 1) * (c1 - c0))).view(input_shape)

        return quantized.permute(0, 3, 1, 2).contiguous()



class Gumbel_Softmax(nn.Module):
    def __init__(self, args, num_embeddings, embedding_dim, straight_through=True):
        super(Gumbel_Softmax).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        bitrate = args.bitrate
        self.eps = args.eps
        self.discarding_threshold = args.discard_threshold
        self.device = args.device
        self.cbr_method = args.cbr_method

        self.straight_through = straight_through
        self.temperature = 1.0
        self.kld_scale = 5e-4

        # codebooks = uniform.Uniform(-1.0 / self._num_embeddings, 1.0 / self._num_embeddings).sample([self._num_embeddings, self._embedding_dim])
        codebooks = torch.load(f'initial_cb_{bitrate}bit.pt', weights_only=True)
        self.codebooks = nn.Embedding(num_embeddings, embedding_dim)
        self.codebooks.weight.data = codebooks
        self.codebooks_used = torch.zeros(self._num_embeddings, dtype=torch.int32)

    def forward(self, z):
        soft_one_hot = F.gumbel_softmax(z, tau=self.temperature, dim=1, hard=self.straight_through)
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebooks.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(z, dim=1)
        diff = self.kld_scale * torch.sum(qy * torch.log(qy * self._num_embeddings + 1e-10), dim=1).mean()

        min_indices = soft_one_hot.argmax(dim=1).flatten()

        # Perplexity Computation
        encoding_indices = min_indices.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=self.device)
        encodings.scatter_(1, encoding_indices, 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        with torch.no_grad():
            self.codebooks_used[min_indices.cpu()] += 1

        return diff, z_q, perplexity, torch.tensor(0.0), torch.tensor(0.0)

    def codebook_replacement(self, num_batches):
        if self.cbr_method == 'new':
            self.codebooks_used, self.codebooks.weight.data = cbr_new(self.codebooks_used, self.codebooks.weight.data, num_batches,
                                                          self.discarding_threshold, self.eps, self._embedding_dim)
        elif self.cbr_method == 'old':
            self.codebooks_used, self.codebooks.weight.data = cbr_old(self.codebooks_used, self.codebooks.weight.data, num_batches,
                                                          self.discarding_threshold, self.eps, self._embedding_dim)

    def inference(self, z):
        with torch.no_grad():
            soft_one_hot = F.gumbel_softmax(z, tau=1e-10, dim=1, hard=self.straight_through)
            z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebooks.weight)

        return z_q


class EMA(nn.Module):
    def __init__(self, args, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(EMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        bitrate = args.bitrate
        self.eps = args.eps
        self.discarding_threshold = args.discard_threshold
        self.device = args.device
        self.cbr_method = args.cbr_method

        # initial_codebook = uniform.Uniform(-1.0 / self._num_embeddings, 1.0 / self._num_embeddings).sample([self._num_embeddings, self._embedding_dim])
        initial_codebook = torch.load(f'initial_cb_{bitrate}bit.pt', weights_only=True)
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data = initial_codebook
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

        self.codebooks_used = torch.zeros(self._num_embeddings, dtype=torch.int32)

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        with torch.no_grad():
            self.codebooks_used[encoding_indices.cpu()] += 1

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs) # Commitment Loss
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity

    def codebook_replacement(self, num_batches):
        if self.cbr_method == 'new':
            self.codebooks_used, self._embedding.weight.data = cbr_new(self.codebooks_used, self._embedding.weight.data, num_batches,
                                                          self.discarding_threshold, self.eps, self._embedding_dim)
        elif self.cbr_method == 'old':
            self.codebooks_used, self._embedding.weight.data = cbr_old(self.codebooks_used, self._embedding.weight.data, num_batches,
                                                          self.discarding_threshold, self.eps, self._embedding_dim)

    def inference(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        return quantized.permute(0, 3, 1, 2).contiguous()
