import  torch
import torch.nn as nn
from torch.distributions import normal
import torch.nn.functional as F
from torch import einsum
import warnings
from typing import Tuple

class STE(nn.Module):
    """
    STE: Straight-Through Estimator module to optimize the codebook of VQ.
    The module encompasses codebook replacement function which discards unused codebook
    entries during training.

    Args:
        - num_embeddings (int): Codebook size (No. of codewords).
        - embedding_dim (int): Dimensionality of embeddings.
        - commitment_coef (float): Weight coefficient for Commitment Loss.
            Recommended commitment_coef = 0.25.
        - replacement_iters (int): Replacement interval (number of training iterations
            to apply codebook replacement). Recommended 50 < replacement_iters < 300.
        - discard_threshold (float): Threshold to discard the codebook entries that are
            used less than this threshold after "replacement_iters" iteration.
            Recommended 0.01 < discard_threshold < 0.05.
        - perturb_eps (float): Adjusts perturbation/shift magnitude from used codewords
            during codebook replacement.
        - uniform_init (bool): Whether to initialize codebook with uniform distribution.
        - allow_warning (bool): Whether to print the warnings.
        - verbose (bool): Whether to print codebook replacement status.

    Returns:
        - z_q (torch.Tensor): Quantized input/latent. shape (N, D)
        - indices (torch.Tensor): Selected codebook indices. shape (N, )
        - perplexity (float): Codebook perplexity (average codebook usage)
        - loss (torch.Tensor): Codebook and commitment loss for VQ.
    """
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            commitment_coef: float = 0.25,
            replacement_iters: int = 100,
            discard_threshold: float = 0.01,
            perturb_eps: float = 1e-9,
            uniform_init: bool = True,
            allow_warning: bool = True,
            verbose: bool = True,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_coef = commitment_coef
        self.replacement_iters = replacement_iters
        self.discard_threshold = discard_threshold
        self.perturb_eps = perturb_eps
        self.uniform_init = uniform_init
        self.allow_warning = allow_warning
        self.verbose = verbose

        self._check_constraints()

        # ---------------- User warnings ----------------
        if allow_warning:
            if replacement_iters < 50:
                warnings.warn(f"`replacement_iters` is set to"
                              f" {replacement_iters}, which is quite small. Values < 50"
                              f" may cause too early and frequent codebook"
                              f" replacements.", UserWarning)
            elif replacement_iters > 300:
                warnings.warn(f"`replacement_iters` is set to"
                              f" {replacement_iters}, which is quite large."
                              f" Values > 300 may cause too late and sporadic codebook"
                              f" replacements.", UserWarning)

            if discard_threshold > 0.05:
                warnings.warn(f"`discard_threshold` is set to"
                              f" {discard_threshold}, which is quite large."
                              f" Values > 0.05 may discard a portion of suitable but"
                              f" rarely used codewords.", UserWarning)

            if perturb_eps > 1e-6:
                warnings.warn(f"`perturb_eps` is set to {perturb_eps}, which"
                              f" is quite large. Values > 1e-6 may cause big"
                              f" perturbation/shift from used codewords.", UserWarning)

        # ---------------- Codebook initialization ----------------
        if uniform_init:
            codebook = (torch.rand((self.num_embeddings, self.embedding_dim))
                        * (1 / self.num_embeddings))
        else:
            codebook = (torch.randn((self.num_embeddings, self.embedding_dim))
                        * (1 / self.num_embeddings))

        self.codebook = torch.nn.Parameter(codebook, requires_grad=True)

        # ---------------- Tensors used for codebook replacement ----------------
        self.register_buffer("codebook_usage", torch.zeros(self.num_embeddings,
                                                           dtype=torch.int32))
        self.register_buffer("iter_counter", torch.zeros(1, dtype=torch.int32))

    # ---------------- Forward pass (Core API) ----------------
    def forward(self, z: torch.Tensor) \
                            -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """
        Args:
            - z (torch.Tensor): input/latent. shape (N, D)

        Returns:
            - z_q (torch.Tensor): Quantized input/latent. shape (N, D)
            - indices (torch.Tensor): Selected codebook indices. shape (N, )
            - perplexity (float): Codebook perplexity (average codebook usage).
            - loss (torch.Tensor): Codebook and commitment loss for VQ.
        """
        self._check_input(z)

        # Calculate distances
        distances = (torch.sum(z.pow(2), dim=1, keepdim=True)
                     + torch.sum(self.codebook.pow(2), dim=1)
                     - 2 * torch.matmul(z, self.codebook.t()))

        indices = torch.argmin(distances, dim=1)

        z_quantized = self.codebook[indices]

        e_latent_loss = F.mse_loss(z_quantized.detach(), z)  # Commitment Loss
        q_latent_loss = F.mse_loss(z_quantized, z.detach())  # Codebook Loss
        loss = q_latent_loss + (self.commitment_coef * e_latent_loss)

        z_q = z + (z_quantized - z).detach()

        # Perplexity Computation
        perplexity = self._compute_perplexity(indices)

        # Track used indices for codebook replacement
        with torch.no_grad():
            self.codebook_usage[indices] += 1
            self.iter_counter += 1
            if self.iter_counter.item() % self.replacement_iters == 0:
                self._replace_unused_entries()  # Applies codebook replacement

        return z_q, indices, perplexity, loss

    # ---------------- Quantization for Inference ----------------
    @torch.no_grad()
    def inference(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Deterministic hard quantization by mapping the input to the nearest codeword.
        Args:
            - z (torch.Tensor): input/latent. shape (N, D)

        Returns:
            - z_q_hard (torch.Tensor): Hard quantized input/latent. shape (N, D)
            - indices (torch.Tensor): Selected codebook indices. shape (N, )
            - perplexity (float): Codebook perplexity (average codebook usage).
        """
        self._check_input(z)
        distances = (torch.sum(z.pow(2), dim=1, keepdim=True)
                     + torch.sum(self.codebook.pow(2), dim=1)
                     - 2.0 * torch.matmul(z, self.codebook.t()))
        indices = torch.argmin(distances, dim=1)
        perplexity = self._compute_perplexity(indices)
        z_q_hard = self.codebook[indices]
        return z_q_hard, indices, perplexity

    # ---------------- Utility functions ----------------
    def _check_constraints(self) -> None:
        if self.commitment_coef < 0.0:
            raise ValueError(f"`commitment_coef` is set to {self.commitment_coef}."
                             f" It must be a positive float value.")

    def _check_input(self, z: torch.Tensor) -> None:
        if z.ndim != 2:
            raise ValueError("STE input must have the shape of (N, D), where N is"
                             " the No. of input samples,and D is the embedding"
                             " dimensionality.")
        if z.size(1) != self.embedding_dim:
            raise ValueError(f"STE input.shape[1] must match the embedding"
                             f" dimensionality that is {self.embedding_dim}.")

    def _compute_perplexity(self, indices: torch.Tensor) -> float:
        encodings = torch.zeros(indices.shape[0], self.num_embeddings,
                                device=indices.device)
        encodings.scatter_(1, indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity.item()

    def _replace_unused_entries(self) -> None:
        with torch.no_grad():
            usage_ratio = self.codebook_usage / self.replacement_iters
            unused_indices = torch.where(usage_ratio < self.discard_threshold)[0]
            used_indices = torch.where(usage_ratio >= self.discard_threshold)[0]

            if unused_indices.numel() == 0 or used_indices.numel() == 0:
                self.codebook_usage.zero_()
                return

            unused_count = unused_indices.numel()
            used_probs = self.codebook_usage[used_indices] / torch.sum(
                self.codebook_usage[used_indices])
            randomly_sampled_indices = used_probs.multinomial(
                num_samples=unused_count,
                replacement=True)
            sampled_indices = used_indices[randomly_sampled_indices]
            used_codewords = self.codebook[sampled_indices].clone()

            self.codebook[unused_indices] = (used_codewords +
                                             self.perturb_eps * torch.randn_like(
                        used_codewords)).clone()
            self.codebook_usage.zero_()

            if self.verbose:
                print("\n***** Replaced " + str(unused_count) + " codewords *****")

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"commitment_coef={self.commitment_coef}, "
            f"replacement_iters={self.replacement_iters}, "
            f"discard_threshold={self.discard_threshold}, "
            f"perturb_eps={self.perturb_eps}, "
            f"uniform_init={self.uniform_init}, "
            f"allow_warning={self.allow_warning}, "
            f"verbose={self.verbose}"
        )


class RT(nn.Module):
    """
    RT: Rotation Trick module to optimize the codebook of VQ.
    The module encompasses codebook replacement function which discards unused codebook
    entries during training.

    Args:
        - num_embeddings (int): Codebook size (No. of codewords).
        - embedding_dim (int): Dimensionality of embeddings.
        - commitment_coef (float): Weight coefficient for Commitment Loss.
            Recommended commitment_coef = 0.25.
        - replacement_iters (int): Replacement interval (number of training iterations
            to apply codebook replacement). Recommended 50 < replacement_iters < 300.
        - discard_threshold (float): Threshold to discard the codebook entries that are
            used less than this threshold after "replacement_iters" iteration.
            Recommended 0.01 < discard_threshold < 0.05.
        - perturb_eps (float): Adjusts perturbation/shift magnitude from used codewords
            during codebook replacement.
        - uniform_init (bool): Whether to initialize codebook with uniform distribution.
        - allow_warning (bool): Whether to print the warnings.
        - verbose (bool): Whether to print codebook replacement status.

    Returns:
        - z_q (torch.Tensor): Quantized input/latent. shape (N, D)
        - indices (torch.Tensor): Selected codebook indices. shape (N, )
        - perplexity (float): Codebook perplexity (average codebook usage)
        - loss (torch.Tensor): Codebook and commitment loss for VQ.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            commitment_coef: float = 0.25,
            replacement_iters: int = 100,
            discard_threshold: float = 0.01,
            perturb_eps: float = 1e-9,
            uniform_init: bool = True,
            allow_warning: bool = True,
            verbose: bool = True,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_coef = commitment_coef
        self.replacement_iters = replacement_iters
        self.discard_threshold = discard_threshold
        self.perturb_eps = perturb_eps
        self.uniform_init = uniform_init
        self.allow_warning = allow_warning
        self.verbose = verbose

        self._check_constraints()

        # ---------------- User warnings ----------------
        if allow_warning:
            if replacement_iters < 50:
                warnings.warn(f"`replacement_iters` is set to"
                              f" {replacement_iters}, which is quite small. Values < 50"
                              f" may cause too early and frequent codebook"
                              f" replacements.", UserWarning)
            elif replacement_iters > 300:
                warnings.warn(f"`replacement_iters` is set to"
                              f" {replacement_iters}, which is quite large."
                              f" Values > 300 may cause too late and sporadic codebook"
                              f" replacements.", UserWarning)

            if discard_threshold > 0.05:
                warnings.warn(f"`discard_threshold` is set to"
                              f" {discard_threshold}, which is quite large."
                              f" Values > 0.05 may discard a portion of suitable but"
                              f" rarely used codewords.", UserWarning)

            if perturb_eps > 1e-6:
                warnings.warn(f"`perturb_eps` is set to {perturb_eps}, which"
                              f" is quite large. Values > 1e-6 may cause big"
                              f" perturbation/shift from used codewords.", UserWarning)

        # ---------------- Codebook initialization ----------------
        if uniform_init:
            codebook = (torch.rand((self.num_embeddings, self.embedding_dim))
                        * (1 / self.num_embeddings))
        else:
            codebook = (torch.randn((self.num_embeddings, self.embedding_dim))
                        * (1 / self.num_embeddings))

        self.codebook = torch.nn.Parameter(codebook, requires_grad=True)

        # ---------------- Tensors used for codebook replacement ----------------
        self.register_buffer("codebook_usage", torch.zeros(self.num_embeddings,
                                                           dtype=torch.int32))
        self.register_buffer("iter_counter", torch.zeros(1, dtype=torch.int32))

    # ---------------- Forward pass (Core API) ----------------
    def forward(self, z: torch.Tensor) \
                            -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """
        Args:
            - z (torch.Tensor): input/latent. shape (N, D)

        Returns:
            - z_q (torch.Tensor): Quantized input/latent. shape (N, D)
            - indices (torch.Tensor): Selected codebook indices. shape (N, )
            - perplexity (float): Codebook perplexity (average codebook usage).
            - loss (torch.Tensor): Codebook and commitment loss for VQ.
        """
        self._check_input(z)

        # Calculate distances
        distances = (torch.sum(z.pow(2), dim=1, keepdim=True)
                     + torch.sum(self.codebook.pow(2), dim=1)
                     - 2 * torch.matmul(z, self.codebook.t()))

        indices = torch.argmin(distances, dim=1)

        z_quantized = self.codebook[indices]

        pre_norm_q = self.get_very_efficient_rotation(z / (torch.norm(z, dim=1, keepdim=True) + 1e-6),
                                  z_quantized / (torch.norm(z_quantized, dim=1, keepdim=True) + 1e-6),
                                                      z.unsqueeze(1)).squeeze()
        z_q = pre_norm_q * (torch.norm(z_quantized, dim=1, keepdim=True)
                                / (torch.norm(z, dim=1, keepdim=True) + 1e-6)).detach()

        # Perplexity Computation
        perplexity = self._compute_perplexity(indices)

        e_latent_loss = F.mse_loss(z_quantized.detach(), z)  # Commitment Loss
        q_latent_loss = F.mse_loss(z_quantized, z.detach())  # Codebook Loss
        loss = q_latent_loss + (self.commitment_coef * e_latent_loss)

        # Track used indices for codebook replacement
        with torch.no_grad():
            self.codebook_usage[indices] += 1
            self.iter_counter += 1
            if self.iter_counter.item() % self.replacement_iters == 0:
                self._replace_unused_entries()  # Applies codebook replacement

        return z_q, indices, perplexity, loss

    # ---------------- Quantization for Inference ----------------
    @torch.no_grad()
    def inference(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Deterministic hard quantization by mapping the input to the nearest codeword.
        Args:
            - z (torch.Tensor): input/latent. shape (N, D)

        Returns:
            - z_q_hard (torch.Tensor): Hard quantized input/latent. shape (N, D)
            - indices (torch.Tensor): Selected codebook indices. shape (N, )
            - perplexity (float): Codebook perplexity (average codebook usage).
        """
        self._check_input(z)
        distances = (torch.sum(z.pow(2), dim=1, keepdim=True)
                     + torch.sum(self.codebook.pow(2), dim=1)
                     - 2.0 * torch.matmul(z, self.codebook.t()))
        indices = torch.argmin(distances, dim=1)
        perplexity = self._compute_perplexity(indices)
        z_q_hard = self.codebook[indices]
        return z_q_hard, indices, perplexity

    # ---------------- Utility functions ----------------
    def _check_constraints(self) -> None:
        if self.commitment_coef < 0.0:
            raise ValueError(f"`commitment_coef` is set to {self.commitment_coef}."
                             f" It must be a positive float value.")

    def _check_input(self, z: torch.Tensor) -> None:
        if z.ndim != 2:
            raise ValueError("RT input must have the shape of (N, D), where N is"
                             " the No. of input samples,and D is the embedding"
                             " dimensionality.")
        if z.size(1) != self.embedding_dim:
            raise ValueError(f"RT input.shape[1] must match the embedding"
                             f" dimensionality that is {self.embedding_dim}.")

    @staticmethod
    def get_very_efficient_rotation(u: torch.Tensor, q: torch.Tensor, e: torch.Tensor)\
                                                                        -> torch.Tensor:
        w = ((u + q) / torch.norm(u + q, dim=1, keepdim=True)).detach()
        e = e - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)),
                              w.unsqueeze(1)) + 2 * torch.bmm(
            torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
        return e

    def _compute_perplexity(self, indices: torch.Tensor) -> float:
        encodings = torch.zeros(indices.shape[0], self.num_embeddings,
                                device=indices.device)
        encodings.scatter_(1, indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity.item()

    def _replace_unused_entries(self) -> None:
        with torch.no_grad():
            usage_ratio = self.codebook_usage / self.replacement_iters
            unused_indices = torch.where(usage_ratio < self.discard_threshold)[0]
            used_indices = torch.where(usage_ratio >= self.discard_threshold)[0]

            if unused_indices.numel() == 0 or used_indices.numel() == 0:
                self.codebook_usage.zero_()
                return

            unused_count = unused_indices.numel()
            used_probs = self.codebook_usage[used_indices] / torch.sum(
                self.codebook_usage[used_indices])
            randomly_sampled_indices = used_probs.multinomial(
                num_samples=unused_count,
                replacement=True)
            sampled_indices = used_indices[randomly_sampled_indices]
            used_codewords = self.codebook[sampled_indices].clone()

            self.codebook[unused_indices] = (used_codewords +
                                             self.perturb_eps * torch.randn_like(
                        used_codewords)).clone()
            self.codebook_usage.zero_()

            if self.verbose:
                print("\n***** Replaced " + str(unused_count) + " codewords *****")

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"commitment_coef={self.commitment_coef}, "
            f"replacement_iters={self.replacement_iters}, "
            f"discard_threshold={self.discard_threshold}, "
            f"perturb_eps={self.perturb_eps}, "
            f"uniform_init={self.uniform_init}, "
            f"allow_warning={self.allow_warning}, "
            f"verbose={self.verbose}"
        )


class NSVQ(nn.Module):
    """
    NSVQ: Noise Substitution Vector Quantization module to optimize the codebook of VQ.
    The module encompasses codebook replacement function which discards unused codebook
    entries during training.

    Args:
        - num_embeddings (int): Codebook size (No. of codewords).
        - embedding_dim (int): Dimensionality of embeddings.
        - replacement_iters (int): Replacement interval (number of training iterations
            to apply codebook replacement). Recommended 50 < replacement_iters < 300.
        - discard_threshold (float): Threshold to discard the codebook entries that are
            used less than this threshold after "replacement_iters" iteration.
            Recommended 0.01 < discard_threshold < 0.05.
        - perturb_eps (float): Adjusts perturbation/shift magnitude from used codewords
            during codebook replacement.
        - uniform_init (bool): Whether to initialize codebook with uniform distribution.
        - allow_warning (bool): Whether to print the warnings.
        - verbose (bool): Whether to print codebook replacement status.

    Returns:
        - z_q (torch.Tensor): Quantized input/latent. shape (N, D)
        - indices (torch.Tensor): Selected codebook indices. shape (N, )
        - perplexity (float): Codebook perplexity (average codebook usage)
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            replacement_iters: int = 100,
            discard_threshold: float = 0.01,
            perturb_eps: float = 1e-9,
            uniform_init: bool = True,
            allow_warning: bool = True,
            verbose: bool = True,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.replacement_iters = replacement_iters
        self.discard_threshold = discard_threshold
        self.perturb_eps = perturb_eps
        self.uniform_init = uniform_init
        self.allow_warning = allow_warning
        self.verbose = verbose

        # ---------------- User warnings ----------------
        if allow_warning:
            if replacement_iters < 50:
                warnings.warn(f"`replacement_iters` is set to"
                              f" {replacement_iters}, which is quite small. Values < 50"
                              f" may cause too early and frequent codebook"
                              f" replacements.", UserWarning)
            elif replacement_iters > 300:
                warnings.warn(f"`replacement_iters` is set to"
                              f" {replacement_iters}, which is quite large."
                              f" Values > 300 may cause too late and sporadic codebook"
                              f" replacements.", UserWarning)

            if discard_threshold > 0.05:
                warnings.warn(f"`discard_threshold` is set to"
                              f" {discard_threshold}, which is quite large."
                              f" Values > 0.05 may discard a portion of suitable but"
                              f" rarely used codewords.", UserWarning)

            if perturb_eps > 1e-6:
                warnings.warn(f"`perturb_eps` is set to {perturb_eps}, which"
                              f" is quite large. Values > 1e-6 may cause big"
                              f" perturbation/shift from used codewords.", UserWarning)

        # ---------------- Codebook initialization ----------------
        if uniform_init:
            codebook = (torch.rand((self.num_embeddings, self.embedding_dim))
                        * (1 / self.num_embeddings))
        else:
            codebook = (torch.randn((self.num_embeddings, self.embedding_dim))
                        * (1 / self.num_embeddings))

        self.codebook = torch.nn.Parameter(codebook, requires_grad=True)

        # ---------------- Tensors used for codebook replacement ----------------
        self.register_buffer("codebook_usage", torch.zeros(self.num_embeddings,
                                                           dtype=torch.int32))
        self.register_buffer("iter_counter", torch.zeros(1, dtype=torch.int32))

    # ---------------- Forward pass (Core API) ----------------
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Args:
            - z (torch.Tensor): input/latent. shape (N, D)

        Returns:
            - z_q (torch.Tensor): Quantized input/latent. shape (N, D)
            - indices (torch.Tensor): Selected codebook indices. shape (N, )
            - perplexity (float): Codebook perplexity (average codebook usage).
        """
        self._check_input(z)

        # Calculate distances
        distances = (torch.sum(z.pow(2), dim=1, keepdim=True)
                     + torch.sum(self.codebook.pow(2), dim=1)
                     - 2 * torch.matmul(z, self.codebook.t()))

        indices = torch.argmin(distances, dim=1)

        z_quantized = self.codebook[indices]

        random_vector = normal.Normal(0, 1).sample(z.shape).to(z.device)
        norm_quantization_residual = torch.linalg.norm(z - z_quantized, dim=1, keepdim=True)
        norm_random_vector = torch.linalg.norm(random_vector, dim=1, keepdim=True)
        vq_error = (norm_quantization_residual / (norm_random_vector + 1e-12)) * random_vector

        z_q = z + vq_error

        # Perplexity Computation
        perplexity = self._compute_perplexity(indices)

        # Track used indices for codebook replacement
        with torch.no_grad():
            self.codebook_usage[indices] += 1
            self.iter_counter += 1
            if self.iter_counter.item() % self.replacement_iters == 0:
                self._replace_unused_entries()  # Applies codebook replacement

        return z_q, indices, perplexity

    # ---------------- Quantization for Inference ----------------
    @torch.no_grad()
    def inference(self, z: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, float]:
        """
        Deterministic hard quantization by mapping the input to the nearest codeword.
        Args:
            - z (torch.Tensor): input/latent. shape (N, D)

        Returns:
            - z_q_hard (torch.Tensor): Hard quantized input/latent. shape (N, D)
            - indices (torch.Tensor): Selected codebook indices. shape (N, )
            - perplexity (float): Codebook perplexity (average codebook usage).
        """
        self._check_input(z)
        distances = (torch.sum(z.pow(2), dim=1, keepdim=True)
                     + torch.sum(self.codebook.pow(2), dim=1)
                     - 2.0 * torch.matmul(z, self.codebook.t()))
        indices = torch.argmin(distances, dim=1)
        perplexity = self._compute_perplexity(indices)
        z_q_hard = self.codebook[indices]
        return z_q_hard, indices, perplexity

    # ---------------- Utility functions ----------------
    def _check_input(self, z: torch.Tensor) -> None:
        if z.ndim != 2:
            raise ValueError("NSVQ input must have the shape of (N, D), where N is"
                             " the No. of input samples,and D is the embedding"
                             " dimensionality.")
        if z.size(1) != self.embedding_dim:
            raise ValueError(f"NSVQ input.shape[1] must match the embedding"
                             f" dimensionality that is {self.embedding_dim}.")

    def _compute_perplexity(self, indices: torch.Tensor) -> float:
        encodings = torch.zeros(indices.shape[0], self.num_embeddings,
                                device=indices.device)
        encodings.scatter_(1, indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity.item()

    def _replace_unused_entries(self) -> None:
        with torch.no_grad():
            usage_ratio = self.codebook_usage / self.replacement_iters
            unused_indices = torch.where(usage_ratio < self.discard_threshold)[0]
            used_indices = torch.where(usage_ratio >= self.discard_threshold)[0]

            if unused_indices.numel() == 0 or used_indices.numel() == 0:
                self.codebook_usage.zero_()
                return

            unused_count = unused_indices.numel()
            used_probs = self.codebook_usage[used_indices] / torch.sum(
                self.codebook_usage[used_indices])
            randomly_sampled_indices = used_probs.multinomial(
                num_samples=unused_count,
                replacement=True)
            sampled_indices = used_indices[randomly_sampled_indices]
            used_codewords = self.codebook[sampled_indices].clone()

            self.codebook[unused_indices] = (used_codewords +
                                             self.perturb_eps * torch.randn_like(
                        used_codewords)).clone()
            self.codebook_usage.zero_()

            if self.verbose:
                print("\n***** Replaced " + str(unused_count) + " codewords *****")

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"replacement_iters={self.replacement_iters}, "
            f"discard_threshold={self.discard_threshold}, "
            f"perturb_eps={self.perturb_eps}, "
            f"uniform_init={self.uniform_init}, "
            f"allow_warning={self.allow_warning}, "
            f"verbose={self.verbose}"
        )


class GumbelSoftmax(nn.Module):
    """
    GumbelSoftmax: Straight-Through Gumbel Softmax module to optimize the codebook of VQ.
    The module encompasses codebook replacement function which discards unused codebook
    entries during training.

    Args:
        - num_embeddings (int): Codebook size (No. of codewords).
        - embedding_dim (int): Dimensionality of embeddings.
        - straight_through (bool): Whether to sample hard categorical
            using "Straight-through" trick.
        - replacement_iters (int): Replacement interval (number of training iterations
            to apply codebook replacement). Recommended 50 < replacement_iters < 300.
        - discard_threshold (float): Threshold to discard the codebook entries that are
            used less than this threshold after "replacement_iters" iteration.
            Recommended 0.01 < discard_threshold < 0.05.
        - perturb_eps (float): Adjusts perturbation/shift magnitude from used codewords
            during codebook replacement.
        - uniform_init (bool): Whether to initialize codebook with uniform distribution.
        - allow_warning (bool): Whether to print the warnings.
        - verbose (bool): Whether to print codebook replacement status.

    Returns:
        - z_q (torch.Tensor): Quantized input/latent. shape (N, D)
        - indices (torch.Tensor): Selected codebook indices. shape (N, )
        - perplexity (float): Codebook perplexity (average codebook usage)
        - loss (torch.Tensor): Kullback–Leibler divergence to prior loss.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            straight_through = True,
            replacement_iters: int = 100,
            discard_threshold: float = 0.01,
            perturb_eps: float = 1e-9,
            uniform_init: bool = True,
            allow_warning: bool = True,
            verbose: bool = True,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.replacement_iters = replacement_iters
        self.discard_threshold = discard_threshold
        self.perturb_eps = perturb_eps
        self.uniform_init = uniform_init
        self.allow_warning = allow_warning
        self.verbose = verbose

        self.straight_through = straight_through
        # Set for instantiating the model. Must be annealed from 1.0 to 0.1 during training
        self.temperature = 1.0
        self.kld_scale = 5e-4

        self._check_constraints()

        # ---------------- User warnings ----------------
        if allow_warning:
            if replacement_iters < 50:
                warnings.warn(f"`replacement_iters` is set to"
                              f" {replacement_iters}, which is quite small. Values < 50"
                              f" may cause too early and frequent codebook"
                              f" replacements.", UserWarning)
            elif replacement_iters > 300:
                warnings.warn(f"`replacement_iters` is set to"
                              f" {replacement_iters}, which is quite large."
                              f" Values > 300 may cause too late and sporadic codebook"
                              f" replacements.", UserWarning)

            if discard_threshold > 0.05:
                warnings.warn(f"`discard_threshold` is set to"
                              f" {discard_threshold}, which is quite large."
                              f" Values > 0.05 may discard a portion of suitable but"
                              f" rarely used codewords.", UserWarning)

            if perturb_eps > 1e-6:
                warnings.warn(f"`perturb_eps` is set to {perturb_eps}, which"
                              f" is quite large. Values > 1e-6 may cause big"
                              f" perturbation/shift from used codewords.", UserWarning)

        # ---------------- Codebook initialization ----------------
        if uniform_init:
            codebook = (torch.rand((self.num_embeddings, self.embedding_dim))
                        * (1 / self.num_embeddings))
        else:
            codebook = (torch.randn((self.num_embeddings, self.embedding_dim))
                        * (1 / self.num_embeddings))

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data = codebook

        # ---------------- Tensors used for codebook replacement ----------------
        self.register_buffer("codebook_usage", torch.zeros(self.num_embeddings,
                                                           dtype=torch.int32))
        self.register_buffer("iter_counter", torch.zeros(1, dtype=torch.int32))

    # ---------------- Forward pass (Core API) ----------------
    def forward(self, z: torch.Tensor) \
                            -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:

        soft_one_hot = F.gumbel_softmax(z, tau=self.temperature, dim=1, hard=self.straight_through)
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(z, dim=1)
        kl_loss = self.kld_scale * torch.sum(qy * torch.log(qy * self.num_embeddings + 1e-10), dim=1).mean()

        indices = soft_one_hot.argmax(dim=1).flatten()

        # Perplexity Computation
        perplexity = self._compute_perplexity(indices)

        # Track used indices for codebook replacement
        with torch.no_grad():
            self.codebook_usage[indices] += 1
            self.iter_counter += 1
            if self.iter_counter.item() % self.replacement_iters == 0:
                self._replace_unused_entries()  # Applies codebook replacement

        return z_q, indices, perplexity, kl_loss

    # ---------------- Quantization for Inference ----------------
    @torch.no_grad()
    def inference(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Deterministic hard quantization by mapping the input to the nearest codeword.
        Args:
            - z (torch.Tensor): input/latent. shape (N, D)

        Returns:
            - z_q_hard (torch.Tensor): Hard quantized input/latent. shape (N, D)
            - indices (torch.Tensor): Selected codebook indices. shape (N, )
            - perplexity (float): Codebook perplexity (average codebook usage).
        """
        self._check_input(z)

        soft_one_hot = F.gumbel_softmax(z, tau=1e-10, dim=1, hard=self.straight_through)
        z_q_hard = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)

        indices = soft_one_hot.argmax(dim=1).flatten()

        # Perplexity Computation
        perplexity = self._compute_perplexity(indices)

        return z_q_hard, indices, perplexity

    # ---------------- Utility functions ----------------
    def _check_constraints(self) -> None:
        if (self.temperature < 0.0) or (self.temperature > 1.0):
            raise ValueError(f"`temperature` is set to {self.temperature}."
                             f" It must be in the range of 0 < temperature < 1.")

    def _check_input(self, z: torch.Tensor) -> None:
        if z.ndim != 2:
            raise ValueError("GumbelSoftmax input must have the shape of (N, D),"
                             " where N is the No. of input samples,and D is the"
                             " embedding dimensionality.")
        if z.size(1) != self.embedding_dim:
            raise ValueError(f"GumbelSoftmax input.shape[1] must match the embedding"
                             f" dimensionality that is {self.embedding_dim}.")

    def _compute_perplexity(self, indices: torch.Tensor) -> float:
        encodings = torch.zeros(indices.shape[0], self.num_embeddings,
                                device=indices.device)
        encodings.scatter_(1, indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity.item()

    def _replace_unused_entries(self) -> None:
        with torch.no_grad():
            usage_ratio = self.codebook_usage / self.replacement_iters
            unused_indices = torch.where(usage_ratio < self.discard_threshold)[0]
            used_indices = torch.where(usage_ratio >= self.discard_threshold)[0]

            if unused_indices.numel() == 0 or used_indices.numel() == 0:
                self.codebook_usage.zero_()
                return

            unused_count = unused_indices.numel()
            used_probs = self.codebook_usage[used_indices] / torch.sum(
                self.codebook_usage[used_indices])
            randomly_sampled_indices = used_probs.multinomial(
                num_samples=unused_count,
                replacement=True)
            sampled_indices = used_indices[randomly_sampled_indices]
            used_codewords = self.codebook.weight.data[sampled_indices].clone()

            self.codebook.weight.data[unused_indices] = (used_codewords +
                         self.perturb_eps * torch.randn_like(used_codewords)).clone()
            self.codebook_usage.zero_()

            if self.verbose:
                print("\n***** Replaced " + str(unused_count) + " codewords *****")


    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"straight_through={self.straight_through}, "
            f"replacement_iters={self.replacement_iters}, "
            f"discard_threshold={self.discard_threshold}, "
            f"perturb_eps={self.perturb_eps}, "
            f"uniform_init={self.uniform_init}, "
            f"allow_warning={self.allow_warning}, "
            f"verbose={self.verbose}"
        )


class EMA(nn.Module):
    """
    EMA: Exponential Moving Averages module to optimize the codebook of VQ.
    The module encompasses codebook replacement function which discards unused codebook
    entries during training.

    Args:
        - num_embeddings (int): Codebook size (No. of codewords).
        - embedding_dim (int): Dimensionality of embeddings.
        - commitment_coef (float): Weight coefficient for Commitment Loss.
            Recommended commitment_coef = 0.25.
        - decay (float): Decay rate for EMA codebook update. Recommended 0.8 < decay < 1
        - replacement_iters (int): Replacement interval (number of training iterations
            to apply codebook replacement). Recommended 50 < replacement_iters < 300.
        - discard_threshold (float): Threshold to discard the codebook entries that are
            used less than this threshold after "replacement_iters" iteration.
            Recommended 0.01 < discard_threshold < 0.05.
        - perturb_eps (float): Adjusts perturbation/shift magnitude from used codewords
            during codebook replacement.
        - uniform_init (bool): Whether to initialize codebook with uniform distribution.
        - allow_warning (bool): Whether to print the warnings.
        - verbose (bool): Whether to print codebook replacement status.

    Returns:
        - z_q (torch.Tensor): Quantized input/latent. shape (N, D)
        - indices (torch.Tensor): Selected codebook indices. shape (N, )
        - perplexity (float): Codebook perplexity (average codebook usage)
        - loss (torch.Tensor): Commitment loss for VQ.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            commitment_coef: float = 0.25,
            decay: float = 0.99,
            replacement_iters: int = 100,
            discard_threshold: float = 0.01,
            perturb_eps: float = 1e-9,
            uniform_init: bool = True,
            allow_warning: bool = True,
            verbose: bool = True,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_coef = commitment_coef
        self.decay = decay
        self.replacement_iters = replacement_iters
        self.discard_threshold = discard_threshold
        self.perturb_eps = perturb_eps
        self.uniform_init = uniform_init
        self.allow_warning = allow_warning
        self.verbose = verbose

        self.ema_epsilon = 1e-5

        self._check_constraints()

        # ---------------- User warnings ----------------
        if allow_warning:
            if replacement_iters < 50:
                warnings.warn(f"`replacement_iters` is set to"
                              f" {replacement_iters}, which is quite small. Values < 50"
                              f" may cause too early and frequent codebook"
                              f" replacements.", UserWarning)
            elif replacement_iters > 300:
                warnings.warn(f"`replacement_iters` is set to"
                              f" {replacement_iters}, which is quite large."
                              f" Values > 300 may cause too late and sporadic codebook"
                              f" replacements.", UserWarning)

            if discard_threshold > 0.05:
                warnings.warn(f"`discard_threshold` is set to"
                              f" {discard_threshold}, which is quite large."
                              f" Values > 0.05 may discard a portion of suitable but"
                              f" rarely used codewords.", UserWarning)

            if perturb_eps > 1e-6:
                warnings.warn(f"`perturb_eps` is set to {perturb_eps}, which"
                              f" is quite large. Values > 1e-6 may cause big"
                              f" perturbation/shift from used codewords.", UserWarning)

            if decay < 0.8:
                warnings.warn(f"`decay` is set to {decay}."
                              f" Recommended 0.8 < decay < 1", UserWarning)

        # ---------------- Codebook initialization ----------------
        if uniform_init:
            codebook = (torch.rand((self.num_embeddings, self.embedding_dim))
                        * (1 / self.num_embeddings))
        else:
            codebook = (torch.randn((self.num_embeddings, self.embedding_dim))
                        * (1 / self.num_embeddings))

        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.codebook.weight.data = codebook

        # ---------------- Tensors used for EMA codebook update ----------------
        self.register_buffer('_ema_cluster_size', torch.zeros(self.num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))
        self._ema_w.data.normal_()

        # ---------------- Tensors used for codebook replacement ----------------
        self.register_buffer("codebook_usage", torch.zeros(self.num_embeddings,
                                                           dtype=torch.int32))
        self.register_buffer("iter_counter", torch.zeros(1, dtype=torch.int32))

        # ---------------- Forward pass (Core API) ----------------
    def forward(self, z: torch.Tensor) \
                            -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """
        Args:
            - z (torch.Tensor): input/latent. shape (N, D)

        Returns:
            - z_q (torch.Tensor): Quantized input/latent. shape (N, D)
            - indices (torch.Tensor): Selected codebook indices. shape (N, )
            - perplexity (float): Codebook perplexity (average codebook usage).
            - loss (torch.Tensor): Commitment loss for VQ.
        """
        self._check_input(z)

        # Calculate distances
        distances = (torch.sum(z.pow(2), dim=1, keepdim=True)
                     + torch.sum(self.codebook.weight.pow(2), dim=1)
                     - 2 * torch.matmul(z, self.codebook.weight.t()))

        indices = torch.argmin(distances, dim=1)

        # Perplexity Computation
        perplexity = self._compute_perplexity(indices)

        encoding_indices = indices.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        z_quantized = torch.matmul(encodings, self.codebook.weight)

        # Use EMA to update the embedding vectors
        self._ema_cluster_size = self._ema_cluster_size * self.decay + \
                                 (1 - self.decay) * torch.sum(encodings, 0)

        # Laplace smoothing of the cluster size
        n = torch.sum(self._ema_cluster_size.data)
        self._ema_cluster_size = (
                (self._ema_cluster_size + self.ema_epsilon)
                / (n + self.num_embeddings * self.ema_epsilon) * n)

        dw = torch.matmul(encodings.t(), z)
        self._ema_w = nn.Parameter(self._ema_w * self.decay + (1 - self.decay) * dw)

        self.codebook.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(z_quantized.detach(), z) # Commitment Loss
        loss = self.commitment_coef * e_latent_loss

        # Straight Through Estimator
        z_q = z + (z_quantized - z).detach()

        # Track used indices for codebook replacement
        with torch.no_grad():
            self.codebook_usage[indices] += 1
            self.iter_counter += 1
            if self.iter_counter.item() % self.replacement_iters == 0:
                self._replace_unused_entries()  # Applies codebook replacement

        return z_q, indices, perplexity, loss

    # ---------------- Quantization for Inference ----------------
    @torch.no_grad()
    def inference(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Deterministic hard quantization by mapping the input to the nearest codeword.
        Args:
            - z (torch.Tensor): input/latent. shape (N, D)

        Returns:
            - z_q_hard (torch.Tensor): Hard quantized input/latent. shape (N, D)
            - indices (torch.Tensor): Selected codebook indices. shape (N, )
            - perplexity (float): Codebook perplexity (average codebook usage).
        """
        self._check_input(z)

        # Calculate distances
        distances = (torch.sum(z.pow(2), dim=1, keepdim=True)
                     + torch.sum(self.codebook.weight.pow(2), dim=1)
                     - 2 * torch.matmul(z, self.codebook.weight.t()))

        indices = torch.argmin(distances, dim=1)

        # Perplexity Computation
        perplexity = self._compute_perplexity(indices)

        encoding_indices = indices.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings,
                                device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        z_q_hard = torch.matmul(encodings, self.codebook.weight)

        return z_q_hard, indices, perplexity

    # ---------------- Utility functions ----------------
    def _check_constraints(self) -> None:
        if self.commitment_coef < 0.0:
            raise ValueError(f"`commitment_coef` is set to {self.commitment_coef}."
                             f" It must be a positive float value.")

        if (self.decay < 0.0) or (self.decay > 1.0):
            raise ValueError(f"`decay` is set to {self.decay}."
                             f" It must be in the range of 0 < decay < 1."
                             f" Recommended that 0.8 < decay < 1.")

    def _check_input(self, z: torch.Tensor) -> None:
        if z.ndim != 2:
            raise ValueError("EMA input must have the shape of (N, D), where N is"
                             " the No. of input samples,and D is the embedding"
                             " dimensionality.")
        if z.size(1) != self.embedding_dim:
            raise ValueError(f"EMA input.shape[1] must match the embedding"
                             f" dimensionality that is {self.embedding_dim}.")

    def _compute_perplexity(self, indices: torch.Tensor) -> float:
        encodings = torch.zeros(indices.shape[0], self.num_embeddings,
                                device=indices.device)
        encodings.scatter_(1, indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity.item()

    def _replace_unused_entries(self) -> None:
        with torch.no_grad():
            usage_ratio = self.codebook_usage / self.replacement_iters
            unused_indices = torch.where(usage_ratio < self.discard_threshold)[0]
            used_indices = torch.where(usage_ratio >= self.discard_threshold)[0]

            if unused_indices.numel() == 0 or used_indices.numel() == 0:
                self.codebook_usage.zero_()
                return

            unused_count = unused_indices.numel()
            used_probs = self.codebook_usage[used_indices] / torch.sum(
                self.codebook_usage[used_indices])
            randomly_sampled_indices = used_probs.multinomial(
                num_samples=unused_count,
                replacement=True)
            sampled_indices = used_indices[randomly_sampled_indices]
            used_codewords = self.codebook.weight.data[sampled_indices].clone()

            self.codebook.weight.data[unused_indices] = (used_codewords +
                                             self.perturb_eps * torch.randn_like(
                        used_codewords)).clone()
            self.codebook_usage.zero_()

            if self.verbose:
                print("\n***** Replaced " + str(unused_count) + " codewords *****")

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"commitment_coef={self.commitment_coef}, "
            f"decay={self.decay}, "
            f"replacement_iters={self.replacement_iters}, "
            f"discard_threshold={self.discard_threshold}, "
            f"perturb_eps={self.perturb_eps}, "
            f"uniform_init={self.uniform_init}, "
            f"allow_warning={self.allow_warning}, "
            f"verbose={self.verbose}"
        )