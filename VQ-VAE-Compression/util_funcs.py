import torch
import numpy as np

def cbr_new(codebooks_used, codebooks, num_batches, discarding_threshold, eps, embedding_dim):
    with torch.no_grad():
        unused_indices = torch.where((codebooks_used.cpu() / num_batches) < discarding_threshold)[0]
        used_indices = torch.where((codebooks_used.cpu() / num_batches) >= discarding_threshold)[0]

        unused_count = unused_indices.shape[0]
        used_probs = codebooks_used[used_indices] / torch.sum(codebooks_used[used_indices])
        sampled_indices = torch.from_numpy(np.random.choice(used_indices.numpy(), size=(unused_count,), p=used_probs.numpy()))
        used_codebooks = codebooks[sampled_indices].clone()

        codebooks[unused_indices] *= 0
        codebooks[unused_indices] += used_codebooks[range(unused_count)] + eps * torch.randn((unused_count, embedding_dim), device=used_codebooks.device).clone()

        print(f'\n************* Replaced ' + str(unused_count) + f' codewords *************')
        codebooks_used[:] = 0

    return codebooks_used, codebooks


def set_temperature(epoch, max_epoch=100, start_temp=1.0, min_temp=0.1):
    """
    Sets temperature based on exponential decay from start_temp to min_temp over max_epoch.
    """
    decay_rate = (min_temp / start_temp) ** (1.0 / max_epoch)
    temperature = max(start_temp * (decay_rate ** epoch), min_temp)
    return temperature