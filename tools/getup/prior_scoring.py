"""Frozen-prior scoring shared by training evaluation and local diagnostics."""
import torch


@torch.no_grad()
def score_windows(agent, windows, seed, repeats=4, batch_size=256):
    """Mean of per-noise exp rewards, NOT exp of the averaged SDS error."""
    device = torch.device(agent._device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    raw_sum = norm_sum = reward_sum = None
    count_before = agent._sds_normalizer.get_count().clone()
    with torch.random.fork_rng(devices=devices):
        for repeat in range(repeats):
            torch.random.default_generator.manual_seed(seed + repeat)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed + repeat)
            raw_parts = []
            for chunk in windows.split(batch_size):
                x = chunk.to(device).reshape(len(chunk), agent._env._num_disc_obs_steps, -1)
                x = agent._prior_model.normalize(x).reshape(len(chunk), -1)
                raw_parts.append(agent._prior_model.ESM_SDS_loss(x, t_lst=agent._diffusion_steps))
            raw = torch.cat(raw_parts)
            normalized = agent._sds_normalizer.normalize(raw)
            reward = torch.exp(-normalized.mean(dim=1) * agent._sds_loss_scale) * agent._smp_reward_scale
            raw_sum = raw if raw_sum is None else raw_sum + raw
            norm_sum = normalized if norm_sum is None else norm_sum + normalized
            reward_sum = reward if reward_sum is None else reward_sum + reward
    if not torch.equal(count_before, agent._sds_normalizer.get_count()):
        raise RuntimeError("Diagnostic unexpectedly changed SDS normalizer count.")
    return (raw_sum / repeats).cpu(), (norm_sum / repeats).cpu(), (reward_sum / repeats).cpu()
