"""Accumulate token-weighted gradients over bounded minibatches."""

from .batch import autocast_context, loss_for_batch


def backward_minibatches(model, cpu_batch, model_device, *, minibatch_size, use_bf16):
    total_tokens = int(cpu_batch.target_mask.sum().item())
    if total_tokens <= 0:
        raise ValueError("packed batches must contain valid targets")
    loss_sum = None
    for start in range(0, len(cpu_batch.input_ids), minibatch_size):
        cpu_minibatch = cpu_batch.slice(start, start + minibatch_size)
        valid_tokens = int(cpu_minibatch.target_mask.sum().item())
        if valid_tokens == 0:
            continue
        batch = cpu_minibatch.to(model_device)
        with autocast_context(model_device, use_bf16):
            loss = loss_for_batch(model, batch, valid_tokens)
        (loss * (valid_tokens / total_tokens)).backward()
        contribution = loss.detach() * valid_tokens
        loss_sum = contribution if loss_sum is None else loss_sum + contribution
    return loss_sum / total_tokens, total_tokens
