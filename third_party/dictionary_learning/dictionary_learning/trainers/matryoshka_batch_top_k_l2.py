from collections import namedtuple

import torch as t

from .matryoshka_batch_top_k import MatryoshkaBatchTopKTrainer


class MatryoshkaBatchTopKTrainerL2(MatryoshkaBatchTopKTrainer):
    def __init__(self, *args, weight_l2: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_l2 = weight_l2

    def loss(self, x, step=None, logging=False):
        f, active_indices_F, post_relu_acts_BF = self.ae.encode(
            x, return_active=True, use_threshold=False
        )

        if step is not None and step > self.threshold_start_step:
            self.update_threshold(f)

        x_reconstruct = t.zeros_like(x) + self.ae.b_dec
        total_l2_loss = 0.0
        l2_losses = t.tensor([]).to(self.device)

        W_dec_chunks = t.split(self.ae.W_dec, self.ae.group_sizes.tolist(), dim=0)
        f_chunks = t.split(f, self.ae.group_sizes.tolist(), dim=1)

        for i in range(self.ae.active_groups):
            W_dec_slice = W_dec_chunks[i]
            acts_slice = f_chunks[i]
            x_reconstruct = x_reconstruct + acts_slice @ W_dec_slice

            l2_loss = (x - x_reconstruct).pow(2).sum(dim=-1).mean() * self.group_weights[i]
            total_l2_loss += l2_loss
            l2_losses = t.cat([l2_losses, l2_loss.unsqueeze(0)])

        min_l2_loss = l2_losses.min().item()
        max_l2_loss = l2_losses.max().item()
        mean_l2_loss = l2_losses.mean()

        self.effective_l0 = self.k

        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        auxk_loss = self.get_auxiliary_loss(
            (x - x_reconstruct).detach(), post_relu_acts_BF
        )

        weight_norm = self.ae.W_enc.pow(2).sum() + self.ae.W_dec.pow(2).sum()
        weight_penalty = self.weight_l2 * weight_norm

        loss = mean_l2_loss + self.auxk_alpha * auxk_loss + weight_penalty

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_reconstruct,
                f,
                {
                    "l2_loss": mean_l2_loss.item(),
                    "auxk_loss": auxk_loss.item(),
                    "loss": loss.item(),
                    "min_l2_loss": min_l2_loss,
                    "max_l2_loss": max_l2_loss,
                    "l2_w": weight_penalty.item(),
                },
            )

    def update(self, step, x):
        if step == 0:
            median = self.geometric_median(x)
            self.ae.b_dec.data = median

        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # Skip gradient projection and decoder normalization to allow
        # weight regularization to have its intended effect on decoder weights
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.update_annealed_k(step, self.ae.activation_dim, self.k_anneal_steps)

        return loss.item()

    @property
    def config(self):
        config = super().config.copy()
        config["trainer_class"] = "MatryoshkaBatchTopKTrainerL2"
        config["l2_w"] = self.weight_l2
        return config
