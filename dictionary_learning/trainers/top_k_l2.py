from collections import namedtuple

import torch as t

from .top_k import TopKTrainer, geometric_median


class TopKTrainerL2(TopKTrainer):
    def __init__(self, *args, weight_l2: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_l2 = weight_l2

    def loss(self, x, step=None, logging=False):
        f, top_acts_BK, top_indices_BK, post_relu_acts_BF = self.ae.encode(
            x, return_topk=True, use_threshold=False
        )

        if step is not None and step > self.threshold_start_step:
            self.update_threshold(top_acts_BK)

        x_hat = self.ae.decode(f)
        e = x - x_hat

        self.effective_l0 = top_acts_BK.size(1)

        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[top_indices_BK.flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = (
            self.get_auxiliary_loss(e.detach(), post_relu_acts_BF)
            if self.auxk_alpha > 0
            else 0
        )

        weight_norm = self.ae.encoder.weight.pow(2).sum() + self.ae.decoder.weight.pow(2).sum()
        weight_penalty = self.weight_l2 * weight_norm

        loss = l2_loss + self.auxk_alpha * auxk_loss + weight_penalty

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "l2_loss": l2_loss.item(),
                    "auxk_loss": auxk_loss.item() if isinstance(auxk_loss, t.Tensor) else float(auxk_loss),
                    "loss": loss.item(),
                    "l2_w": weight_penalty.item(),
                },
            )

    def update(self, step, x):
        # Initialise the decoder bias
        if step == 0:
            median = geometric_median(x)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median

        # compute the loss
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # Skip gradient projection and decoder normalization to allow
        # weight regularization to have its intended effect on decoder weights
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        # do a training step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.update_annealed_k(step, self.ae.activation_dim, self.k_anneal_steps)

        return loss.item()

    @property
    def config(self):
        config = super().config.copy()
        config["trainer_class"] = "TopKTrainerL2"
        config["l2_w"] = self.weight_l2
        return config
