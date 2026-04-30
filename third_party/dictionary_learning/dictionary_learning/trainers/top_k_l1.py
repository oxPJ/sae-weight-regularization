from collections import namedtuple

import torch as t

from .top_k import TopKTrainer


class TopKTrainerL1(TopKTrainer):
    def __init__(self, *args, weight_l1: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_l1 = weight_l1

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

        weight_norm = self.ae.encoder.weight.abs().sum() + self.ae.decoder.weight.abs().sum()
        weight_penalty = self.weight_l1 * weight_norm

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
                    "l1_w": weight_penalty.item(),
                },
            )

    @property
    def config(self):
        config = super().config.copy()
        config["trainer_class"] = "TopKTrainerL1"
        config["l1_w"] = self.weight_l1
        return config


