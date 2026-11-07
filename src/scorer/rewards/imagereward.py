"""ImageReward (Xu et al., NeurIPS 2023): BLIP ViT-L with a cross-attention text encoder and an MLP, trained on
137k expert comparisons of text-to-image results. It reads the prompt (truncated to 35 tokens) and the generated
image only, never the original: a text-to-image reward model used as a baseline for edits.

Code: THUDM/ImageReward at COMMIT (installed in its own environment). `score` below is the batched form of
`ImageReward.score` in that code (same tokenizer settings, preprocessing, MLP and normalisation by the reward mean
and deviation), for speed.
"""

from ...paths import MODELS_DIR
from .base import RewardModel

COMMIT = "78f8051059b34fef8e0032ccb40561c23a2ddbbb"


class ImageRewardModel(RewardModel):
    name = "imagereward"
    model_id = f"THUDM/ImageReward ImageReward-v1.0 (code {COMMIT[:7]})"
    batch_size = 32
    install = (f"git+https://github.com/THUDM/ImageReward.git@{COMMIT}",)

    def load(self, device: str) -> None:
        import ImageReward as image_reward

        self.device = device
        self.model = image_reward.load("ImageReward-v1.0", device=device,
                                       download_root=str(MODELS_DIR / "imagereward")).eval()

    def score(self, batch):
        import torch

        texts = [item.text for item, _, _ in batch]
        with torch.inference_mode():
            tokens = self.model.blip.tokenizer(texts, padding="max_length", truncation=True, max_length=35,
                                               return_tensors="pt").to(self.device)
            pixels = torch.stack([self.model.preprocess(edit) for _, _, edit in batch]).to(self.device)
            image_embeds = self.model.blip.visual_encoder(pixels)
            image_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)
            output = self.model.blip.text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask,
                                                  encoder_hidden_states=image_embeds,
                                                  encoder_attention_mask=image_mask, return_dict=True)
            rewards = self.model.mlp(output.last_hidden_state[:, 0, :].float())
            rewards = (rewards - self.model.mean) / self.model.std
        return [{"score": round(float(value), 5)} for value in rewards.squeeze(-1).tolist()]
