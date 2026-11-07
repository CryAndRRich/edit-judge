"""EditReward (Wu et al., ICLR 2026): Qwen2.5-VL-7B fine-tuned as a reward model for instruction-guided image
editing on 200k+ expert comparisons. It reads (source image, instruction, edited image) twice, with two rubric
prompts (instruction following, visual quality); each pass ends at a special <|Reward|> token whose hidden state
goes through its own head, and the score is the mean of the two heads' outputs (column 0; column 1 is a log
deviation).

Code: TIGER-AI-Lab/EditReward at COMMIT, with the settings of its config/EditReward-Qwen2.5-7B-VL.yaml
(ranknet_multi_head, mean pooling, special reward token, output_dim 2) and of EditRewardVLInferencer
(256 x 28 x 28 pixels per image, the same prompts). Changes for 2 x T4, recorded in ../docs/scorer.md:
- float16 instead of bfloat16 (T4 has no bfloat16), sdpa attention (no flash-attn);
- the model is spread over both GPUs (device_map="auto") instead of `model.to(device)` on one GPU (16 GB of
  weights do not fit one T4);
- the checkpoint is copied into the model tensor by tensor (the upstream code loads the whole 16.6 GB state dict
  in memory, more than Kaggle's RAM next to the base model).
"""

from .base import RewardModel

COMMIT = "77a93aaa461fe9187e0ff841b59ecc0d0620bb7f"
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
CHECKPOINT = "TIGER-Lab/EditReward-Qwen2.5-VL-7B"
IMAGE_PIXELS = 256 * 28 * 28


class EditReward(RewardModel):
    name = "editreward"
    model_id = f"{CHECKPOINT} on {BASE_MODEL} (code {COMMIT[:7]})"
    batch_size = 4
    install = (f"git+https://github.com/TIGER-AI-Lab/EditReward.git@{COMMIT} --no-deps",
               "transformers==4.57.0 trl==0.8.6 peft fire omegaconf megfile qwen-vl-utils matplotlib tensorboard "
               "datasets pandas einops opencv-python-headless")

    def load(self, device: str) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from transformers import AutoProcessor

        from EditReward.dataset.data_collator_qwen_edit import prompt_with_special_token
        from EditReward.dataset.prompts import INSTRUCTION_EDIT_FOLLOWING, INSTRUCTION_EDIT_QUALITY
        from EditReward.dataset.utils import process_vision_info
        from EditReward.model.qwen2_5_vl_trainer import Qwen2_5_VLRewardModelBT_MultiHead

        self.prompts = (INSTRUCTION_EDIT_FOLLOWING, INSTRUCTION_EDIT_QUALITY)
        self.suffix, self.process_vision_info = prompt_with_special_token, process_vision_info
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL, padding_side="right")
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<|Reward|>"]})
        special_ids = self.processor.tokenizer.convert_tokens_to_ids(["<|Reward|>"])
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = Qwen2_5_VLRewardModelBT_MultiHead.from_pretrained(
            BASE_MODEL, output_dim=2, reward_token="special", special_token_ids=special_ids,
            rm_head_type="ranknet_multi_head", rm_head_kwargs=None, pooling_strategy="mean", torch_dtype=dtype,
            attn_implementation="sdpa", use_cache=False, device_map="auto" if device == "cuda" else None)
        model.resize_token_embeddings(len(self.processor.tokenizer))
        parameters = dict(model.named_parameters()) | dict(model.named_buffers())
        loaded = set()             # only the weights file (not the whole repository)
        with safe_open(hf_hub_download(CHECKPOINT, "model.safetensors"), framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                name = key.removeprefix("model.") if key.startswith("model.") and key not in parameters else key
                if name not in parameters:
                    raise SystemExit(f"EditReward checkpoint key {key!r} not in the model")
                target = parameters[name]
                with torch.no_grad():
                    target.copy_(checkpoint.get_tensor(key).to(device=target.device, dtype=target.dtype))
                loaded.add(name)
        missing = sorted(set(dict(model.named_parameters())) - loaded)
        if missing:
            raise SystemExit(f"EditReward checkpoint misses {len(missing)} parameters, e.g. {missing[:3]}")
        for head in model.rm_heads.values():
            head.to(torch.float32)
        self.model = model.eval()
        self.input_device = model.get_input_embeddings().weight.device

    def batch_for(self, batch, prompt: str):
        messages = [[{"role": "user", "content": [
            {"type": "image", "image": original, "min_pixels": IMAGE_PIXELS, "max_pixels": IMAGE_PIXELS},
            {"type": "image", "image": edit, "min_pixels": IMAGE_PIXELS, "max_pixels": IMAGE_PIXELS},
            {"type": "text", "text": prompt.format(text_prompt=item.text) + self.suffix}]}]
            for item, original, edit in batch]
        images, _ = self.process_vision_info(messages)
        texts = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.input_device)

    def score(self, batch):
        import torch

        with torch.inference_mode():
            logits = self.model(return_dict=True, batch_dim1=self.batch_for(batch, self.prompts[0]),
                                batch_dim2=self.batch_for(batch, self.prompts[1]))["logits"].float().cpu()
        return [{"score": round(float(row[0]), 5), "log_sigma": round(float(row[1]), 5)} for row in logits]
