"""VLM judges with the VIEScore protocol as EditScore uses it (Ku et al., ACL 2024; EditScore, ICLR 2026):
- semantic consistency (SC): the model sees the original and the edited image and the instruction, and gives
  [editing success, degree of overediting] from 0 to SCORE_RANGE, after a short reasoning;
- perceptual quality (PQ): the model sees the edited image only and gives [naturalness, artifacts];
- each part is min(scores) / (SCORE_RANGE / 10), and the overall score is sqrt(SC * PQ), from 0 to 10.

Two judges share this code, so their difference is what EditScore's fine-tuning adds:
- `qwen-judge`: Qwen2.5-VL-7B-Instruct zero-shot;
- `editscore`: the same model with the EditScore-7B LoRA merged in (EditScore/EditScore-7B, trained on edit
  preferences).

Code: VectorSpaceLab/EditScore at COMMIT (its prompts, its chat template with <img1>/<img2> labels and its output
parser are imported, not copied). Changes for speed and repeatability on 2 x T4, recorded in ../docs/scorer.md:
- float16 instead of bfloat16, the 7B model spread over both GPUs (device_map="auto", as upstream);
- greedy decoding instead of sampling at temperature 0.7 (num_pass 1, as upstream's default), so runs repeat;
- the SC and PQ prompts of a batch of edits in one batched `generate` call (upstream: one call per prompt);
- images limited to MAX_PIXELS (512 x 512, the input size of EditScore's own RL setup), about 330 visual tokens;
- an answer that cannot be parsed scores the middle of the scale (as upstream's batch_evaluate) and is marked
  "parsed": false, instead of upstream's retry with sampling and random guess.
"""

import math

from .base import RewardModel

COMMIT = "4609c5d2ebb62fdebf665d3c924686d896ef1f74"
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
SCORE_RANGE = 25
MAX_PIXELS = 512 * 512
MIN_PIXELS = 64 * 28 * 28
MAX_NEW_TOKENS = 512


class VIEJudge(RewardModel):
    lora: str | None = None
    batch_size = 4
    install = ("transformers>=4.56,<5 accelerate",
               f"git+https://github.com/VectorSpaceLab/EditScore.git@{COMMIT} --no-deps",
               "json-repair regex qwen-vl-utils peft")

    def load(self, device: str) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        from editscore import vie_prompts
        from editscore.mllm_tools.qwen25vl import apply_chat_template
        from editscore.utils import mllm_output_to_dict

        dtype = torch.float16 if device == "cuda" else torch.float32
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, dtype=dtype, device_map="auto" if device == "cuda" else None)
        if self.lora:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.lora).merge_and_unload()
        self.model = model.eval()
        self.processor = AutoProcessor.from_pretrained(
            BASE_MODEL, size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS})
        self.processor.tokenizer.padding_side = "left"          # batched generation
        context = vie_prompts._context_no_delimit_reasoning_first
        # exactly as EditScore.__init__ builds them (the "10" of the scale becomes SCORE_RANGE)
        self.consistency_prompt = "\n".join([context, vie_prompts._prompts_0shot_two_image_edit_rule,
                                             vie_prompts._prompts_0shot_tie_rule_SC.replace("10", str(SCORE_RANGE))])
        self.quality_prompt = "\n".join([context, vie_prompts._prompts_0shot_rule_PQ.replace("10", str(SCORE_RANGE))])
        self.chat_template, self.parse = apply_chat_template, mllm_output_to_dict

    def generate(self, texts: list[str], images: list) -> list[str]:
        import torch

        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return [answer.strip() for answer in self.processor.batch_decode(
            output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)]

    def scores_of(self, answer: str, text: str) -> tuple[list[float], bool]:
        parsed = self.parse(answer, give_up_parsing=False, text_prompt=text, score_range=SCORE_RANGE)
        values = parsed.get("score") if isinstance(parsed, dict) else None
        try:
            values = [float(value) for value in values]
        except (TypeError, ValueError):
            values = None
        if not values or any(not 0 <= value <= SCORE_RANGE for value in values):
            return [SCORE_RANGE / 2], False
        return values, True

    def score(self, batch):
        texts, images = [], []
        for item, original, edit in batch:              # SC prompts: original and edited image
            texts.append(self.chat_template(self.consistency_prompt.replace("<instruction>", item.text), 2))
            images += [original, edit]
        for item, _, edit in batch:                     # PQ prompts: the edited image only
            texts.append(self.chat_template(self.quality_prompt, 1))
            images.append(edit)
        answers = self.generate(texts, images)
        results = []
        for index, (item, _, _) in enumerate(batch):
            consistency_answer, quality_answer = answers[index], answers[len(batch) + index]
            consistency, consistency_parsed = self.scores_of(consistency_answer, item.text)
            quality, quality_parsed = self.scores_of(quality_answer, item.text)
            scale = SCORE_RANGE / 10
            semantic, perceptual = min(consistency) / scale, min(quality) / scale
            results.append({"score": round(math.sqrt(semantic * perceptual), 4),
                            "prompt_following": round(consistency[0] / scale, 3),
                            "overediting": round(consistency[1] / scale, 3) if len(consistency) > 1 else None,
                            "perceptual_quality": round(perceptual, 3),
                            "parsed": consistency_parsed and quality_parsed,
                            "sc_answer": consistency_answer[:1500], "pq_answer": quality_answer[:1500]})
        return results


class QwenJudge(VIEJudge):
    name = "qwen-judge"
    model_id = f"{BASE_MODEL}, zero-shot, EditScore prompts (code {COMMIT[:7]})"


class EditScoreJudge(VIEJudge):
    name = "editscore"
    lora = "EditScore/EditScore-7B"
    model_id = f"{lora} LoRA on {BASE_MODEL} (code {COMMIT[:7]})"
