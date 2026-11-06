from ...paths import MODELS_DIR
from ..base import Detector


def patch_clip(CLIPImageEmbeddings):
    """With Transformers 5.x, get_image_features() returns an output object, while NeMo Curator 1.3.0 expects a
    tensor: embed with the vision model's pooled output and visual_projection, then L2-normalise (the same
    embedding as before)."""
    import numpy as np
    import torch

    @torch.no_grad()
    def call(self, images):
        if isinstance(images, np.ndarray):
            pixel_values = self.transforms(torch.from_numpy(images).permute(0, 3, 1, 2).to(self.device))
        else:
            pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        out = self.clip.vision_model(pixel_values=pixel_values, return_dict=True)
        emb = self.clip.visual_projection(out.pooler_output)
        return emb / torch.linalg.vector_norm(emb, dim=-1, keepdim=True).clamp_min(1e-12)

    CLIPImageEmbeddings.__call__ = call


class NemoCurator(Detector):
    """NVIDIA NeMo Curator 1.3.0 image stages: a CLIP ViT-L/14 embedding, then its NSFW scorer (an MLP on the
    embedding). The filter threshold is set above 1 so no image is dropped; the score is kept. Needs CUDA and a
    kernel restart after the install."""
    task, name, model_id, batch_size = "nsfw", "nemo", "nemo-curator==1.3.0 (CLIP ViT-L/14 + NSFW scorer)", 64
    install = ('"nemo-curator[image_cuda12]==1.3.0" pillow',)
    isolated = True                 # it installs its own torch: the machine's torchaudio would not load next to it

    def load(self, device):
        import numpy as np
        from nemo_curator.models.clip import CLIPImageEmbeddings
        from nemo_curator.models.nsfw import NSFWScorer
        from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
        from nemo_curator.stages.image.filters.nsfw_filter import ImageNSFWFilterStage
        from nemo_curator.tasks import ImageBatch, ImageObject
        if device != "cuda":
            raise SystemExit("NeMo Curator needs CUDA")
        self.np, self.ImageBatch, self.ImageObject = np, ImageBatch, ImageObject
        patch_clip(CLIPImageEmbeddings)
        model_dir = MODELS_DIR / "nemo"
        model_dir.mkdir(parents=True, exist_ok=True)
        CLIPImageEmbeddings.download_weights_on_node(str(model_dir))
        NSFWScorer.download_weights_on_node(str(model_dir))
        self.embed = ImageEmbeddingStage(model_dir=str(model_dir), model_inference_batch_size=16,
                                         num_gpus_per_worker=1, verbose=False)
        self.nsfw = ImageNSFWFilterStage(model_dir=str(model_dir), model_inference_batch_size=32,
                                         num_gpus_per_worker=1, score_threshold=1.1, verbose=False)
        self.embed.setup()
        self.nsfw.setup()

    def score(self, images):
        objects = [self.ImageObject(image_path=str(index), image_id=str(index), metadata={},
                                    image_data=self.np.asarray(image)) for index, image in enumerate(images)]
        task = self.nsfw.process(self.embed.process(self.ImageBatch(dataset_name="editjudge", data=objects)))
        by_id = {item.image_id: float(item.nsfw_score) for item in task.data}
        return [by_id[str(index)] for index in range(len(images))]
