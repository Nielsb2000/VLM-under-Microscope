import argparse
try:
    from fastchat.model import add_model_args
except ImportError:
    # Fallback if fastchat is not installed
    def add_model_args(parser):
        parser.add_argument("--model-path", type=str, required=True)
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--num-gpus", type=int, default=1)
        parser.add_argument("--max-gpu-memory", type=str, help="Max GPU memory")
        parser.add_argument("--load-8bit", action="store_true")
        parser.add_argument("--cpu-offloading", action="store_true")
        parser.add_argument("--revision", type=str, default="main")
        parser.add_argument("--dtype", type=str, default=None)

class InferenceArgumentParser:
    def __init__(self, version, description="Inference arg parser."):
        self.version = version
        self.parser = argparse.ArgumentParser(description=description)

        self._add_common_args()

        if version == "lm":
            self._add_lm_args()
        elif version == "vlm":
            self._add_vlm_args()
        else:
            raise ValueError(f"Unknown version: {version}")
    
    def _add_common_args(self):
        # Arguments common to all models
        self.parser.add_argument("--dataset_id", type=str, default="MilaWang/SpatialEval",
                                 help="Dataset identifier for Hugging Face.")
        self.parser.add_argument("--temperature", type=float, default=0.2)
        self.parser.add_argument("--top_p", type=float, default=0.9)
        self.parser.add_argument("--repetition_penalty", type=float, default=1.0)
        self.parser.add_argument("--max_new_tokens", type=int, default=1024)
        self.parser.add_argument("--output_folder", type=str, default="outputs")
        self.parser.add_argument("--task", type=str, default="all", choices=["all", "spatialmap", "mazenav", "spatialgrid", "spatialreal"],
                                 help="Set specific task to evaluate or evaluate all tasks.")
        self.parser.add_argument("--completion", action="store_true", help="Add completion prompt.")
        self.parser.add_argument("--w_reason", action="store_true", help="Add reason prompt.")
        self.parser.add_argument("--first_k", type=int, default=None, help="Test first k samples for each question type. If not specified, test all samples.")
        self.parser.add_argument("--offset_k", type=int, default=0, help="Skip the first offset_k samples before applying first_k. Use for non-overlapping multi-run evaluation.")
        self.parser.add_argument("--runs", type=int, default=1,
                     help="Number of repeated runs with the same images (no random sampling). Use for deterministic multi-run evaluation.")
        self.parser.add_argument("--mc_runs", type=int, default=0,
                     help="Number of Monte Carlo runs with different random subsets of first_k samples. If >0, enables randomized MC evaluation. Requires --first_k.")
        self.parser.add_argument("--mc_seed", type=int, default=42,
                     help="Base random seed for Monte Carlo sampling. Run i uses seed mc_seed+i. Only used if --mc_runs > 0.")
        self.parser.add_argument("--workers", type=int, default=1,
                                 help="Number of parallel workers for API calls. "
                                      "Values >1 speed up GPT/cloud models significantly (e.g. --workers 8). "
                                      "Ignored for local GPU models (always runs sequentially). Default: 1.")
        self.parser.add_argument("--debug", action="store_true", default=False,
                                 help="Enable debug logging of all agent intermediate messages (tool calls, responses). "
                                      "Log is written to spatial_eval/logs/debug_<model>_<variant>_<timestamp>.log.")
    
    def _add_lm_args(self):
        add_model_args(self.parser)
        self.parser.add_argument("--mode", default="tqa", choices=["tqa"], 
                                 help="Set mode for test input modality (only 'tqa' allowed for language model).")
    
    def _add_vlm_args(self):
        self.parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "xpu", "npu"], default="cuda",
                                help="The device type.")
        self.parser.add_argument("--mode", choices=["tqa", "vqa", "vtqa"], 
                                 help="Set mode for test input modality.")
        self.parser.add_argument("--random_image", action="store_true", help="Use random image for inference.")
        self.parser.add_argument("--noise_image", action="store_true", help="Use noise image for inference.")
        self.parser.add_argument("--model_path", type=str,
                                 help="Local model path for storing model checkpoints or model identifier for Hugging Face.")
        self.parser.add_argument("--model_base", type=str, default=None, help="Base model.")
        self.parser.add_argument("--use_skills", action="store_true", default=False,
                                 help="Use deepagents with spatial reasoning skills instead of plain GPT.")
        self.parser.add_argument("--skills_variant", type=str, default=None,
                                 choices=[
                                     "img-only", "img-qa", "img-context",
                                     "img-qa-val",
                                     "img-only-n3", "img-only-n10", "img-only-n30", "img-only-n50", "img-only-n100",
                                     "img-qa-val-v2",
                                     "img-qa-val-v2-offset-n3",
                                     "img-qa-val-v2-offset",
                                     "img-qa-val-v2-offset-n30",
                                     "img-only-tool-n3",
                                     "img-only-tool-n10",
                                     "img-only-tool-n30",
                                     "img-only-annotated",
                                     "img-annotated-context",
                                     "sam3",
                                 ],
                                 help="Which skill variant to use with --use_skills. "
                                      "img-only: image-path examples (3 imgs). "
                                      "img-qa: image + Q&A (biased, 3 imgs). "
                                      "img-context: image + domain context (unbiased, 3 imgs). "
                                      "img-qa-val: validation test — 10 images with full Q&A, tested on same images. "
                                      "img-only-n10/n30/n50/n100: range test — N example images taken from the tail of the dataset (no overlap with test set when --first_k ≤ 100). "
                                      "img-qa-val-v2: preload architecture — agent calls read_example(0..9) before answering (same images, contamination upper-bound). "
                                      "img-qa-val-v2-offset-n3/n10/n30: few-shot preload with N examples + images, test offset by N*3 samples. "
                                      "img-only-tool-n3/n10/n30: image-only preload via tool — agent sees N example images (no answers). "
                                      "offset_k is auto-computed as N×mc_runs (e.g. n30+mc3→offset=90) unless --offset_k is set explicitly. "
                                      "sam3: SAM 3 segmentation tool — agent autonomously decides what to segment based on the question; task-agnostic. "
                                      "If omitted uses the baseline models/skills/ folder.")
        self.parser.add_argument("--use_sam2", action="store_true", default=False,
                                 help="Enable SAM2 segmentation pre-pass: segments the image, extracts bounding-box "
                                      "centroids and mask overlays, and feeds them to the model before answering. "
                                      "Adds '_sam2' suffix to the output filename. Only effective for VQA/VTQA modes "
                                      "with GPT models (gpt-5.5, gpt-5.2, gpt-4o, etc.).")
        self.parser.add_argument("--sam2_grid_n", type=int, default=4,
                                 help="Grid size for undirected SAM2 segmentation (N\u00d7N prompt points). "
                                      "Default 4 gives 16 evenly-spaced points across the image. "
                                      "Only used when --use_sam2 is set.")
        self.parser.add_argument("--sam2_model_id", type=str, default="facebook/sam2.1-hiera-base-plus",
                                 help="SAM2 HuggingFace model ID. Smaller models are faster: "
                                      "facebook/sam2.1-hiera-base-plus (recommended for speed), "
                                      "facebook/sam2.1-hiera-small, facebook/sam2.1-hiera-tiny. "
                                      "Only used when --use_sam2 is set.")

    def parse_args(self):
        return self.parser.parse_args()
