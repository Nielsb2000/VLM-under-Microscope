from PIL import Image
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, InstructBlipProcessor, InstructBlipForConditionalGeneration
import json
import os
import warnings
import concurrent.futures
import threading
import time
from utils.format_filename import format_output_path_vlm
from configs.inference_configs import InferenceArgumentParser
from datasets import load_dataset
from sam2_integration import build_sam2_result_basic, boxes_to_text, centroids_to_text, load_sam2_predictor, DEFAULT_SAM2_MODEL_ID

IMAGE_TOKEN_INDEX = -200

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def format_bunny_vqa_prompt_hf(text, args):
    if args.w_reason:
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{text}\nFirst, provide a concise answer in one sentence. Then, elaborate on the reasoning behind your answer in a detailed, step-by-step explanation. ASSISTANT:"
    elif args.completion:
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{text} Answer: ASSISTANT:"
    else:
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{text} ASSISTANT:"

def format_bunny_tqa_prompt_hf(text, args):
    if args.w_reason:
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {text}\nFirst, provide a concise answer in one sentence. Then, elaborate on the reasoning behind your answer in a detailed, step-by-step explanation. ASSISTANT:"
    elif args.completion:
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {text}\n Answer: ASSISTANT:"
    else:
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {text} ASSISTANT:"

def load_bunny_model_tokenizer(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    return model, tokenizer

def load_qwen_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # use bf16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
    # use fp16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
    # use cpu only
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
    # use cuda device
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=args.device, trust_remote_code=True).eval()

    return model, tokenizer

def load_cog_model_tokenizer(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(args.device).eval()
    tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    return model, tokenizer

def load_instructblip_model_processor(args):
    model = InstructBlipForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    processor = InstructBlipProcessor.from_pretrained(args.model_path)

    return model, processor


def _sam2_composite_path(output_file_path: str, task: str, item_id: str) -> str:
    out_dir = os.path.join(os.path.dirname(str(output_file_path)), "sam2_overlays")
    os.makedirs(out_dir, exist_ok=True)
    safe_id = item_id.replace("/", "_").replace(".", "_")
    return os.path.join(out_dir, f"{task}_{safe_id}_sam2.png")


@torch.inference_mode()
def main(args, model, processor, dataset, output_file_path, sam2_predictor=None, sam2_device=None):
    question_groups = {}

    for item in dataset:
        question_id = item['id'].split('.')[-1]

        if question_id not in question_groups:
            question_groups[question_id] = []
        
        question_groups[question_id].append(item)

    # ── Determine which image indices to evaluate ──────────────────────────
    # Collect all available image indices (position 2 in "task.mode.idx.qtype")
    all_img_indices = sorted({
        int(it['id'].split('.')[2])
        for group in question_groups.values()
        for it in group
    })
    offset = getattr(args, 'offset_k', 0) or 0
    pool = all_img_indices[offset:]

    rng = getattr(args, '_mc_rng', None)
    if args.first_k is None:
        selected_indices = set(pool)
    elif rng is not None:
        # Monte Carlo mode: random sample without replacement
        k = min(args.first_k, len(pool))
        selected_indices = set(rng.sample(pool, k))
    else:
        # Default sequential mode (unchanged behaviour)
        selected_indices = set(pool[:args.first_k])

    # Flatten all items into one ordered list for easier parallel dispatch
    flat_items = []
    for question_id, items in question_groups.items():
        sliced = [it for it in items if int(it['id'].split('.')[2]) in selected_indices]
        flat_items.extend(sliced)

    _image_cache = {}
    _sam2_result_cache: dict = {}   # img_index → Sam2TaskResult (shared across question types)
    _sam2_lock = threading.Lock()   # serialises SAM2 GPU calls; GPT answers can run in parallel
    sam2_enabled = getattr(args, 'use_sam2', False)

    # ── Parallelism: GPT models are I/O-bound; SAM2 GPU calls are serialised via lock ─
    workers = getattr(args, 'workers', 1)
    is_gpt = "gpt" in args.model_path.lower()
    effective_workers = workers if (workers > 1 and is_gpt) else 1
    if workers > 1 and sam2_enabled:
        print(f"Note: SAM2 enabled with {effective_workers} workers — SAM2 GPU calls serialised, GPT answers parallel.")
    elif workers > 1 and not is_gpt:
        print(f"Note: --workers={workers} ignored for non-GPT models; running sequentially.")

    def _process_item(item):
        """Prepare prompt + image for one dataset item, call the model, return result dict."""
        item_id = item['id']

        if args.mode == "tqa":
            image_path = None
        else:
            if args.random_image:
                if "mazenav" in item_id.lower():
                    image_path = "assets/random_maze_nav.png"
                elif "spatialgrid" in item_id.lower():
                    image_path = "assets/random_spatial_grid.png"
                elif "spatialmap" in item_id.lower():
                    image_path = "assets/random_spatial_map.png"
                else:
                    raise ValueError(f"Unknown dataset type for random image: {args.task}")
            elif args.noise_image:
                image_path = "assets/noise.png"
            else:
                image_path = item['image']

        if args.mode == "tqa":
            image = None
        else:
            image = _image_cache.get(image_path) if (isinstance(image_path, str) and image_path in _image_cache) else load_image(image_path)

        if "bunny" in args.model_path.lower() and "merged" not in args.model_path.lower():
            if args.mode == "tqa":
                prompt = format_bunny_tqa_prompt_hf(item['text'], args)
            else:
                prompt = format_bunny_vqa_prompt_hf(item['text'], args)
        elif any(k in args.model_path.lower() for k in ("qwen", "cog", "instructblip", "llava", "merged", "gpt")):
            if args.w_reason:
                prompt = f"{item['text']}\nFirst, provide a concise answer in one sentence. Then, elaborate on the reasoning behind your answer in a detailed, step-by-step explanation."
            elif args.completion:
                prompt = f"{item['text']}\nAnswer:"
            else:
                prompt = item['text']
        else:
            raise ValueError(f"The maze dataset does not support the model {args.model_path}.")

        if "bunny" in args.model_path.lower() and "merged" not in args.model_path.lower():
            if image is not None:
                text_chunks = [processor(chunk).input_ids for chunk in prompt.split('<image>')]
                input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0)
                image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)
                output_ids = model.generate(input_ids, images=image_tensor, use_cache=True)[0]
                answer_text = processor.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
            else:
                input_ids = tokenizer_image_token(prompt, processor, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).to(model.device)
                output_ids = model.generate(input_ids, use_cache=True)[0]
                answer_text = processor.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

        elif "qwen" in args.model_path.lower():
            if args.mode == "tqa":
                query = processor.from_list_format([{'text': prompt}])
            else:
                query = processor.from_list_format([
                    {'image': image_path},
                    {'text': prompt},
                ])
            answer_text, history = model.chat(processor, query=query, history=None)
        elif "cog" in args.model_path.lower():
            history = []
            if args.mode == "tqa":
                text_only_first_query = True
                
            history = []
            if image is None:
                if text_only_first_query:
                    text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"
                    query = text_only_template.format(prompt)
                    text_only_first_query = False
                else:
                    old_prompt = ''
                    for _, (old_query, response) in enumerate(history):
                        old_prompt += old_query + " " + response + "\n"
                    query = old_prompt + "USER: {} ASSISTANT:".format(query)

            if image is None:
                input_by_model = model.build_conversation_input_ids(processor, query=query, history=history, template_version='base')
            else:
                input_by_model = model.build_conversation_input_ids(processor, query=prompt, history=history,
                                                                images=[image])
            inputs = {
                'input_ids': input_by_model['input_ids'].unsqueeze(0).to(args.device),
                'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(args.device),
                'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(args.device),
                'images': [[input_by_model['images'][0].to(args.device).to(torch.bfloat16)]] if image is not None else None,
            }
            if 'cross_images' in input_by_model and input_by_model['cross_images']:
                inputs['cross_images'] = [[input_by_model['cross_images'][0].to(args.device).to(torch.bfloat16)]]

            gen_kwargs = {"max_length": 2048, "do_sample": False}
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                answer_text = processor.decode(outputs[0])
                answer_text = answer_text.split("</s>")[0]

        elif "instructblip" in args.model_path.lower():
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(args.device)

            output_id = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            )
            answer_text = processor.batch_decode(output_id, skip_special_tokens=True)[0].strip()

        elif "llava" in args.model_path.lower() or ("bunny" in args.model_path.lower() and "merged" in args.model_path.lower()) or "gpt" in args.model_path.lower():
            sam2_meta = None
            sam2_composite_path = ""
            gpt_prompt = prompt
            gpt_image = image
            if sam2_enabled and image is not None:
                _cache_key = item_id.split('.')[2]  # image index — reliable across question types
                sam2_result = _sam2_result_cache.get(_cache_key)
                if sam2_result is None:
                    with _sam2_lock:  # serialise SAM2 GPU calls; double-check after acquiring
                        sam2_result = _sam2_result_cache.get(_cache_key)
                        if sam2_result is None:
                            sam2_result = build_sam2_result_basic(
                                image,
                                task=args.task,
                                sam2_model_id=getattr(args, 'sam2_model_id', DEFAULT_SAM2_MODEL_ID),
                                device=sam2_device or args.device,
                                predictor=sam2_predictor,
                                predictor_device=sam2_device,
                                grid_n=getattr(args, 'sam2_grid_n', 4),
                            )
                            _sam2_result_cache[_cache_key] = sam2_result
                else:
                    print(f"  [sam2] cache hit img={_cache_key}")
                sam2_composite_path = _sam2_composite_path(str(output_file_path), args.task, item_id)
                sam2_result.composite_image.save(sam2_composite_path)
                sam2_meta = {
                    "task": sam2_result.task,
                    "landmarks_canvas": sam2_result.landmarks_canvas,
                    "landmarks": sam2_result.landmarks,
                    "centroids": {
                        k: [[cx, cy, score] for (cx, cy, score) in v]
                        for k, v in sam2_result.centroids.items()
                    },
                    "boxes": {
                        k: [[x0, y0, x1, y1, score] for (x0, y0, x1, y1, score) in v]
                        for k, v in sam2_result.boxes.items()
                    },
                }
                gpt_prompt = (
                    f"{prompt}\n\n"
                    f"SAM2 centroids:\n{centroids_to_text(sam2_result.centroids)}\n\n"
                    f"SAM2 boxes:\n{boxes_to_text(sam2_result.boxes)}"
                )
                gpt_image = sam2_result.composite_image
            _t_ans = time.perf_counter()
            prompt, answer_text = model.generate(gpt_prompt, gpt_image, args.temperature)
            if sam2_enabled and image is not None:
                print(f"  [sam2] answer generation:  {time.perf_counter() - _t_ans:.1f}s")

        else:
            raise ValueError(f"Model id {args.model_path} is not supported.")

        result = {
            "id": item_id,
            "answer": answer_text,
            "oracle_answer": item['oracle_answer'],
            "oracle_option": item['oracle_option'],
            "oracle_full_answer": item['oracle_full_answer'],
            "prompt": prompt,
            "image": image_path if isinstance(image_path, str) else "",
        }
        if sam2_enabled and image is not None and sam2_meta is not None:
            result.update({
                "sam2_enabled": True,
                "sam2_overlay_path": sam2_composite_path,
                "sam2_meta": sam2_meta,
            })
        return result

    # ── Execute: parallel for GPT models, sequential for local/GPU models ──
    results = [None] * len(flat_items)

    if effective_workers > 1:
        print(f"Running {len(flat_items)} items with {effective_workers} parallel workers...")
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_idx = {
                executor.submit(_process_item, item): i
                for i, item in enumerate(flat_items)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                completed += 1
                if completed % 10 == 0 or completed == len(flat_items):
                    print(f"Completed {completed}/{len(flat_items)} items.")
    else:
        for index, item in enumerate(flat_items):
            results[index] = _process_item(item)
            if index % 10 == 0:
                print(f"Processed {index} items.")
                print(f"{results[index]['prompt']}")
                print(f"{results[index]['answer']}")

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for result in results:
            if result is not None:
                outfile.write(json.dumps(result) + '\n')

    print(f"Results saved to {output_file_path}")


def _compute_run_accuracy(jsonl_path) -> float:
    """Compute accuracy from a finished JSONL output file using the same logic as evaluation.py."""
    import sys as _sys
    _eval_dir = os.path.join(os.path.dirname(__file__), "evals")
    if _eval_dir not in _sys.path:
        _sys.path.insert(0, _eval_dir)
    from evals.evaluation import _check_answer  # noqa: E402
    correct = total = 0
    with open(jsonl_path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            c, _ = _check_answer(row.get("answer", ""), row)
            correct += c
            total += 1
    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    args = InferenceArgumentParser("vlm").parse_args()

    dataset = load_dataset(args.dataset_id, args.mode, split="test")
    if args.task != "all":
        dataset = dataset.filter(lambda x: args.task in x['id'])

    from utils.load_image import load_image
    
    _PRELOAD_VARIANTS = (
        "img-qa-val-v2",
        "img-qa-val-v2-offset-n3", "img-qa-val-v2-offset", "img-qa-val-v2-offset-n30",
        "img-only-tool-n3", "img-only-tool-n10", "img-only-tool-n30",
    )
    if ("gpt-4" in args.model_path.lower() or "gpt4" in args.model_path.lower() or "gpt-5" in args.model_path.lower() or "gpt-5.5" in args.model_path.lower()) and getattr(args, 'use_skills', False) and getattr(args, 'skills_variant', None) in _PRELOAD_VARIANTS:
        from models.deepagent_preload_model import DeepAgentPreload
        variant = getattr(args, 'skills_variant', None)
        img_only = variant.startswith("img-only-tool")
        fewshot = not img_only and variant != "img-qa-val-v2"
        n_examples = {
            "img-qa-val-v2": 10,
            "img-qa-val-v2-offset-n3": 3, "img-qa-val-v2-offset": 10, "img-qa-val-v2-offset-n30": 30,
            "img-only-tool-n3": 3, "img-only-tool-n10": 10, "img-only-tool-n30": 30,
        }[variant]
        # Auto-compute offset for img-only-tool variants so test images never
        # overlap with the N example images.  Formula: n_examples × mc_runs
        # reserves enough pool space for all MC runs combined.
        # (n3,mc=3→9; n10,mc=3→30; n30,mc=3→90). Only applied when the user
        # has not explicitly set --offset_k.
        if img_only and getattr(args, 'offset_k', 0) == 0:
            auto_offset = n_examples * getattr(args, 'mc_runs', 1)
            args.offset_k = auto_offset
            print(f"[img-only-tool] auto offset_k={auto_offset} "
                  f"(n_examples={n_examples} × mc_runs={getattr(args, 'mc_runs', 1)})")
        model = DeepAgentPreload(model_name=args.model_path, max_tokens=args.max_new_tokens,
                                 task=args.task, fewshot=fewshot, n_examples=n_examples,
                                 img_only=img_only)
        processor = None
    elif ("gpt-4" in args.model_path.lower() or "gpt4" in args.model_path.lower() or "gpt-5" in args.model_path.lower() or "gpt-5.5" in args.model_path.lower()) and getattr(args, 'use_skills', False) and getattr(args, 'skills_variant', None) == "sam3":
        from models.deepagent_sam3_model import DeepAgentSAM3
        model = DeepAgentSAM3(model_name=args.model_path, max_tokens=args.max_new_tokens)
        processor = None
    elif ("gpt-4" in args.model_path.lower() or "gpt4" in args.model_path.lower() or "gpt-5" in args.model_path.lower() or "gpt-5.5" in args.model_path.lower()) and getattr(args, 'use_skills', False):
        from models.deepagent_model import DeepAgentGPT
        _debug_log_path = None
        if getattr(args, 'debug', False):
            import datetime as _dt
            _variant_tag = getattr(args, 'skills_variant', None) or 'default'
            _model_tag = args.model_path.replace('/', '-')
            _ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            _debug_log_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "logs",
                f"debug_{_model_tag}_{_variant_tag}_{args.task}_{args.mode}_{_ts}.log",
            )
            print(f"[debug] Agent message trace will be written to: {_debug_log_path}")
        model = DeepAgentGPT(model_name=args.model_path, max_tokens=args.max_new_tokens,
                             skills_variant=getattr(args, 'skills_variant', None),
                             debug_log_path=_debug_log_path)
        processor = None
    elif "gpt-4" in args.model_path.lower() or "gpt4" in args.model_path.lower() or "gpt-5" in args.model_path.lower() or "gpt-5.5" in args.model_path.lower():
        from models.gpt4_model import GPT4Vision
        model = GPT4Vision(model_name=args.model_path, max_tokens=args.max_new_tokens)
        processor = None
        
    elif "llava" in args.model_path.lower():
        from models.llava_model import Llava
        model = Llava(args.model_path, args.model_base)
        processor = None
    
    elif "bunny" in args.model_path.lower() and "merged" in args.model_path.lower():
        # support local model, assume we name the model with merged suffix, e.g., bunny-phi-2-eva-merged
        from models.bunny_model import Bunny, get_bunny_model_type
        model_name = args.model_path.split("/")[-1]
        model_type = get_bunny_model_type(model_name)
        model = Bunny(args.model_path, args.model_base, model_type)
        processor = None
        
    elif "bunny" in args.model_path.lower() and "merged" not in args.model_path.lower():
        # generally support bunny models from huggingface
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        warnings.filterwarnings('ignore')
        torch.set_default_device(args.device)
        model, processor = load_bunny_model_tokenizer(args)
        
    elif "qwen" in args.model_path.lower():
        model, processor = load_qwen_model_tokenizer(args)
        
    elif "cog" in args.model_path.lower():
        model, processor = load_cog_model_tokenizer(args)
        
    elif "instructblip" in args.model_path.lower():
        model, processor = load_instructblip_model_processor(args)
    else:
        raise ValueError(f"Model {args.model_path} is not supported.")

    # ── SAM2 predictor: load once, reuse across all runs ───────────────────
    _sam2_predictor = None
    _sam2_device = None
    if getattr(args, 'use_sam2', False):
        _sam2_model_id = getattr(args, 'sam2_model_id', DEFAULT_SAM2_MODEL_ID)
        _sam2_predictor, _sam2_device = load_sam2_predictor(
            _sam2_model_id,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    # --- Support both deterministic runs and MC randomized runs ---
    import statistics as _statistics
    mc_runs = getattr(args, 'mc_runs', 0)
    runs = getattr(args, 'runs', 1)

    if mc_runs > 0:
        if args.first_k is None:
            raise ValueError("--mc_runs requires --first_k to be set")
        import random as _random
        run_accuracies = []
        run_times = []
        for run_i in range(mc_runs):
            args._mc_run_idx = run_i
            args._mc_seed_i = args.mc_seed + run_i
            args._mc_rng = _random.Random(args._mc_seed_i)
            print(f"\n=== MC run {run_i + 1}/{mc_runs} (seed={args._mc_seed_i}) ===")

            output_path = format_output_path_vlm(args)
            t_start = time.perf_counter()
            main(args, model, processor, dataset, output_path, sam2_predictor=_sam2_predictor, sam2_device=_sam2_device)
            elapsed = time.perf_counter() - t_start
            run_times.append(elapsed)

            # Write timing sidecar alongside the JSONL
            timing_path = str(output_path).replace('.jsonl', '.timing.json')
            with open(timing_path, 'w') as _tf:
                json.dump({
                    "run_i": run_i,
                    "seed": getattr(args, '_mc_seed_i', args.mc_seed),
                    "elapsed_seconds": round(elapsed, 2),
                    "n_items": getattr(args, 'first_k', None),
                    "task": args.task,
                    "variant": getattr(args, 'skills_variant', None),
                }, _tf, indent=2)

            acc = _compute_run_accuracy(output_path)
            run_accuracies.append(acc)
            print(f"  Run {run_i + 1} accuracy: {acc:.2%}  time: {elapsed:.1f}s")

        mean_acc = _statistics.mean(run_accuracies)
        std_acc = _statistics.stdev(run_accuracies) if len(run_accuracies) > 1 else 0.0
        print(f"\n=== MC Summary ({mc_runs} runs, first_k={args.first_k} per q-type) ===")
        print(f"  Mean accuracy : {mean_acc:.2%}")
        print(f"  Std deviation : {std_acc:.2%}")
        print(f"  Per-run accs  : {', '.join(f'{a:.2%}' for a in run_accuracies)}")
        print(f"\n=== Timing Summary ===")
        for i, t in enumerate(run_times):
            seed_i = args.mc_seed + i
            print(f"  Run {i + 1}: {t:.1f}s  (seed={seed_i})")
        total_t = sum(run_times)
        mean_t = _statistics.mean(run_times)
        print(f"  Total: {total_t:.1f}s  Mean: {mean_t:.1f}s")
    elif runs > 1:
        run_accuracies = []
        run_times = []
        for run_i in range(runs):
            args._run_idx = run_i
            print(f"\n=== Run {run_i + 1}/{runs} ===")

            output_path = format_output_path_vlm(args)
            t_start = time.perf_counter()
            main(args, model, processor, dataset, output_path, sam2_predictor=_sam2_predictor, sam2_device=_sam2_device)
            elapsed = time.perf_counter() - t_start
            run_times.append(elapsed)

            # Write timing sidecar alongside the JSONL
            timing_path = str(output_path).replace('.jsonl', '.timing.json')
            with open(timing_path, 'w') as _tf:
                json.dump({
                    "run_i": run_i,
                    "elapsed_seconds": round(elapsed, 2),
                    "n_items": getattr(args, 'first_k', None),
                    "task": args.task,
                    "variant": getattr(args, 'skills_variant', None),
                }, _tf, indent=2)

            acc = _compute_run_accuracy(output_path)
            run_accuracies.append(acc)
            print(f"  Run {run_i + 1} accuracy: {acc:.2%}  time: {elapsed:.1f}s")

        mean_acc = _statistics.mean(run_accuracies)
        std_acc = _statistics.stdev(run_accuracies) if len(run_accuracies) > 1 else 0.0
        print(f"\n=== Run Summary ({runs} runs, first_k={args.first_k} per q-type) ===")
        print(f"  Mean accuracy : {mean_acc:.2%}")
        print(f"  Std deviation : {std_acc:.2%}")
        print(f"  Per-run accs  : {', '.join(f'{a:.2%}' for a in run_accuracies)}")
        print(f"\n=== Timing Summary ===")
        for i, t in enumerate(run_times):
            print(f"  Run {i + 1}: {t:.1f}s")
        total_t = sum(run_times)
        mean_t = _statistics.mean(run_times)
        print(f"  Total: {total_t:.1f}s  Mean: {mean_t:.1f}s")
    elif runs == 1:
        t_start = time.perf_counter()
        output_path = format_output_path_vlm(args)
        main(args, model, processor, dataset, output_path, sam2_predictor=_sam2_predictor, sam2_device=_sam2_device)
        elapsed = time.perf_counter() - t_start
        print(f"\n  Elapsed: {elapsed:.1f}s")