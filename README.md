# Dataset-and-Benchmark-for-Empathetic-Responses

This repository releases the training and evaluation pipeline used in our empathetic response study, covering dataset preparation, LoRA fine-tuning, and LLM-as-a-Judge evaluation for Chinese forum conversations. The codebase currently supports three model tracks in the paper-facing documentation: ChatGLM-6B, Llama3.1-8B, and Llama2-7B.

## Model Weights Storage
- `fine-tune model`: our fine-tuned model will be updated on [Google Drive](https://drive.google.com/drive/folders/1wOAQQ9c9oeEt5VQxNdpS_chAPTxBOAsv?usp=sharing)

## Project Layout (Folder Level)

- `fine_tune/`: Data preprocessing, dataloading, model loading, and the main fine-tuning workflow for the Llama training path.
- `llm_judge/`: LMMS-Judge (LLM-as-a-Judge) components, including prompt templates, judge registry, API client calls, and JSON result parsing.
- `markdown/`: Prompt documentation and citations for evaluation templates and academic grounding.
- `utils/`: Shared training configurations and utility helpers used by multiple training and metric workflows.

## Environment and Setup

### Runtime assumptions

- Python: 3.10 or 3.11 recommended.
- CUDA/cuDNN: CUDA-enabled GPU environment is assumed for practical fine-tuning; 4-bit quantization is used in training scripts.
- OS: Linux is recommended for BitsAndBytes + PEFT workflows.

### Key libraries

- `torch`, `transformers`, `peft`, `bitsandbytes`
- `numpy`, `pandas`, `nltk`, `tqdm`
- `openai` (for compatible judge APIs), `python-dotenv` (optional convenience)

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

Place raw data under the following relative structure:

```text
./pre data/
  上班这件事/
  人生问题研究社/
  人间情侣观察/
  职场体验讨论小组/
  职场吐槽大会/
```

Each dataset folder is expected to contain three CSV files keyed by the dataset short code:

- `<key>_id_c_r.csv`: high-level columns include `postID`, `comment`, `review`
- `<key>_id_c_l.csv`: high-level columns include `postID`, `comment`, `label`
- `<key>_postid_pre_final.csv`: high-level columns include `postID`, `postkey`

Verified dataset names wired in code:

1. 上班这件事
2. 人生问题研究社
3. 人间情侣观察
4. 职场体验讨论小组
5. 职场吐槽大会

## Training / Fine-Tuning

### One-command starter

Use the provided launcher script:

```bash
bash start.sh
```

### Generic training example (Llama path)

```bash
python fine_tune/main.py \
  --data_name rswtyjs \
  --lr 5e-5 \
  --epoch 3 \
  --task1 Gen \
  --task2 full_comment
```

### ChatGLM-6B dedicated launcher (separate entry point)

```bash
python chatGLM_main.py \
  --data_name sbzjs \
  --lr 5e-5 \
  --epoch 1 \
  --task1 chatglm \
  --task2 full_comment
```

## Inference / Demo

Minimal demo command (loads an existing adapter if present; otherwise it will train then evaluate):

```bash
python fine_tune/main.py \
  --data_name rswtyjs \
  --task1 Gen \
  --task2 demo
```

## Checkpoints and Outputs

- Llama training outputs are saved under:
  - `./llama3/<data_name>/<MM_DD>/llama3_gen_<...>/adapter`
  - `eval_results_<...>.jsonl`
  - `summary_metrics_<...>.json`
- ChatGLM training outputs are saved under:
  - `./chatglm/<data_name>/<MM_DD>/chatglm_gen_<...>/adapter`
  - `eval_results_<...>.jsonl`
  - `summary_metrics_<...>.json`
- Judge outputs are saved under:
  - `./LLM_AS_Judge/LLM_Judge_<judge>_<input_stem>.jsonl`

## Reproducibility

- Training scripts fix random seed to 42.
- 4-bit quantization is enabled for memory efficiency.
- Hardware expectation: at least one CUDA GPU is strongly recommended for fine-tuning.

## Known Issues

- ChatGLM-6B path assumes a local base model layout and environment compatible with quantized loading.
- Legacy dependency combinations for ChatGLM-6B may require version pinning updates on newer CUDA or driver stacks.

## Evaluation with LMMS-Judge

### Evaluation data placement

Recommended structure:

```text
./eval/
  data.json
  results/
```

Input JSON must be an array of objects with at least:

- `post_id`
- `postkey`
- `feedback`
- `response` (recommended for stronger empathy judgment)

### Secrets and environment variables

Required keys depend on the selected judge:

- `DASHSCOPE_API_KEY` for `glm-5.2` and `qwen3.7-max`
- `XAI_API_KEY` for `grok-beta`

Shell export example:

```bash
export DASHSCOPE_API_KEY=YOUR_KEY
export XAI_API_KEY=YOUR_KEY
```

Optional dotenv workflow:

```bash
pip install python-dotenv
cp .env.example .env
```

### Entry point

`run_eval.py` launches the LMMS-Judge workflow.

```bash
python run_eval.py --input ./eval/data.json --judge glm-5.2 --concurrency 10
```

### Prompt citations

See prompt text and paper-grounded mappings in:

- `markdown/prompts_citations.md`

Prompts are designed to reflect empirical positions and findings from the cited literature.

## License and Citation

### License

No license file is currently included. Add a repository license (for example MIT or Apache-2.0) before broader redistribution.

### Citation

Please cite both the paper and this repository in your work.

Repository citation template:

```bibtex
@misc{dataset_benchmark_empathetic_responses,
  title        = {Dataset and Benchmark for Empathetic Responses},
  author       = {QI-Joe and Contributors},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/QI-Joe/Dataset-and-Benchmark-for-Empathetic-Responses}
}
```

## Notes

- The code-level argument defaults and local model path names may differ from paper-facing model names; this README normalizes public names to ChatGLM-6B, Llama3.1-8B, and Llama2-7B.
- A published paper URL is not present in the repository yet. Add the final paper link in this section once available.
