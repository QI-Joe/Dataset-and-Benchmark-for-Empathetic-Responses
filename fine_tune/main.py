import os
import sys
import json
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from transformers.trainer_callback import ProgressCallback
from peft import LoraConfig, get_peft_model, TaskType
import argparse
from datetime import datetime


class LossProgressCallback(ProgressCallback):
    """Show training loss to 2 d.p. on the tqdm progress bar."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            if logs and "loss" in logs:
                self.training_bar.set_postfix(loss=f"{logs['loss']:.2f}")
        super().on_log(args, state, control, logs=logs, **kwargs)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config_gen import GenTrainingConfig

from data_loader import GenerationCollator, gen_loader_warp
from train_module import multi_turn_chat_with_ppl_batched, compute_sentence_bleu

def run_training(model, tokenizer, train_dataset, val_dataset, args_dict):
    print("Building training dataset ...")
    print(f"Training samples: {len(train_dataset)}, Val samples: {len(val_dataset)}\nrun in epoch {args_dict['epoch']} with lr {args_dict['lr']}")

    training_args = TrainingArguments(
        output_dir=os.path.join(args_dict['BASE_PATH'], args_dict['FOLDER_NAME']),
        seed=args_dict['SEED'],
        num_train_epochs=args_dict['epoch'],
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=args_dict['lr'],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    data_collator = GenerationCollator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(LossProgressCallback())

    print("Starting LoRA fine-tuning ...")
    trainer.train()
    print("Fine-tuning complete.\n")

    os.makedirs(args_dict['LORA_ADAPTER_PATH'], exist_ok=True)
    model.save_pretrained(args_dict['LORA_ADAPTER_PATH'])
    tokenizer.save_pretrained(args_dict['LORA_ADAPTER_PATH'])
    print(f"LoRA adapter saved \u2192 {args_dict['LORA_ADAPTER_PATH']}\n")

if __name__ == "__main__":
    # --- Configuration ---
    parser = argparse.ArgumentParser(description="Train and evaluate the generation model.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--ratio", type=float, default=0.2, help="(unused) kept for compatibility.")
    parser.add_argument("--new_model_train", action="store_true", help="Force retrain even if adapter exists.")
    parser.add_argument("--data_name", type=str, default='zctytlxz',
                        choices=["sbzjs", "rswtyjs", "rjqlgc", "zctytlxz", "zctcdh"],
                        help="Forum dataset to use.")
    parser.add_argument("--model", type=str, default="llama3.1-8B-Instruct", help="Base model name or path.")
    parser.add_argument("--task1", type=str, default="Gen", help="Task label for folder naming.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length for training samples.")
    parser.add_argument("--task2", type=str, default="debug_fix_small_batch_test", help="Additional task label for folder naming.")
    parser.add_argument("--fast_train", action="store_true", help="Use small fixed-size splits (50/20/30 blocks) for a quick smoke test.")
    
    parser.add_argument("--semi_supervised", action="store_true", help="Enable semi-supervised learning mode (uses semi_ratio to split labeled/unlabeled).")
    parser.add_argument("--semi_ratio", type=float, default=0.1, help="In semi-supervised mode, fraction of data to use as labeled.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="In few-shot/semi-supervised mode, fraction of blocks to use as validation.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="In few-shot/semi-supervised mode, fraction of blocks to use as test.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length for training samples.")
    parser.add_argument("--epoch", type=int, default=3, help="Number of training epochs.")
    args = parser.parse_args()
    config = GenTrainingConfig()
    
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATE = datetime.now().strftime("%m_%d")

    BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'llama3', args.data_name, DATE)
    FOLDER_NAME = f"llama3_gen_{args.data_name}_{args.lr}_{args.task1}_{args.task2}"
    LORA_ADAPTER_PATH = os.path.join(BASE_PATH, FOLDER_NAME, "adapter")
    args_dict = {
        'lr': args.lr,
        'SEED': SEED,
        'LORA_ADAPTER_PATH': LORA_ADAPTER_PATH,
        'BASE_PATH': BASE_PATH,
        'FOLDER_NAME': FOLDER_NAME,
        'max_seq_len': config.max_seq_length,
        'epoch': args.epoch,
    }
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # Use float16 if your GPU doesn't support bfloat16
    )

    # --- Model Loading ---
    print("Loading model ...")
    model_name = rf"../../LLModel/{args.model}"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./llama3-8B/', force_download=False)

    if tokenizer.pad_token is None:
        # tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", cache_dir='./llama3-8B/', force_download=False, quantization_config=bnb_config
    )

    # Resize embeddings to cover any tokens added to the tokenizer (e.g. <|pad|>).
    # Must happen before LoRA is applied so the new rows are included in the base weights.
    # model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        # modules_to_save=["embed_tokens", "lm_head"],
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    print()

    # --- Data Loading ---
    train_loader, val_loader, test_loader, raw_ds = gen_loader_warp(args.data_name, tokenizer, config)
    train_dataset, val_dataset, test_dataset = raw_ds
    print(f"Loaded training data: {len(train_dataset)} samples\nValidation data: {len(val_dataset)} samples\nTest data: {len(test_dataset)} samples\n")

    # --- Training ---
    if not os.path.exists(LORA_ADAPTER_PATH) or args.new_model_train:
        run_training(model, tokenizer, train_dataset, val_dataset, args_dict)
    else:
        print(f"Found existing LoRA adapter at '{LORA_ADAPTER_PATH}', skipping training.\n")
        model.load_adapter(LORA_ADAPTER_PATH, adapter_name="default")
        model.set_adapter("default")

    # --- Evaluation ---
    ppl_total, sample_ppl_total = 0.0, 0.0
    bleu1_total, bleu2_total, bleu3_total, bleu4_total = 0.0, 0.0, 0.0, 0.0

    all_results = []

    print("Evaluating...")
    test_dataloader = test_loader

    processed_samples = 0
    for batch in test_dataloader:
        generated_list, loss_list, ppl_list, sample_ppl_list = multi_turn_chat_with_ppl_batched(
            model=model,
            tokenizer=tokenizer,
            DEVICE=DEVICE,
            batch=batch,
            emo_head=None,
            max_new_tokens=50,
            temperature=0.45,
            top_p=0.85,
        )

        for b_i in range(len(generated_list)):
            generated = generated_list[b_i]
            reference = batch["reference"][b_i]
            history = batch["postkey"][b_i]
            ppl = ppl_list[b_i]
            sample_ppl = sample_ppl_list[b_i]
            loss = loss_list[b_i]
            sample_idx = batch["sample_idx"][b_i]

            bleu1, bleu2, bleu3, bleu4 = compute_sentence_bleu(generated, reference)

            bleu1_total += bleu1; bleu2_total += bleu2
            bleu3_total += bleu3; bleu4_total += bleu4
            ppl_total += ppl; sample_ppl_total += sample_ppl

            all_results.append({
                "id": sample_idx,
                "postkey": history,
                "reference": reference,
                "generated": generated,
                "metrics": {
                    "ppl": ppl, "sample_ppl": sample_ppl,
                    "bleu1": bleu1, "bleu2": bleu2, "bleu3": bleu3, "bleu4": bleu4,
                }
            })

            processed_samples += 1
            if processed_samples % 100 == 0:
                print(f"Evaluated {processed_samples} samples ...")
                print(f"[Round {processed_samples}] Avg PPL: {ppl_total / processed_samples:.4f}")
                print(f"[Round {processed_samples}] Avg BLEU-1: {bleu1_total / processed_samples:.4f}")
                print(f"[Round {processed_samples}] Avg BLEU-2: {bleu2_total / processed_samples:.4f}")

    n = max(processed_samples, 1)

    unigrams, bigrams = set(), set()
    total_unigrams, total_bigrams = 0, 0
    for item in all_results:
        tokens = list(item["generated"].strip())  # character-level for Chinese
        for i, tok in enumerate(tokens):
            unigrams.add(tok)
            total_unigrams += 1
            if i < len(tokens) - 1:
                bigrams.add((tok, tokens[i + 1]))
                total_bigrams += 1

    corpus_dist_1 = len(unigrams) / total_unigrams if total_unigrams > 0 else 0.0
    corpus_dist_2 = len(bigrams) / total_bigrams if total_bigrams > 0 else 0.0

    print(f"Average PPL:        {ppl_total / n:.4f}")
    print(f"Average Sample PPL: {sample_ppl_total / n:.4f}")
    print(f"Average BLEU-1:     {bleu1_total / n:.4f}")
    print(f"Average BLEU-2:     {bleu2_total / n:.4f}")
    print(f"Corpus Dist-1:      {corpus_dist_1:.4f}")
    print(f"Corpus Dist-2:      {corpus_dist_2:.4f}")

    for item in all_results:
        item["metrics"]["dist1_corpus"] = corpus_dist_1
        item["metrics"]["dist2_corpus"] = corpus_dist_2

    output_dir = os.path.join(BASE_PATH, FOLDER_NAME)
    os.makedirs(output_dir, exist_ok=True)

    # Per-sample JSONL
    output_jsonl_path = os.path.join(output_dir, f"eval_results_{FOLDER_NAME}.jsonl")
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Evaluation results saved to {output_jsonl_path}")

    # Aggregate summary JSON
    summary = {
        "folder": FOLDER_NAME,
        "data_name": args.data_name,
        "lr": args.lr,
        "num_test_samples": processed_samples,
        "avg_ppl":        round(ppl_total        / n, 6),
        "avg_sample_ppl": round(sample_ppl_total  / n, 6),
        "avg_bleu1":      round(bleu1_total        / n, 6),
        "avg_bleu2":      round(bleu2_total        / n, 6),
        "avg_bleu3":      round(bleu3_total        / n, 6),
        "avg_bleu4":      round(bleu4_total        / n, 6),
        "corpus_dist1":   round(corpus_dist_1,       6),
        "corpus_dist2":   round(corpus_dist_2,       6),
    }
    summary_path = os.path.join(output_dir, f"summary_metrics_{FOLDER_NAME}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary metrics saved to  {summary_path}")
