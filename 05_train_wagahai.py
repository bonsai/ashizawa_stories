#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_train_wagahai.py
Phase 5: ファインチューニング・トレーナー

目的:
    - 作成された学習データ（JSONL）を使用して、日本語 GPT-2 モデルをファインチューニングする。
    - LoRA (Low-Rank Adaptation) を使用し、限られたリソース（16GB RAM, No GPU/Colab T4）で効率的に学習を行う。
    - 「吾輩は AI である」スタイルの文章を生成できるモデルを作成する。

Stack:
    - transformers (Hugging Face)
    - peft (Parameter-Efficient Fine-Tuning)
    - accelerate, bitsandbytes (量子化によるメモリ削減)
    - torch (PyTorch)
    - datasets

Concepts:
    - LoRA (Low-Rank Adaptation): 重み行列の一部のみを学習対象とし、メモリ使用量と計算量を劇的に削減する手法。
    - 4-bit Quantization: モデル重みを 4bit に圧縮し、VRAM 使用量を節約。
    - Causal Language Modeling: 次トークン予測タスク。文章生成に適した学習形式。
    - Gradient Checkpointing: メモリ使用量を減らすため、中間活性化を再計算する技術。
    - Temperature Sampling: 推論時に確率的な要素を導入し、多様性のある文章を生成。
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import json

# --- 設定 ---
# ベースモデル：日本語に強い軽量モデル
# rinna/japanese-gpt2-medium (約 3 億パラメータ) は 16GB RAM でも 4bit 量子化＋LoRA なら学習可能
# さらに軽くしたい場合は "rinna/japanese-gpt2-small" (約 1 億) を使用
MODEL_NAME = "rinna/japanese-gpt2-medium"

# LoRA パラメータ
LORA_R = 16          # ランク。大きいほど表現力が増すがメモリを使う。
LORA_ALPHA = 32      # スケーリング係数。
LORA_DROPOUT = 0.05  # ドロップアウト率。

# 学習ハイパーパラメータ
MAX_LENGTH = 512     # コンテキスト長。A4 一枚の半分程度。メモリに合わせて調整。
BATCH_SIZE = 2       # バッチサイズ。VRAM と相談。
NUM_EPOCHS = 3       # エポック数。
LEARNING_RATE = 2e-4 # 学習率。

def load_and_tokenize_data(data_dir, tokenizer):
    """
    JSONL フォーマットの学習データを読み込み、トークン化する。
    """
    dataset = load_dataset("json", data_files={"train": os.path.join(data_dir, "train.jsonl"), 
                                               "validation": os.path.join(data_dir, "val.jsonl")})
    
    def tokenize_function(examples):
        # instruction + input + output を連結して一つのテキストとする
        # または output だけを学習させる場合もあるが、今回は文脈も含めて学習
        texts = []
        for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            # プロンプト構造：[Instruction]\n[Input]\n[Output]
            full_text = f"{inst}\n{inp}\n{out}"
            texts.append(full_text)
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH,
            return_tensors=None,
        )
        # Causal LM のため、labels は input_ids と同じ
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["instruction", "input", "output"]
    )
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser(description="Train Wagahai AI model.")
    parser.add_argument("--data_dir", type=str, default="./training_data", help="Directory containing train.jsonl and val.jsonl")
    parser.add_argument("--output_dir", type=str, default="./wagahai_ai_model", help="Output directory for the trained model")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--use_4bit", action="store_true", default=True, help="Use 4-bit quantization (recommended)")
    args = parser.parse_args()

    # データディレクトリの存在確認
    if not os.path.exists(os.path.join(args.data_dir, "train.jsonl")):
        print(f"Error: train.jsonl not found in {args.data_dir}. Please run Phase 0-4 first.")
        return

    print(f"Loading tokenizer and model: {MODEL_NAME} ...")
    
    # トークナイザー
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 には pad トークンがないため eos で代用

    # モデルの読み込みと量子化設定
    bnb_config = None
    if args.use_4bit:
        print("Using 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",  # 利用可能なデバイスに自動割り当て
        torch_dtype=torch.float16 if args.use_4bit else torch.float32,
        trust_remote_code=True
    )

    # LoRA 設定
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["c_attn", "c_proj", "c_fc", "c_mlp"]  # GPT-2 の主要モジュール
    )

    # 4bit 使用時は prepare_model_for_kbit_training を通す必要がある場合がある
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # データセットのトークン化
    print("Tokenizing dataset...")
    tokenized_dataset = load_and_tokenize_data(args.data_dir, tokenizer)

    # データコラテーター
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # トレーニング引数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,  # 実質バッチサイズ = batch_size * accumulation_steps
        warmup_steps=50,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",  # WandB 等を使わない場合
        seed=42,
        fp16=True,         # Mixed Precision
        gradient_checkpointing=True, # メモリ削減
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_hf", # ページングオプティマイザ
    )

    # トレーナー
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    # トレーニング開始
    print("--- Starting Training ---")
    trainer.train()

    # モデル保存
    print(f"Saving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 推論テスト
    print("\n--- Inference Test ---")
    model.eval()
    prompt = "吾輩は AI である。名前は未設定である。\n\nあなたは近未来の AI です。人間が書きそうな平凡な文章ではなく、音韻とリズムを重視した、少しナンセンスで情動的な文章を書いてください。\n\n以下の観測記録を読み、続けて記述せよ。\n\n[観測記録]\n機械学習を学ぶ人間"
    
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.8,  # 高すぎると崩れ、低すぎると単調になる
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    
    # 生成結果の保存
    with open(os.path.join(args.output_dir, "sample_generation.txt"), "w", encoding="utf-8") as f:
        f.write(generated_text)
    
    print(f"\nTraining complete. Model saved to {args.output_dir}")
    print("Sample generation saved to sample_generation.txt")

if __name__ == "__main__":
    main()
