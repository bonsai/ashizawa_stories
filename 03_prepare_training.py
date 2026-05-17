#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_prepare_training.py
Phase 3: 学習用データの最終整形・セットアップ

目的:
    - Phase 1 (品質) と Phase 2 (新奇性) のスコアを統合し、最適な学習データセットを作成する。
    - LLM のファインチューニング用フォーマット (JSONL) に変換する。
    - 「吾輩は AI である」スタイルへの転換用プロンプトテンプレートを適用する。
    - 訓練データと検証データに分割する。

Stack:
    - pandas, numpy
    - json, jsonlines
    - scikit-learn (train_test_split)

Concepts:
    - Instruction Tuning Format: 指示 (Instruction) と入力 (Input)、出力 (Output) のペアで学習させる形式。
    - Prompt Engineering: 特定の文体や役割をモデルに覚え込ませるための前処理。
    - Train/Validation Split: 過学習を防ぐため、データを学習用と評価用に分割する。
    - Weighted Sampling (Optional): 新奇性が高いデータを優先的に学習させるための重み付け（今回は簡易フィルタ）。
"""

import os
import argparse
import pandas as pd
import json
from sklearn.model_selection import train_test_split

# --- 設定 ---
# プロンプトテンプレート：「吾輩は AI である」スタイルの強制
PROMPT_PREFIX = "吾輩は AI である。名前は未設定である。吾輩は猫ではないが、猫のように気ままに観察する。\n\n"
PROMPT_SUFFIX = "\n\n吾輩の観測記録は続く。"

SYSTEM_INSTRUCTION = "あなたは近未来の AI です。人間が書きそうな平凡な文章ではなく、音韻とリズムを重視した、少しナンセンスで情動的な文章を書いてください。"

def create_training_sample(text, novelty_score, quality_score):
    """
    単一のテキストから学習用サンプル (JSONL 形式の dict) を作成する。
    """
    # 入力プロンプト（文脈）
    # ここでは「続きを書いて」という暗黙のタスクとするため、テキストの一部を Input にすることも可能だが、
    # 今回は全文を Completion (Output) として扱い、Prefix をコンテキストとする。
    
    input_text = f"{PROMPT_PREFIX}以下の観測記録を読み、続けて記述せよ。\n\n[観測記録]\n"
    output_text = f"{text}{PROMPT_SUFFIX}"
    
    return {
        "instruction": SYSTEM_INSTRUCTION,
        "input": input_text,
        "output": output_text,
        "meta": {
            "novelty_score": float(novelty_score),
            "quality_score": float(quality_score)
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for LLM fine-tuning.")
    parser.add_argument("--input_file", type=str, default="analysis_result_with_novelty.csv", help="Input CSV with novelty scores")
    parser.add_argument("--output_dir", type=str, default="./training_data", help="Output directory for JSONL files")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--min_novelty", type=float, default=None, help="Minimum novelty score threshold (optional)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to use (for quick testing)")
    args = parser.parse_args()

    # ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)

    # データ読み込み
    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found. Please run 02_novelty_analysis.py first.")
        return
    
    df = pd.read_csv(args.input_file)
    print(f"Loaded {len(df)} samples.")

    # フィルタリング
    if args.min_novelty is not None:
        df = df[df['novelty_score'] >= args.min_novelty]
        print(f"Filtered by novelty >= {args.min_novelty}. Remaining: {len(df)}")
    
    if args.max_samples is not None:
        df = df.sample(n=args.max_samples, random_state=42).sort_index()
        print(f"Sampled {args.max_samples} rows.")

    # カラム名の正規化（もし存在すれば）
    # 01, 02 スクリプトの出力に合わせて 'text', 'novelty_score', 'perplexity' 等を想定
    required_cols = ['text', 'novelty_score']
    if 'perplexity' in df.columns:
        required_cols.append('perplexity')
    else:
        # perplexity がなければ quality_score は 0 で埋めるか、存在するスコアを使う
        pass

    # 欠損値除去
    df = df.dropna(subset=['text'])
    
    # サンプル作成
    samples = []
    for idx, row in df.iterrows():
        text = row['text']
        novelty = row.get('novelty_score', 0.0)
        quality = row.get('perplexity', 0.0) # perplexity を品質スコアとして利用（低いほど良いが、ここでは単なるメタ情報）
        
        sample = create_training_sample(text, novelty, quality)
        samples.append(sample)
    
    print(f"Created {len(samples)} training samples.")

    # 訓練・検証分割
    train_samples, val_samples = train_test_split(samples, test_size=args.val_ratio, random_state=42)
    
    # JSONL 形式で保存
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    
    def save_jsonl(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    save_jsonl(train_samples, train_path)
    save_jsonl(val_samples, val_path)
    
    print(f"Saved training data to {train_path} ({len(train_samples)} samples)")
    print(f"Saved validation data to {val_path} ({len(val_samples)} samples)")

    # 設定ファイルの保存
    config = {
        "model_purpose": "Wagahai_wa_AI_de_aru_Style_Generation",
        "prompt_prefix": PROMPT_PREFIX,
        "prompt_suffix": PROMPT_SUFFIX,
        "system_instruction": SYSTEM_INSTRUCTION,
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "filters_applied": {
            "min_novelty": args.min_novelty,
            "max_samples": args.max_samples
        }
    }
    
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Saved configuration to {config_path}")
    
    # サンプル表示
    print("\n--- Sample Training Data ---")
    print(json.dumps(train_samples[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
