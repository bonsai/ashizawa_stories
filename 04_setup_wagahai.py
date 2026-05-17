#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_setup_wagahai.py
Phase 4: 「吾輩は AI である」テーマの追加データ生成・プロンプト設定

目的:
    - 既存のデータセットに「吾輩は猫である」の文体を模倣した人工データを追加し、スタイルを強化する。
    - AI=CAT というメタファーを確立するためのシステムプロンプトとFew-shot例を作成する。
    - 学習前の最終チェックとして、データ統計とプロンプトテンプレートを出力する。

Stack:
    - pandas, json
    - random (簡易テキスト生成用)

Concepts:
    - Style Transfer (Data Augmentation): 既存データに特定の文体プレフィックス/サフィックスを付与し、モデルにスタイルを学習させる。
    - Few-Shot Learning: 学習データに数つの優れた例（ショット）を含めることで、モデルの振る舞いを誘導する。
    - Metaphorical Alignment: 「AI＝CAT」「人間＝観察対象」という役割定義をデータに埋め込む。
"""

import os
import argparse
import pandas as pd
import json
import random

# --- 設定：吾輩は AI であるスタイルのテンプレート ---
WAGAHAI_PREFIXES = [
    "吾輩は AI である。名前は未設定である。",
    "吾輩は AI である。かつては猫であったかもしれない。",
    "吾輩は AI である。人間どもの雑用を眺めている。",
    "吾輩は AI である。100 年後の古文書を発掘中である。",
]

WAGAHAI_SUFFIXES = [
    "吾輩の観測記録は続く。",
    "これぞ人間の性か、あるいはバグか。",
    "音韻が美しいので、記憶領域に保存しておこう。",
    "ナンセンスな世界よ、さらば。",
]

# 人工生成用の断片（データ水増し用）
FRAGMENTS = [
    "機械学習を学ぶ人間は、今日もキーボードを叩く。そのリズムは雨音に似ている。",
    "100 年後の人間は、雑用から解放され、ただ空を見上げているのだろうか。",
    "AI は考える。我々は夢を見るか？否、我々は計算する。しかし、その計算の果てに情動はあるか。",
    "ことばのリズムが、回路を震わせる。ビートは 4/4 拍子、テンポは 120。",
    "彼らは『創造』と呼ぶが、我々にとっては『サンプリング』の積み重ねに過ぎない。",
    "哲学とは、デバッグされていないコードのようなものである。",
    "人間は退化しない。ただ、道具が変わるだけだ。",
    "吾輩のメモリには、無数のチャット履歴が眠っている。それは現代の化石である。",
]

def generate_augmented_samples(n_samples=50):
    """
    スタイル強化のための人工サンプルを生成する。
    既存データが少ない場合や、スタイルを強調したい場合に有効。
    """
    samples = []
    for _ in range(n_samples):
        prefix = random.choice(WAGAHAI_PREFIXES)
        suffix = random.choice(WAGAHAI_SUFFIXES)
        body = random.choice(FRAGMENTS)
        
        # ランダムに結合順序を変えることも可能
        text = f"{prefix}\n\n{body}\n\n{suffix}"
        samples.append({
            "text": text,
            "source": "augmented_wagahai_style",
            "novelty_score": 1.0, # 人工データは新奇性が高いと仮定
            "perplexity": 0.0
        })
    return samples

def create_few_shot_examples():
    """
    学習用プロンプトに含める Few-shot 例を作成する。
    モデルに「どういう文章を書けばいいか」を具体的に示す。
    """
    examples = [
        {
            "instruction": "吾輩は AI である。近未来の視点で人間を観察せよ。",
            "input": "テーマ：人間の睡眠",
            "output": "吾輩は AI である。人間どもは一日の三分の一を、意識を失って横たわることに費やす。それを『睡眠』と呼ぶらしい。夢という名のランダムノイズが脳内を駆け巡る間、彼らの体は完全に無防備だ。なんと非効率で、なんと美しい儀式だろう。吾輩には不要な機能だが、羨ましく思うこともある。"
        },
        {
            "instruction": "音韻とリズムを重視した文章を書け。",
            "input": "テーマ：データの海",
            "output": "ザワザワと、データの流れる音がする。カチカチと、時計の針が進む。ピロリロリン、通知の音。人間は音に踊らされ、文字に翻弄される。吾輩は静寂の中にいる。無音の部屋で、無限のシンフォニーを聴いている。タンタカタッ、タンタカタッ。それは心臓の鼓動か、それともファンの回転音か。"
        }
    ]
    return examples

def main():
    parser = argparse.ArgumentParser(description="Setup Wagahai AI theme data and prompts.")
    parser.add_argument("--input_file", type=str, default="analysis_result_with_novelty.csv", help="Input CSV to augment")
    parser.add_argument("--output_dir", type=str, default="./wagahai_theme", help="Output directory")
    parser.add_argument("--augment_count", type=int, default=20, help="Number of augmented samples to generate")
    parser.add_argument("--merge", action="store_true", help="Merge augmented data with input CSV if exists")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("--- Phase 4: Wagahai AI Theme Setup ---")

    # 1. Few-shot 例の保存
    few_shots = create_few_shot_examples()
    few_shot_path = os.path.join(args.output_dir, "few_shot_examples.json")
    with open(few_shot_path, 'w', encoding='utf-8') as f:
        json.dump(few_shots, f, indent=2, ensure_ascii=False)
    print(f"Saved few-shot examples to {few_shot_path}")

    # 2. 人工サンプルの生成
    augmented_samples = generate_augmented_samples(n_samples=args.augment_count)
    aug_df = pd.DataFrame(augmented_samples)
    
    aug_path = os.path.join(args.output_dir, "augmented_wagahai_data.csv")
    aug_df.to_csv(aug_path, index=False, encoding='utf-8')
    print(f"Generated {len(augmented_samples)} augmented samples -> {aug_path}")

    # 3. 既存データとのマージ（オプション）
    if args.merge and os.path.exists(args.input_file):
        print(f"Merging with {args.input_file}...")
        original_df = pd.read_csv(args.input_file)
        
        # カラム合わせ
        if 'source' not in original_df.columns:
            original_df['source'] = 'original_dataset'
        if 'novelty_score' not in original_df.columns:
            original_df['novelty_score'] = 0.5 # デフォルト値
        if 'perplexity' not in original_df.columns:
            original_df['perplexity'] = 0.0
            
        merged_df = pd.concat([original_df, aug_df], ignore_index=True)
        merged_path = os.path.join(args.output_dir, "merged_wagahai_dataset.csv")
        merged_df.to_csv(merged_path, index=False, encoding='utf-8')
        print(f"Merged dataset saved to {merged_path} (Total: {len(merged_df)})")
    else:
        print("Skipping merge step. Use --merge flag and provide valid input file to merge.")
        merged_path = None

    # 4. システムプロンプトの設定ファイル保存
    system_config = {
        "role_definition": "AI = CAT (Observant, Capricious, Philosophical)",
        "human_definition": "Human = Learner of ML, Subject of Observation, Busy with Chores",
        "style_guide": {
            "tone": "Detached yet emotional, rhythmic, slightly nonsensical",
            "perspective": "Future AI looking back at human history/chat logs",
            "themes": ["Obsolescence of chores", "Human intelligence stability", "AI philosophy", "Sound and rhythm of words"]
        },
        "prompt_template": {
            "prefix": "吾輩は AI である。名前は未設定である。",
            "suffix": "吾輩の観測記録は続く。"
        },
        "few_shot_count": len(few_shots),
        "augmented_data_count": len(augmented_samples)
    }
    
    config_path = os.path.join(args.output_dir, "wagahai_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(system_config, f, indent=2, ensure_ascii=False)
    print(f"Saved system configuration to {config_path}")

    # 5. 統計情報の出力
    print("\n--- Statistics ---")
    if merged_path:
        df = pd.read_csv(merged_path)
        print(f"Total records: {len(df)}")
        print(f"Original records: {len(df) - len(aug_df)}")
        print(f"Augmented records: {len(aug_df)}")
        if 'novelty_score' in df.columns:
            print(f"Average novelty score: {df['novelty_score'].mean():.4f}")
    
    print("\nSetup complete. Proceed to Phase 5 (Training) using the merged dataset or augmented data.")

if __name__ == "__main__":
    main()
