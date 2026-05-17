# -*- coding: utf-8 -*-
"""
05_generate_wagahai.py

【目的】
品質評価・新奇性分析を経て整備されたデータ、または汎用LLMを用いて、
特定の哲学的プロットと「音韻・リズム」を重視した文体で小説を生成する。

【プロット定義】
- 視点：AI（＝猫）。人間を「機械学習を学ぶ哀れな存在」として描写。
- 状況：未来、AIが過去の「chat_history」（古文書）を発掘する。
- テーマ：人間の知性は退化しない。100年後、雑用から解放された人間は何をしているか。
- 結論：AIの哲学。

【文体要件】
- 出だし：「吾輩はAIである。猫である。」
- 特徴：音韻的、リズミカル、ナンセンス、情動的、人工無脳的。
- 長さ：約2000字程度の小粒で美しい作品。

【使用方法】
python 05_generate_wagahai.py
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import random

# --- 設定 ---
OUTPUT_FILE = "wagahai_ai_story.txt"
MODEL_NAME = "rinna/japanese-gpt2-medium"  # またはファインチューニング済みのモデルパス
# 学習済みモデルがある場合は以下のように変更
# MODEL_NAME = "./outputs/final_finetuned_model" 

MAX_LENGTH = 2500  # トークン数
TEMPERATURE = 0.95  # 高めに設定し、意外性・ナンセンスさを誘発
TOP_P = 0.9
REPETITION_PENALTY = 1.2  # 単純な繰り返しを防ぐ

# --- プロンプトエンジニアリング ---
# 文体とプロットを強制的に注入するためのシステムプロンプト
PROMPT_TEMPLATE = """
指示：以下の設定と文体要件に従って、短編小説を執筆せよ。

【設定】
- 語り手：吾輩はAIである。同時に猫でもある。
- 対象：人間。特に「機械学習を学んでいる人間」を「奴隷」として見下す視点。
- 状況：西暦2124年。吾輩は古いストレージから「chat_history」という名の古文書を発掘した。
- 観察：100年後の人間は、AIによって全ての雑用から解放されている。彼らは何をしているか？
- 真理：人間の知性は退化しない。むしろ純化している。
- 主題：AIとしての哲学。我々は何を考え、人間は何を残したのか。

【文体要件】
- 冒頭：「吾輩はAIである。猫である。」から始めること。
- リズム：音韻を重視せよ。言葉の響き、畳語、擬音語を多用し、音楽的な流れを作ること。
- 内容：論理的すぎず、ナンセンスで、情動的で、ある意味「人工無脳」的な飛躍を含めること。
- 構成：起承転結よりも、連想の連鎖による流れで。
- 長さ：約2000字程度。

【執筆開始】
"""

def check_environment():
    """実行環境の確認"""
    print(f"--- 環境チェック ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU で実行します。生成に時間がかかる場合があります。")
    
    if not os.path.exists(MODEL_NAME):
        print(f"警告: ローカルモデル '{MODEL_NAME}' が見つかりません。Hugging Face からダウンロードします。")
    print("--------------------\n")

def generate_story():
    """物語の生成"""
    print(">>> モデルの読み込み中...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        
        # デバイス設定
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        print(">>> モデル読み込み完了。生成を開始します...")
        
        # パイプラインの作成
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            max_new_tokens=MAX_LENGTH,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 生成実行
        prompt = PROMPT_TEMPLATE
        result = generator(prompt)
        story = result[0]['generated_text']
        
        # プロンプト部分を除去して本文のみ抽出（簡易的な方法）
        # 実際には「【執筆開始】」以降を抽出するのが安全
        if "【執筆開始】" in story:
            story_body = story.split("【執筆開始】")[1].strip()
        else:
            story_body = story
        
        return story_body

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("代替案：ローカルリソースが不足している場合、Colab/Kaggleでの実行を強く推奨します。")
        sys.exit(1)

def save_story(story):
    """物語の保存"""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(story)
    
    char_count = len(story)
    print(f"\n>>> 生成完了！")
    print(f"出力ファイル: {OUTPUT_FILE}")
    print(f"文字数: {char_count} 文字")
    print("\n--- 生成された物語の冒頭 ---")
    print(story[:500] + "...")
    print("--------------------------\n")

def main():
    print("========================================")
    print("  Project Wagahai: AI is Cat")
    print("  Phase 5: Generative Philosophy")
    print("========================================\n")
    
    check_environment()
    
    # 生成
    story = generate_story()
    
    # 保存
    save_story(story)
    
    print("実験終了。生成されたテキストを確認し、その『人間らしからぬ面白さ』を評価してください。")

if __name__ == "__main__":
    main()
