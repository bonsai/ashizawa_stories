"""
01_evaluate_and_filter.py
小説データの品質評価とフィルタリングスクリプト

機能:
1. CSV データの読み込み
2. ルールベースの事前フィルタリング（おかしな日本語の排除）
3. 言語モデルによる Perplexity 計算（品質評価）
4. 閾値による足切り（悪い文章の排除）
5. クリーンなデータの出力
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import os
from tqdm import tqdm

# ======================
# 設定パラメータ
# ======================
INPUT_CSV = "twnovel_roseaullus/twnovel_dataset_roseaullus.csv"  # 入力 CSV
OUTPUT_SCORED_CSV = "scored_dataset.csv"      # スコア付き出力
OUTPUT_CLEAN_CSV = "cleaned_dataset.csv"      # クリーンなデータ出力
REPORT_FILE = "filter_report.txt"             # レポートファイル

# モデル設定 (軽量な日本語 GPT-2)
MODEL_NAME = "rinna/japanese-gpt2-small"

# フィルタリング閾値
MAX_KANJI_RATIO = 0.9       # 漢字比率の最大値 (90% 以上は不自然)
MIN_HIRAGANA_RATIO = 0.05   # ひらがな比率の最小値 (5% 未満は不自然)
MIN_LENGTH = 10             # 最小文字数
MAX_LENGTH = 512            # 最大文字数 (モデルの入力制限考慮)

# Perplexity 閾値 (足切り)
# PP が高いほど不自然。データの分布を見て決定するか、固定値を設定。
# 暫定的に上位 20% をカットする方式を採用（後で自動計算）
PERPLEXITY_PERCENTILE_CUTOFF = 80  # 下位 80% を採用（上位 20% を除外）


def load_data(filepath):
    """CSV データを読み込む"""
    print(f"データを読み込んでいます: {filepath}")
    df = pd.read_csv(filepath)
    
    # 'text' カラムが存在するか確認
    if 'text' not in df.columns:
        # カラム名が違う場合のフォールバック
        text_cols = [col for col in df.columns if 'text' in col.lower() or 'novel' in col.lower()]
        if text_cols:
            df.rename(columns={text_cols[0]: 'text'}, inplace=True)
            print(f"'{text_cols[0]}' カラムを'text' として使用します")
        else:
            raise ValueError("テキストデータのカラムが見つかりません")
    
    print(f"読み込んだデータ数: {len(df)}")
    return df


def rule_based_filter(df):
    """ルールベースでおかしな日本語を排除する"""
    print("\n=== ルールベースフィルタリング ===")
    initial_count = len(df)
    mask = pd.Series([True] * len(df))
    
    # NaN の除去
    nan_count = df['text'].isna().sum()
    mask &= df['text'].notna()
    if nan_count > 0:
        print(f"- NaN を含むデータを {nan_count} 件除外")
    
    # 長さのチェック
    lengths = df['text'].str.len()
    too_short = (lengths < MIN_LENGTH).sum()
    too_long = (lengths > MAX_LENGTH).sum()
    mask &= (lengths >= MIN_LENGTH) & (lengths <= MAX_LENGTH)
    if too_short > 0:
        print(f"- 短すぎるデータ ({MIN_LENGTH} 文字未満) を {too_short} 件除外")
    if too_long > 0:
        print(f"- 長すぎるデータ ({MAX_LENGTH} 文字超) を {too_long} 件除外")
    
    # 文字種比率のチェック
    def check_char_ratios(text):
        if not isinstance(text, str):
            return False
        
        total = len(text)
        if total == 0:
            return False
        
        # 漢字数
        kanji_count = len(re.findall(r'[\u4e00-\u9fff]', text))
        # ひらがな数
        hiragana_count = len(re.findall(r'[\u3040-\u309f]', text))
        
        kanji_ratio = kanji_count / total
        hiragana_ratio = hiragana_count / total
        
        # 条件を満たさない場合は False (除外)
        if kanji_ratio > MAX_KANJI_RATIO:
            return False
        if hiragana_ratio < MIN_HIRAGANA_RATIO:
            return False
        
        return True
    
    ratio_mask = df['text'].apply(check_char_ratios)
    ratio_failed = (~ratio_mask).sum()
    mask &= ratio_mask
    if ratio_failed > 0:
        print(f"- 文字種比率が不自然なデータを {ratio_failed} 件除外")
    
    # 特殊文字のチェック (制御文字など)
    def has_control_chars(text):
        if not isinstance(text, str):
            return True  # NaN などは既に弾かれている
        # 一般的な制御文字 (改行タブ以外) を検出
        if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', text):
            return True
        return False
    
    control_mask = ~df['text'].apply(has_control_chars)
    control_failed = (~control_mask).sum()
    mask &= control_mask
    if control_failed > 0:
        print(f"- 制御文字を含むデータを {control_failed} 件除外")
    
    filtered_df = df[mask].copy()
    final_count = len(filtered_df)
    removed_count = initial_count - final_count
    
    print(f"\nルールベースフィルタリング完了:")
    print(f"  初期データ数：{initial_count}")
    print(f"  除外数：{removed_count}")
    print(f"  残存数：{final_count}")
    
    return filtered_df


def calculate_perplexity(df, model_name):
    """言語モデルを用いて各行の Perplexity を計算する"""
    print(f"\n=== Perplexity 計算中 (モデル：{model_name}) ===")
    
    # デバイスの設定 (GPU があれば GPU、なければ CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用するデバイス：{device}")
    
    # トークナイザとモデルの読み込み
    print("モデルを読み込んでいます...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    perplexities = []
    
    # バッチ処理で計算 (メモリ節約のため 1 行ずつ、または小さいバッチ)
    batch_size = 4 if device.type == 'cuda' else 1
    
    texts = df['text'].tolist()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Perplexity 計算中"):
            batch_texts = texts[i:i+batch_size]
            
            # トークナイズ
            encodings = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            ).to(device)
            
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']
            target_ids = input_ids.clone()
            
            # 損失計算
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target_ids
            )
            
            loss = outputs.loss
            
            # Perplexity = exp(loss)
            # バッチ内の平均損失から Perplexity を計算
            ppl = torch.exp(loss)
            
            # バッチサイズ分をリストに追加
            if batch_size == 1:
                perplexities.append(ppl.item())
            else:
                # バッチ内の各要素の Perplexity を概算
                # 厳密には各サンプルごとに計算すべきだが、速度優先でバッチ損失を使用
                # より正確に行う場合はループを回す必要がある
                for _ in batch_texts:
                    perplexities.append(ppl.item())
    
    df['perplexity'] = perplexities
    print(f"Perplexity 計算完了。平均 PP: {np.mean(perplexities):.2f}, 中央値：{np.median(perplexities):.2f}")
    
    return df


def apply_threshold(df, percentile_cutoff):
    """Perplexity の閾値で足切りを行う"""
    print(f"\n=== 閾値による足切り (上位 {100-percentile_cutoff}% を除外) ===")
    
    threshold = np.percentile(df['perplexity'], percentile_cutoff)
    print(f"閾値 (Perplexity): {threshold:.2f}")
    
    initial_count = len(df)
    filtered_df = df[df['perplexity'] <= threshold].copy()
    final_count = len(filtered_df)
    removed_count = initial_count - final_count
    
    print(f"\n足切り完了:")
    print(f"  対象データ数：{initial_count}")
    print(f"  除外数：{removed_count}")
    print(f"  残存数：{final_count}")
    
    return filtered_df, threshold


def save_results(scored_df, clean_df, threshold, report_file):
    """結果を保存しレポートを出力する"""
    print(f"\n=== 結果の保存 ===")
    
    # スコア付きデータの保存
    scored_df.to_csv(OUTPUT_SCORED_CSV, index=False, encoding='utf-8-sig')
    print(f"スコア付きデータを保存しました：{OUTPUT_SCORED_CSV}")
    
    # クリーンなデータの保存 (不要なカラムを落とす場合もあるが、ここでは全カラム保存)
    clean_df.to_csv(OUTPUT_CLEAN_CSV, index=False, encoding='utf-8-sig')
    print(f"クリーンなデータを保存しました：{OUTPUT_CLEAN_CSV}")
    
    # レポートの出力
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== 小説データクレンジング・評価レポート ===\n\n")
        f.write(f"入力ファイル：{INPUT_CSV}\n")
        f.write(f"使用モデル：{MODEL_NAME}\n\n")
        
        f.write("[フィルタリング基準]\n")
        f.write(f"- 最大漢字比率：{MAX_KANJI_RATIO}\n")
        f.write(f"- 最小ひらがな比率：{MIN_HIRAGANA_RATIO}\n")
        f.write(f"- 最小文字数：{MIN_LENGTH}\n")
        f.write(f"- 最大文字数：{MAX_LENGTH}\n")
        f.write(f"- Perplexity カットオフ：上位 {100-PERPLEXITY_PERCENTILE_CUTOFF}%\n\n")
        
        f.write("[結果]\n")
        f.write(f"最終的なデータ数：{len(clean_df)}\n")
        f.write(f"Perplexity 閾値：{threshold:.2f}\n")
        f.write(f"Perplexity 統計 (クリーンデータ):\n")
        f.write(f"  平均：{clean_df['perplexity'].mean():.2f}\n")
        f.write(f"  中央値：{clean_df['perplexity'].median():.2f}\n")
        f.write(f"  最小：{clean_df['perplexity'].min():.2f}\n")
        f.write(f"  最大：{clean_df['perplexity'].max():.2f}\n")
    
    print(f"レポートを保存しました：{report_file}")


def main():
    print("小説データの品質評価とフィルタリングを開始します...\n")
    
    # 1. データの読み込み
    df = load_data(INPUT_CSV)
    
    # 2. ルールベースフィルタリング
    df_filtered = rule_based_filter(df)
    
    if len(df_filtered) == 0:
        print("エラー：フィルタリング後にデータが残っていません。パラメータを見直してください。")
        return
    
    # 3. Perplexity の計算
    df_scored = calculate_perplexity(df_filtered, MODEL_NAME)
    
    # 4. 閾値による足切り
    df_clean, threshold = apply_threshold(df_scored, PERPLEXITY_PERCENTILE_CUTOFF)
    
    # 5. 結果の保存
    save_results(df_scored, df_clean, threshold, REPORT_FILE)
    
    print("\n=== 全工程完了 ===")
    print(f"クリーンなデータセット：{len(df_clean)} 件")
    print("次のステップ：このデータを使用して LLM のファインチューニングを行ってください。")


if __name__ == "__main__":
    main()
