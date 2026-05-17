import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ==========================================
# 設定
# ==========================================
INPUT_CSV = 'twnovel_roseaullus/twnovel_dataset_roseaullus.csv'
OUTPUT_CLEAN_CSV = 'twnovel_roseaullus/cleaned_dataset.csv'
OUTPUT_SCORED_CSV = 'twnovel_roseaullus/scored_dataset.csv'

# モデル設定 (軽量な日本語モデルを使用)
# rinna/japanese-gpt-1b は比較的小さく、品質評価に適しています
MODEL_NAME = "rinna/japanese-gpt-1b"
DEVICE = "cpu"  # ローカル環境用。Colab/Kaggleなら "cuda" に変更可能
BATCH_SIZE = 4   # メモリに合わせて調整
MAX_LENGTH = 512 # 評価に使う最大トークン数

# 足切り閾値 (パープレキシティ)
# 後ほど分布を見て決定することも可能です
PERPLEXITY_THRESHOLD = 15.0 

# ==========================================
# 1. データ読み込みと基本的なクリーニング
# ==========================================
print("データを読み込んでいます...")
df = pd.read_csv(INPUT_CSV)

# textカラムが存在するか確認
if 'text' not in df.columns:
    # カラム名が違う場合のフォールバック（例：先頭カラムなど）
    text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    print(f"'text' カラムが見つかりませんでした。'{text_col}' を使用します。")
    df['text'] = df[text_col]

# NaN の除去
df = df.dropna(subset=['text'])
df['text'] = df['text'].astype(str)

print(f"初期データ数: {len(df)}")

# ==========================================
# 2. ルールベースのフィルタリング (おかしな日本語の一次排除)
# ==========================================
print("ルールベースのフィルタリングを実行中...")

def is_valid_text(text):
    if len(text) < 10: # 短すぎるものは除外
        return False
    if len(text) > MAX_LENGTH * 4: # 長すぎるものは一旦除外（モデル入力制限のため）
        return False
    
    # 特殊文字や制御文字のチェック
    if re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', text):
        return False
    
    # 文字種比率のチェック (極端な偏りがあるものは怪しい)
    kanji = len(re.findall(r'[\u4e00-\u9fff]', text))
    hiragana = len(re.findall(r'[\u3040-\u309f]', text))
    katakana = len(re.findall(r'[\u30a0-\u30ff]', text))
    total = len(text)
    
    if total == 0:
        return False
        
    ratio_kanji = kanji / total
    ratio_hira = hiragana / total
    
    # 漢字が多すぎる（80%以上）、またはひらがな・カタカナがほぼない（10%未満）場合は怪しい
    # ※小説ジャンルによりますが、一般的な目安
    if ratio_kanji > 0.85:
        return False
    if (ratio_hira + katakana) < 0.05:
        return False
        
    return True

mask = df['text'].apply(is_valid_text)
df_filtered = df[mask].reset_index(drop=True)
print(f"ルールベースフィルタ後: {len(df_filtered)} (除外数: {len(df) - len(df_filtered)})")

# ==========================================
# 3. 言語モデルによる品質評価 (パープレキシティ)
# ==========================================
print(f"品質評価モデル ({MODEL_NAME}) を読み込んでいます... (時間がかかります)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)
model.eval()

def calculate_perplexity_batch(texts):
    """
    テキストのバッチに対してパープレキシティを計算する
    """
    encodings = tokenizer(
        texts, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=MAX_LENGTH
    )
    
    input_ids = encodings.input_ids.to(DEVICE)
    attention_mask = encodings.attention_mask.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss # average loss per token
        
        # Perplexity = exp(loss)
        perplexities = torch.exp(loss)
        
    return perplexities.cpu().numpy()

scores = []
print("文章の品質評価中 (パープレキシティ計算)...")

# バッチ処理で実行
for i in tqdm(range(0, len(df_filtered), BATCH_SIZE)):
    batch_texts = df_filtered['text'].iloc[i:i+BATCH_SIZE].tolist()
    try:
        ppls = calculate_perplexity_batch(batch_texts)
        scores.extend(ppls)
    except Exception as e:
        # エラーが出た場合は最大スコアを入れておく（除外対象とする）
        scores.extend([float('inf')] * len(batch_texts))
        print(f"バッチ処理中にエラーが発生しました: {e}")

df_filtered['perplexity_score'] = scores

# ==========================================
# 4. 足切りと結果保存
# ==========================================
print(f"パープレキシティの統計情報:")
print(df_filtered['perplexity_score'].describe())

# 足切り実行
final_df = df_filtered[df_filtered['perplexity_score'] <= PERPLEXITY_THRESHOLD].reset_index(drop=True)

print(f"\n足切り閾値: {PERPLEXITY_THRESHOLD}")
print(f"最終的なデータ数: {len(final_df)}")
print(f"除外されたデータ数: {len(df_filtered) - len(final_df)}")

# スコア付きデータを保存（分析用）
df_filtered.to_csv(OUTPUT_SCORED_CSV, index=False)
print(f"スコア付き全データを保存: {OUTPUT_SCORED_CSV}")

# クリーンなデータを保存
final_df.drop(columns=['perplexity_score']).to_csv(OUTPUT_CLEAN_CSV, index=False)
print(f"クリーンなデータを保存: {OUTPUT_CLEAN_CSV}")

print("処理が完了しました。")
