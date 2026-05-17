#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_novelty_analysis.py
Phase 2: 新奇性分析・グルーピング・偏りの可視化

目的:
    - 品質フィルタ済みのデータに対して、埋め込みベクトルを用いたクラスタリングを行い「叙述の型」を把握する。
    - 各データポイントの「新奇性スコア（Novelty Score）」を計算し、偏りを防ぐとともに「意外な組み合わせ」を発見する。
    - 出力：新奇性スコア付き CSV、クラスタレポート、可視化画像

Stack:
    - scikit-learn (PCA, KMeans, NearestNeighbors)
    - umap-learn (次元削減・可視化)
    - matplotlib, seaborn (プロット)
    - pandas, numpy

Concepts:
    - Embedding: テキストを意味空間上のベクトルに変換。
    - Clustering (K-Means): 類似した文章をグループ化し「型」を発見。
    - Novelty Score (Local Outlier Factor / Distance to Centroid): 
      クラスタ中心からの距離や、近傍との疎遠度に基づき「珍しさ」を数値化。
      ここでは簡易的に「属するクラスタからの平均距離」を新奇性とする。
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel
import torch
import umap

# --- 設定 ---
MODEL_NAME = "rinna/japanese-gpt2-medium"  # 特徴抽出に使用するモデル（軽量なもの）
# 実際には BERT 系の方が埋め込み性能が良いが、リソース制約のため GPT2 の隠れ層を利用するか、
# もし環境に余裕があれば "cl-tohoku/bert-base-japanese" などを使用推奨。
# ここでは簡易化のため、ランダム投影または前計算された特徴量を想定するロジックにするか、
# 軽量化のため TF-IDF で代用する選択肢もあるが、今回は意味空間を重視し小型 BERT を使うことを前提としたコード構造にする。
# ※実行環境に "cl-tohoku/bert-base-japanese" がない場合、この部分はエラーになる可能性があるため、
#   代替として TF-IDF ベースの実装も併記しておく。

USE_BERT = False  # True にすると BERT を使用。False なら TF-IDF でフォールバック（高速・軽量）

def get_embeddings_tfidf(texts, max_features=500):
    """TF-IDF による簡易埋め込み（軽量・高速）"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    return X.toarray(), vectorizer

def get_embeddings_bert(texts, model_name="cl-tohoku/bert-base-japanese"):
    """BERT による埋め込み（高精度だが重め）"""
    print(f"Loading BERT model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    embeddings = []
    batch_size = 16
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            outputs = model(**inputs)
            # [CLS] トークンの埋め込みを使用
            cls_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeds)
    
    return np.vstack(embeddings), None

def calculate_novelty_scores(X, n_clusters=10, n_neighbors=5):
    """
    クラスタリングを行い、各点が属するクラスタの中心からの距離を「新奇性スコア」とする。
    距離が遠いほど「意外な組み合わせ」または「外れ値（面白い可能性）」とみなす。
    """
    print("Clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    print("Calculating distances to cluster centers...")
    novelty_scores = []
    for i, point in enumerate(X):
        center = kmeans.cluster_centers_[clusters[i]]
        dist = np.linalg.norm(point - center)
        novelty_scores.append(dist)
    
    return np.array(novelty_scores), clusters

def visualize_clusters(X, clusters, novelty_scores, output_prefix):
    """UMAP と PCA による可視化"""
    print("Visualizing clusters...")
    
    # UMAP (高次元データの可視化に強力)
    try:
        reducer = umap.UMAP(random_state=42, n_components=2, metric='cosine')
        embedding_2d = reducer.fit_transform(X)
    except Exception as e:
        print(f"UMAP failed ({e}), falling back to PCA.")
        pca = PCA(n_components=2, random_state=42)
        embedding_2d = pca.fit_transform(X)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=novelty_scores, cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Novelty Score (Distance)')
    plt.title(f'Cluster Visualization (Novelty Map)\nClusters: {len(set(clusters))}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(f"{output_prefix}_novelty_map.png")
    plt.close()
    
    # クラスタ別の散布図
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=clusters, cmap='tab10', alpha=0.6, s=10)
    plt.title(f'Cluster Groups')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(f"{output_prefix}_clusters.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze novelty and clustering of text data.")
    parser.add_argument("--input_file", type=str, default="cleaned_dataset.csv", help="Input cleaned CSV file")
    parser.add_argument("--output_prefix", type=str, default="analysis_result", help="Output file prefix")
    parser.add_argument("--n_clusters", type=int, default=15, help="Number of clusters for K-Means")
    parser.add_argument("--use_bert", action="store_true", help="Use BERT for embeddings (slower but better)")
    args = parser.parse_args()

    # データ読み込み
    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found. Please run 01_evaluate_and_clean.py first.")
        return
    
    df = pd.read_csv(args.input_file)
    texts = df['text'].dropna().astype(str).tolist()
    print(f"Loaded {len(texts)} texts for analysis.")

    # 埋め込み生成
    if args.use_bert:
        X, _ = get_embeddings_bert(texts)
    else:
        print("Using TF-IDF for lightweight embedding...")
        X, _ = get_embeddings_tfidf(texts)
    
    print(f"Embedding shape: {X.shape}")

    # 新奇性スコアとクラスタの計算
    novelty_scores, clusters = calculate_novelty_scores(X, n_clusters=args.n_clusters)

    # 結果を DataFrame に追加
    df['cluster_id'] = clusters
    df['novelty_score'] = novelty_scores

    # ソートして保存（新奇性が高い順）
    df_sorted = df.sort_values(by='novelty_score', ascending=False)
    df_sorted.to_csv(f"{args.output_prefix}_with_novelty.csv", index=False, encoding='utf-8')
    print(f"Saved scored dataset to {args.output_prefix}_with_novelty.csv")

    # 統計レポート
    report_lines = [
        f"=== Novelty Analysis Report ===",
        f"Total samples: {len(df)}",
        f"Number of clusters: {args.n_clusters}",
        f"Novelty Score Stats:",
        f"  Mean: {novelty_scores.mean():.4f}",
        f"  Std:  {novelty_scores.std():.4f}",
        f"  Max:  {novelty_scores.max():.4f}",
        f"  Min:  {novelty_scores.min():.4f}",
        f"",
        f"Top 5 Novel Texts (Indices):",
    ]
    top_indices = df_sorted.head(5).index.tolist()
    for idx in top_indices:
        report_lines.append(f"- Index {idx}: Score={df.loc[idx, 'novelty_score']:.4f}")
        report_lines.append(f"  Text: {df.loc[idx, 'text'][:50]}...")
    
    with open(f"{args.output_prefix}_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Saved report to {args.output_prefix}_report.txt")

    # 可視化
    visualize_clusters(X, clusters, novelty_scores, args.output_prefix)
    print(f"Saved visualization plots.")

if __name__ == "__main__":
    main()
