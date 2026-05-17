#!/bin/bash
set -e

echo "========================================="
echo "  AI=CAT: Wagahai Novel Generator Pipeline"
echo "========================================="

INPUT_CSV="twnovel_dataset_roseaullus.csv"

echo "[Phase 0] 環境セットアップ..."
python 00_setup_colab.py

echo "[Phase 1] データの品質評価とフィルタリング..."
if [ ! -f "$INPUT_CSV" ]; then
    echo "エラー: 入力ファイル $INPUT_CSV が見つかりません。"
    exit 1
fi
python 01_evaluate_and_clean.py --input_file "$INPUT_CSV"
CLEANED_CSV="cleaned_dataset.csv"

echo "[Phase 2] 新奇性分析とクラスタリング..."
if [ ! -f "$CLEANED_CSV" ]; then
    echo "エラー: クリーンデータ $CLEANED_CSV が見つかりません。"
    exit 1
fi
python 02_novelty_analysis.py --input_file "$CLEANED_CSV"
NOVELTY_CSV="analysis_result_with_novelty.csv"

echo "[Phase 3] 学習用データの整形と分割..."
if [ ! -f "$NOVELTY_CSV" ]; then
    echo "エラー: 新奇性分析結果 $NOVELTY_CSV が見つかりません。"
    exit 1
fi
python 03_prepare_training.py --input_file "$NOVELTY_CSV"

echo "[Phase 4] '吾輩はAIである' テーマのデータ強化..."
python 04_setup_wagahai.py --input_file "$NOVELTY_CSV" --merge

echo "[Phase 5] ファインチューニング開始..."
MERGED_DATA="./wagahai_theme/merged_wagahai_dataset.csv"
if [ -f "$MERGED_DATA" ]; then
    echo "マージ済みデータを使用して学習します..."
    python 05_train_wagahai.py --input_data "$MERGED_DATA" --output_dir "./wagahai_ai_model" --num_epochs 3
else
    echo "マージ済みデータが見つからないため、Phase 3 のデータで学習します..."
    python 05_train_wagahai.py --input_data "./training_data/train.jsonl" --output_dir "./wagahai_ai_model" --num_epochs 3
fi

echo "========================================="
echo "  パイプライン完了！"
echo "  生成モデル: ./wagahai_ai_model/"
echo "========================================="