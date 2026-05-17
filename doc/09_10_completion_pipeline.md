# 09-10 完成形生成・投稿パイプライン

## 概要

3ラウンドの生成・評価を行い、最高得点のバージョンを完成形として複数形式で出力し、プラットフォームに自動投稿するパイプライン。

## ファイル構成

| ファイル | 機能 | 備考 |
|---------|------|------|
| `09_generate_and_compare.py` | 3ラウンド実行→最高版を txt/html/pdf 出力 | メイン |
| `10_auto_submit.py` | 生成した小説をプラットフォームに投稿 | 補助（手動投稿も可） |

## 使用フロー

```
09_generate_and_compare.py
        ↓
[Round 1] 生成 → 評価
[Round 2] 生成 → 評価
[Round 3] 生成 → 評価
        ↓
比較・最高版選定
        ↓
出力形式変換
        ├─ results/best_story.txt
        ├─ results/best_story.html
        ├─ results/best_story.pdf
        └─ results/comparison_report.json
        ↓
10_auto_submit.py (オプション)
        ↓
プラットフォーム投稿
```

## 実行方法

### 1️⃣ 3ラウンド生成・比較

```bash
python 09_generate_and_compare.py
```

**出力:**
- `results/best_story.txt` - テキスト版
- `results/best_story.html` - ブラウザ表示用
- `results/best_story.pdf` - PDF版（オプション）
- `results/comparison_report.json` - 3ラウンドの詳細スコア

### 2️⃣ 自動投稿（オプション）

#### A) ブラウザで手動投稿

```bash
python 10_auto_submit.py --platform browser
```

**動作:**
- デフォルトブラウザで投稿ページを開く
- 本文をクリップボードにコピー
- 手動で投稿フォームに貼り付け

#### B) カクヨム自動投稿

```bash
# 認証情報を環境変数に設定
export KAKUYOMU_EMAIL="your_email@example.com"
export KAKUYOMU_PASSWORD="your_password"

# 投稿実行
python 10_auto_submit.py --platform kakuyomu
```

**要件:**
- Selenium インストール: `pip install selenium webdriver-manager`
- プラウザドライバー自動ダウンロード対応

## オプション

### 09_generate_and_compare.py

```bash
python 09_generate_and_compare.py \
  --rounds 3 \
  --output-dir ./results
```

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--rounds` | 3 | ラウンド数 |
| `--output-dir` | ./results | 出力ディレクトリ |

### 10_auto_submit.py

```bash
python 10_auto_submit.py \
  --platform kakuyomu \
  --story-file results/best_story.txt \
  --title "吾輩はAIである"
```

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--platform` | browser | 投稿先（kakuyomu/syosetu/note/browser） |
| `--story-file` | results/best_story.txt | 投稿するファイル |
| `--title` | 吾輩はAIである | 投稿時のタイトル |

## 出力形式

### txt形式
- プレーンテキスト
- 用途: テキストエディタ、ターミナル表示

### html形式
```html
<!DOCTYPE html>
<html lang="ja">
  <head>スタイル定義</head>
  <body>
    <h1>吾輩はAIである</h1>
    <div class="story-content">本文</div>
    <div class="evaluation">評価結果テーブル</div>
  </body>
</html>
```
- 用途: Webブラウザ表示、SNS共有

### pdf形式（reportlab）
- A4サイズ、日本語対応
- メタデータ：生成日時、評価スコア
- 用途: 印刷、公式投稿時の配付資料

**注:** PDF出力には `reportlab` が必要
```bash
pip install reportlab
```

## 評価メトリクス

### AI視点の5観点スコア

各項目0-10点：

1. **自己言及性** - AIが自分自身を分析
2. **論理パラドックス度** - 矛盾の深さ
3. **計算的美しさ** - アルゴリズム的構造の美しさ
4. **拡張性・再利用性** - 般化可能性
5. **身体性への違和感** - デジタル×物理の矛盾

**総合評価** = 5観点の平均

### 比較レポート形式

```json
{
  "execution_date": "2026-05-18T15:30:00...",
  "best_score": 7.85,
  "results": [
    {
      "round": 1,
      "score": 6.92,
      "scores_breakdown": {
        "自己言及性": 6.8,
        "論理パラドックス度": 7.2,
        ...
      }
    },
    ...
  ],
  "best_round": 2,
  "story_length": 2145
}
```

## SUBMITフォーム自動化について

### 現状

✅ **実装済み**
- カクヨム（Selenium + ブラウザ自動化）
- ブラウザ手動投稿（パイプ）

⚠️  **要認証**
- 小説家になろう（API実装予定）
- note（API実装予定）

### 問題と解決

| 問題 | 原因 | 解決策 |
|------|------|--------|
| JavaScript動的フォーム | SPA型プラットフォーム | Selenium + 待機 |
| CAPTCHA | ボット対策 | 手動確認 |
| ログイン状態の維持 | セッション管理 | クッキー保存 |
| 利用規約違反 | 過度な自動化 | 手動投稿推奨 |

### 推奨フロー

```
自動生成 (✓ 問題なし)
    ↓
形式変換 (✓ 問題なし)
    ↓
投稿 (⚠️ 慎重に)
    ├─ 強く推奨: ブラウザ手動投稿
    ├─ 要確認: 自動投稿（利用規約確認）
    └─ 非推奨: API無効化時の無理やり自動化
```

## サンプル実行

### 最小限の実行

```bash
# ローカル環境での実行（3ラウンド、txt/html/pdf出力）
python 09_generate_and_compare.py
```

### Colab対応実行

```python
# Colabノートブック内
!cd /content/ashizawa_stories && python 09_generate_and_compare.py

# 結果のダウンロード
from google.colab import files
files.download('results/best_story.pdf')
```

## トラブルシューティング

### Q: PDF出力に失敗する

**A:** reportlab をインストール
```bash
pip install reportlab
```

### Q: カクヨム自動投稿が失敗する

**A:** Seleniumドライバーを確認
```bash
pip install webdriver-manager
```

### Q: HTMLが文字化けする

**A:** ファイルをUTF-8で保存（既に対応済み）

### Q: 評価スコアが毎回変わる

**A:** LLMのサンプリング温度が高い（仕様）
調整が必要な場合は `TEMPERATURE` を変更

## 次のステップ

1. **複数タイトルの連続生成**
   - `for` ループで複数作品を生成
   - 各作品ごとに最高版を保存

2. **評価基準のカスタマイズ**
   - `_score_*` 関数を修正
   - 業界固有の評価軸を追加

3. **プラットフォーム対応拡大**
   - Wattpad, AO3 対応
   - API実装完備

4. **分析機能強化**
   - 時系列グラフ化
   - テーマ別の統計分析
   - 読者フィードバック自動取得

---

**更新日**: 2026年5月18日  
**バージョン**: 0.1.0 (Beta)
