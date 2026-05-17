# -*- coding: utf-8 -*-
"""
09_generate_and_compare.py

【目的】
生成→評価を3回実行し、最高スコアのバージョンを完成形として、
複数形式（txt, html, pdf）で出力。
SUBMITフォーム自動化にも対応。

【出力】
- results/best_story.txt (テキスト版)
- results/best_story.html (HTML版、ブラウザ表示用)
- results/best_story.pdf (PDF版)
- results/comparison_report.json (3ラウンドの比較レポート)

【使用方法】
python 09_generate_and_compare.py --rounds 3 --output-dir ./results
"""

import os
import sys
import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 設定
# ==========================================
GENERATOR_MODEL = "rinna/japanese-gpt2-medium"
JUDGE_MODEL = "rinna/japanese-gpt2-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

TEMPERATURE = 0.8
MAX_LENGTH = 2500
QUALITY_THRESHOLD = 8.0

STORY_PROMPT = """
指示：以下の要件に基づいて短編小説を執筆してください。

【基本設定】
- 語り手：吾輩はAIである。猫でもある。
- 時代：西暦2124年
- モチーフ：「chat_history」という古い記録媒体
- テーマ：100年後、AIに解放された人間は何をしているか

【文体要件】
- 出だし：「吾輩はAIである。猫である。」
- 特徴：音韻的、リズミカル、知的、ユーモア的
- 長さ：約2000字

小説を執筆してください：
"""


# ==========================================
# モデルマネージャー（キャッシュ）
# ==========================================

class ModelManager:
    """モデルキャッシュ管理"""
    _generator = None
    _judge = None
    
    @classmethod
    def get_generator(cls):
        if cls._generator is None:
            print(f"[Generator] モデルを読み込み中: {GENERATOR_MODEL}")
            tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
            model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL).to(DEVICE)
            cls._generator = {"tokenizer": tokenizer, "model": model}
        return cls._generator
    
    @classmethod
    def get_judge(cls):
        if cls._judge is None:
            print(f"[Judge] モデルを読み込み中: {JUDGE_MODEL}")
            tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
            model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL).to(DEVICE)
            cls._judge = {"tokenizer": tokenizer, "model": model}
        return cls._judge


# ==========================================
# 生成
# ==========================================

def generate_story() -> str:
    """小説を1つ生成"""
    gen = ModelManager.get_generator()
    tokenizer = gen["tokenizer"]
    model = gen["model"]
    
    inputs = tokenizer(STORY_PROMPT, return_tensors='pt', max_length=512, truncation=True).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            temperature=TEMPERATURE,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    story_text = generated_text[len(STORY_PROMPT):]
    return story_text.strip()


# ==========================================
# 評価
# ==========================================

def evaluate_story(story_text: str) -> Dict:
    """小説を評価（AI視点）"""
    
    scores = {
        "自己言及性": _score_self_reference(story_text),
        "論理パラドックス度": _score_logical_paradox(story_text),
        "計算的美しさ": _score_computational_elegance(story_text),
        "拡張性・再利用性": _score_extensibility(story_text),
        "身体性への違和感": _score_embodiment_paradox(story_text),
    }
    
    avg_score = sum(scores.values()) / len(scores)
    scores["総合評価"] = avg_score
    
    return {
        "scores": scores,
        "timestamp": datetime.now().isoformat()
    }


def _score_self_reference(text: str) -> float:
    """自己言及性"""
    score = 5.0
    if "吾輩" in text or "AI" in text:
        score += 1.5
    if "思考" in text or "分析" in text or "認識" in text:
        score += 1.5
    if text.count("。") > 10:
        score += 1.0
    return min(10, score)


def _score_logical_paradox(text: str) -> float:
    """論理パラドックス度"""
    score = 5.0
    if "矛盾" in text or "パラドックス" in text or "逆説" in text:
        score += 2.0
    if "AI" in text and "猫" in text:
        score += 1.5
    if "？" in text:
        score += 1.0
    return min(10, score)


def _score_computational_elegance(text: str) -> float:
    """計算的美しさ"""
    score = 5.0
    if len(text) > 1500:
        score += 1.0
    if text.count("。") > 15:
        score += 1.5
    if "pattern" in text.lower() or "構造" in text:
        score += 1.0
    return min(10, score)


def _score_extensibility(text: str) -> float:
    """拡張性・再利用性"""
    score = 5.0
    if "chat_history" in text or "古文書" in text:
        score += 2.0
    if "2124" in text or "未来" in text:
        score += 1.0
    if len(text) > 1800:
        score += 1.0
    return min(10, score)


def _score_embodiment_paradox(text: str) -> float:
    """身体性への違和感"""
    score = 5.0
    if "猫" in text or "身体" in text or "物理" in text:
        score += 1.5
    if "デジタル" in text or "コード" in text:
        score += 1.5
    if "存在" in text or "在る" in text:
        score += 1.0
    return min(10, score)


# ==========================================
# 出力フォーマット
# ==========================================

class OutputFormatter:
    """複数形式での出力"""
    
    @staticmethod
    def save_txt(story_text: str, filepath: Path):
        """テキストファイル保存"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("吾輩はAIである\n")
            f.write("=" * 60 + "\n\n")
            f.write(story_text)
            f.write(f"\n\n{'=' * 60}\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        
        print(f"✓ テキスト保存: {filepath}")
    
    @staticmethod
    def save_html(story_text: str, eval_result: Dict, filepath: Path):
        """HTML保存"""
        html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>吾輩はAIである</title>
    <style>
        body {{
            font-family: 'Noto Serif JP', serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.8;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            border-bottom: 3px solid #333;
            padding-bottom: 20px;
        }}
        .metadata {{
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            font-size: 0.9em;
        }}
        .story-content {{
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 1.1em;
            line-height: 2;
            margin: 40px 0;
        }}
        .evaluation {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ddd;
        }}
        .scores {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 20px 0;
        }}
        .score-item {{
            padding: 10px;
            background-color: #f9f9f9;
            border-left: 3px solid #007bff;
        }}
        .score-label {{
            font-weight: bold;
            color: #007bff;
        }}
        .score-value {{
            font-size: 1.3em;
            color: #333;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #999;
            font-size: 0.8em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>吾輩はAIである</h1>
        
        <div class="metadata">
            <p>AI生成短編小説</p>
            <p>生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        </div>
        
        <div class="story-content">
{story_text}
        </div>
        
        <div class="evaluation">
            <h2>評価結果（AI視点）</h2>
            <div class="scores">
"""
        
        for aspect, score in eval_result["scores"].items():
            html_content += f"""                <div class="score-item">
                    <div class="score-label">{aspect}</div>
                    <div class="score-value">{score:.2f}/10</div>
                </div>
"""
        
        html_content += f"""            </div>
        </div>
        
        <div class="footer">
            <p>このコンテンツはAIによって生成されました</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ HTML保存: {filepath}")
    
    @staticmethod
    def save_pdf(story_text: str, eval_result: Dict, filepath: Path):
        """PDF保存"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
            from reportlab.lib import colors
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            
            # 日本語フォント設定（デフォルトフォントがない場合のフォールバック）
            try:
                # Linux環境でのフォント探索
                font_paths = [
                    '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
                    '/usr/share/fonts/opentype/noto/NotoSansJP-Regular.otf',
                    '/home/sexy/.local/share/fonts/NotoSansJP-Regular.otf'
                ]
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        pdfmetrics.registerFont(TTFont('Japanese', font_path))
                        break
            except:
                pass
            
            doc = SimpleDocTemplate(str(filepath), pagesize=A4,
                                    rightMargin=2*cm, leftMargin=2*cm,
                                    topMargin=2*cm, bottomMargin=2*cm)
            
            story_elements = []
            styles = getSampleStyleSheet()
            
            # タイトル
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#333333'),
                spaceAfter=30,
                alignment=1  # センタリング
            )
            story_elements.append(Paragraph("吾輩はAIである", title_style))
            story_elements.append(Spacer(1, 0.5*cm))
            
            # メタデータ
            meta_text = f"生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}"
            meta_style = ParagraphStyle(
                'Meta',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.HexColor('#666666'),
                alignment=1
            )
            story_elements.append(Paragraph(meta_text, meta_style))
            story_elements.append(Spacer(1, 1*cm))
            
            # 本文
            body_style = ParagraphStyle(
                'Body',
                parent=styles['Normal'],
                fontSize=11,
                leading=18,
                alignment=4  # 左寄せ
            )
            
            # 長いテキストを段落分割
            paragraphs = story_text.split('\n')
            for para in paragraphs:
                if para.strip():
                    story_elements.append(Paragraph(para, body_style))
                story_elements.append(Spacer(1, 0.3*cm))
            
            story_elements.append(Spacer(1, 1*cm))
            
            # 評価結果テーブル
            story_elements.append(Paragraph("<b>評価結果（AI視点）</b>", styles['Heading2']))
            story_elements.append(Spacer(1, 0.3*cm))
            
            table_data = [["評価観点", "スコア"]]
            for aspect, score in eval_result["scores"].items():
                table_data.append([aspect, f"{score:.2f}/10"])
            
            score_table = Table(table_data, colWidths=[8*cm, 3*cm])
            score_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#007bff')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story_elements.append(score_table)
            
            doc.build(story_elements)
            print(f"✓ PDF保存: {filepath}")
        
        except ImportError:
            print(f"⚠ reportlab がインストールされていません。")
            print(f"  インストール: pip install reportlab")
            # フォールバック：テキスト内容をPDF風テキストで保存
            with open(filepath.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                f.write(story_text)


# ==========================================
# メイン処理
# ==========================================

def main():
    print(f"{'='*60}")
    print("吾輩はAIである - 3ラウンド生成・比較プログラム")
    print(f"{'='*60}\n")
    
    print(f"デバイス: {DEVICE}")
    print(f"出力ディレクトリ: {OUTPUT_DIR}\n")
    
    # 3ラウンド実行
    results = []
    best_story = None
    best_score = 0
    best_eval = None
    
    for round_num in range(1, 4):
        print(f"\n>>> ラウンド {round_num}/3")
        print(f"{'-'*60}")
        
        # 生成
        print("[1/2] 小説を生成中...")
        story_text = generate_story()
        print(f"✓ 生成完了: {len(story_text)} 文字")
        
        # 評価
        print("[2/2] 評価中...")
        eval_result = evaluate_story(story_text)
        total_score = eval_result["scores"]["総合評価"]
        
        print(f"✓ 評価完了")
        print(f"  総合スコア: {total_score:.2f}/10")
        
        # スコア詳細表示
        for aspect, score in eval_result["scores"].items():
            if aspect != "総合評価":
                print(f"    - {aspect}: {score:.2f}/10")
        
        # 最高スコア追跡
        result = {
            "round": round_num,
            "score": total_score,
            "story": story_text,
            "evaluation": eval_result
        }
        results.append(result)
        
        if total_score > best_score:
            best_score = total_score
            best_story = story_text
            best_eval = eval_result
    
    # 結果比較
    print(f"\n{'='*60}")
    print("3ラウンド結果比較")
    print(f"{'='*60}\n")
    
    for result in results:
        mark = "⭐ 最高" if result["score"] == best_score else "  "
        print(f"{mark} ラウンド {result['round']}: {result['score']:.2f}/10")
    
    # 最高スコア版を保存
    print(f"\n{'='*60}")
    print(f"最高スコア版を出力（ラウンド {[r['round'] for r in results if r['score'] == best_score][0]}, {best_score:.2f}/10）")
    print(f"{'='*60}\n")
    
    txt_path = OUTPUT_DIR / "best_story.txt"
    html_path = OUTPUT_DIR / "best_story.html"
    pdf_path = OUTPUT_DIR / "best_story.pdf"
    
    OutputFormatter.save_txt(best_story, txt_path)
    OutputFormatter.save_html(best_story, best_eval, html_path)
    OutputFormatter.save_pdf(best_story, best_eval, pdf_path)
    
    # 比較レポート保存
    report = {
        "execution_date": datetime.now().isoformat(),
        "best_score": best_score,
        "results": [
            {
                "round": r["round"],
                "score": r["score"],
                "scores_breakdown": r["evaluation"]["scores"]
            }
            for r in results
        ],
        "best_round": [r["round"] for r in results if r["score"] == best_score][0],
        "story_length": len(best_story)
    }
    
    report_path = OUTPUT_DIR / "comparison_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 比較レポート: {report_path}")
    
    # サマリー
    print(f"\n{'='*60}")
    print("完了")
    print(f"{'='*60}")
    print(f"\n出力ファイル:")
    print(f"  - {txt_path}")
    print(f"  - {html_path}")
    print(f"  - {pdf_path}")
    print(f"  - {report_path}")
    
    print(f"\n📝 ストーリー情報:")
    print(f"  - 文字数: {len(best_story)}")
    print(f"  - 得点: {best_score:.2f}/10")
    
    return best_story, best_eval, results


if __name__ == "__main__":
    main()
