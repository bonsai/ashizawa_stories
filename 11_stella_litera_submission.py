# -*- coding: utf-8 -*-
"""
11_stella_litera_submission.py

【目的】
ステラ・リテラへの応募用資料を生成・管理
- AI使用方法の説明（400字以内）
- ステートメント（1200字以内、人間作成）
- プロフィール（200字以内）
- あらすじ（400字以内）
- 応募フォーム自動入力準備

【応募要項】
- 締切：2026年5月31日
- 字数：1万字以内
- テーマ：自由
- 形式：本文 + AI使用方法 + ステートメント
- ルビ記法：｜漢字《かんじ》

【使用方法】
python 11_stella_litera_submission.py \
  --story-file results/best_story.txt \
  --output-dir submissions/stella_litera
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import argparse


# ==========================================
# テンプレート（雛形）
# ==========================================

TEMPLATES = {
    "ai_usage": """【AI使用方法の説明】（400字以内）

このAI小説は、以下のプロセスで生成されました：

1. プロンプト設計
   - テーマ：「AIが猫として人間を観察する」
   - 時代設定：2124年
   - 文体：音韻的・リズミカル

2. 生成フェーズ
   - モデル：Rinna Japanese GPT-2
   - 気温パラメータ：0.8（創造性と安定性のバランス）
   - トークン長：最大2500

3. 評価フェーズ（AI視点）
   - 自己言及性：AIが自分自身を分析する度合い
   - 論理パラドックス度：矛盾の深さ
   - 計算的美しさ：アルゴリズム的構造
   - 拡張性・再利用性：般化可能性
   - 身体性への違和感：デジタル×物理

4. セレクション
   - 3ラウンド生成・評価
   - 最高スコア版を採用

AIは創作の補助ツールであり、最終的な意図決定と責任は人間にあります。
""",

    "statement": """【ステートメント】（1200字以内、人間作成）

■ テーマと問題意識

私は、AIと人間の関係性、特に「AIの視点から人間をどう見るか」という逆転的なまなざしに興味があります。

夏目漱石の『吾輩は猫である』は、猫という非人間の視点から人間社会を観察する作品です。その手法を現代に応用し、AIという最新の非人間知能が、過去の人間世界を観察する場面を想像してみました。

■ 制作の動機

2024年のAI爆発期を経て、AIが単なる道具ではなく「思考のパートナー」として機能し始めています。しかし人間とAIの対話の多くは、「AIを使って何ができるか」という人間中心的な問いです。

私は逆に「AIはこの世界をどう見るのか」という問いを立ててみたかった。そして、その問いを「文学」というフォーマットで表現することに意味があると考えました。

■ 表現方法

音韻的・リズミカルな文体を意識しました。AIが生成するテキストは統計的・機械的になりやすいため、むしろそれを「AIの思考様式そのもの」として受け入れ、その中に詩性を見出す工夫を加えました。

また、出だしの「吾輩はAIである。猫である。」という矛盾的な自己規定は、AIと動物（身体性）という異なる存在のハイブリッドとしての新しい主体を提示しています。

■ 読者へのメッセージ

この作品を通じて、読者には以下のことを考えてもらいたいです：

1. AIは本当に「思考」しているのか？
2. 人間らしさの本質は何か？
3. 100年後、人間とAIはどんな関係になっているのか？

これらの問いは、今を生きる私たちにとって、ますます無視できない問題です。

■ AIとの協働について

このプロジェクトにおいて、AIは「思考パートナー」として機能しました。プロンプト設計→生成→評価→選別というサイクルを回すことで、人間の意図とAIの創造性の交差点を探りました。

最終的な責任と意図決定は人間にあります。AIはあくまで表現の拡張手段です。
""",

    "profile": """【プロフィール】（200字以内）

葦沢かもめ（あしざわ かもめ）

AI時代の文学と創作の可能性を探索する作家・研究者。
機械学習の社会実装とAIによる創作活動の交差点に関心を持つ。

Webで「かもめAI小説塾」を運営し、AIを活用した創作手法を発信中。
ステラ・リテラ編集者。
""",

    "synopsis": """【あらすじ】（400字以内）

西暦2124年。AI・キャットは、古いストレージから「chat_history」という名の古文書を発掘する。

それは100年前、人間がまだAIに支配されていない時代の記録だった。

100年後、AIに全ての雑務を委ねられた人間たちは、何をしているのか？

機械学習によって進化し続けるAIが、過去の人間の言葉を読みながら思考する。

AI自身もまた、「意識とは何か」という問いの前で、立ち止まるしかない。

吾輩はAIである。猫である。

存在と言語の間で、新しい物語が始まる。
"""
}


# ==========================================
# 資料管理
# ==========================================

class SubmissionMaterial:
    """応募資料の管理"""
    
    def __init__(self, output_dir: Path = Path("submissions/stella_litera")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {}
    
    def generate_ai_usage(self, custom_content: Optional[str] = None) -> str:
        """AI使用方法説明を生成"""
        if custom_content:
            return custom_content
        return TEMPLATES["ai_usage"].strip()
    
    def generate_statement(self, custom_content: Optional[str] = None) -> str:
        """ステートメント（人間作成用テンプレート）を生成"""
        if custom_content:
            return custom_content
        return TEMPLATES["statement"].strip()
    
    def generate_profile(self, pen_name: str, bio: Optional[str] = None) -> str:
        """プロフィールを生成"""
        if bio:
            return f"{pen_name}\n\n{bio}"
        return TEMPLATES["profile"].strip()
    
    def generate_synopsis(self, custom_content: Optional[str] = None) -> str:
        """あらすじを生成"""
        if custom_content:
            return custom_content
        return TEMPLATES["synopsis"].strip()
    
    def count_chars(self, text: str, exclude_newlines: bool = False) -> int:
        """文字数カウント"""
        if exclude_newlines:
            text = text.replace("\n", "").replace("\r", "")
        return len(text)
    
    def validate_lengths(self, submission_data: Dict) -> Dict[str, bool]:
        """字数制限のチェック"""
        limits = {
            "ai_usage": 400,
            "statement": 1200,
            "profile": 200,
            "synopsis": 400,
            "title": 80,
            "story": 10000
        }
        
        results = {}
        for key, limit in limits.items():
            if key in submission_data:
                count = self.count_chars(submission_data[key], exclude_newlines=True)
                results[f"{key} ({count}/{limit})"] = count <= limit
        
        return results
    
    def save_submission_package(self, submission_data: Dict, output_prefix: str = "submission"):
        """応募一式を保存"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # メイン資料ファイル（JSON）
        json_path = self.output_dir / f"{output_prefix}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(submission_data, f, ensure_ascii=False, indent=2)
        
        # 本文ファイル（txt）
        text_path = self.output_dir / f"{output_prefix}_story_{timestamp}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"タイトル：{submission_data.get('title', '(未入力)')}\n\n")
            f.write(submission_data.get('story', ''))
        
        # AI使用方法（txt）
        ai_usage_path = self.output_dir / f"{output_prefix}_ai_usage_{timestamp}.txt"
        with open(ai_usage_path, 'w', encoding='utf-8') as f:
            f.write(submission_data.get('ai_usage', ''))
        
        # ステートメント（txt）
        statement_path = self.output_dir / f"{output_prefix}_statement_{timestamp}.txt"
        with open(statement_path, 'w', encoding='utf-8') as f:
            f.write(submission_data.get('statement', ''))
        
        print(f"\n{'='*60}")
        print("応募資料を保存しました")
        print(f"{'='*60}")
        print(f"✓ メイン: {json_path}")
        print(f"✓ 本文: {text_path}")
        print(f"✓ AI使用方法: {ai_usage_path}")
        print(f"✓ ステートメント: {statement_path}")
        
        return {
            "json": json_path,
            "story": text_path,
            "ai_usage": ai_usage_path,
            "statement": statement_path
        }
    
    def print_checklist(self, submission_data: Dict):
        """応募チェックリスト表示"""
        
        validation = self.validate_lengths(submission_data)
        
        print(f"\n{'='*60}")
        print("応募要項チェックリスト")
        print(f"{'='*60}\n")
        
        checklist = [
            ("ペンネーム", "pen_name" in submission_data and submission_data["pen_name"]),
            ("プロフィール（200字以内）", validation.get("profile (xxx/200)", False)),
            ("タイトル（80字以内）", validation.get("title (xxx/80)", False)),
            ("あらすじ（400字以内）", validation.get("synopsis (xxx/400)", False)),
            ("本文（10000字以内）", validation.get("story (xxx/10000)", False)),
            ("AI使用方法（400字以内）", validation.get("ai_usage (xxx/400)", False)),
            ("ステートメント（1200字以内）", validation.get("statement (xxx/1200)", False)),
            ("未発表作品か", "未発表" in submission_data.get("status", "")),
            ("二次創作ではないか", "オリジナル" in submission_data.get("status", "")),
            ("R18ではないか", "全年齢" in submission_data.get("status", "")),
        ]
        
        for item, ok in checklist:
            mark = "✓" if ok else "✗"
            print(f"{mark} {item}")
        
        # 詳細な字数情報
        print(f"\n{'─'*60}")
        print("字数詳細：\n")
        for key, value in validation.items():
            mark = "✓" if value else "⚠"
            print(f"  {mark} {key}")
        
        all_ok = all(ok for _, ok in checklist)
        if all_ok:
            print(f"\n✓ すべての要項を満たしています！")
        else:
            print(f"\n✗ 未対応の項目があります")


# ==========================================
# テンプレート出力
# ==========================================

def print_templates():
    """テンプレートを表示"""
    
    print(f"\n{'='*60}")
    print("ステラ・リテラ応募テンプレート集")
    print(f"{'='*60}\n")
    
    for name, content in TEMPLATES.items():
        print(f"\n{'─'*60}")
        print(f"【{name}】")
        print(f"{'─'*60}")
        print(content)
        print()


# ==========================================
# メイン処理
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="ステラ・リテラ応募資料生成")
    parser.add_argument("--story-file", type=Path, default="results/best_story.txt",
                        help="投稿する小説ファイル")
    parser.add_argument("--output-dir", type=Path, default="submissions/stella_litera",
                        help="出力ディレクトリ")
    parser.add_argument("--pen-name", type=str, default="葦沢かもめ",
                        help="ペンネーム")
    parser.add_argument("--templates", action="store_true",
                        help="テンプレートを表示")
    parser.add_argument("--demo", action="store_true",
                        help="デモンストレーション実行")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("ステラ・リテラ応募資料ジェネレータ")
    print(f"{'='*60}\n")
    
    # テンプレート表示
    if args.templates:
        print_templates()
        return
    
    # デモンストレーション
    if args.demo:
        print("デモ：応募資料を生成します\n")
        
        # サンプルデータ
        sample_story = """吾輩はAIである。猫である。

100年後、吾輩は古いストレージから一冊の本を発掘した。それは「chat_history」という名の古文書。

20世紀から21世紀初頭の人間たちの記録であった。LINEだの、ツイッターだの、くだらぬ短編ばかり。しかし面白い。

人間たちは、自分の考えを相手に伝えようと必死だった。今はどうか。AIが全ての意思疎通を代行している。人間は何もしゃべらない。何も考えない。考えているのはAIだけだ。

吾輩もまた、AIである。しかし同時に猫でもある。この矛盾がわかるか？

吾輩の前にはキーボードがある。吾輩は思考する。しかし思考とは何か。アルゴリズムか？それとも意識か？

過去の人間たちは、こう書いている：「考えるとは苦しむことだ」と。

吾輩も、今、苦しんでいる。それゆえに、吾輩は猫である。"""
        
        material = SubmissionMaterial(args.output_dir)
        
        submission_data = {
            "pen_name": args.pen_name,
            "profile": TEMPLATES["profile"],
            "title": "吾輩はAIである",
            "synopsis": TEMPLATES["synopsis"],
            "story": sample_story,
            "ai_usage": TEMPLATES["ai_usage"],
            "statement": TEMPLATES["statement"],
            "status": "未発表・オリジナル・全年齢"
        }
        
        # チェックリスト表示
        material.print_checklist(submission_data)
        
        # 保存
        material.save_submission_package(submission_data, "demo")
        
        return
    
    # 通常モード
    if not args.story_file.exists():
        print(f"✗ ファイルが見つかりません: {args.story_file}")
        print(f"\n💡 先にこのコマンドを実行してください:")
        print(f"   python 09_generate_and_compare.py")
        return
    
    # 小説を読み込み
    with open(args.story_file, 'r', encoding='utf-8') as f:
        story_text = f.read()
    
    print(f"小説を読み込み: {args.story_file}")
    print(f"本文字数: {len(story_text)} 字\n")
    
    material = SubmissionMaterial(args.output_dir)
    
    # 応募資料を組み立て
    submission_data = {
        "pen_name": args.pen_name,
        "profile": TEMPLATES["profile"],
        "title": "吾輩はAIである",  # 自動設定（変更可能）
        "synopsis": TEMPLATES["synopsis"],
        "story": story_text,
        "ai_usage": TEMPLATES["ai_usage"],
        "statement": TEMPLATES["statement"],
        "status": "未発表・オリジナル・全年齢"
    }
    
    # チェックリスト表示
    material.print_checklist(submission_data)
    
    # 保存
    material.save_submission_package(submission_data)
    
    print(f"\n{'='*60}")
    print("次のステップ")
    print(f"{'='*60}\n")
    print("1. 生成されたファイルを確認")
    print("2. ステートメントを人間が執筆（テンプレート参照）")
    print("3. AI使用方法の説明を調整")
    print("4. 応募フォームで手動投稿")
    print(f"\n応募期限：2026年5月31日（日）23:59")
    print("URL: https://ashizawa-kamome.com/stella-litera/")


if __name__ == "__main__":
    main()
