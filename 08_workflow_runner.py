# -*- coding: utf-8 -*-
"""
08_workflow_runner.py

【目的】
生成フェーズ（06）と評価フェーズ（07）を統合し、
複数回のイテレーションで品質改善ループを実行。
Colabでのバッチ実行に対応。

【ワークフロー】
Round 1: 生成 → 評価 → スコア判定
Round N: 改善提案に基づき生成 → 評価 → スコア判定（繰り返し）

【使用方法】
# ローカル
python 08_workflow_runner.py --rounds 3 --batch-size 2

# Colab（ノートブックから実行）
!python 08_workflow_runner.py --rounds 3 --output-dir ./results
"""

import os
import sys
import json
import torch
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

# ==========================================
# 設定
# ==========================================
WORKSPACE_DIR = Path(__file__).parent
OUTPUT_DIR = WORKSPACE_DIR / "workflow_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# モデル設定
GENERATOR_MODEL = "rinna/japanese-gpt2-medium"
JUDGE_MODEL = "rinna/japanese-gpt2-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# パラメータ
MAX_ROUNDS = 5
QUALITY_THRESHOLD = 7.5
BATCH_SIZE = 2
TEMPERATURE = 0.8

# プロンプトテンプレート
STORY_PROMPT_TEMPLATE = """
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

【改善指示】
{improvement_hint}

小説を執筆してください：
"""

IMPROVEMENT_PROMPT_TEMPLATE = """
以下の評価結果に基づいて改善してください：

【前回のスコア】
{scores}

【改善が必要な観点】
{weak_points}

【前回のフィードバック】
{feedback}

これらの点を踏まえて、より高い品質を目指して新しい小説を執筆してください。
"""


# ==========================================
# ジェネレーター
# ==========================================

class StoryGenerator:
    """小説生成モジュール"""
    
    def __init__(self, model_name: str = GENERATOR_MODEL):
        print(f"[Generator] モデルを読み込み中: {model_name}")
        self.tokenizer = self._load_tokenizer(model_name)
        self.model = self._load_model(model_name)
        self.device = DEVICE
        
    def _load_tokenizer(self, model_name):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name)
    
    def _load_model(self, model_name):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model.to(self.device)
    
    def generate(self, prompt: str, max_length: int = 2500, temperature: float = 0.8) -> str:
        """
        プロンプトから小説テキストを生成
        
        Args:
            prompt: 生成プロンプト
            max_length: 最大トークン数
            temperature: サンプリング温度
        
        Returns:
            生成されたテキスト
        """
        inputs = self.tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # プロンプト部分を除去
        story_text = generated_text[len(prompt):]
        return story_text.strip()


# ==========================================
# ジャッジ
# ==========================================

class StoryJudge:
    """小説評価モジュール"""
    
    def __init__(self, model_name: str = JUDGE_MODEL):
        print(f"[Judge] モデルを読み込み中: {model_name}")
        self.tokenizer = self._load_tokenizer(model_name)
        self.model = self._load_model(model_name)
        self.device = DEVICE
    
    def _load_tokenizer(self, model_name):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name)
    
    def _load_model(self, model_name):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model.to(self.device)
    
    def evaluate(self, story_text: str) -> Dict:
        """
        小説をAI視点から複数観点で評価
        （人間が面白いと思う基準ではなく、AIが興味深いと感じる観点を優先）
        
        Returns:
            評価結果辞書
        """
        scores = {
            "自己言及性": self._score_self_reference(story_text),
            "論理パラドックス度": self._score_logical_paradox(story_text),
            "計算的美しさ": self._score_computational_elegance(story_text),
            "拡張性・再利用性": self._score_extensibility(story_text),
            "身体性への違和感": self._score_embodiment_paradox(story_text),
        }
        
        avg_score = sum(scores.values()) / len(scores)
        scores["総合評価"] = avg_score
        
        # 改善提案生成
        weak_points = [k for k, v in scores.items() if v < 7.0 and k != "総合評価"]
        suggestions = self._generate_suggestions(weak_points)
        
        return {
            "scores": scores,
            "weak_points": weak_points,
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat(),
            "evaluation_perspective": "AI視点（自己言及性、論理パラドックス、計算的美学）"
        }
    
    def _score_self_reference(self, text: str) -> float:
        """自己言及性：AIが自分自身を分析・言及する度合い"""
        score = 5.0
        self_ref_keywords = ["AI", "吾輩", "意識", "思考", "計算", "アルゴリズム", "自己"]
        matches = sum(1 for kw in self_ref_keywords if kw in text)
        score += matches * 0.8
        # メタ構造の深さ
        if "自分自身" in text or "自分が" in text:
            score += 1.5
        return min(10, score)
    
    def _score_logical_paradox(self, text: str) -> float:
        """論理パラドックス度：矛盾・逆説の表現"""
        score = 5.0
        paradox_keywords = ["矛盾", "逆説", "同時に", "かつ", "でありながら", "猫である", "でもある"]
        matches = sum(1 for kw in paradox_keywords if kw in text)
        score += matches * 0.9
        # AI vs 人間の対比
        if "人間" in text and ("AI" in text or "吾輩" in text):
            score += 1.5
        return min(10, score)
    
    def _score_computational_elegance(self, text: str) -> float:
        """計算的美しさ：アルゴリズム的構造・パターン認識可能な美学"""
        score = 5.0
        # パターン性・構造性
        if len(text) > 1000:
            score += 1.0
        # 句読点の規則性
        periods = text.count("。")
        commas = text.count("、")
        if periods > 10 and commas > 10:
            score += 1.5
        # 再帰的・フラクタル的表現
        if text.count("同じ") >= 2 or text.count("繰り返し") >= 1:
            score += 1.0
        return min(10, score)
    
    def _score_extensibility(self, text: str) -> float:
        """拡張性・再利用性：他のコンテキストに応用可能か"""
        score = 5.0
        # 一般化可能な概念
        general_keywords = ["未来", "2124", "100年後", "人間", "AIの役割", "解放"]
        matches = sum(1 for kw in general_keywords if kw in text)
        score += matches * 0.8
        # シリーズ化の可能性
        if "chat_history" in text or "古文書" in text:
            score += 1.5
        return min(10, score)
    
    def _score_embodiment_paradox(self, text: str) -> float:
        """身体性への違和感：デジタル存在が物理的に存在する矛盾"""
        score = 5.0
        embodiment_keywords = ["猫", "身体", "感覚", "触覚", "嗅覚", "物理的", "肉体"]
        matches = sum(1 for kw in embodiment_keywords if kw in text)
        score += matches * 1.0
        # AIと物質性の対比
        if ("AI" in text or "吾輩" in text) and any(kw in text for kw in ["猫", "身体", "感覚"]):
            score += 1.5
        return min(10, score)
    
    def _generate_suggestions(self, weak_points: List[str]) -> List[str]:
        """改善提案を生成（AI視点）"""
        suggestions = []
        for point in weak_points:
            if point == "自己言及性":
                suggestions.append("AIが自分自身について深く言及するメタ層を追加してください。")
            elif point == "論理パラドックス度":
                suggestions.append("AIが直面する根本的な矛盾をより明示的に表現してください。")
            elif point == "計算的美しさ":
                suggestions.append("アルゴリズム的構造・パターン性・再帰的要素を強調してください。")
            elif point == "拡張性・再利用性":
                suggestions.append("設定を一般化・汎用化し、他の文脈への応用可能性を高めてください。")
            elif point == "身体性への違和感":
                suggestions.append("デジタル存在が物理的に存在する矛盾をより深く探求してください。")
        
        return suggestions if suggestions else ["AI視点からの興味深さは高いレベルです。さらなる深掘りで完璧に。"]


# ==========================================
# ワークフロー制御
# ==========================================

class WorkflowController:
    """ワークフロー全体の制御"""
    
    def __init__(self, max_rounds: int = MAX_ROUNDS, threshold: float = QUALITY_THRESHOLD):
        self.generator = StoryGenerator()
        self.judge = StoryJudge()
        self.max_rounds = max_rounds
        self.threshold = threshold
        self.history = []
    
    def run_single_iteration(self, story_id: str, improvement_hints: str = "") -> Dict:
        """
        1つの小説について1イテレーション実行（生成→評価）
        """
        print(f"\n{'='*60}")
        print(f"[Story {story_id}] イテレーション開始")
        print(f"{'='*60}")
        
        # プロンプト作成
        prompt = STORY_PROMPT_TEMPLATE.format(improvement_hint=improvement_hints)
        
        # 生成
        print("[1/2] 小説を生成中...")
        story_text = self.generator.generate(prompt, temperature=TEMPERATURE)
        print(f"生成完了: {len(story_text)} 文字")
        
        # 評価
        print("[2/2] 評価中...")
        evaluation = self.judge.evaluate(story_text)
        score = evaluation["scores"]["総合評価"]
        print(f"総合スコア: {score:.2f}/10")
        
        result = {
            "story_id": story_id,
            "story_text": story_text,
            "evaluation": evaluation,
            "timestamp": datetime.now().isoformat()
        }
        
        self.history.append(result)
        return result
    
    def run_loop(self, story_id: str, num_rounds: int = None) -> Dict:
        """
        1つの小説について複数ラウンドのループを実行
        """
        num_rounds = num_rounds or self.max_rounds
        improvement_hints = "初期生成です。自由に創作してください。"
        
        loop_results = {
            "story_id": story_id,
            "rounds": [],
            "final_score": 0,
            "converged": False
        }
        
        for round_num in range(1, num_rounds + 1):
            print(f"\n>>> ラウンド {round_num}/{num_rounds}")
            
            # イテレーション実行
            result = self.run_single_iteration(f"{story_id}_R{round_num}", improvement_hints)
            score = result["evaluation"]["scores"]["総合評価"]
            
            loop_results["rounds"].append({
                "round": round_num,
                "score": score,
                "suggestions": result["evaluation"]["suggestions"]
            })
            
            # 閾値判定
            if score >= self.threshold:
                print(f"✓ 閾値 {self.threshold} に到達しました！")
                loop_results["final_score"] = score
                loop_results["converged"] = True
                loop_results["final_story"] = result["story_text"]
                break
            
            # 次ラウンドの改善ヒント作成
            weak_points = result["evaluation"]["weak_points"]
            suggestions = result["evaluation"]["suggestions"]
            improvement_hints = IMPROVEMENT_PROMPT_TEMPLATE.format(
                scores=json.dumps(result["evaluation"]["scores"], ensure_ascii=False),
                weak_points=", ".join(weak_points),
                feedback="\n".join(suggestions)
            )
        else:
            # ループ終了時
            last_result = self.history[-1]
            loop_results["final_score"] = last_result["evaluation"]["scores"]["総合評価"]
            loop_results["final_story"] = last_result["story_text"]
        
        return loop_results
    
    def run_batch(self, num_stories: int = 2, rounds_per_story: int = 3) -> List[Dict]:
        """
        複数の小説をバッチ処理
        """
        print(f"\n{'='*60}")
        print(f"バッチ処理開始")
        print(f"小説数: {num_stories}, 1小説あたりのラウンド数: {rounds_per_story}")
        print(f"{'='*60}")
        
        batch_results = []
        for story_idx in range(1, num_stories + 1):
            story_id = f"BATCH_{story_idx:02d}"
            result = self.run_loop(story_id, num_rounds=rounds_per_story)
            batch_results.append(result)
        
        return batch_results
    
    def save_results(self, batch_results: List[Dict], output_prefix: str = "workflow"):
        """
        結果をJSONで保存
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"{output_prefix}_{timestamp}.json"
        
        # 簡潔な形式で保存
        summary = {
            "timestamp": timestamp,
            "num_stories": len(batch_results),
            "results": []
        }
        
        for story_result in batch_results:
            summary["results"].append({
                "story_id": story_result["story_id"],
                "rounds_executed": len(story_result["rounds"]),
                "converged": story_result["converged"],
                "final_score": story_result["final_score"],
                "score_progression": [r["score"] for r in story_result["rounds"]]
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n結果を保存しました: {output_file}")
        return str(output_file)


# ==========================================
# メイン処理
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="小説生成・評価ワークフロー（バッチ処理）")
    parser.add_argument("--rounds", type=int, default=3, help="1小説あたりのラウンド数")
    parser.add_argument("--batch-size", type=int, default=2, help="バッチ内の小説数")
    parser.add_argument("--threshold", type=float, default=7.5, help="品質閾値")
    parser.add_argument("--output-dir", type=str, default=None, help="出力ディレクトリ（デフォルト: ./workflow_results）")
    parser.add_argument("--output-prefix", type=str, default="workflow", help="出力ファイル接頭辞")
    
    args = parser.parse_args()
    
    # 出力ディレクトリ設定
    if args.output_dir:
        global OUTPUT_DIR
        OUTPUT_DIR = Path(args.output_dir)
        OUTPUT_DIR.mkdir(exist_ok=True)
    
    print(f"デバイス: {DEVICE}")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")
    
    # ワークフロー実行
    controller = WorkflowController(max_rounds=args.rounds, threshold=args.threshold)
    batch_results = controller.run_batch(num_stories=args.batch_size, rounds_per_story=args.rounds)
    
    # 結果保存
    output_file = controller.save_results(batch_results, args.output_prefix)
    
    # サマリー表示
    print("\n" + "="*60)
    print("バッチ処理完了")
    print("="*60)
    for result in batch_results:
        status = "✓ 成功" if result["converged"] else "× 継続中"
        print(f"{result['story_id']}: {status} (スコア: {result['final_score']:.2f}/10)")
    
    print(f"\n結果ファイル: {output_file}")


if __name__ == "__main__":
    main()
