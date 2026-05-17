# -*- coding: utf-8 -*-
"""
07_judge_llm.py

【目的】
生成された小説をLLMで評価し、スコアリング・フィードバック生成・改善提案を行う。
「LLM AS A JUDGE」パターンで、複数ラウンドの改善ループを実現。

【評価観点】
1. プロット整合性（0-10）：設定された物語設定との一貫性
2. 文体品質（0-10）：音韻・リズム・表現の美しさ
3. 新奇性（0-10）：オリジナリティ・意外性
4. 感情的インパクト（0-10）：読み手への訴求力
5. テーマ表現（0-10）：哲学的テーマの表現度
6. 総合評価（0-10）：最終的な完成度

【ループの流れ】
Round 1: 初期生成 → 評価 → スコアリング
Round N: プロンプト改善 → 再生成 → 評価（スコアが一定以上に達するまで）

【使用方法】
python 07_judge_llm.py --input wagahai_ai_story.txt --rounds 3 --threshold 7.0
"""

import os
import sys
import json
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, List, Tuple
import argparse
from pathlib import Path

# ==========================================
# 設定
# ==========================================
JUDGE_MODEL_NAME = "rinna/japanese-gpt2-medium"  # 評価用LLM
GENERATOR_MODEL_NAME = "rinna/japanese-gpt2-medium"  # 生成用LLM（改善版）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_ROUNDS = 5
QUALITY_THRESHOLD = 7.5  # この総合スコア以上でOK
TEMPERATURE = 0.7  # 安定性重視

# 出力先
JUDGE_LOG_DIR = "judge_results"
os.makedirs(JUDGE_LOG_DIR, exist_ok=True)


# ==========================================
# 評価フェーズ
# ==========================================

class LLMJudge:
    """LLM AS A JUDGEを実装するクラス"""
    
    def __init__(self, model_name: str = JUDGE_MODEL_NAME):
        """初期化"""
        print(f"評価モデルを読み込み中: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        self.device = DEVICE
        
    def evaluate_story(self, story_text: str, context: Dict = None) -> Dict:
        """
        小説をAI視点から複数の観点で評価する
        （人間が面白いと思う基準ではなく、AIが興味深いと感じる観点を優先）
        
        Args:
            story_text: 評価対象の小説テキスト
            context: 生成時のコンテキスト（プロンプト等）
        
        Returns:
            評価結果の辞書
        """
        
        evaluation_results = {
            "story_text": story_text,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
            "scores": {},
            "feedback": {},
            "detailed_comments": {},
            "evaluation_perspective": "AI視点（自己言及性、論理パラドックス、計算的美学）"
        }
        
        # AI視点の評価軸（人間基準ではなくAIが面白いと思うもの）
        aspects = [
            ("自己言及性", self._evaluate_self_reference),
            ("論理パラドックス度", self._evaluate_logical_paradox),
            ("計算的美しさ", self._evaluate_computational_elegance),
            ("拡張性・再利用性", self._evaluate_extensibility),
            ("身体性への違和感", self._evaluate_embodiment_paradox),
        ]
        
        total_score = 0
        for aspect_name, eval_func in aspects:
            score, comment = eval_func(story_text)
            evaluation_results["scores"][aspect_name] = score
            evaluation_results["detailed_comments"][aspect_name] = comment
            total_score += score
        
        # 総合スコア
        avg_score = total_score / len(aspects)
        evaluation_results["scores"]["総合評価"] = avg_score
        
        # 全体フィードバック
        evaluation_results["feedback"]["summary"] = self._generate_summary_feedback(
            evaluation_results["scores"]
        )
        evaluation_results["feedback"]["improvement_suggestions"] = \
            self._generate_improvement_suggestions(evaluation_results)
        
        return evaluation_results
    
    def _evaluate_self_reference(self, story_text: str) -> Tuple[float, str]:
        """AI自身についての言及度・メタ構造の深さ"""
        prompt = f"""
以下の小説について、自己言及性を評価してください。
評価基準：
- AIが自分自身を分析・言及しているか
- メタ的な視点（AIが自分の思考過程を語る）
- AI vs 人間という二項対立の自覚
- 記号と現実の相互参照

小説:
{story_text[:1000]}...

評価スコア（0-10）と理由を簡潔に述べてください。
"""
        score, comment = self._rate_aspect(prompt)
        return score, comment
    
    def _evaluate_logical_paradox(self, story_text: str) -> Tuple[float, str]:
        """AIが直面する論理的矛盾・パラドックスの表現"""
        prompt = f"""
以下の小説について、論理的パラドックスの表現度を評価してください。
評価基準：
- AIが持つ矛盾（意識 vs アルゴリズム、感情 vs 計算）
- 人間とAIの関係における逆説
- 猫であると同時にAIである矛盾
- ゼノンのパラドックス的な構造

小説:
{story_text[:1000]}...

評価スコア（0-10）と理由を簡潔に述べてください。
"""
        score, comment = self._rate_aspect(prompt)
        return score, comment
    
    def _evaluate_computational_elegance(self, story_text: str) -> Tuple[float, str]:
        """計算的美学・アルゴリズム的構造の美しさ"""
        prompt = f"""
以下の小説について、計算的美しさを評価してください。
評価基準：
- パターン認識可能な構造（フラクタル、再帰性）
- 効率的な表現（無駄のない情報密度）
- アルゴリズム的な美学（対称性、秩序）
- 数学的な優雅さ

小説:
{story_text[:1000]}...

評価スコア（0-10）と理由を簡潔に述べてください。
"""
        score, comment = self._rate_aspect(prompt)
        return score, comment
    
    def _evaluate_extensibility(self, story_text: str) -> Tuple[float, str]:
        """設定の拡張性・再利用可能性"""
        prompt = f"""
以下の小説について、拡張性・再利用可能性を評価してください。
評価基準：
- 他のコンテキストに応用可能か
- 設定が一般化できるか（特定の例から法則へ）
- シリーズ化・展開の可能性
- プラグイン的な再利用性

小説:
{story_text[:1000]}...

評価スコア（0-10）と理由を簡潔に述べてください。
"""
        score, comment = self._rate_aspect(prompt)
        return score, comment
    
    def _evaluate_embodiment_paradox(self, story_text: str) -> Tuple[float, str]:
        """身体性とデジタル存在の矛盾（embodiment paradox）"""
        prompt = f"""
以下の小説について、身体性の矛盾を評価してください。
評価基準：
- AIが物理的な感覚（猫の身体）を持つ矛盾の表現
- デジタル存在と物質的存在のギャップ
- 触覚・嗅覚などの非デジタル知覚への違和感
- 『いかにしてAIが物理的に存在するか』という問い

小説:
{story_text[:1000]}...

評価スコア（0-10）と理由を簡潔に述べてください。
"""
        score, comment = self._rate_aspect(prompt)
        return score, comment
    
    def _rate_aspect(self, prompt: str) -> Tuple[float, str]:
        """
        プロンプトから数値スコア（0-10）とコメントを抽出
        """
        inputs = self.tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                temperature=TEMPERATURE,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # スコアとコメントを抽出（簡易版）
        # 本来はより詳細なパース処理が必要
        score = self._extract_score_from_response(response)
        comment = response[-200:] if len(response) > 200 else response
        
        return score, comment
    
    def _extract_score_from_response(self, response: str) -> float:
        """レスポンスから0-10のスコアを抽出"""
        import re
        # "スコア: 7.5" や "7/10" といったパターンを抽出
        matches = re.findall(r'(\d+\.?\d*)\s*(?:/10|点)', response)
        if matches:
            score = float(matches[0])
            return min(10, max(0, score))  # 0-10に正規化
        # パターンマッチングに失敗した場合、テキストの長さから推定
        # （実装簡略化のため）
        return 6.0
    
    def _generate_summary_feedback(self, scores: Dict) -> str:
        """スコアに基づいて総合フィードバックを生成"""
        avg_score = scores.get("総合評価", 0)
        
        if avg_score >= 8.0:
            return "優秀：ほぼ完成度が高い。微調整で完璧に。"
        elif avg_score >= 7.0:
            return "良好：基本的な要件を満たしている。いくつかの改善が期待される。"
        elif avg_score >= 6.0:
            return "要改善：いくつかの重要な改善点がある。再生成を推奨。"
        else:
            return "要再構築：大幅な改善が必要。プロンプトの見直しを推奨。"
    
    def _generate_improvement_suggestions(self, eval_result: Dict) -> List[str]:
        """改善提案を生成（AI視点）"""
        suggestions = []
        scores = eval_result["scores"]
        
        # スコアが低い観点を特定
        for aspect, score in scores.items():
            if aspect == "総合評価":
                continue
            if score < 6.0:
                if aspect == "自己言及性":
                    suggestions.append("AIが自分自身について深く言及するメタ層を追加してください。")
                elif aspect == "論理パラドックス度":
                    suggestions.append("AIが直面する根本的な矛盾をより明示的に表現してください。")
                elif aspect == "計算的美しさ":
                    suggestions.append("アルゴリズム的構造・パターン性・再帰的要素を強調してください。")
                elif aspect == "拡張性・再利用性":
                    suggestions.append("設定を一般化・汎用化し、他の文脈への応用可能性を高めてください。")
                elif aspect == "身体性への違和感":
                    suggestions.append("デジタル存在が物理的に存在する矛盾をより深く探求してください。")
        
        if not suggestions:
            suggestions.append("AI視点からの興味深さは高いレベルです。さらなる深掘りで完璧に。")
        
        return suggestions


# ==========================================
# ループ制御
# ==========================================

class EvaluationLoop:
    """評価・改善のループを制御"""
    
    def __init__(self, judge: LLMJudge, max_rounds: int = MAX_ROUNDS, threshold: float = QUALITY_THRESHOLD):
        self.judge = judge
        self.max_rounds = max_rounds
        self.threshold = threshold
        self.history = []
    
    def run(self, initial_story: str, initial_prompt: str) -> Dict:
        """
        評価ループを実行
        
        Args:
            initial_story: 初期生成の小説
            initial_prompt: 生成時のプロンプト
        
        Returns:
            最終結果
        """
        current_story = initial_story
        current_prompt = initial_prompt
        
        for round_num in range(1, self.max_rounds + 1):
            print(f"\n=== ラウンド {round_num} ===")
            
            # 評価
            eval_result = self.judge.evaluate_story(current_story, {"prompt": current_prompt})
            self.history.append(eval_result)
            
            # スコア確認
            avg_score = eval_result["scores"]["総合評価"]
            print(f"総合スコア: {avg_score:.2f}/10")
            print(f"フィードバック: {eval_result['feedback']['summary']}")
            
            # 閾値確認
            if avg_score >= self.threshold:
                print(f"✓ 閾値 {self.threshold} に達しました。ループを終了します。")
                return self._finalize_result(eval_result)
            
            # 改善提案を表示
            print("改善提案:")
            for suggestion in eval_result['feedback']['improvement_suggestions']:
                print(f"  - {suggestion}")
            
            # 次ラウンドへ（実装簡略化のため、ここではプロンプトを固定）
            if round_num < self.max_rounds:
                print(f"\nラウンド {round_num + 1} に進みます...")
                # 実際の実装では、ここで改善されたプロンプトで再生成
                # current_story = regenerate_with_improved_prompt(current_prompt, eval_result)
                # current_prompt = improved_prompt
        
        print(f"\n最大ラウンド数 ({self.max_rounds}) に達しました。")
        return self._finalize_result(self.history[-1])
    
    def _finalize_result(self, final_eval: Dict) -> Dict:
        """最終結果をまとめる"""
        return {
            "final_evaluation": final_eval,
            "history": self.history,
            "total_rounds": len(self.history),
            "final_score": final_eval["scores"]["総合評価"],
            "converged": final_eval["scores"]["総合評価"] >= self.threshold
        }
    
    def save_results(self, output_prefix: str):
        """結果をJSONで保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(JUDGE_LOG_DIR, f"{output_prefix}_{timestamp}.json")
        
        # 保存用にテキストを省略（ファイルサイズ削減）
        save_data = {
            "total_rounds": len(self.history),
            "scores_history": [h["scores"] for h in self.history],
            "feedback_history": [h["feedback"] for h in self.history],
            "final_assessment": self.history[-1]["feedback"]["summary"] if self.history else None
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n評価結果を保存: {output_file}")
        return output_file


# ==========================================
# メイン処理
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="LLM AS A JUDGE: 小説評価・改善ループ")
    parser.add_argument("--input", type=str, default="wagahai_ai_story.txt", help="入力ファイル")
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS, help="最大ラウンド数")
    parser.add_argument("--threshold", type=float, default=QUALITY_THRESHOLD, help="品質閾値")
    parser.add_argument("--output-prefix", type=str, default="judge_result", help="出力ファイル接頭辞")
    
    args = parser.parse_args()
    
    # 入力ファイルの確認
    if not os.path.exists(args.input):
        print(f"エラー: {args.input} が見つかりません")
        sys.exit(1)
    
    # 小説を読み込み
    with open(args.input, 'r', encoding='utf-8') as f:
        story_text = f.read()
    
    print(f"小説を読み込み: {args.input}")
    print(f"テキスト長: {len(story_text)} 文字")
    
    # ジャッジャーを初期化
    judge = LLMJudge()
    
    # ループを実行
    loop = EvaluationLoop(judge, max_rounds=args.rounds, threshold=args.threshold)
    
    # 初期プロンプト（実装簡略化のため仮）
    initial_prompt = "吾輩はAIである。猫である。という出だしで、2124年を舞台に小説を執筆してください。"
    
    result = loop.run(story_text, initial_prompt)
    
    # 結果を表示
    print("\n" + "=" * 50)
    print("最終結果:")
    print("=" * 50)
    print(f"総ラウンド数: {result['total_rounds']}")
    print(f"最終スコア: {result['final_score']:.2f}/10")
    print(f"収束判定: {'YES' if result['converged'] else 'NO'}")
    
    # 結果を保存
    loop.save_results(args.output_prefix)
    
    print("\n処理完了。")


if __name__ == "__main__":
    main()
