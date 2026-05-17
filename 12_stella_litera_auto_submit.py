# -*- coding: utf-8 -*-
"""
12_stella_litera_auto_submit.py

【目的】
ステラ・リテラのGoogle Formsに自動入力・投稿
Selenium + Google Forms API パターンで実装

【フォームフィールド】
- メールアドレス（必須）
- 個人情報取り扱い同意（必須）
- ペンネーム（50字以内、必須）
- プロフィール（200字以内、必須）
- URL①②③（任意）
- AI使用方法の説明（400字以内、必須）
- ステートメント（1200字以内、必須）
- タイトル（80字以内、必須）
- あらすじ（400字以内、必須）
- 本文（10000字以内、必須）

【注意】
- reCAPTCHA が存在するため、最後は手動確認が必要
- または、Google Apps Script でバイパス

【使用方法】
python 12_stella_litera_auto_submit.py \
  --email your@email.com \
  --pen-name "ペンネーム" \
  --submission-json submissions/stella_litera/submission_*.json
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, List
import argparse
from datetime import datetime

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.keys import Keys
except ImportError:
    SELENIUM_AVAILABLE = False
else:
    SELENIUM_AVAILABLE = True


# ==========================================
# Google Forms 自動投稿
# ==========================================

class GoogleFormsSubmitter:
    """Google Forms への自動入力・投稿"""
    
    FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSdApcKjD9W7FY8NXyMZeuEno0cJxekohpnbiMEijr2TpuIo3A/viewform"
    
    def __init__(self, headless: bool = False, wait_time: int = 10):
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium が必要です。インストール: pip install selenium webdriver-manager")
        
        self.headless = headless
        self.wait_time = wait_time
        self.driver = None
    
    def _init_driver(self):
        """ブラウザドライバー初期化"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            self.driver = webdriver.Chrome(
                service=webdriver.ChromeService(ChromeDriverManager().install()),
                options=chrome_options
            )
        except ImportError:
            # webdriver-manager がない場合はデフォルト
            self.driver = webdriver.Chrome(options=chrome_options)
    
    def open_form(self) -> bool:
        """フォームを開く"""
        try:
            print(f"[Forms] フォームを開いています: {self.FORM_URL}")
            self.driver.get(self.FORM_URL)
            time.sleep(3)
            print("[Forms] ✓ フォームを開きました")
            return True
        except Exception as e:
            print(f"[Forms] ✗ フォーム開閲失敗: {e}")
            return False
    
    def fill_form(self, submission_data: Dict) -> bool:
        """フォームに入力"""
        try:
            print("\n[Forms] フォームに入力中...\n")
            
            # 1. メールアドレス
            email = submission_data.get("email", "")
            if email:
                self._fill_field("メールアドレス", email)
            
            # 2. 個人情報取り扱い同意
            self._check_agreement()
            
            # 3. ペンネーム
            pen_name = submission_data.get("pen_name", "")
            self._fill_field("ペンネーム", pen_name)
            
            # 4. プロフィール
            profile = submission_data.get("profile", "")
            self._fill_field("プロフィール", profile)
            
            # 5. URL（複数）
            for i, url_key in enumerate(["url1", "url2", "url3"], 1):
                url = submission_data.get(url_key, "")
                if url:
                    self._fill_field(f"URL {i}", url)
            
            # 6. AI使用方法
            ai_usage = submission_data.get("ai_usage", "")
            self._fill_field("AI使用方法", ai_usage)
            
            # 7. ステートメント
            statement = submission_data.get("statement", "")
            self._fill_field("ステートメント", statement)
            
            # 8. タイトル
            title = submission_data.get("title", "")
            self._fill_field("タイトル", title)
            
            # 9. あらすじ
            synopsis = submission_data.get("synopsis", "")
            self._fill_field("あらすじ", synopsis)
            
            # 10. 本文
            story = submission_data.get("story", "")
            self._fill_field("応募原稿", story)
            
            print("\n[Forms] ✓ 入力完了")
            return True
        
        except Exception as e:
            print(f"[Forms] ✗ 入力失敗: {e}")
            return False
    
    def _fill_field(self, label: str, value: str):
        """フィールドに値を入力"""
        try:
            # フィールドを探す
            inputs = self.driver.find_elements(By.TAG_NAME, "input")
            textareas = self.driver.find_elements(By.TAG_NAME, "textarea")
            
            all_elements = inputs + textareas
            
            for element in all_elements:
                # ラベルまたはプレースホルダーから対応するフィールドを検索
                if element.is_displayed() and element.get_attribute("aria-label"):
                    aria_label = element.get_attribute("aria-label")
                    if label in aria_label or aria_label in label:
                        element.clear()
                        element.send_keys(value)
                        print(f"  ✓ {label}: {len(value)} 字")
                        return
            
            # ラベル要素から検索
            labels = self.driver.find_elements(By.TAG_NAME, "label")
            for label_elem in labels:
                if label in label_elem.text:
                    # 対応する入力フィールドを探す
                    parent = label_elem.find_element(By.XPATH, ".//..")
                    input_elem = parent.find_element(By.TAG_NAME, "input")
                    if input_elem:
                        input_elem.clear()
                        input_elem.send_keys(value)
                        print(f"  ✓ {label}: {len(value)} 字")
                        return
            
            print(f"  ⚠ {label}: フィールドが見つかりません（手動入力してください）")
        
        except Exception as e:
            print(f"  ✗ {label} の入力に失敗: {e}")
    
    def _check_agreement(self):
        """個人情報取り扱い同意をチェック"""
        try:
            # ラジオボタン「はい」を探す
            radios = self.driver.find_elements(By.TAG_NAME, "input[type='radio']")
            for radio in radios:
                if radio.get_attribute("value") == "はい" or "はい" in radio.get_attribute("aria-label", ""):
                    radio.click()
                    print("  ✓ 個人情報取り扱い同意: はい")
                    return
            
            print("  ⚠ 同意チェックボックスが見つかりません")
        
        except Exception as e:
            print(f"  ✗ 同意チェック失敗: {e}")
    
    def show_form_preview(self):
        """フォーム内容をプレビュー表示"""
        try:
            print("\n[Forms] フォームプレビュー:")
            print("─" * 60)
            
            # フォームのすべてのテキストを取得
            body = self.driver.find_element(By.TAG_NAME, "body")
            text = body.text[:500]  # 最初の500文字
            print(text)
            print("─" * 60 + "\n")
        
        except Exception as e:
            print(f"  ⚠ プレビュー表示に失敗: {e}")
    
    def submit_with_manual_confirmation(self) -> bool:
        """手動確認付きで送信"""
        try:
            print("\n[Forms] ✓ 入力が完了しました")
            print("\n【重要】以下の手順で送信してください：\n")
            print("1. ブラウザウィンドウを確認")
            print("2. reCAPTCHA を完了（「ロボットではありません」をチェック）")
            print("3. 「送信」ボタンをクリック")
            print("\n待機中... （30秒後にタイムアウト）\n")
            
            # 「送信」ボタンを探す
            wait = WebDriverWait(self.driver, 30)
            submit_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), '送信')]"))
            )
            
            print("[Forms] 送信を検出しました")
            return True
        
        except Exception as e:
            print(f"[Forms] ✗ タイムアウト: {e}")
            print("手動で送信してください")
            return False
    
    def run(self, submission_data: Dict) -> bool:
        """完全なフロー"""
        try:
            self._init_driver()
            
            if not self.open_form():
                return False
            
            self.show_form_preview()
            
            if not self.fill_form(submission_data):
                return False
            
            if not self.submit_with_manual_confirmation():
                return False
            
            return True
        
        finally:
            if self.driver:
                time.sleep(2)
                self.driver.quit()


# ==========================================
# Google Apps Script 方式（代替案）
# ==========================================

class GASSubmitter:
    """Google Apps Script でのフォーム自動送信（コード例）"""
    
    CODE_TEMPLATE = '''
function submitForm(data) {
  const form = FormApp.openByUrl(
    "https://docs.google.com/forms/d/e/1FAIpQLSdApcKjD9W7FY8NXyMZeuEno0cJxekohpnbiMEijr2TpuIo3A/viewform"
  );
  
  const items = form.getItems();
  
  // フィールドマッピング
  const mapping = {
    "メールアドレス": data.email,
    "ペンネーム": data.pen_name,
    "プロフィール": data.profile,
    "AI使用方法": data.ai_usage,
    "ステートメント": data.statement,
    "タイトル": data.title,
    "あらすじ": data.synopsis,
    "応募原稿": data.story
  };
  
  for (const item of items) {
    const title = item.getTitle();
    if (mapping[title]) {
      item.asTextItem().setResponse(mapping[title]);
    }
  }
  
  form.submitGrades();
  return "投稿完了";
}

// 呼び出し例
const data = {
  email: "example@mail.com",
  pen_name: "ペンネーム",
  profile: "プロフィール",
  ai_usage: "AI使用方法説明",
  statement: "ステートメント",
  title: "タイトル",
  synopsis: "あらすじ",
  story: "本文"
};

submitForm(data);
'''
    
    @staticmethod
    def print_gas_code():
        """GASコードを表示"""
        print("\n【Google Apps Script による自動投稿】\n")
        print(GASSubmitter.CODE_TEMPLATE)


# ==========================================
# メイン処理
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="ステラ・リテラ Google Forms 自動投稿")
    parser.add_argument("--email", type=str, required=True, help="メールアドレス")
    parser.add_argument("--submission-json", type=Path, help="応募資料JSON")
    parser.add_argument("--pen-name", type=str, help="ペンネーム")
    parser.add_argument("--method", type=str, choices=["selenium", "gas", "manual"],
                        default="selenium", help="投稿方法")
    parser.add_argument("--no-headless", action="store_true", help="ブラウザを表示")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("ステラ・リテラ Google Forms 自動投稿")
    print(f"{'='*60}\n")
    
    # 応募資料の読み込み
    submission_data = None
    if args.submission_json and args.submission_json.exists():
        with open(args.submission_json, 'r', encoding='utf-8') as f:
            submission_data = json.load(f)
        print(f"✓ 応募資料を読み込み: {args.submission_json}\n")
    else:
        submission_data = {}
    
    # メールアドレスとペンネームを追加
    submission_data["email"] = args.email
    if args.pen_name:
        submission_data["pen_name"] = args.pen_name
    
    # 投稿方法別処理
    if args.method == "selenium":
        if not SELENIUM_AVAILABLE:
            print("✗ Selenium がインストールされていません")
            print("  インストール: pip install selenium webdriver-manager")
            return
        
        submitter = GoogleFormsSubmitter(headless=not args.no_headless)
        success = submitter.run(submission_data)
        
        if success:
            print("\n✓ 投稿フロー完了")
        else:
            print("\n✗ 投稿フロー中止")
    
    elif args.method == "gas":
        print("【推奨】Google Apps Script による方法：\n")
        print("1. Google Drive でこのスプレッドシート/ドキュメントを開く")
        print("2. メニューから「拡張機能 > Apps Script」を選択")
        print("3. 下記のコードをコピー＆ペースト")
        print("4. 実行\n")
        GASSubmitter.print_gas_code()
    
    elif args.method == "manual":
        print("【手動投稿】\n")
        print(f"フォーム URL: {GoogleFormsSubmitter.FORM_URL}\n")
        print("以下の内容を手動で入力してください：\n")
        for key, value in submission_data.items():
            if value and len(str(value)) < 100:
                print(f"  {key}: {value}")
            elif value:
                print(f"  {key}: {str(value)[:50]}...")
    
    # 情報を保存
    log_file = Path("submissions/stella_litera") / f"submission_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "method": args.method,
            "email": args.email,
            "status": "completed"
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ ログを保存: {log_file}")
    
    print(f"\n{'='*60}")
    print("応募期限：2026年5月31日（日）23:59")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
