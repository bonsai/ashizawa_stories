# -*- coding: utf-8 -*-
"""
10_auto_submit.py

【目的】
生成した小説をプラットフォームに自動投稿
対応プラットフォーム：
- カクヨム (Kakuyomu)
- 小説家になろう (Syosetu)
- note

【注意】
- 利用規約を確認の上、自動投稿してください
- 認証情報は環境変数から読み込みます
- ファイル投稿とメタデータ投稿に分かれます

【使用方法】
# 認証情報の設定
export KAKUYOMU_EMAIL="your_email@example.com"
export KAKUYOMU_PASSWORD="your_password"

python 10_auto_submit.py --platform kakuyomu --story-file best_story.txt
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import argparse

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
except ImportError:
    SELENIUM_AVAILABLE = False
else:
    SELENIUM_AVAILABLE = True


# ==========================================
# Kakuyomu自動投稿
# ==========================================

class KakuyomuSubmitter:
    """カクヨム投稿自動化"""
    
    LOGIN_URL = "https://kakuyomu.jp/login"
    WRITE_URL = "https://kakuyomu.jp/works/new"
    
    def __init__(self, email: str, password: str, headless: bool = True):
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium が必要です。インストール: pip install selenium")
        
        self.email = email
        self.password = password
        self.headless = headless
        self.driver = None
    
    def _init_driver(self):
        """ブラウザドライバー初期化"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        self.driver = webdriver.Chrome(options=chrome_options)
    
    def login(self) -> bool:
        """ログイン処理"""
        try:
            print("[Kakuyomu] ログインページに移動...")
            self.driver.get(self.LOGIN_URL)
            
            # メールアドレス入力
            email_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "email"))
            )
            email_field.send_keys(self.email)
            
            # パスワード入力
            password_field = self.driver.find_element(By.NAME, "password")
            password_field.send_keys(self.password)
            
            # ログインボタン
            login_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'ログイン')]")
            login_button.click()
            
            # ログイン完了待機
            time.sleep(3)
            print("[Kakuyomu] ✓ ログイン成功")
            return True
        
        except Exception as e:
            print(f"[Kakuyomu] ✗ ログイン失敗: {e}")
            return False
    
    def submit_work(self, title: str, content: str, description: str = "") -> bool:
        """作品投稿"""
        try:
            print("[Kakuyomu] 新規作品投稿ページに移動...")
            self.driver.get(self.WRITE_URL)
            
            # タイトル入力
            title_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "title"))
            )
            title_field.send_keys(title)
            
            # 本文入力
            content_field = self.driver.find_element(By.NAME, "content")
            content_field.send_keys(content)
            
            # 説明入力（あれば）
            if description:
                desc_field = self.driver.find_element(By.NAME, "description")
                desc_field.send_keys(description)
            
            # 投稿ボタン
            submit_button = self.driver.find_element(By.XPATH, "//button[contains(text(), '投稿')]")
            submit_button.click()
            
            time.sleep(2)
            print("[Kakuyomu] ✓ 投稿成功")
            return True
        
        except Exception as e:
            print(f"[Kakuyomu] ✗ 投稿失敗: {e}")
            return False
    
    def submit(self, title: str, content: str, description: str = "") -> bool:
        """完全な投稿フロー"""
        try:
            self._init_driver()
            
            if not self.login():
                return False
            
            if not self.submit_work(title, content, description):
                return False
            
            return True
        
        finally:
            if self.driver:
                self.driver.quit()


# ==========================================
# 小説家になろう自動投稿（JSON API）
# ==========================================

class SyosetuSubmitter:
    """小説家になろう投稿自動化（シンプル版）"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.syosetu.com/submit"
    
    def submit(self, title: str, content: str, description: str = "") -> bool:
        """
        API経由での投稿
        
        注: 実装はプラットフォームのAPI仕様に依存
        """
        try:
            import requests
            
            payload = {
                "api_key": self.api_key,
                "title": title,
                "content": content,
                "description": description,
                "tags": ["AI生成", "短編"],
                "age_limit": 0  # 全年齢
            }
            
            # プラットフォームの実装に合わせて調整が必要
            print("[Syosetu] API投稿機能は未実装です")
            print("        利用規約を確認の上、手動投稿してください")
            return False
        
        except Exception as e:
            print(f"[Syosetu] ✗ 投稿失敗: {e}")
            return False


# ==========================================
# Note自動投稿（APIベース）
# ==========================================

class NoteSubmitter:
    """Note投稿自動化"""
    
    API_URL = "https://note.com/api/v2/notes"
    
    def __init__(self, auth_token: str):
        self.auth_token = auth_token
    
    def submit(self, title: str, content: str, description: str = "") -> bool:
        """
        Note API経由での投稿
        
        注: 実装はプラットフォームのAPI仕様に依存
        """
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "title": title,
                "body": content,
                "preview": description or content[:100],
                "price": 0,  # 無料公開
                "is_publish": True
            }
            
            print("[Note] API投稿機能は未実装です")
            print("      利用規約を確認の上、手動投稿してください")
            return False
        
        except Exception as e:
            print(f"[Note] ✗ 投稿失敗: {e}")
            return False


# ==========================================
# フォーム自動化管理
# ==========================================

class SubmissionManager:
    """投稿管理"""
    
    PLATFORMS = ["kakuyomu", "syosetu", "note", "browser"]
    
    def __init__(self):
        self.credentials = self._load_credentials()
    
    def _load_credentials(self) -> Dict[str, str]:
        """環境変数から認証情報を読み込む"""
        return {
            "kakuyomu_email": os.getenv("KAKUYOMU_EMAIL"),
            "kakuyomu_password": os.getenv("KAKUYOMU_PASSWORD"),
            "syosetu_api_key": os.getenv("SYOSETU_API_KEY"),
            "note_auth_token": os.getenv("NOTE_AUTH_TOKEN"),
        }
    
    def submit(self, platform: str, story_file: Path, title: str = "吾輩はAIである") -> bool:
        """指定プラットフォームに投稿"""
        
        # ファイル読み込み
        if not story_file.exists():
            print(f"✗ ファイルが見つかりません: {story_file}")
            return False
        
        with open(story_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"\n{'='*60}")
        print(f"投稿準備")
        print(f"{'='*60}")
        print(f"プラットフォーム: {platform}")
        print(f"タイトル: {title}")
        print(f"本文長: {len(content)} 文字\n")
        
        if platform == "kakuyomu":
            return self._submit_kakuyomu(title, content)
        elif platform == "syosetu":
            return self._submit_syosetu(title, content)
        elif platform == "note":
            return self._submit_note(title, content)
        elif platform == "browser":
            return self._open_browser(platform, title, content)
        else:
            print(f"✗ 不明なプラットフォーム: {platform}")
            return False
    
    def _submit_kakuyomu(self, title: str, content: str) -> bool:
        """カクヨムに投稿"""
        email = self.credentials.get("kakuyomu_email")
        password = self.credentials.get("kakuyomu_password")
        
        if not email or not password:
            print("✗ KAKUYOMU_EMAIL と KAKUYOMU_PASSWORD を環境変数に設定してください")
            return False
        
        if not SELENIUM_AVAILABLE:
            print("✗ Selenium がインストールされていません")
            print("  インストール: pip install selenium webdriver-manager")
            return False
        
        submitter = KakuyomuSubmitter(email, password)
        return submitter.submit(title, content)
    
    def _submit_syosetu(self, title: str, content: str) -> bool:
        """小説家になろうに投稿"""
        api_key = self.credentials.get("syosetu_api_key")
        
        if not api_key:
            print("✗ SYOSETU_API_KEY を環境変数に設定してください")
            return False
        
        submitter = SyosetuSubmitter(api_key)
        return submitter.submit(title, content)
    
    def _submit_note(self, title: str, content: str) -> bool:
        """Noteに投稿"""
        auth_token = self.credentials.get("note_auth_token")
        
        if not auth_token:
            print("✗ NOTE_AUTH_TOKEN を環境変数に設定してください")
            return False
        
        submitter = NoteSubmitter(auth_token)
        return submitter.submit(title, content)
    
    def _open_browser(self, platform: str, title: str, content: str) -> bool:
        """ブラウザで投稿ページを開く（手動投稿用）"""
        import webbrowser
        
        print(f"\n⚠ {platform} への自動投稿は非対応です")
        print(f"  ブラウザを開いて手動投稿してください\n")
        
        # 各プラットフォームのURL
        urls = {
            "kakuyomu": "https://kakuyomu.jp/works/new",
            "syosetu": "https://www.syosetu.com/userwork/create/",
            "note": "https://note.com/new",
        }
        
        if platform in urls:
            print(f"→ ブラウザを開きます: {urls[platform]}")
            webbrowser.open(urls[platform])
            
            # クリップボードにコピー
            try:
                import subprocess
                process = subprocess.Popen(['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE)
                process.communicate(content.encode('utf-8'))
                print("\n✓ 本文をクリップボードにコピーしました")
                print("  ブラウザで Ctrl+V で貼り付けしてください")
            except:
                print("\n💡 コピー方法: 本文をコピーして投稿フォームに貼り付けてください")
            
            print(f"\n投稿情報:")
            print(f"  タイトル: {title}")
            print(f"  本文長: {len(content)} 文字")
            
            return True
        
        return False


# ==========================================
# メイン処理
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="AI生成小説の自動投稿")
    parser.add_argument("--platform", type=str, default="browser", 
                        choices=["kakuyomu", "syosetu", "note", "browser"],
                        help="投稿先プラットフォーム")
    parser.add_argument("--story-file", type=Path, default="results/best_story.txt",
                        help="投稿する小説ファイル")
    parser.add_argument("--title", type=str, default="吾輩はAIである",
                        help="投稿時のタイトル")
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print("AI小説自動投稿システム")
    print(f"{'='*60}\n")
    
    manager = SubmissionManager()
    success = manager.submit(args.platform, args.story_file, args.title)
    
    if success:
        print("\n✓ 投稿処理が完了しました")
    else:
        print("\n✗ 投稿処理に失敗しました")
        print("\n💡 手動投稿について:")
        print("  1. best_story.html をブラウザで開く")
        print("  2. 投稿先プラットフォームにアクセス")
        print("  3. 投稿フォームに内容をコピー&ペースト")


if __name__ == "__main__":
    main()
