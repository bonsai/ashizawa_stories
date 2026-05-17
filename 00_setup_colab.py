"""
00_setup_colab.py
Google Colab/Kaggle 環境のセットアップスクリプト

機能:
1. Google Drive のマウント
2. 空き容量の確認
3. 必要なライブラリのインストール確認と設置
4. ワークスペースの準備
"""

import os
import subprocess
import sys
import shutil

def check_drive_space():
    """Google Drive の空き容量を確認する"""
    print("=== Google Drive の空き容量確認 ===")
    try:
        # Drive がマウントされているか確認
        drive_path = "/content/drive"
        if not os.path.exists(drive_path):
            print("Google Drive がマウントされていません。以下のコードをセルで実行してください:")
            print("from google.colab import drive")
            print("drive.mount('/content/drive')")
            return False
        
        # 空き容量を取得 (df コマンド使用)
        result = subprocess.run(['df', '-h', drive_path], capture_output=True, text=True)
        print(result.stdout)
        
        # 数値解析
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 4:
                available = parts[3]
                print(f"利用可能な容量: {available}")
                
                # 簡易的なチェック (5GB以上あるか)
                # 実際には単位(G, M等)をパースする必要があるが、ここでは表示のみ
                return True
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return False
    return False

def install_dependencies():
    """必要なライブラリをインストールする"""
    print("\n=== 依存ライブラリのインストール ===")
    
    requirements = [
        "pandas",
        "numpy",
        "transformers",
        "torch",
        "accelerate",
        "datasets"
    ]
    
    for lib in requirements:
        try:
            # 既にインストールされているか確認
            __import__(lib.replace('-', '_'))
            print(f"✓ {lib} は既にインストールされています")
        except ImportError:
            print(f"Installing {lib}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "-q"])
            print(f"✓ {lib} をインストールしました")

def setup_workspace():
    """作業用ディレクトリの設定"""
    print("\n=== ワークスペースの設定 ===")
    
    # Drive 上にプロジェクトディレクトリを作成
    project_dir = "/content/drive/MyDrive/twnovel_ml_project"
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
        print(f"プロジェクトディレクトリを作成しました: {project_dir}")
    else:
        print(f"プロジェクトディレクトリは既に存在します: {project_dir}")
    
    # データ保存用ディレクトリ
    data_dir = os.path.join(project_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"データディレクトリを作成しました: {data_dir}")
    
    # モデル保存用ディレクトリ
    model_dir = os.path.join(project_dir, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"モデルディレクトリを作成しました: {model_dir}")
    
    return project_dir, data_dir, model_dir

def main():
    print("Colab/Kaggle 環境セットアップを開始します...")
    
    # 1. Drive のマウントと容量確認
    if check_drive_space():
        print("Drive の容量確認完了")
    else:
        print("Drive のマウントが必要です。上記の指示に従ってマウントしてください。")
    
    # 2. 依存ライブラリのインストール
    install_dependencies()
    
    # 3. ワークスペースの設定
    project_dir, data_dir, model_dir = setup_workspace()
    
    print("\n=== セットアップ完了 ===")
    print(f"プロジェクトルート: {project_dir}")
    print(f"データ保存先: {data_dir}")
    print(f"モデル保存先: {model_dir}")
    print("\n次のステップ: 01_evaluate_and_filter.py を実行してデータのクレンジングを行ってください。")

if __name__ == "__main__":
    main()
