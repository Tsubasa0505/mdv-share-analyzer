# Renderへのデプロイ更新手順

## 準備
1. GitHubアカウントが必要です
2. Renderアカウントが必要です

## 手順

### 1. GitHubリポジトリの作成
1. GitHub.com にアクセス
2. 新しいリポジトリを作成（例: mdv-share-analyzer）
3. Public または Private を選択

### 2. ローカルからGitHubへプッシュ

```bash
# Gitの初期設定（初回のみ）
git config --global user.name "あなたの名前"
git config --global user.email "your-email@example.com"

# リポジトリの初期化とコミット
git init
git add .
git commit -m "Add statistics guide feature"

# GitHubリポジトリと接続
git branch -M main
git remote add origin https://github.com/[ユーザー名]/[リポジトリ名].git
git push -u origin main
```

### 3. Renderでの設定

#### 新規デプロイの場合：
1. https://render.com にログイン
2. "New +" → "Web Service" をクリック
3. GitHubアカウントと連携
4. リポジトリを選択
5. 以下の設定を行う：
   - Name: mdv-share-analyzer
   - Runtime: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
6. "Create Web Service" をクリック

#### 既存サービスの更新の場合：
1. Renderダッシュボードにログイン
2. 既存のサービスを選択
3. "Settings" タブで GitHub リポジトリを接続
4. "Auto-Deploy" を有効化

### 4. 自動デプロイの設定
- Auto-Deploy を有効にすると、GitHubにプッシュするたびに自動的にデプロイされます

### 5. 今後の更新方法

```bash
# 変更をコミット
git add .
git commit -m "Update description"

# GitHubにプッシュ（自動的にRenderにデプロイされる）
git push
```

## 環境変数（必要な場合）
Renderのダッシュボードで以下を設定：
- Environment → Add Environment Variable
- 必要な環境変数を追加

## トラブルシューティング

### ビルドエラーの場合
- requirements.txt が最新か確認
- Python バージョンを確認（render.yaml で 3.11.0 を指定）

### アクセスできない場合
- Renderのログを確認
- ポート設定を確認（$PORT を使用）

## 動作確認
デプロイ完了後、以下のURLでアクセス可能：
https://[サービス名].onrender.com