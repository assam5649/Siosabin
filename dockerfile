# ベースイメージの指定
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# 依存ファイルをコピー
COPY requirements.txt .

# 依存関係のインストール
RUN pip install -r requirements.txt