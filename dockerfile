# ベースイメージの指定
FROM python:latest

# 作業ディレクトリを設定
WORKDIR /app

# 依存ファイルをコピー
COPY requirements.txt .

# 依存関係のインストール
RUN pip install -r requirements.txt