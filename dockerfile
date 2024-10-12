# ベースイメージの指定
FROM python:3.8.20

RUN apt-get update && apt-get install -y cron

# 作業ディレクトリを設定
WORKDIR /app

# 依存ファイルをコピー
COPY requirements.txt .

# 依存関係のインストール
RUN pip install -r requirements.txt

# スクリプトをコンテナにコピー
COPY entry.py /usr/local/bin/entry.py

# cronジョブを設定
RUN echo "0 6,18 * * * python /usr/local/bin/get.py >> /var/log/cron.log 2>&1" > /etc/cron.d/my-cron

# cronジョブの権限を設定
RUN chmod 0644 /etc/cron.d/my-cron

# cronを起動
CMD ["cron", "-f"]