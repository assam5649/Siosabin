until mysqladmin ping -h db -P 3306 -u root -ppass; do
  echo 'waiting for mysqld to be connectable...'
  sleep 5
done
echo "app is starting...!"
pip install -r ./requirements.txt

python ./main.py
