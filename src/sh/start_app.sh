until mysqladmin ping -h db -P 3306 -u root -ppass; do
  echo 'waiting for mysqld to be connectable...'
  sleep 2
done
echo "app is starting...!"
pip install requests
pip install mysql-connector-python

python ./main.py