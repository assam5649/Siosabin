until mysqladmin ping -h db -P 3306 -u root -ppass; do
  echo 'waiting for mysqld to be connectable...'
  sleep 2
done
echo "app is starting...!"
# exec python run ../main.go