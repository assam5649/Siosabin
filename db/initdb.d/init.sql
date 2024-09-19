USE dbdata;

CREATE TABLE users
(
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    password VARCHAR(255) NOT NULL,
    location VARCHAR(255),
    remaining INT,
    salinity FLOAT
);

CREATE TABLE data
(
    id INT PRIMARY KEY AUTO_INCREMENT,
    max_temp FLOAT,
    min_temp FLOAT,
    ave_temp FLOAT,
    ave_humidity INT,
    ave_windvelocity FLOAT,
    max_windvelocity FLOAT
);

INSERT INTO users
    (id, name, password, location, remaining, salinity)
VALUES
    (1, 'a', 'aPass', 'POINT(137.10 35.20)', 70, 0.2);

INSERT INTO data
    (id, max_temp, min_temp, ave_temp, ave_humidity, ave_windvelocity, max_windvelocity)
VALUES
    (1, 25.1, 22.0, 23.2, 80, 2.4, 2.6);