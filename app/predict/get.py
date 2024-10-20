import requests
import json
import mysql.connector
from mysql.connector import Error, IntegrityError
from datetime import datetime
import numpy as np

def forecast():
    url = "https://www.jma.go.jp/bosai/forecast/data/forecast/471000.json"
    response = requests.get(url)
    data = response.json() 
    report_datetime = data[0]["reportDatetime"]

    dt = datetime.fromisoformat(report_datetime)

    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour

    def extract_weather_and_temps(data):
        result = {
            "hontou_chunanbu_weathers": [],
            "naha_temps_min": [],
            "naha_temps_max": []
        }

        for entry in data:
            # 本島中南部のweathersを抽出
            for series in entry["timeSeries"]:
                for area in series["areas"]:
                    if area["area"]["name"] == "本島中南部" and "weathers" in area:
                        result["hontou_chunanbu_weathers"] = area["weathers"]

            # 那覇のtempsMinとtempsMaxを抽出
            for series in entry["timeSeries"]:
                for area in series["areas"]:
                    if area["area"]["name"] == "那覇":
                        if "tempsMin" in area:
                            result["naha_temps_min"] = area["tempsMin"]
                        if "tempsMax" in area:
                            result["naha_temps_max"] = area["tempsMax"]

        return result


    result = extract_weather_and_temps(data)

    forecast = result["hontou_chunanbu_weathers"][1]

    def judge(forecast):
        if '雨' in forecast:
            return 0
        else:
            return 1

    data = {
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "precipitation": judge(forecast),
        "tempMax": result["naha_temps_max"][1],
        "tempMin": result["naha_temps_min"][1]
    }

    return data

def forecastDays():
    url = "https://www.jma.go.jp/bosai/forecast/data/forecast/471000.json"
    response = requests.get(url)
    data = response.json()
    time_series = []
    weather_codes = []
    temps_max = []
    temps_min = []
    rain_weather_codes = []

    with open("./RainWeatherCode.txt", "r") as f:
        for l in f:
            rain_weather_codes.append(l.replace(", ", " ").split())

    rain_weather_codes = np.array(rain_weather_codes)
    rain_weather_codes = rain_weather_codes.flatten()

    for i in range(len(data[1]["timeSeries"][0]["timeDefines"])):
        time_series.append(data[1]["timeSeries"][0]["timeDefines"][i])
        time_series[i] = datetime.fromisoformat(time_series[i])
        weather_codes.append(data[1]["timeSeries"][0]["areas"][0]["weatherCodes"][i])
        temps_max.append(data[1]["timeSeries"][1]["areas"][0]["tempsMax"][i])
        temps_min.append(data[1]["timeSeries"][1]["areas"][0]["tempsMin"][i])

    for i in range(len(data[1]["timeSeries"][0]["timeDefines"])):
        if weather_codes[i] == rain_weather_codes[i]:
            weather_codes[i] = 0
        else:
            weather_codes[i] = 1

    result = []

    for i in range(len(data[1]["timeSeries"][0]["timeDefines"]) - 1):
        result.append({
            "year": time_series[i+1].year,
            "month": time_series[i+1].month,
            "day": time_series[i+1].day,
            "precipitation": weather_codes[i+1],
            "tempMax": temps_max[i+1],
            "tempMin": temps_min[i+1]
        })

    return result