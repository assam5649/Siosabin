import requests
import json
import mysql.connector
from mysql.connector import Error, IntegrityError
from datetime import datetime

def forecast():
    url = "https://www.jma.go.jp/bosai/forecast/data/forecast/471000.json"
    response = requests.get(url)
    data = response.json()
    print(type(data)) 
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