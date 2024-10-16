from app.predict.get import forecast, forecastDays
from app.predict.services import insert_features, insert_featuresDays
from app.predict.utils import connect, connectDays, connectDaysPredict, salinity
from app.predict.predict import predict
from app.predict.predictDays import predictDays
from app.predict.continual import updateModel
from app.predict.continualDays import updateModelDays

insert_features(forecast())
insert_featuresDays(forecastDays())

updateModel(connect(), salinity())
updateModelDays(connectDays(), salinity())

predict(connect())
for i in connectDaysPredict():
    predictDays(i)