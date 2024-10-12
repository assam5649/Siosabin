from app.predict.get import forecast
from app.predict.services import insert_features
from app.predict.utils import connect
from app.predict.predict import predict
from app.predict.continual import update_model

insert_features(forecast())
predict(connect())

# update_model()