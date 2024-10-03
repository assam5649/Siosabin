from flask import Blueprint
from . import models

predict = Blueprint('predict', __name__)