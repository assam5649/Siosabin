from flask import Blueprint
from . import models

predict = Blueprint('predict', __name__)

from . import routes