from flask import Blueprint
from . import models

main = Blueprint('main', __name__)

from . import routes