from flask import Blueprint
from . import models

auth = Blueprint('auth', __name__)

from . import routes