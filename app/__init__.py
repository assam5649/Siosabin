from flask import Flask

def create_app():
    app = Flask(__name__)

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint, url_prefix='/main')

    from .main import models
    models.initdb()

    # from .auth import auth as auth_blueprint
    # app.register_blueprint(auth_blueprint, urlprefix='/auth')

    return app