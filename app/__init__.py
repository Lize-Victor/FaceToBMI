from flask import Flask
import sys
sys.path.append('..')  # Adjust the path to import config from the parent directory
from config import config

def create_app():
    app = Flask(__name__)
    app.config.from_object(config['development'])

    from . import routes
    app.register_blueprint(routes.bp)

    return app