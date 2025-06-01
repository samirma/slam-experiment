from flask import Flask
from .routes import main_blueprint
from dotenv import load_dotenv
import os

def create_app():
    app = Flask(__name__)

    # Load environment variables from .env file located in the project root
    # The project root is one level up from the 'app' directory where main.py is.
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(dotenv_path=dotenv_path)

    app.register_blueprint(main_blueprint)
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
