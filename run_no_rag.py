
import os
from mi_aplicacion import create_app

config_name = os.getenv('FLASK_CONFIG', 'dev')
app = create_app(config_name, skip_rag=True)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
