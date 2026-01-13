import pytest
import os
import sys

# Asegurar que el directorio raíz del proyecto esté en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mi_aplicacion import create_app

@pytest.fixture
def app(tmp_path):
    """Crea una instancia de la aplicación para pruebas."""
    # Crear un archivo de base de datos temporal
    db_file = tmp_path / "test_db.sqlite"
    
    # Sobreescribir la configuración para usar este archivo
    # Nota: create_app ya carga la config 'test', pero aquí forzamos el archivo
    # para evitar problemas con :memory: y conexiones múltiples
    app = create_app(config_name='test', skip_rag=True)
    app.config['DATABASE_FILE'] = str(db_file)
    
    # Inicializar la base de datos explícitamente si es necesario
    # (aunque create_app lo intenta, al cambiar el path aquí aseguramos que apunte al temp)
    with app.app_context():
        from mi_aplicacion.app_logic import init_sqlite_db, init_users_table
        init_sqlite_db(str(db_file))
        try:
            init_users_table(str(db_file))
        except:
            pass
            
    # Contexto de aplicación para que las pruebas tengan acceso a 'current_app'
    with app.app_context():
        yield app

@pytest.fixture
def client(app):
    """Crea un cliente de prueba para realizar peticiones."""
    return app.test_client()

@pytest.fixture
def runner(app):
    """Crea un runner de prueba para comandos CLI (si existen)."""
    return app.test_cli_runner()
