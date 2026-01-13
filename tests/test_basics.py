def test_app_exists(app):
    """Verifica que la aplicación se crea correctamente."""
    assert app is not None

def test_app_is_testing(app):
    """Verifica que la aplicación está en modo de pruebas."""
    assert app.config['TESTING']

def test_index_page(client):
    """Verifica que la página de inicio carga (o redirige al login)."""
    response = client.get('/')
    # Puede ser 200 (si no hay auth) o 302 (si redirige a login)
    assert response.status_code in [200, 302]

def test_login_page_loads(client):
    """Verifica que la página de login carga correctamente."""
    # Asumiendo que la ruta es /login
    response = client.get('/login')
    assert response.status_code == 200
    # Buscar textos que realmente están en login.html
    assert b"Ingreso" in response.data or b"Usuario" in response.data
