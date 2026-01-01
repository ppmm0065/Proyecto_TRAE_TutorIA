
import os
import sys
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Añadir el directorio actual al path para importar mi_aplicacion
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mi_aplicacion import create_app
    from mi_aplicacion.app_logic import train_predictive_risk_model, build_training_dataset_from_db
except ImportError as e:
    logger.error(f"Error importando la aplicación: {e}")
    sys.exit(1)

def main():
    logger.info("Iniciando entrenamiento del modelo predictivo de riesgo...")
    
    app = create_app('dev', skip_rag=True)
    
    with app.app_context():
        db_path = app.config['DATABASE_FILE']
        artifacts_dir = app.config['MODEL_ARTIFACTS_DIR']
        
        logger.info(f"Usando DB: {db_path}")
        logger.info(f"Usando directorio de artefactos: {artifacts_dir}")
        
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir, exist_ok=True)
            
        # Debug dataset
        logger.info("Construyendo dataset...")
        ds = build_training_dataset_from_db(db_path)
        if not ds:
            logger.error("Dataset es None")
        else:
            logger.info(f"Dataset X len: {len(ds.get('X', []))}")
            logger.info(f"Snapshots: {len(ds.get('snapshots', []))}")
            
        metrics = train_predictive_risk_model(db_path, artifacts_dir)
        
        if metrics:
            logger.info("Entrenamiento exitoso.")
            logger.info(f"Métricas: {metrics}")
        else:
            logger.warning("El entrenamiento no generó métricas.")

if __name__ == "__main__":
    main()
