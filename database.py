import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

logger = logging.getLogger("BioGazeDB")
logger.setLevel(logging.INFO)

# --- Configuración de Base de Datos (SHEETS / CHDEV) ---
try:
    from chlib import chutil
    # from chlib.DatabaseConnection import DatabaseConnection # No lo usamos directamente para mantener compatibilidad con SQLAlchemy

    # Cargar configuración del entorno Sheets
    conf = chutil.loadConfigFile()
    
    DB_USER = conf.get("DB_USERNAME")
    DB_PASS = conf.get("DB_PASSWORD")
    DB_HOST = conf.get("DB_HOST")
    DB_PORT = conf.get("DB_PORT")
    DB_NAME = conf.get("DB_DATABASE")
    
    print(f"[DB] Usando configuración de Sheets/CHDev: {DB_HOST}:{DB_PORT}/{DB_NAME}")

except ImportError:
    # Fallback o Error si no estamos en el entorno correcto
    print("[DB] Advertencia: 'chlib' no encontrado. Usando configuración por defecto (puede fallar si no es local).")
    # Valores dummy o locales si se desea probar sin chlib
    DB_HOST = os.environ.get("DB_HOST", "host.docker.internal")
    DB_PORT = os.environ.get("DB_PORT", "3306")
    DB_USER = os.environ.get("DB_USER", "root")
    DB_PASS = os.environ.get("DB_PASS", "")
    DB_NAME = os.environ.get("DB_NAME", "minrel03_sac")
# --- Configuración de Base de Datos (LOCAL - COMENTADO) ---
DB_HOST = os.environ.get("DB_HOST", "127.0.0.1") 
DB_PORT = os.environ.get("DB_PORT", "3306")
DB_USER = os.environ.get("DB_USER", "root") 
DB_PASS = os.environ.get("DB_PASS", "")     
DB_NAME = os.environ.get("DB_NAME", "minrel03_sac")
logger.info(DB_HOST, DB_PORT, DB_USER, DB_NAME)

# String de conexión para MySQL/MariaDB usando el driver pymysql
# Construimos la URL de SQLAlchemy usando los valores obtenidos (ya sea de chlib o fallback)
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Creamos el motor de base de datos
# pool_recycle=3600 ayuda a evitar desconexiones por inactividad
engine = create_engine(DATABASE_URL, pool_recycle=3600, pool_pre_ping=True)

# Creamos la fábrica de sesiones
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db_connection():
    """
    Retorna una conexión cruda al motor (para usar con conn.execute como en tu script).
    """
    return engine.connect()
