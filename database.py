import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger("BioGazeDB")
logger.setLevel(logging.INFO)

# --- Configuración de Base de Datos (SHEETS / CHDEV) ---
from chlib import chutil

conf = chutil.loadConfigFile()
DB_USER = conf.get("DB_USERNAME")
DB_PASS = conf.get("DB_PASSWORD")
DB_HOST = conf.get("DB_HOST")
DB_PORT = conf.get("DB_PORT")
DB_NAME = conf.get("DB_DATABASE")

print(f"[DB] Usando configuración de Sheets/CHDev: {DB_HOST}:{DB_PORT}/{DB_NAME}")
logger.info(f"[DB] Conectando a {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# String de conexión para MySQL/MariaDB usando el driver pymysql
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Creamos el motor de base de datos
engine = create_engine(DATABASE_URL, pool_recycle=3600, pool_pre_ping=True)

# Creamos la fábrica de sesiones
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db_connection():
    """
    Retorna una conexión cruda al motor.
    """
    return engine.connect()