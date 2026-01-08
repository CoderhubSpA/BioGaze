import os
import time
import logging
import shutil
import uuid
import asyncio
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Security, Depends, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime

load_dotenv()  # Cargar variables de entorno ANTES de importar módulos que usen DB

# Importamos la interfaz y el adaptador (Inyección de Dependencias manual)
from interfaces import IPhotoValidator, ValidationResult
from adapters import BioGazeAdapter
from db_repository import save_validation_result
from s3_storage import S3StorageManager, s3_manager

# --- Configuración ---
UPLOAD_DIR = "temp_uploads"
AUDIT_DIR = "audit_storage"
LOG_FILE = "api_transactions.log"
API_KEY_NAME = "x-api-key"
# En producción, esto debe venir de os.environ
API_KEY = os.environ.get("BIOGAZE_API_KEY", "minrel-biogaze-secret-key-2025")

# --- Seguridad ---
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="No se pudieron validar las credenciales")

# --- Configuración de Logs ---
logger = logging.getLogger("BioGazeAPI")
logger.setLevel(logging.INFO)

# Evitar duplicar handlers si se recarga el módulo
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Handler de Archivo
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler de Consola (para Docker logs)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# --- Estado Global ---
# Aquí definimos la variable como la INTERFAZ, no la clase concreta
validator_engine: IPhotoValidator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestor de contexto del ciclo de vida.
    Aquí es donde decidimos QUÉ motor usar. Si mañana cambiamos a AWS,
    solo cambiamos la línea `validator_engine = AwsAdapter()`.
    """
    global validator_engine, s3_manager
    # logger.info("Iniciando API de Validación...")
    
    # Asegurar que existan los directorios
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(AUDIT_DIR, exist_ok=True)
    
    try:
        # INYECCIÓN DE DEPENDENCIA: Aquí elegimos BioGazeAdapter
        validator_engine = BioGazeAdapter()
        validator_engine.load_models()
        # logger.info("Motor de Validación inicializado correctamente.")
        
        # Inicializar gestor de S3 (solo si está habilitado)
        s3_enabled = os.getenv("S3_ENABLED", "true").lower() in {"true", "1", "yes"}
        if s3_enabled:
            s3_manager = S3StorageManager()
            logger.info("S3 storage habilitado")
        else:
            s3_manager = None
            logger.info("S3 storage deshabilitado - solo almacenamiento local")
            
    except Exception as e:
        logger.error(f"Fallo al inicializar el Motor de Validación: {e}")
        raise RuntimeError("No se pudieron inicializar los modelos de IA") from e
    
    yield
    
    # logger.info("Apagando API...")

# --- Aplicación API ---
app = FastAPI(
    title="API de Validación de Fotos MINREL",
    description="API para validación de fotos según estándares OACI.",
    version="2.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Validación"
        }
    ],
    swagger_ui_parameters={"defaultModelsExpandDepth": -1} # Oculta la sección "Schemas"
)

# --- Modelos Pydantic para Respuesta de API ---
# Estos modelos son el contrato con el Frontend de MINREL.
# Se ha simplificado para mostrar solo lo necesario (sin métricas técnicas numéricas).

class APIResponse(BaseModel):
    timestamp: str
    estado: str
    motivos_rechazo: List[str]

# --- Endpoints ---

async def store_photo_and_save_to_db(temp_path: str, original_filename: str, new_filename: str, validation_result: dict, request_id: str):
    """
    Tarea asíncrona en segundo plano para subir a S3 y guardar en BD.
    Esta función se ejecuta de forma independiente y no bloquea la respuesta de la API.
    
    Args:
        temp_path (str): Ruta temporal del archivo
        original_filename (str): Nombre original del archivo
        new_filename (str): Nombre UUID del archivo (ej: uuid.jpg)
        validation_result (dict): Resultado de la validación
        request_id (str): ID de la solicitud para logging
    """
    try:
        # 1. Subir a S3
        if s3_manager is None:
            logger.error(f"S3Manager no inicializado para ID {request_id}")
            return
        
        # Determinar tipo MIME
        file_ext = os.path.splitext(new_filename)[1].lower()
        content_type = "image/jpeg" if file_ext in [".jpg", ".jpeg"] else "image/png"
        
        # Ruta en S3: minrel03_sac/fotografias/uuid.jpg
        s3_key = f"minrel03_sac/fotografias/{new_filename}"
        
        # Subir a S3
        s3_url = await s3_manager.upload_file_async(temp_path, s3_key, content_type)
        
        if s3_url:
            # logger.info(f"Imagen subida a S3 para ID {request_id}: {s3_url}")
            
            # 2. Guardar en BD con la URL de S3
            await save_validation_result(temp_path, original_filename, validation_result, s3_url, new_filename)
            logger.info(f"Validación guardada en BD para ID {request_id}")
        else:
            logger.error(f"No se pudo subir imagen a S3 para ID {request_id}")
            
    except Exception as e:
        logger.error(f"Error en tarea asíncrona para ID {request_id}: {str(e)}")
    finally:
        # Limpieza de archivo temporal
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/api/v1/validar-foto", response_model=APIResponse, tags=["Validación"], summary="Validar Fotografía")
async def validate_photo(
    file: UploadFile = File(..., description="Archivo de imagen (JPEG/PNG)"),
    background_tasks: BackgroundTasks = None,
    api_key: str = Depends(get_api_key)
):
    """
    Endpoint principal. Recibe la imagen y delega la validación al motor configurado.
    Retorna la respuesta del estado de validación y los motivos de rechazo.
    
    IMPORTANTE: La validación de la foto se procesa de forma síncrona para dar respuesta inmediata.
    El guardado en S3 y BD se ejecuta en segundo plano de forma asíncrona.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"Solicitud de validación recibida. ID: {request_id}, Archivo: {file.filename}")

    # 1. Validar Tipo de Archivo
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        logger.warning(f"Tipo de contenido inválido: {file.content_type} para ID: {request_id}")
        raise HTTPException(status_code=415, detail="Tipo de medio no soportado. Solo se permiten JPG/PNG.")

    # 2. Generar nombre UUID para el archivo (evita conflictos y sobrescrituras)
    file_ext = os.path.splitext(file.filename)[1]
    new_filename = f"{request_id}{file_ext}"  # UUID + extensión original
    temp_path = os.path.join(UPLOAD_DIR, new_filename)

    try:
        # 3. Guardar Archivo Temporalmente
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 4. Procesar Imagen usando la Interfaz (PROCESO SÍNCRONO - RESPUESTA RÁPIDA)
        if validator_engine is None:
             raise HTTPException(status_code=500, detail="Motor de Validación no inicializado")

        # LLAMADA AGNÓSTICA: No sabemos si es BioGaze o AWS, solo llamamos a .validate()
        result: ValidationResult = validator_engine.validate(temp_path)
        
        # 5. Construir Respuesta (SE RETORNA DE INMEDIATO)
        estado_str = "ACEPTADO" if result.compliant else "RECHAZADO"
        
        response_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "estado": estado_str,
            "motivos_rechazo": result.reasons
        }

        # 6. Preparar datos para guardado asíncrono
        db_validation_result = {
            "compliant": result.compliant,
            "reasons": result.reasons
        }
        
        # 7. TAREA EN SEGUNDO PLANO (NO BLOQUEA LA RESPUESTA)
        # Esto se ejecuta de forma asíncrona después de retornar la respuesta
        asyncio.create_task(
            store_photo_and_save_to_db(
                temp_path, 
                file.filename, 
                new_filename, 
                db_validation_result, 
                request_id
            )
        )

        # 8. Registrar tiempo de procesamiento (solo validación, no guardado)
        processing_time = time.time() - start_time
        logger.info(f"Procesado ID: {request_id} | Estado: {estado_str} | Tiempo: {processing_time:.2f}s")
        
        # RETORNAR RESPUESTA INMEDIATAMENTE
        return response_data

    except Exception as e:
        logger.error(f"Error procesando ID: {request_id}: {str(e)}")
        # Si hay error, limpiamos el archivo temporal
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error Interno del Servidor: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
