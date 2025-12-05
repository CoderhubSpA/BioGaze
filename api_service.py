import os
import time
import logging
import shutil
import uuid
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from datetime import datetime

# Importamos la interfaz y el adaptador (Inyección de Dependencias manual)
from interfaces import IPhotoValidator, ValidationResult
from adapters import BioGazeAdapter

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
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("BioGazeAPI")

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
    global validator_engine
    # logger.info("Iniciando API de Validación...")
    
    # Asegurar que existan los directorios
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(AUDIT_DIR, exist_ok=True)
    
    try:
        # INYECCIÓN DE DEPENDENCIA: Aquí elegimos BioGazeAdapter
        validator_engine = BioGazeAdapter()
        validator_engine.load_models()
        # logger.info("Motor de Validación inicializado correctamente.")
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

@app.post("/api/v1/validar-foto", response_model=APIResponse, tags=["Validación"], summary="Validar Fotografía")
async def validate_photo(
    file: UploadFile = File(..., description="Archivo de imagen (JPEG/PNG)"),
    api_key: str = Depends(get_api_key)
):
    """
    Endpoint principal. Recibe la imagen y delega la validación al motor configurado.
    Retorna la respuesta del estado de validación y los motivos de rechazo.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"Solicitud de validación recibida. ID: {request_id}, Archivo: {file.filename}")

    # 1. Validar Tipo de Archivo
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        logger.warning(f"Tipo de contenido inválido: {file.content_type} para ID: {request_id}")
        raise HTTPException(status_code=415, detail="Tipo de medio no soportado. Solo se permiten JPG/PNG.")

    # 2. Guardar Archivo Temporalmente
    file_ext = os.path.splitext(file.filename)[1]
    temp_filename = f"{request_id}{file_ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Procesar Imagen usando la Interfaz
        if validator_engine is None:
             raise HTTPException(status_code=500, detail="Motor de Validación no inicializado")

        # LLAMADA AGNÓSTICA: No sabemos si es BioGaze o AWS, solo llamamos a .validate()
        result: ValidationResult = validator_engine.validate(temp_path)
        
        # 4. Construir Respuesta
        # Traducimos el estado a Español
        estado_str = "ACEPTADO" if result.compliant else "RECHAZADO"
        
        response_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "estado": estado_str,
            "motivos_rechazo": result.reasons
        }

        # 5. Almacenamiento de Auditoría
        today_str = datetime.now().strftime("%Y-%m-%d")
        audit_folder = os.path.join(AUDIT_DIR, today_str, estado_str)
        os.makedirs(audit_folder, exist_ok=True)
        
        final_path = os.path.join(audit_folder, temp_filename)
        shutil.move(temp_path, final_path)
        
        logger.info(f"Imagen almacenada para auditoría en: {final_path}")

        # 6. Registrar Resultado
        processing_time = time.time() - start_time
        logger.info(f"Procesado ID: {request_id} | Estado: {estado_str} | Tiempo: {processing_time:.2f}s")
        
        return response_data

    except Exception as e:
        logger.error(f"Error procesando ID: {request_id}: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Error Interno del Servidor: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
