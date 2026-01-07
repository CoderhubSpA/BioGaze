import uuid
import os
import json
import hashlib
import asyncio
from sqlalchemy import text
from datetime import datetime
from database import get_db_connection

# Constantes hardcodeadas del sistema (según tu script original)
DEFAULT_AREA_ID = '8d276a55-618a-413e-be95-c517e9ddbdef'
DEFAULT_CREATED_BY = 'd5f14e37-2527-11eb-8dfb-f23c920f9a68'

async def save_validation_result(file_path: str, original_filename: str, validation_result: dict, s3_url: str, new_filename: str):
    """
    Guarda el resultado de la validación y el documento en la base de datos de forma asíncrona.
    
    Args:
        file_path (str): Ruta física temporal del archivo.
        original_filename (str): Nombre original del archivo subido.
        validation_result (dict): Diccionario con el resultado de la validación (compliant, reasons, etc).
        s3_url (str): URL completa del archivo en S3.
        new_filename (str): Nombre UUID del archivo (ej: uuid.jpg).
    """
    
    # 1. Generar IDs y Hashes
    document_id = str(uuid.uuid4())
    validation_id = str(uuid.uuid4())
    
    # Leer contenido del archivo de forma asíncrona (no bloquear event loop)
    loop = asyncio.get_event_loop()
    file_content = await loop.run_in_executor(None, lambda: open(file_path, "rb").read())
    file_size = len(file_content)
    
    # Generar unique_hash combinando el nombre UUID del archivo (new_filename) + contenido
    # Esto asegura que cada validación genere un hash único, incluso si es la misma imagen
    # Permite pruebas con la misma imagen sin violar la constraint unique_hash_version de BD
    hash_input = new_filename.encode('utf-8') + file_content
    unique_hash = hashlib.md5(hash_input).hexdigest()
    
    # Extraemos extensión del nuevo nombre
    _, extension = os.path.splitext(new_filename)
    extension = extension.replace(".", "").lower() # ej: "jpg"

    # 2. Preparar datos para tabla DOCUMENT
    # src apunta ahora a la URL de S3 donde está almacenada la imagen
    document_data = {
        "id": document_id,
        "unique_hash": unique_hash,
        "name": new_filename,  # Usamos el nombre UUID, no el original
        "version": 1,
        "extension": extension,
        "entity_id": None, # No tenemos entity_id en este contexto, se deja NULL
        "src": s3_url,  # URL completa del archivo en S3
        "valid": 1,
        "area_id": DEFAULT_AREA_ID,
        "owner_id": None, # No tenemos owner_id del usuario logueado
        "size": file_size,
        "created_by": DEFAULT_CREATED_BY
    }

    # 3. Preparar datos para tabla VALIDACIONES_FOTOGRAFIAS
    # Serializamos el JSON completo de la respuesta
    json_response = json.dumps(validation_result, ensure_ascii=False)
    
    # Determinamos estado y motivos
    estado = "ACEPTADO" if validation_result.get("compliant") else "RECHAZADO"
    motivos = ", ".join(validation_result.get("reasons", [])) if validation_result.get("reasons") else None

    validation_data = {
        "id": validation_id,
        "valid": 1,
        "resultado": estado,
        "motivos_rechazo": motivos,
        "respuesta_json": json_response,
        "document_id": document_id,
        "created_by": DEFAULT_CREATED_BY,
        "area_id": DEFAULT_AREA_ID
    }

    # 4. Ejecutar Transacción en Base de Datos de forma asíncrona
    # Ejecutamos las operaciones de BD en un executor para no bloquear
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _save_to_database_sync, document_data, validation_data)


def _save_to_database_sync(document_data: dict, validation_data: dict):
    """
    Función sincrónica para guardar en BD (llamada desde executor).
    """
    conn = get_db_connection()
    trans = conn.begin()
    
    try:
        # A. Insertar en DOCUMENT
        query_doc = text("""
            INSERT INTO document (
                id, unique_hash, name, version, extension, entity_id, src, valid, area_id, owner_id, size, created_by
            ) VALUES (
                :id, :unique_hash, :name, :version, :extension, :entity_id, :src, :valid, :area_id, :owner_id, :size, :created_by
            )
        """)
        conn.execute(query_doc, document_data)
        
        # B. Insertar en VALIDACIONES_FOTOGRAFIAS
        query_val = text("""
            INSERT INTO validaciones_fotografias (
                id, valid, resultado, motivos_rechazo, respuesta_json, document_id, created_by, area_id
            ) VALUES (
                :id, :valid, :resultado, :motivos_rechazo, :respuesta_json, :document_id, :created_by, :area_id
            )
        """)
        conn.execute(query_val, validation_data)
        
        trans.commit()
        print(f"[DB] Validación guardada exitosamente. Doc ID: {document_data['id']}, Val ID: {validation_data['id']}")
        return True
        
    except Exception as e:
        trans.rollback()
        print(f"[DB] Error al guardar en base de datos: {str(e)}")
        raise e
    finally:
        conn.close()
