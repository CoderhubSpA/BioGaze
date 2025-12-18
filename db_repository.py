import uuid
import os
import json
import hashlib
from sqlalchemy import text
from datetime import datetime
from database import get_db_connection

# Constantes hardcodeadas del sistema (según tu script original)
DEFAULT_AREA_ID = '8d276a55-618a-413e-be95-c517e9ddbdef'
DEFAULT_CREATED_BY = 'd5f14e37-2527-11eb-8dfb-f23c920f9a68'

def save_validation_result(file_path: str, original_filename: str, validation_result: dict):
    """
    Guarda el resultado de la validación y el documento en la base de datos.
    
    Args:
        file_path (str): Ruta física donde quedó guardada la imagen (en audit_storage).
        original_filename (str): Nombre original del archivo subido.
        validation_result (dict): Diccionario con el resultado de la validación (compliant, reasons, etc).
    """
    
    # 1. Generar IDs y Hashes
    document_id = str(uuid.uuid4())
    validation_id = str(uuid.uuid4())
    
    # Generamos un hash único para el archivo (útil para integridad)
    with open(file_path, "rb") as f:
        file_content = f.read()
        unique_hash = hashlib.md5(file_content).hexdigest()
        file_size = len(file_content)
    
    # Extraemos extensión
    _, extension = os.path.splitext(original_filename)
    extension = extension.replace(".", "").lower() # ej: "jpg"

    # 2. Preparar datos para tabla DOCUMENT
    # Nota: 'src' en tu script era "/document/" + id. Mantenemos ese formato lógico.
    # Aunque físicamente el archivo esté en audit_storage, la BD guarda una referencia lógica.
    document_data = {
        "id": document_id,
        "unique_hash": unique_hash,
        "name": original_filename,
        "version": 1,
        "extension": extension,
        "entity_id": None, # No tenemos entity_id en este contexto, se deja NULL
        "src": f"/document/{document_id}", 
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

    # 4. Ejecutar Transacción en Base de Datos
    conn = get_db_connection()
    trans = conn.begin() # Iniciar transacción
    
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
        
        trans.commit() # Confirmar cambios
        print(f"[DB] Validación guardada exitosamente. Doc ID: {document_id}, Val ID: {validation_id}")
        return True
        
    except Exception as e:
        trans.rollback() # Revertir si hay error
        print(f"[DB] Error al guardar en base de datos: {str(e)}")
        raise e # Re-lanzamos para que el logger de la API lo capture
    finally:
        conn.close()
