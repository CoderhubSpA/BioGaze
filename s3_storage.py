"""
Módulo para gestionar el almacenamiento de imágenes en S3 compatible (Linode Object Storage).
Usa requests con firma AWS v4 directamente para máxima compatibilidad.
"""
import os
import asyncio
import hashlib
import hmac
from datetime import datetime
from typing import Optional
import requests
import logging

logger = logging.getLogger("BioGazeAPI.S3")


class S3StorageManager:
    """
    Gestor de almacenamiento en S3 para las fotografías de validación.
    Usa requests con firma AWS Signature v4 para compatibilidad con Linode.
    """
    
    def __init__(self):
        """
        Inicializa el gestor de S3 con las credenciales del entorno.
        Variables de entorno requeridas:
        - AWS_ACCESS_KEY_ID
        - AWS_SECRET_ACCESS_KEY
        - AWS_REGION
        - S3_BUCKET_NAME
        - AWS_ENDPOINT_URL (para proveedores compatibles como Linode)
        """
        self.aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.environ.get("AWS_REGION", "us-east-1")
        self.bucket_name = os.environ.get("S3_BUCKET_NAME")
        self.endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
        
        if not all([self.aws_access_key, self.aws_secret_key, self.bucket_name]):
            raise ValueError("Faltan credenciales de S3 en las variables de entorno")
        
        if not self.endpoint_url:
            raise ValueError("AWS_ENDPOINT_URL es requerido para Linode Object Storage")
        
        # Sesión de requests para reutilizar conexiones
        self.session = requests.Session()
        
        endpoint_info = f"Endpoint: {self.endpoint_url}"
        # logger.info(f"S3 Manager inicializado. Bucket: {self.bucket_name}, Región: {self.aws_region}, {endpoint_info}")
    
    def _sign(self, key: bytes, msg: str) -> bytes:
        """Firma HMAC-SHA256."""
        return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()
    
    def _get_signature_key(self, date_stamp: str) -> bytes:
        """Genera la clave de firma para AWS v4."""
        k_date = self._sign(('AWS4' + self.aws_secret_key).encode('utf-8'), date_stamp)
        k_region = self._sign(k_date, self.aws_region)
        k_service = self._sign(k_region, 's3')
        k_signing = self._sign(k_service, 'aws4_request')
        return k_signing
    
    def _create_signed_headers(self, method: str, host: str, uri: str, 
                                payload: bytes, content_type: str) -> dict:
        """Crea headers firmados para AWS S3 v4."""
        t = datetime.utcnow()
        amz_date = t.strftime('%Y%m%dT%H%M%SZ')
        date_stamp = t.strftime('%Y%m%d')
        
        # Payload hash
        payload_hash = hashlib.sha256(payload).hexdigest()
        
        # Canonical request
        canonical_headers = (
            f'content-type:{content_type}\n'
            f'host:{host}\n'
            f'x-amz-content-sha256:{payload_hash}\n'
            f'x-amz-date:{amz_date}\n'
        )
        signed_headers = 'content-type;host;x-amz-content-sha256;x-amz-date'
        
        canonical_request = (
            f'{method}\n'
            f'{uri}\n'
            f'\n'  # query string vacío
            f'{canonical_headers}\n'
            f'{signed_headers}\n'
            f'{payload_hash}'
        )
        
        # String to sign
        algorithm = 'AWS4-HMAC-SHA256'
        credential_scope = f'{date_stamp}/{self.aws_region}/s3/aws4_request'
        string_to_sign = (
            f'{algorithm}\n'
            f'{amz_date}\n'
            f'{credential_scope}\n'
            f'{hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()}'
        )
        
        # Signature
        signing_key = self._get_signature_key(date_stamp)
        signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        
        # Authorization header
        authorization_header = (
            f'{algorithm} '
            f'Credential={self.aws_access_key}/{credential_scope}, '
            f'SignedHeaders={signed_headers}, '
            f'Signature={signature}'
        )
        
        return {
            'Content-Type': content_type,
            'x-amz-date': amz_date,
            'x-amz-content-sha256': payload_hash,
            'Authorization': authorization_header,
            'Host': host
        }
    
    async def upload_file_async(self, file_path: str, s3_key: str, 
                                 content_type: str = "image/jpeg") -> Optional[str]:
        """
        Sube un archivo a S3 de forma asíncrona.
        
        Args:
            file_path: Ruta local del archivo a subir
            s3_key: Ruta/nombre del archivo en S3
            content_type: Tipo MIME del archivo
            
        Returns:
            URL del archivo en S3 si fue exitoso, None si falló
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._upload_file_sync,
                file_path,
                s3_key,
                content_type
            )
            return result
        except Exception as e:
            logger.error(f"Error en upload_file_async: {str(e)}")
            return None
    
    def _upload_file_sync(self, file_path: str, s3_key: str, content_type: str) -> Optional[str]:
        """Método sincrónico para subir archivo."""
        try:
            file_size = os.path.getsize(file_path)
            # logger.info(f"Iniciando subida - Key: {s3_key}, Tamaño: {file_size} bytes")
            
            # Leer archivo
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Construir URL y headers (path style)
            host = self.endpoint_url
            uri = f"/{self.bucket_name}/{s3_key}"
            url = f"https://{host}{uri}"
            
            headers = self._create_signed_headers('PUT', host, uri, file_content, content_type)
            
            # Subir
            response = self.session.put(url, data=file_content, headers=headers, timeout=120)
            
            if response.status_code in [200, 204]:
                s3_url = self.generate_s3_url(s3_key)
                logger.info(f"Fotografía subida: {s3_url}")
                return s3_url
            else:
                logger.error(f"Error HTTP {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"Error en _upload_file_sync: {type(e).__name__}: {str(e)}")
            return None
    
    def generate_s3_url(self, s3_key: str) -> str:
        """Genera la URL pública de un objeto en S3."""
        return f"https://{self.endpoint_url}/{self.bucket_name}/{s3_key}"


# Instancia global del gestor
s3_manager: Optional[S3StorageManager] = None


def get_s3_manager() -> S3StorageManager:
    """Obtiene la instancia global del gestor de S3."""
    if s3_manager is None:
        raise RuntimeError("S3StorageManager no ha sido inicializado")
    return s3_manager
