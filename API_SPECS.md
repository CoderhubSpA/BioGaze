# Especificaciones Técnicas: API de Validación Fotográfica BioGaze (MINREL)

## 1. Descripción General
Este documento define la interfaz de programación de aplicaciones (API) para el servicio de validación automática de fotografías del Sistema de Atención Consular. El servicio recibe una imagen digital, la procesa mediante motores de Inteligencia Artificial (BioGaze) y retorna un veredicto de cumplimiento basado en normas ISO/ICAO.

## 2. Protocolo de Comunicación
- **Protocolo:** HTTP/1.1
- **Formato de Intercambio:** JSON (UTF-8)
- **Seguridad:** API Key en Header (`x-api-key`)

## 3. Definición de Endpoints

### 3.1. Validar Fotografía
Recibe una imagen y ejecuta el pipeline completo de validación.

- **Método:** `POST`
- **URL:** `/api/v1/validar-foto`
- **Content-Type:** `multipart/form-data`

#### Encabezados (Headers)
| Header | Valor | Obligatorio | Descripción |
|--------|-------|-------------|-------------|
| `x-api-key` | `[CLAVE_SECRETA]` | Sí | Clave de autenticación del servicio. |

#### Parámetros de Entrada (Form Data)
| Campo | Tipo | Obligatorio | Descripción |
|-------|------|-------------|-------------|
| `file` | File | Sí | Archivo de imagen (JPG, PNG). |

#### Respuesta Exitosa (200 OK)

El servicio retorna un objeto JSON simplificado con el veredicto final.

**Ejemplo ACEPTADO:**
```json
{
  "timestamp": "2025-12-05T14:30:00Z",
  "estado": "ACEPTADO",
  "motivos_rechazo": []
}
```

**Ejemplo RECHAZADO:**
```json
{
  "timestamp": "2025-12-05T14:35:00Z",
  "estado": "RECHAZADO",
  "motivos_rechazo": [
    "Boca abierta detectada",
    "Imagen desenfocada"
  ]
}
```

#### Respuestas de Error

| Código HTTP | Descripción | Cuerpo de Respuesta (Ejemplo) |
|-------------|-------------|-------------------------------|
| **403 Forbidden** | La API Key proporcionada es inválida o falta. | `{"detail": "No se pudieron validar las credenciales"}` |
| **415 Unsupported Media Type** | El archivo enviado no es una imagen válida (solo JPG/PNG). | `{"detail": "Tipo de medio no soportado. Solo se permiten JPG/PNG."}` |
| **422 Unprocessable Entity** | Falta el campo `file` o el formato de la petición es incorrecto. | `{"detail": [...]}` (Error de validación estándar de FastAPI) |
| **500 Internal Server Error** | Error interno del servidor o del motor de IA. | `{"detail": "Motor de Validación no inicializado"}` |

## 4. Trazabilidad y Logs
El sistema genera un registro (log) por cada transacción procesada en `api_transactions.log`, almacenando:
- ID de solicitud (UUID).
- Timestamp.
- Nombre del archivo.
- Resultado de la validación.

Además, las imágenes procesadas se almacenan temporalmente en carpetas de auditoría clasificadas por fecha y resultado (`audit_storage/YYYY-MM-DD/ACEPTADO` o `RECHAZADO`).

## 5. Arquitectura de Software
El servicio está construido sobre:
- **Lenguaje:** Python 3.12+
- **Framework Web:** FastAPI
- **Motor IA:** BioGaze (YOLOv8, Dlib, OpenCV, MediaPipe)
- **Contenedor:** Docker
