from abc import ABC, abstractmethod
from typing import Dict, List, Any
from pydantic import BaseModel

# Definimos estructuras de datos genéricas que la API entiende
# Esto aísla a la API de los formatos específicos de cada modelo

class ValidationResult(BaseModel):
    compliant: bool
    reasons: List[str]
    details: Dict[str, Any]
    technical_metrics: Dict[str, Any]

class IPhotoValidator(ABC):
    """
    Interfaz abstracta para cualquier motor de validación de fotos.
    Cualquier modelo futuro (BioGaze v2, AWS Rekognition, etc.) debe heredar de esta clase.
    """
    
    @abstractmethod
    def load_models(self):
        """Carga los modelos en memoria."""
        pass

    @abstractmethod
    def validate(self, image_path: str) -> ValidationResult:
        """
        Valida una imagen y retorna un resultado estandarizado.
        """
        pass
