import polars as pl
from typing import Dict, Optional
from pathlib import Path
import logging

# Configurar logging para mejor debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoaderPolars:
    """
    Cargador de datos optimizado usando Polars para mejor rendimiento
    y manejo eficiente de memoria con datasets grandes.
    """
    
    def __init__(self, parquet_file_path: str):
        """
        Inicializa el cargador de datos con Polars.
        
        Args:
            parquet_file_path (str): Ruta al directorio con archivos Parquet
        """
        self.parquet_file_path = Path(parquet_file_path)
        self.lazy_frames = {}  # Para almacenar LazyFrames
        self.dataframes = {}   # Para almacenar DataFrames materializados
        
        if not self.parquet_file_path.exists():
            raise FileNotFoundError(f"El directorio no existe: {self.parquet_file_path}")

    def load_data_lazy(self) -> Dict[str, pl.LazyFrame]:
        """
        Carga los datos como LazyFrames para procesamiento eficiente.
        Los LazyFrames no cargan datos en memoria hasta que se ejecutan.
        
        Returns:
            Dict[str, pl.LazyFrame]: Diccionario con LazyFrames
        """
        file_names = ["df_concesiones", "df_subterraneas", "df_superficiales"]
        
        for file_name in file_names:
            file_path = self.parquet_file_path / f"{file_name}.parquet"
            if file_path.exists():
                try:
                    # Crear LazyFrame - no carga datos en memoria aún
                    self.lazy_frames[file_name] = pl.scan_parquet(str(file_path))
                    logger.info(f"LazyFrame creado para {file_name}")
                except Exception as e:
                    logger.error(f"Error creando LazyFrame para {file_name}: {e}")
            else:
                logger.warning(f"No se encontró el archivo {file_path}")
        
        return self.lazy_frames

    def load_concesiones_lazy(self) -> pl.LazyFrame:
        """
        Carga datos de concesiones como LazyFrame para procesamiento eficiente.
        
        Returns:
            pl.LazyFrame: LazyFrame con datos de concesiones
        """
        file_path = self.parquet_file_path / "df_concesiones.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        return pl.scan_parquet(str(file_path))

    def load_concesiones(self, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
        """
        Carga datos de concesiones con opción lazy o eager.
        
        Args:
            lazy (bool): Si True, retorna LazyFrame; si False, DataFrame
            
        Returns:
            pl.DataFrame | pl.LazyFrame: Datos de concesiones
        """
        file_path = self.parquet_file_path / "df_concesiones.parquet"
        
        if lazy:
            return pl.scan_parquet(str(file_path))
        else:
            return pl.read_parquet(str(file_path))

    def load_subterraneas(self, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
        """
        Carga datos de aguas subterráneas.
        
        Args:
            lazy (bool): Si True, retorna LazyFrame; si False, DataFrame
            
        Returns:
            pl.DataFrame | pl.LazyFrame: Datos de aguas subterráneas
        """
        file_path = self.parquet_file_path / "df_subterraneas.parquet"
        
        if lazy:
            return pl.scan_parquet(str(file_path))
        else:
            return pl.read_parquet(str(file_path))

    def load_superficiales(self, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
        """
        Carga datos de aguas superficiales.
        
        Args:
            lazy (bool): Si True, retorna LazyFrame; si False, DataFrame
            
        Returns:
            pl.DataFrame | pl.LazyFrame: Datos de aguas superficiales
        """
        file_path = self.parquet_file_path / "df_superficiales.parquet"
        
        if lazy:
            return pl.scan_parquet(str(file_path))
        else:
            return pl.read_parquet(str(file_path))

    def get_data_info(self, dataset_name: str, lazy: bool = True) -> dict:
        """
        Obtiene información básica sobre un dataset específico.
        
        Args:
            dataset_name (str): Nombre del dataset ('concesiones', 'subterraneas', 'superficiales')
            lazy (bool): Si usar evaluación lazy para contar filas
            
        Returns:
            dict: Información del dataset
        """
        file_map = {
            'concesiones': 'df_concesiones.parquet',
            'subterraneas': 'df_subterraneas.parquet', 
            'superficiales': 'df_superficiales.parquet'
        }
        
        if dataset_name not in file_map:
            raise ValueError(f"Dataset '{dataset_name}' no válido. Opciones: {list(file_map.keys())}")
        
        file_path = self.parquet_file_path / file_map[dataset_name]
        
        if lazy:
            lf = pl.scan_parquet(str(file_path))
            # Obtener schema sin materializar todo el DataFrame
            schema = lf.schema
            # Contar filas de manera eficiente
            num_rows = lf.select(pl.len()).collect().item()
        else:
            df = pl.read_parquet(str(file_path))
            schema = df.schema
            num_rows = len(df)
            
        file_size_mb = file_path.stat().st_size / 1024**2
        
        return {
            'archivo': str(file_path),
            'filas': num_rows,
            'columnas': len(schema),
            'tamaño_archivo_mb': round(file_size_mb, 2),
            'columnas_nombres': list(schema.keys()),
            'tipos_datos': {col: str(dtype) for col, dtype in schema.items()},
            'motor': 'Polars'
        }

    def get_all_datasets_info(self) -> Dict[str, dict]:
        """
        Obtiene información de todos los datasets disponibles.
        
        Returns:
            Dict[str, dict]: Información de todos los datasets
        """
        datasets = ['concesiones', 'subterraneas', 'superficiales']
        info = {}
        
        for dataset in datasets:
            try:
                info[dataset] = self.get_data_info(dataset, lazy=True)
            except Exception as e:
                logger.error(f"Error obteniendo info de {dataset}: {e}")
                
        return info

    def stream_processing_example(self, dataset_name: str, chunk_size: int = 100000) -> pl.LazyFrame:
        """
        Ejemplo de procesamiento por streaming para datasets muy grandes.
        Útil cuando el dataset no cabe en memoria.
        
        Args:
            dataset_name (str): Nombre del dataset
            chunk_size (int): Tamaño del chunk para procesamiento
            
        Returns:
            pl.LazyFrame: Resultado del procesamiento streaming
        """
        file_map = {
            'concesiones': 'df_concesiones.parquet',
            'subterraneas': 'df_subterraneas.parquet', 
            'superficiales': 'df_superficiales.parquet'
        }
        
        file_path = self.parquet_file_path / file_map[dataset_name]
        
        # Ejemplo: contar registros por alguna columna usando streaming
        return (
            pl.scan_parquet(str(file_path))
            .select([
                pl.len().alias("total_registros"),
                pl.col("*").first().alias("primer_registro")
            ])
        )

    def memory_efficient_operations(self, operations_chain: list) -> pl.LazyFrame:
        """
        Ejecuta una cadena de operaciones de forma eficiente en memoria.
        
        Args:
            operations_chain (list): Lista de operaciones a ejecutar
            
        Returns:
            pl.LazyFrame: Resultado de las operaciones
        """
        # Este es un ejemplo de cómo encadenar operaciones eficientemente
        lf = self.load_concesiones_lazy()
        
        # Las operaciones se optimizan automáticamente
        for operation in operations_chain:
            lf = operation(lf)
            
        return lf