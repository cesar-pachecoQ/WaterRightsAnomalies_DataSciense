import pandas as pd
import os
from pathlib import Path
from typing import Optional, List


class DataConvert:
    """
    Clase para convertir archivos CSV a formato Parquet.
    Maneja la conversión de datos de concesiones de agua del proyecto.
    """

    def __init__(self, csv_file_path: str):
        """
        Inicializa el convertidor de datos.
        
        Args:
            csv_file_path (str): Ruta al archivo CSV que se va a convertir
            output_dir (str, optional): Directorio donde guardar los archivos Parquet.
                                      Si no se especifica, usa DataSets_Parquets/
        """
        self.csv_file_path = Path(csv_file_path)
        self.dataframe = None
        
        project_root = self.csv_file_path.parent.parent
        self.output_dir = project_root / "DataSets_CSVs" / "DataSets_Parquets"
        
        # Crear directorio de salida si no existe
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validar que el archivo CSV existe
        if not self.csv_file_path.exists():
            raise FileNotFoundError(f"El archivo CSV no existe: {self.csv_file_path}")
        # Validar que el archivo es un CSV
        if not self.csv_file_path.suffix.lower() == '.csv':
            raise ValueError(f"El archivo debe ser un CSV: {self.csv_file_path}")

    def read_data(self) -> pd.DataFrame:
        """
        Lee los datos del archivo CSV en un DataFrame de pandas.
        
        Returns:
            pd.DataFrame: Los datos leídos del archivo CSV
        """
        try:
            self.dataframe = pd.read_csv(self.csv_file_path)
            print(f"Datos leídos exitosamente: {len(self.dataframe)} filas")
            return self.dataframe
        # manejar errores de lectura
        except Exception as e:
            raise Exception(f"Error al leer el archivo CSV: {e}")

    def get_data_info(self) -> dict:
        """
        Obtiene información básica sobre el dataset.
        
        Returns:
            dict: Información del dataset (filas, columnas, memoria, etc.)
        """
        if self.dataframe is None:
            self.read_data()
        
        return {
            'filas': len(self.dataframe),
            'columnas': len(self.dataframe.columns),
            'memoria_mb': self.dataframe.memory_usage(deep=True).sum() / 1024**2,
            'columnas_nombres': list(self.dataframe.columns),
            'tipos_datos': self.dataframe.dtypes.to_dict()
        }

    def convert_to_parquet(self, custom_name: Optional[str] = None) -> str:
        """
        Convierte el DataFrame actual a un archivo Parquet.
        
        Args:
            custom_name (str, optional): Nombre personalizado para el archivo.
                                       Si no se especifica, usa el nombre del CSV original.
        
        Returns:
            str: Ruta del archivo Parquet creado
        """
        if self.dataframe is None:
            self.read_data()
        
        # Determinar nombre del archivo de salida
        if custom_name:
            if not custom_name.endswith('.parquet'):
                custom_name += '.parquet'
            output_filename = custom_name
        else:
            # Usar el nombre del archivo CSV original pero con extensión .parquet
            output_filename = self.csv_file_path.stem + '.parquet'
        
        output_path = self.output_dir / output_filename
        
        try:
            # Convertir a Parquet con compresión optimizada
            self.dataframe.to_parquet(
                output_path, 
                compression='zstd',  # zstd es más eficiente que snappy
                engine='pyarrow', 
                index=False
            )
            
            # Obtener tamaños de archivo para comparación
            csv_size = self.csv_file_path.stat().st_size / 1024**2  # MB
            parquet_size = output_path.stat().st_size / 1024**2  # MB
            compression_ratio = (1 - parquet_size/csv_size) * 100
            
            print(f" Conversión exitosa:")
            print(f"    Archivo: {output_path}")
            print(f"    Filas: {len(self.dataframe):,}")
            print(f"    Tamaño CSV: {csv_size:.2f} MB")
            print(f"    Tamaño Parquet: {parquet_size:.2f} MB")
            print(f"     Compresión: {compression_ratio:.1f}%")
            
            return str(output_path)
            
        except Exception as e:
            raise Exception(f"Error al convertir a Parquet: {e}")

    @classmethod
    def convert_multiple_csvs(cls, csv_directory: str, file_patterns: Optional[List[str]] = None) -> List[str]:
        """
        Convierte múltiples archivos CSV a Parquet de una vez.
        
        Args:
            csv_directory (str): Directorio que contiene los archivos CSV
            file_patterns (List[str], optional): Patrones de nombres de archivo a incluir.
                                               Por defecto convierte todos los .csv
        
        Returns:
            List[str]: Lista de rutas de archivos Parquet creados
        """
        csv_dir = Path(csv_directory)
        if not csv_dir.exists():
            raise FileNotFoundError(f"El directorio no existe: {csv_dir}")
        
        # Buscar archivos CSV
        if file_patterns:
            csv_files = []
            for pattern in file_patterns:
                csv_files.extend(csv_dir.glob(pattern))
        else:
            csv_files = list(csv_dir.glob("*.csv"))
        
        if not csv_files:
            print("⚠️  No se encontraron archivos CSV para convertir")
            return []
        
        converted_files = []
        print(f"🔄 Convirtiendo {len(csv_files)} archivos CSV...")
        
        for csv_file in csv_files:
            try:
                converter = cls(str(csv_file))
                parquet_path = converter.convert_to_parquet()
                converted_files.append(parquet_path)
                print()  # Línea en blanco entre conversiones
            except Exception as e:
                print(f" Error convirtiendo {csv_file.name}: {e}")
        
        print(f"Conversión completada: {len(converted_files)} archivos convertidos")
        return converted_files
        
    