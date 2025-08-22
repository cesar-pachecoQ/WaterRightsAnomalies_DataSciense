import pandas as pd
from typing import Dict
from pathlib import Path
import os

class DataLoader:
    def __init__(self, parquet_file_path: str):
        self.parquet_file_path = Path(parquet_file_path)  # Convertir a Path object
        self.dataframe = None

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Carga los datos desde archivos Parquet en un diccionario de DataFrames.

        Returns:
            Dict[str, pd.DataFrame]: Diccionario con los DataFrames cargados
        """
        try:
            file_names = ["df_concesiones", "df_subterraneas", "df_superficiales"]
            dataframes = {}
            for file_name in file_names:
                file_path = self.parquet_file_path / f"{file_name}.parquet"
                if file_path.exists():
                    dataframes[file_name] = pd.read_parquet(file_path)
                else:
                    print(f"Advertencia: No se encontró el archivo {file_path}")
            return dataframes
        except Exception as e:
            print(f"Error detallado: {e}")
            raise Exception(f"Error al cargar el archivo Parquet: {e}")
    def load_concesiones(self) -> pd.DataFrame:
        """
        Carga los datos de concesiones desde un archivo Parquet en un DataFrame de pandas.

        Returns:
            pd.DataFrame: Los datos de concesiones leídos del archivo Parquet
        """
        try:
            file_name = "df_concesiones"
            self.dataframe = pd.read_parquet(self.parquet_file_path / (file_name + ".parquet"))
            return self.dataframe
        except Exception as e:
            raise Exception(f"Error al cargar el archivo Parquet: {e}")
    def load_subterraneas(self) -> pd.DataFrame:
        """
        Carga los datos de aguas subterráneas desde un archivo Parquet en un DataFrame de pandas.

        Returns:
            pd.DataFrame: Los datos de aguas subterráneas leídos del archivo Parquet
        """
        try:
            file_name = "df_subterraneas"
            self.dataframe = pd.read_parquet(self.parquet_file_path / (file_name + ".parquet"))
            return self.dataframe
        except Exception as e:
            raise Exception(f"Error al cargar el archivo Parquet: {e}")
    def load_superficiales(self) -> pd.DataFrame:
        """
        Carga los datos de aguas superficiales desde un archivo Parquet en un DataFrame de pandas.

        Returns:
            pd.DataFrame: Los datos de aguas superficiales leídos del archivo Parquet
        """
        try:
            file_name = "df_superficiales"
            self.dataframe = pd.read_parquet(self.parquet_file_path / (file_name + ".parquet"))
            return self.dataframe
        except Exception as e:
            raise Exception(f"Error al cargar el archivo Parquet: {e}")

    def get_data_info(self) -> dict:
        """
        Obtiene información básica sobre el dataset.

        Returns:
            dict: Información del dataset (filas, columnas, memoria, etc.)
        """
        if self.dataframe is None:
            self.load_data()

        return {
            'filas': len(self.dataframe),
            'columnas': len(self.dataframe.columns),
            'memoria_mb': self.dataframe.memory_usage(deep=True).sum() / 1024**2,
            'columnas_nombres': list(self.dataframe.columns),
            'tipos_datos': self.dataframe.dtypes.to_dict()
        }