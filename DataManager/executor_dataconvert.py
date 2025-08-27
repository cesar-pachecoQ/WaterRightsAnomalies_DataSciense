#!/usr/bin/env python3
"""
Ejemplo de uso de la clase DataConvert mejorada
Para el proyecto de anomalías en concesiones de agua
"""

from DataConvert import DataConvert
import os

def main():
    # Directorio base del proyecto
    base_dir = "/home/cesar_r/Documentos/Proyectos/IberoSocialData/WaterRightsAnomalies_DataSciense"
    csv_dir = os.path.join(base_dir, "DataSets_CSVs")
    
    print(" Iniciando conversión de archivos CSV a Parquet")
    print("=" * 50)
    
    # Ejemplo 1: Convertir un archivo individual
    print("\n Ejemplo 1: Conversión individual")
    try:
        # Convertir el archivo principal de concesiones
        csv_path = os.path.join(csv_dir, "df_concesiones.csv")
        converter = DataConvert(csv_path)
        
        # Mostrar información del dataset
        info = converter.get_data_info()
        print(f" Dataset info:")
        print(f"   - Filas: {info['filas']:,}")
        print(f"   - Columnas: {info['columnas']}")
        print(f"   - Memoria: {info['memoria_mb']:.2f} MB")
        
        # Convertir a Parquet
        parquet_path = converter.convert_to_parquet()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    
    # Ejemplo 2: Convertir múltiples archivos
    print("\n Ejemplo 2: Conversión múltiple")
    try:
        # Convertir todos los CSV del directorio
        converted_files = DataConvert.convert_multiple_csvs(csv_dir)
        
        print(f"\n Archivos convertidos:")
        for file_path in converted_files:
            print(f"    {os.path.basename(file_path)}")
            
    except Exception as e:
        print(f" Error en conversión múltiple: {e}")
    
    print("\n Proceso completado!")

if __name__ == "__main__":
    main()
