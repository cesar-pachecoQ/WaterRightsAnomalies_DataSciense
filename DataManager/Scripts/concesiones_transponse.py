#!/usr/bin/env python3
# transpose_parquets.py
import os
from pathlib import Path
import polars as pl
import gc
import tempfile
import shutil
import sys
import time
import logging

# Configurar logging con más detalle
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Para que salga por stdout
        logging.FileHandler('transpose_log.txt')  # También guardar en archivo
    ]
)

logger = logging.getLogger(__name__)

# Opcional: limita arenas de glibc (mejor comportamiento de RAM en Linux)
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
# Opcional: limita hilos si lo deseas (no reduce RAM, pero estabiliza carga)
os.environ.setdefault("POLARS_MAX_THREADS", "8")

pl.enable_string_cache()

DATA = {
    'concesiones'  : Path('/home/cesar_r/Documentos/Proyectos/IberoSocialData/WaterRightsAnomalies_DataSciense/DataSets_CSVs/DataSets_Parquets/df_concesiones.parquet'),
    'subterraneas' : Path('/home/cesar_r/Documentos/Proyectos/IberoSocialData/WaterRightsAnomalies_DataSciense/DataSets_CSVs/DataSets_Parquets/df_subterraneas.parquet'),
    'superficiales': Path('/home/cesar_r/Documentos/Proyectos/IberoSocialData/WaterRightsAnomalies_DataSciense/DataSets_CSVs/DataSets_Parquets/df_superficiales.parquet'),
}

OUT_NAMES = {
    'concesiones'  : 'df_concesiones_transposed.parquet',
    'subterraneas' : 'df_subterraneas_transposed.parquet',
    'superficiales': 'df_superficiales_transposed.parquet',
}

# Configuración de compresión Zstd
ZSTD_LEVEL = 6  # 1-9 suele ser razonable; >9 sube mucho el costo CPU por poca ganancia

def human(n):
    for u in ("B","KiB","MiB","GiB","TiB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PiB"

def transpose_one(key: str, in_path: Path, out_name: str):
    t0 = time.time()
    if not in_path.exists():
        logger.error(f"[{key}] No existe el archivo: {in_path}")
        return

    out_path = in_path.parent / out_name
    logger.info(f"[{key}] Iniciando procesamiento de: {in_path}")
    
    try:
        # Lee el parquet completo (necesario para transponer)
        logger.info(f"[{key}] Leyendo archivo parquet...")
        df = pl.read_parquet(in_path)
        
        # Recompacta para reducir fragmentación antes de transponer
        df = df.rechunk()
        logger.info(f"[{key}] Shape original: {df.height:,} filas × {df.width:,} cols")

        # La transposición en Polars
        logger.info(f"[{key}] Iniciando transposición (esto puede requerir mucha RAM)...")
        df_t = df.transpose(include_header=True)
        logger.info(f"[{key}] Transposición completada. Shape transpuesto: {df_t.height:,} filas × {df_t.width:,} cols")

        # Liberar memoria del dataframe original
        del df
        gc.collect()
        logger.debug(f"[{key}] Memoria del dataframe original liberada")

        # Escribe con archivo temporal y rename atómico
        tmpdir = tempfile.mkdtemp(prefix=f"transpose_{key}_")
        tmp_out = Path(tmpdir) / (out_name + ".tmp")
        
        try:
            logger.info(f"[{key}] Escribiendo archivo transpuesto a: {out_path}")
            df_t.write_parquet(
                tmp_out,
                compression="zstd",
                statistics=True,
                compression_level=ZSTD_LEVEL,
                use_pyarrow=False,   # Polars nativo suele ir bien
            )
            # Movimiento atómico al destino final
            shutil.move(str(tmp_out), str(out_path))
            elapsed_time = time.time() - t0
            logger.info(f"[{key}] ✅ Procesamiento completado exitosamente. Archivo guardado en: {out_path} (Tiempo: {elapsed_time:.1f}s)")
            
        finally:
            # Limpieza del directorio temporal
            try:
                if tmp_out.exists():
                    tmp_out.unlink(missing_ok=True)
                Path(tmpdir).rmdir()
                logger.debug(f"[{key}] Archivos temporales limpiados")
            except Exception as cleanup_error:
                logger.warning(f"[{key}] Error limpiando archivos temporales: {cleanup_error}")

    except Exception as e:
        logger.error(f"[{key}] Error durante el procesamiento: {str(e)}")
        raise
    finally:
        # Libera memoria agresivamente
        try:
            del df_t
        except:
            pass
        gc.collect()
        logger.debug(f"[{key}] Memoria liberada")

def main():
    logger.info("=== Iniciando proceso de transposición de archivos parquet ===")
    
    # Validación rápida de espacio libre
    target_dirs = {p.parent for p in DATA.values()}
    for d in target_dirs:
        try:
            st = os.statvfs(d)
            free = st.f_bavail * st.f_frsize
            logger.info(f"Espacio libre en {d}: {human(free)}")
        except Exception as e:
            logger.warning(f"No se pudo verificar espacio libre en {d}: {e}")

    # Procesar cada archivo
    total_files = len(DATA)
    for i, (key, in_path) in enumerate(DATA.items(), 1):
        logger.info(f"Procesando archivo {i}/{total_files}: {key}")
        
        try:
            transpose_one(key, in_path, OUT_NAMES[key])
        except MemoryError:
            logger.error(f"[{key}] ❌ MemoryError: No hay suficiente RAM para transponer este archivo. Considera alternativas.")
        except Exception as e:
            logger.error(f"[{key}] ❌ Error inesperado: {e}")
        finally:
            gc.collect()
            logger.debug(f"Limpieza de memoria después de procesar {key}")

    logger.info("=== Proceso de transposición completado ===")

if __name__ == "__main__":
    main()