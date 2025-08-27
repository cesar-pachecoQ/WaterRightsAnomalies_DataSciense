#!/usr/bin/env python3
"""
concesiones_melt.py
-------------------
Convierte tablas "anchas" (wide) a formato "largo" (long) usando Polars de forma perezosa/streaming,
evitando materializar el dataset completo en RAM.

Caso típico: columnas ANOMALIA_* (value_vars) + columnas de identidad (id_vars como CAMPO, TEMA, etc.).

Características:
- Lee Parquet con scan_parquet() (Lazy).
- Selecciona sólo las columnas necesarias.
- Hace melt() y escribe el resultado a Parquet.
- Soporta procesamiento por BLOQUES de columnas (batch) para evitar picos de RAM si melt no es "streaming-friendly".
- Escribe un único archivo *_long.parquet si no hay batching; si hay batching, escribe partes en *_long_parts/part_XXX.parquet.

Ejemplos de uso:
1) Un archivo, sin batching (intenta sink):
   python concesiones_melt.py \
     --inputs DataSets_CSVs/DataSets_Parquets/df_concesiones.parquet \
     --value-regex '^ANOMALIA_.*$' \
     --id-cols CAMPO TEMA \
     --outdir DataSets_CSVs/DataSets_Parquets

2) Varios archivos, con batching de 200 columnas por bloque:
   python concesiones_melt.py \
     --inputs DataSets_CSVs/DataSets_Parquets/df_concesiones.parquet \
              DataSets_CSVs/DataSets_Parquets/df_subterraneas.parquet \
              DataSets_CSVs/DataSets_Parquets/df_superficiales.parquet \
     --value-regex '^ANOMALIA_.*$' \
     --id-cols CAMPO TEMA \
     --batch-size 200 \
     --outdir DataSets_CSVs/DataSets_Parquets

Notas:
- Asegúrate de tener polars instalado: `pip install polars` (>=0.20 recomendado).
- Si el patrón de columnas de "valor" no es ANOMALIA_*, ajusta --value-regex o usa --value-cols.
"""

import argparse
import os
import re
from pathlib import Path
from itertools import islice
import sys

try:
    import polars as pl
except Exception as e:
    print("ERROR: No se pudo importar polars. Instálalo con `pip install polars`.", file=sys.stderr)
    raise


def log(msg: str):
    print(f"[concesiones_melt] {msg}", flush=True)


def batched(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk


def infer_columns(schema_cols, id_cols_cli, value_cols_cli, value_regex):
    """
    Decide id_vars y value_vars a partir del esquema y parámetros del CLI.
    """
    schema_cols = list(schema_cols)

    # Si el usuario pasó value_cols explícitas, úsalas tal cual.
    if value_cols_cli:
        value_cols = [c for c in value_cols_cli if c in schema_cols]
    else:
        # Si hay regex, matchea
        value_cols = [c for c in schema_cols if re.search(value_regex, c)] if value_regex else []

    # id_cols: si vienen del CLI, úsalas; si no, las complementarias
    if id_cols_cli:
        id_cols = [c for c in id_cols_cli if c in schema_cols]
    else:
        id_cols = [c for c in schema_cols if c not in value_cols]

    return id_cols, value_cols


def melt_lazy_single(input_path: Path, outdir: Path, id_cols, value_cols, drop_nulls=False):
    """
    Intenta un solo melt en Lazy y escribir un único archivo *_long.parquet usando sink_parquet().
    Si no se puede (por falta de soporte en streaming), cae a collect(streaming=True).
    """
    base = input_path.stem
    out_path = outdir / f"{base}_long.parquet"
    log(f"Archivo: {input_path} → {out_path}")
    lf = pl.scan_parquet(str(input_path))

    # Selecciona sólo columnas necesarias
    cols = list(dict.fromkeys(list(id_cols) + list(value_cols)))
    lf = lf.select([pl.col(c) for c in cols])

    # Melt
    melted = lf.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        variable_name="variable",
        value_name="valor"
    )

    if drop_nulls:
        melted = melted.filter(pl.col("valor").is_not_null())

    # Intentar sink directo
    try:
        log("Intentando sink_parquet() (evita materializar en RAM)...")
        melted.sink_parquet(str(out_path), compression="zstd", statistics=True)
        log("OK sink_parquet()")
        return [out_path]
    except Exception as e:
        log(f"Advertencia: sink_parquet() no disponible/compatible aquí: {e}")
        log("Cayendo a collect(streaming=True) + write_parquet()... (puede usar más RAM)")
        df = melted.collect(streaming=True)
        df.write_parquet(str(out_path), compression="zstd", statistics=True)
        log("OK write_parquet()")
        return [out_path]


def melt_lazy_batched(input_path: Path, outdir: Path, id_cols, value_cols, batch_size, drop_nulls=False):
    """
    Procesa value_cols en bloques para evitar picos de RAM.
    Escribe partes en *_long_parts/part_XXX.parquet
    """
    base = input_path.stem
    parts_dir = outdir / f"{base}_long_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    log(f"Archivo: {input_path} → {parts_dir}/part_XXX.parquet (batch_size={batch_size})")

    lf = pl.scan_parquet(str(input_path))

    written = []
    for i, cols_batch in enumerate(batched(value_cols, batch_size)):
        log(f"  Lote {i}: {len(cols_batch)} columnas de valor")
        # sólo id + el batch actual
        lf_batch = lf.select([pl.col(c) for c in id_cols + cols_batch])
        melted = lf_batch.melt(
            id_vars=id_cols,
            value_vars=cols_batch,
            variable_name="variable",
            value_name="valor"
        )
        if drop_nulls:
            melted = melted.filter(pl.col("valor").is_not_null())

        part_path = parts_dir / f"part_{i:03}.parquet"
        # Intentar sink; si falla, collect(streaming=True)
        try:
            melted.sink_parquet(str(part_path), compression="zstd", statistics=True)
        except Exception:
            df = melted.collect(streaming=True)
            df.write_parquet(str(part_path), compression="zstd", statistics=True)
        written.append(part_path)

    log(f"Escritas {len(written)} partes.")
    return written


def main():
    parser = argparse.ArgumentParser(description="MELT wide→long de Parquet usando Polars Lazy/Streaming.")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Rutas a archivos Parquet de entrada.")
    parser.add_argument("--outdir", default=".", help="Directorio de salida (por defecto, actual).")
    parser.add_argument("--id-cols", nargs="*", default=None,
                        help="Columnas de identidad (id_vars). Si se omite, se infieren como 'todas menos value_vars'.")
    parser.add_argument("--value-cols", nargs="*", default=None,
                        help="Columnas de valor explícitas (value_vars). Si se omite, se usa --value-regex.")
    parser.add_argument("--value-regex", default="^ANOMALIA_.*$",
                        help="Regex para detectar columnas de valor (por defecto '^ANOMALIA_.*$'). Ignorado si se pasa --value-cols.")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Tamaño de lote para procesar columnas de valor por bloques. 0 = un solo melt.")
    parser.add_argument("--drop-nulls", action="store_true",
                        help="Si se pasa, descarta filas con valor nulo tras el melt.")

    args = parser.parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    any_written = []
    for in_path in args.inputs:
        p = Path(in_path).resolve()
        if not p.exists():
            log(f"ERROR: no existe {p}")
            continue

        # Leer esquema de forma perezosa y decidir columnas
        lf_schema = pl.scan_parquet(str(p)).schema
        schema_cols = list(lf_schema.keys())

        id_cols, value_cols = infer_columns(
            schema_cols=schema_cols,
            id_cols_cli=args.id_cols,
            value_cols_cli=args.value_cols,
            value_regex=args.value_regex
        )

        if not value_cols:
            log(f"Advertencia: no se detectaron columnas de valor en {p.name}. "
                f"Usa --value-cols ... o ajusta --value-regex.")
            # En este caso, escribiríamos una copia vacía (sin melt), mejor saltar:
            continue

        # Decide estrategia
        if args.batch_size and args.batch_size > 0:
            written = melt_lazy_batched(
                input_path=p,
                outdir=outdir,
                id_cols=id_cols,
                value_cols=value_cols,
                batch_size=args.batch_size,
                drop_nulls=args.drop_nulls
            )
        else:
            written = melt_lazy_single(
                input_path=p,
                outdir=outdir,
                id_cols=id_cols,
                value_cols=value_cols,
                drop_nulls=args.drop_nulls
            )

        any_written.extend(written)

    if not any_written:
        log("No se escribió ningún archivo. Revisa rutas y patrones.")
        sys.exit(1)

    log("Listo.")
    for w in any_written:
        print(str(w))

if __name__ == "__main__":
    main()
