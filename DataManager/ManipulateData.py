import unicodedata
import re
import polars as pl

# --- Catálogos (ajústalos a tu gusto) ---

STOPWORDS = {
    # conectores comunes
    "DE","DEL","LA","EL","LOS","LAS","Y","E","AL","A","EN","POR","PARA","CON",
}

# Términos mercantiles (razones sociales) e institucionales frecuentes en México
MERCANTILES_O_INST = {
    # Razones sociales y variantes
    "SA","SAPI","CV","RL","SRL","SC","AC","SCL","SNC","SPR",
    "SOCIEDAD","ANONIMA","RESPONSABILIDAD","LIMITADA","PROMOTORA","INVERSION",
    "COOPERATIVA","COOP","CAPITAL","VARIABLE",
    # Institucional / administración pública / ejidal
    "H","AYUNTAMIENTO","MUNICIPAL","MUNICIPIO","LOC","EJIDO","COMUNIDAD","COLONIA",
    "DELEGACION","ALCALDIA",
}

class ManipulateData:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def _estandarizar_texto(self,
                            texto: str,
                            sin_espacios: bool = False,
                            sin_stopword: bool = False,
                            sin_terminos_mercantiles: bool = False,
                            quitar_tokens_1_letra: bool = False) -> str:
        """Estandariza un texto individual (titular). Devuelve str o None."""
        # 0) Manejo básico
        if texto is None or not isinstance(texto, str):
            return None

        # 1) Normalizar y quitar acentos/diacríticos (incluye ñ->n)
        texto = unicodedata.normalize('NFD', texto)
        texto = texto.encode('ascii', 'ignore').decode('utf-8')

        # 2) Sustituciones simples
        texto = texto.replace("Ø", "O").replace("ø", "O")
        texto = texto.replace("&", " Y ")

        # 3) Eliminar signos de puntuación (dejamos letras, números y espacios)
        #    Si prefieres ser más agresivo: r"[^A-Za-z0-9\s]"
        texto = re.sub(r'[.,()"\'()]', ' ', texto)

        # 4) Quitar espacios extra
        texto = re.sub(r'\s+', ' ', texto).strip()

        # 5) Mayúsculas
        texto = texto.upper()

        # 6) Filtrado por tokens
        toks = texto.split()

        if sin_terminos_mercantiles:
            toks = [t for t in toks if t not in MERCANTILES_O_INST]

        if sin_stopword:
            toks = [t for t in toks if t not in STOPWORDS]

        if quitar_tokens_1_letra:
            # Quita tokens de 1 letra (suelen ser residuos de S.A. de C.V.: S, A, C, V, etc.)
            toks = [t for t in toks if len(t) > 1]

        texto = " ".join(toks)

        # 7) Opcional: eliminar espacios por completo (para comparaciones más estrictas)
        if sin_espacios:
            texto = texto.replace(" ", "")

        return texto

    def estandarizar_titular(self,
                         columna: str,
                         sin_espacios: bool = False,
                         sin_stopword: bool = False,
                         sin_terminos_mercantiles: bool = False,
                         quitar_tokens_1_letra: bool = False) -> pl.DataFrame:
        """Aplica estandarización a una columna del DataFrame."""
        self.df = self.df.with_columns(
            pl.col(columna).map_elements(
                lambda x: self._estandarizar_texto(
                    x,
                    sin_espacios=sin_espacios,
                    sin_stopword=sin_stopword,
                    sin_terminos_mercantiles=sin_terminos_mercantiles,
                    quitar_tokens_1_letra=quitar_tokens_1_letra
                ),
                return_dtype=pl.Utf8
            ).alias(columna)  
        )
        return self.df