import polars as pl
import re
from rapidfuzz import fuzz
import difflib
import jellyfish
import math
import ManipulateData


class MetodoLexico:
    def __init__(self, a: str, b: str):
        self.a = a or ""
        self.b = b or ""
        self.a_std = None
        self.b_std = None

        # Intento 1: versión de ManipulateData que acepta (df, a, b)
        try:
            self.cleaner_data = ManipulateData.ManipulateData(df=pl.DataFrame(), a=self.a, b=self.b)
            self._estandarizar_pair = lambda **kwargs: self.cleaner_data.estandarizar_dos_strings(**kwargs)
        except TypeError:
            # Intento 2: versión "clásica" que sólo acepta (df)
            self.cleaner_data = ManipulateData.ManipulateData(pl.DataFrame())
            def _fallback(**kwargs):
                # usamos el método interno por cada string, sin tocar los originales
                a_std = self.cleaner_data._estandarizar_texto(self.a, **kwargs)
                b_std = self.cleaner_data._estandarizar_texto(self.b, **kwargs)
                return a_std, b_std
            self._estandarizar_pair = _fallback

    def normalization_and_soft_standardization(self):
        a_std, b_std = self._estandarizar_pair(
            sin_espacios=False,
            sin_stopword=True,
            sin_terminos_mercantiles=True,
            quitar_tokens_1_letra=True
        )
        self.a_std = a_std or ""
        self.b_std = b_std or ""
        return self.a_std, self.b_std

    # ---------------------------
    # 2) Prefiltro de longitud
    # ---------------------------
    def length_prefilter(self, base_abs: int = 5, rel_long: float = 0.5) -> bool:
        if self.a_std is None or self.b_std is None:
            self.normalization_and_soft_standardization()
        len_a = len(self.a_std or "")
        len_b = len(self.b_std or "")
        lmax = max(len_a, len_b)
        if lmax <= 3:
            return False
        abs_thr = max(base_abs, int(round(0.20 * lmax)))
        rel_thr = 0.40 if lmax < 10 else rel_long
        gap_abs = abs(len_a - len_b)
        gap_rel = gap_abs / lmax
        return (gap_abs > abs_thr) and (gap_rel > rel_thr)

    # ---------------------------
    # 3) Métricas léxicas
    # ---------------------------
    def _tokens(self, s: str):
        return [t for t in re.split(r"\s+", s.strip()) if t]

    def jaccard_similarity(self) -> float:
        """
        Jaccard sobre tokens de las versiones estandarizadas 'suaves'.
        Regresa porcentaje [0,100].
        """
        if self.a_std is None or self.b_std is None:
            self.normalization_and_soft_standardization()

        A, B = set(self._tokens(self.a_std)), set(self._tokens(self.b_std))
        if not A and not B:
            return 0.0
        inter = len(A & B)
        union = len(A | B) if (A or B) else 1
        return 100.0 * inter / union

    def _qgram_vec(self, s: str, q: int = 3):
        if not s:
            return {}
        s = s.upper()
        grams = [s[i:i+q] for i in range(max(0, len(s)-q+1))]
        # cuenta por gram
        d = {}
        for g in grams:
            d[g] = d.get(g, 0) + 1
        return d

    def _cosine_counts(self, a: dict, b: dict):
        if not a or not b:
            return 0.0
        keys = set(a) | set(b)
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
        na = math.sqrt(sum(v*v for v in a.values()))
        nb = math.sqrt(sum(v*v for v in b.values()))
        return 0.0 if na == 0 or nb == 0 else (dot / (na * nb))

    def cosine_qgrams(self, q: int = 3) -> float:
        """
        Coseno sobre q-gramas de caracteres usando las versiones estandarizadas 'suaves'.
        Regresa porcentaje [0,100].
        """
        if self.a_std is None or self.b_std is None:
            self.normalization_and_soft_standardization()

        va = self._qgram_vec(self.a_std, q=q)
        vb = self._qgram_vec(self.b_std, q=q)
        return 100.0 * self._cosine_counts(va, vb)

    def jaro_winkler_distance(self) -> float:
        # Mantener sobre los ORIGINALES (sensibles a minúsculas/acentos/puntuación)
        return jellyfish.jaro_winkler_similarity(self.a, self.b) * 100.0

    def token_set_ratio(self) -> float:
        # También sobre originales, como referencia cercana a tu flujo actual
        return fuzz.token_set_ratio(self.a, self.b)

    # ---------------------------
    # 4) Score y clase
    # ---------------------------
    def score_y_clase(self):
        """
        Reglas conservadoras:
          - Mismo titular (clase=1) si:
              (JW ≥ 93 y TokenSet/Sort ≥ 93)  ó
              (TokenSet ≥ 95 y CosQgrams ≥ 92) ó
              (JW ≥ 90 y Jaccard ≥ 75)
          - Distinto (clase=0) si:
              (JW < 85 y TokenSet < 85)
          - En otro caso: indeterminado (clase=2)
        Score (0-100): 40% JW + 40% TokenSet + 20% CosQgrams
        Si clase==1 -> calcular lista de caracteres conflictivos; si clase==0 -> lista vacía.
        """
        # 0) Prefiltro por longitud (NO modificar según tu petición)
        if self.length_prefilter():
            return 0.0, 0, []  # score, clase, conflictos

        # 1) Asegurar versiones estandarizadas suaves
        if self.a_std is None or self.b_std is None:
            self.normalization_and_soft_standardization()

        # 2) Métricas
        jw = self.jaro_winkler_distance()          # 0..100
        ts = self.token_set_ratio()                # 0..100
        cq = self.cosine_qgrams(q=3)               # 0..100
        jac = self.jaccard_similarity()            # 0..100

        # 3) Reglas (conservadoras)
        same = ((jw >= 93 and ts >= 93) or
                (ts >= 95 and cq >= 92) or
                (jw >= 90 and jac >= 75))
        diff = (jw < 85 and ts < 85)

        if same:
            clase = 1
        elif diff:
            clase = 0
        else:
            clase = 2  # indeterminado, por si quieres usarlo más adelante

        # 4) Score agregado
        score = 0.4 * jw + 0.4 * ts + 0.2 * cq

        # 5) Caracteres conflictivos SOLO si son el mismo titular
        if clase == 1:
            list_conflicters = self.conflicting_characters()
        else:
            list_conflicters = []

        return score, clase, list_conflicters

    # ---------------------------
    # 5) Diferencias de caracteres
    # ---------------------------
    def conflicting_characters(self):
        """
        Devuelve lista de pares (char_from, char_to) usando opcodes:
          - replace: ('x','y')
          - delete:  ('x','∅')
          - insert:  ('∅','y')
        Se calcula sobre las CADENAS ORIGINALES (a,b).
        """
        a, b = self.a, self.b
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        pairs = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                continue
            a_sub = a[i1:i2]
            b_sub = b[j1:j2]
            if tag == 'replace':
                m = min(len(a_sub), len(b_sub))
                for k in range(m):
                    pairs.append((a_sub[k], b_sub[k]))
                for ch in a_sub[m:]:
                    pairs.append((ch, '∅'))
                for ch in b_sub[m:]:
                    pairs.append(('∅', ch))
            elif tag == 'delete':
                for ch in a_sub:
                    pairs.append((ch, '∅'))
            elif tag == 'insert':
                for ch in b_sub:
                    pairs.append(('∅', ch))
        # Únicos, conservando primer aparición
        seen, uniq = set(), []
        for p in pairs:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq
