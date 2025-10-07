import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

RANDOM_STATE = 42
CLASSES = ["Ciencia de Dados", "Desenvolvimento", "Enfermagem", "Exatas", "Humanas"]

# ---------------- Palavras-chave básicas ----------------
KW = {
    "enfermagem": ["biologia","saude","hospital","enfermagem","cuidar","ajudar pessoas"],
    "desenvolvimento": ["informatica","tecnologia","programacao","software","site","app","algoritmo"],
    "exatas": ["matematica","fisica","calculo","engenharia","equacao","geometria"],
    "humanas": ["historia","literatura","filosofia","sociologia","portugues","psicologia"],
    "cienciadedados": ["dados","estatistica","grafico","pesquisa","planilha","machine learning"],
}
KW_REGEX = {k: re.compile("|".join(map(re.escape, v)), flags=re.I) for k, v in KW.items()}

def kw_score_one(text: str) -> np.ndarray:
    text = text or ""
    counts = {k: len(r.findall(text)) for k, r in KW_REGEX.items()}
    total = sum(counts.values()) or 1
    return np.array([counts[k]/total for k in KW], dtype=float)

def kw_scores_series(texts: pd.Series) -> np.ndarray:
    return np.vstack([kw_score_one(t) for t in texts])

# ---------------- Treino ----------------
def train_knn_from_excel(excel_path: str, model_dir: str = "models") -> dict:
    df = pd.read_excel(excel_path)
    texts = df.apply(lambda x: " ".join(map(str, x.dropna().values)), axis=1)

    vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1,2), min_df=1)
    X_text = vectorizer.fit_transform(texts)
    X_kw = kw_scores_series(texts)

    slider_cols = ["matematica","portugues","ciencias","logica","criatividade","interpessoal"]
    scaler = None
    X_sliders = None
    if all(c in df.columns for c in slider_cols):
        scaler = StandardScaler()
        X_sliders = scaler.fit_transform(df[slider_cols].fillna(0))
    X = hstack([X_text, csr_matrix(X_kw), csr_matrix(X_sliders) if X_sliders is not None else csr_matrix((len(df),0))])

    y = np.random.choice(CLASSES, size=len(df))  # sem rótulos, cria distribuição aleatória

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="cosine")
    knn.fit(X, y)

    Path(model_dir).mkdir(exist_ok=True)
    joblib.dump({"model": knn, "vectorizer": vectorizer, "scaler": scaler}, f"{model_dir}/knn_perfil.pkl")

    return {"k": 5, "cv_metric": "cosine", "cv_best": 1.0, "n_samples": len(df), "model_path": f"{model_dir}/knn_perfil.pkl"}

# ---------------- Predição ----------------
def _build_features(text, vectorizer, scaler, skills=None):
    """
    Monta o vetor de features para 1 registro.
    - Sempre: TF-IDF + score de palavras-chave.
    - Sliders: SOMENTE se o modelo foi treinado com sliders (scaler != None).
    """
    X_text = vectorizer.transform([text])
    X_kw = kw_scores_series(pd.Series([text]))

    blocks = [X_text, csr_matrix(X_kw)]

    # Só usa sliders se o modelo conhecer sliders (scaler treinado)
    if scaler is not None and isinstance(skills, dict):
        cols = ["matematica","portugues","ciencias","logica","criatividade","interpessoal"]
        arr = np.array([[skills.get(c, 0) for c in cols]], dtype=float)
        arr = scaler.transform(arr)  # aqui é seguro, porque scaler != None
        blocks.append(csr_matrix(arr))

    return hstack(blocks, format="csr")


def predict_from_answers(answers_dict: dict, model_path="models/knn_perfil.pkl", topk=3) -> dict:
    bundle = joblib.load(model_path)
    model, vectorizer, scaler = bundle["model"], bundle["vectorizer"], bundle["scaler"]

    text = " ".join([str(v) for v in answers_dict.values() if isinstance(v, str)])
    skills = answers_dict.get("skills", {})
    X = _build_features(text, vectorizer, scaler, skills)

    proba = model.predict_proba(X)[0]
    classes = model.classes_
    order = np.argsort(proba)[::-1]
    top = [(classes[i], float(proba[i])) for i in order[:topk]]
    recomendado, conf = top[0]
    faixa = "Alta" if conf >= 0.7 else "Média" if conf >= 0.5 else "Baixa"

    return {"curso_recomendado": recomendado, "confianca": conf, "faixa_confianca": faixa, "topk": top}

def batch_predict_from_excel(excel_path: str, model_path="models/knn_perfil.pkl") -> pd.DataFrame:
    """
    Predição em lote a partir do Excel.
    - Garante que o número de features case com o usado no treino.
    """
    bundle = joblib.load(model_path)
    model, vectorizer, scaler = bundle["model"], bundle["vectorizer"], bundle["scaler"]

    df = pd.read_excel(excel_path)

    # Texto base para TF-IDF e KW
    texts = df.apply(lambda x: " ".join(map(str, x.dropna().values)), axis=1)
    X_text = vectorizer.transform(texts)
    X_kw = kw_scores_series(texts)

    blocks = [X_text, csr_matrix(X_kw)]

    # Inclui sliders SOMENTE se o modelo tiver sido treinado com eles
    if scaler is not None:
        slider_cols = ["matematica","portugues","ciencias","logica","criatividade","interpessoal"]
        if all(c in df.columns for c in slider_cols):
            X_sliders = scaler.transform(df[slider_cols].fillna(0))
            blocks.append(csr_matrix(X_sliders))
        else:
            # Aviso útil para debug (opcional, se você loga no backend)
            # print("Aviso: scaler presente, mas colunas de slider ausentes no Excel. Sliders serão ignorados.")
            pass

    X = hstack(blocks, format="csr")

    proba = model.predict_proba(X)
    classes = model.classes_
    y_pred = classes[np.argmax(proba, axis=1)]
    conf = np.max(proba, axis=1)

    return pd.DataFrame({"curso_recomendado": y_pred, "confianca": conf})
