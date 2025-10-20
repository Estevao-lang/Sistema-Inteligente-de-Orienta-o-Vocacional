# model_utils_simple.py
from typing import Dict, List

WEIGHTS = {
    "Ciencia de Dados": {"matematica":0.32,"logica":0.32,"ciencias":0.16,"criatividade":0.10,"portugues":0.10},
    "Desenvolvimento":  {"logica":0.36,"matematica":0.26,"criatividade":0.18,"portugues":0.10,"interpessoal":0.10},
    "Enfermagem":       {"ciencias":0.36,"interpessoal":0.26,"portugues":0.16,"criatividade":0.12,"matematica":0.10},
    "Exatas":           {"matematica":0.52,"logica":0.30,"ciencias":0.18},
    "Humanas":          {"portugues":0.42,"interpessoal":0.30,"criatividade":0.18,"logica":0.10},
}
AREAS: List[str] = list(WEIGHTS.keys())

def _score(skills: Dict[str, float]) -> Dict[str, float]:
    raw = {}
    for area, w in WEIGHTS.items():
        s = 0.0
        for k, alpha in w.items():
            s += (float(skills.get(k, 0)) / 10.0) * alpha
        raw[area] = s
    m = max(raw.values()) if raw else 1.0
    return {a: (v / m if m > 0 else 0.0) for a, v in raw.items()}

def train_knn_from_excel(excel_path: str) -> dict:
    return {
        "k": None, "cv_metric": None, "cv_best": None,
        "n_samples": None, "model_path": None, "classes": AREAS,
        "note": "Versão simples: sem treino, usa regras por pesos."
    }

def predict_from_answers(answers_dict: dict, **_) -> dict:
    skills = answers_dict.get("skills") or {}
    scores = _score(skills)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    curso, conf = ordered[0]
    topk = [(a, round(p, 3)) for a, p in ordered[:3]]
    faixa = "Alta" if conf >= 0.70 else "Média" if conf >= 0.50 else "Baixa"
    return {"curso_recomendado": curso, "confianca": round(conf, 3), "faixa_confianca": faixa, "topk": topk}

def batch_predict_from_excel(excel_path: str, **_) -> "object":
    raise NotImplementedError("Versão simples não processa lote.")
