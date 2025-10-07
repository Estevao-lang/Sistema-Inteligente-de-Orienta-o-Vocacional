import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename

from model_utils import (
    train_knn_from_excel,
    predict_from_answers,
    batch_predict_from_excel,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")
BASE_DIR = Path(__file__).parent.resolve()

DEFAULT_EXCEL = os.environ.get("EXCEL_PATH", str(BASE_DIR / "data" / "converter.xlsx"))
EXPORT_DIR = BASE_DIR / "exports"; EXPORT_DIR.mkdir(exist_ok=True)
MODELS_DIR = BASE_DIR / "models";  MODELS_DIR.mkdir(exist_ok=True)

# -------- Páginas ----------
@app.get("/")
def home():
    return render_template("index.html", default_excel=DEFAULT_EXCEL)

@app.get("/predict")
def predict_page():
    return render_template("predict.html")

@app.get("/batch")
def batch_page():
    return render_template("batch.html", current_excel=DEFAULT_EXCEL)

# -------- APIs ----------
@app.get("/health")
def health():
    return jsonify({"ok": True, "message": "KNN Perfil API is running."})

@app.post("/api/train")
def api_train():
    data = request.get_json(silent=True) or {}
    excel_path = data.get("excel_path", DEFAULT_EXCEL)
    if not Path(excel_path).exists():
        return jsonify({"ok": False, "error": f"Excel não encontrado: {excel_path}"}), 400
    summary = train_knn_from_excel(excel_path)
    return jsonify({"ok": True, "summary": summary})

@app.post("/api/batch_predict")
def api_batch_predict():
    data = request.get_json(silent=True) or {}
    excel_path = data.get("excel_path", DEFAULT_EXCEL)
    if not Path(excel_path).exists():
        return jsonify({"ok": False, "error": f"Excel não encontrado: {excel_path}"}), 400
    try:
        df = batch_predict_from_excel(excel_path)
        out_path = EXPORT_DIR / "recomendacoes_alunos.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        return send_file(out_path, as_attachment=True)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# -------- Form de previsão (single) ----------
@app.post("/predict")
def predict_submit():
    nome = request.form.get("nome", "").strip()

    def to_int(k, default=5):
        try:
            return int(float(request.form.get(k, default)))
        except Exception:
            return default

    skills = {
        "matematica":   to_int("matematica"),
        "portugues":    to_int("portugues"),
        "ciencias":     to_int("ciencias"),
        "logica":       to_int("logica"),
        "criatividade": to_int("criatividade"),
        "interpessoal": to_int("interpessoal"),
    }

    # se existe pipeline ML no disco, usa; se não, tenta treinar com DEFAULT_EXCEL
    model_path = MODELS_DIR / "knn_perfil.pkl"
    if not model_path.exists():
        try:
            train_knn_from_excel(DEFAULT_EXCEL, model_dir=str(MODELS_DIR))
            flash("Modelo treinado automaticamente a partir do Excel padrão.", "success")
        except Exception as e:
            flash(f"Falha ao treinar automaticamente: {e}", "error")
            return redirect(url_for("batch_page"))

    respostas = {"texto": f"Nome: {nome}", "skills": skills}

    try:
        result = predict_from_answers(respostas, model_path=str(model_path))
        return render_template("predict.html",
                               nome=nome or "estudante",
                               skills=skills,
                               resultado=result)
    except FileNotFoundError:
        flash("Modelo não encontrado. Vá em Lote & Treino → Treinar.", "error")
        return redirect(url_for("batch_page"))
    except Exception as e:
        flash(f"Erro ao prever: {e}", "error")
        return redirect(url_for("predict_page"))

# -------- Batch (UI) ----------
@app.post("/batch/train")
def batch_train():
    # 1) upload de arquivo, se houver
    file = request.files.get("excel_file")
    excel_path = None
    if file and file.filename:
        fname = secure_filename(file.filename)
        save_path = EXPORT_DIR / f"upload_train_{fname}"
        file.save(save_path)
        excel_path = str(save_path)

    # 2) fallback para caminho de texto ou DEFAULT
    if not excel_path:
        excel_path = request.form.get("excel_path") or DEFAULT_EXCEL

    if not Path(excel_path).exists():
        flash(f"Excel não encontrado: {excel_path}", "error")
        return redirect(url_for("batch_page"))

    summary = train_knn_from_excel(excel_path)
    flash("Treino concluído.", "success")
    return render_template("batch.html", current_excel=excel_path, summary=summary)

@app.post("/batch/predict")
def batch_predict_page():
    # 1) upload de arquivo, se houver
    file = request.files.get("excel_file")
    excel_path = None
    if file and file.filename:
        fname = secure_filename(file.filename)
        save_path = EXPORT_DIR / f"upload_batch_{fname}"
        file.save(save_path)
        excel_path = str(save_path)

    # 2) fallback para caminho de texto ou DEFAULT
    if not excel_path:
        excel_path = request.form.get("excel_path") or DEFAULT_EXCEL

    if not Path(excel_path).exists():
        flash(f"Excel não encontrado: {excel_path}", "error")
        return redirect(url_for("batch_page"))

    # 3) gera CSV (se a sua versão de model_utils não suportar lote, mostra aviso)
    try:
        df = batch_predict_from_excel(excel_path)
    except NotImplementedError:
        flash("Versão simples: processamento em lote não está disponível.", "error")
        return redirect(url_for("batch_page"))

    out_path = EXPORT_DIR / "recomendacoes_alunos.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    flash("Arquivo gerado com sucesso.", "success")
    return send_file(out_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
