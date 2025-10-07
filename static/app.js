// Só necessário na página inicial (index.html) se você usar os botões AJAX
document.addEventListener("DOMContentLoaded", () => {
  const trainForm = document.getElementById("train-form");
  const trainOut = document.getElementById("train-output");
  const batchForm = document.getElementById("batch-form");
  const batchOut = document.getElementById("batch-output");

  if (trainForm && trainOut) {
    trainForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      trainOut.textContent = "Treinando...";
      const formData = new FormData(trainForm);
      const excel_path = formData.get("excel_path");
      const res = await fetch("/api/train", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ excel_path })
      });
      const data = await res.json();
      trainOut.textContent = JSON.stringify(data, null, 2);
    });
  }

  if (batchForm && batchOut) {
    batchForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      batchOut.innerHTML = "Gerando...";
      const formData = new FormData(batchForm);
      const excel_path = formData.get("excel_path");
      const res = await fetch("/api/batch_predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ excel_path })
      });
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url; a.download = "recomendacoes_alunos.csv";
        document.body.appendChild(a); a.click(); a.remove();
        batchOut.innerHTML = "CSV baixado com sucesso.";
      } else {
        const err = await res.json().catch(() => ({}));
        batchOut.innerHTML = `<span class="text-rose-500">Erro: ${err.error || "Falha ao gerar CSV"}</span>`;
      }
    });
  }
});
