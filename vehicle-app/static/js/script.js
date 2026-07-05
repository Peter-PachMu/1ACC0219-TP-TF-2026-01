const uploadZone = document.getElementById("upload-zone");
const uploadInner = document.getElementById("upload-inner");
const fileInput = document.getElementById("file-input");
const previewImg = document.getElementById("preview-img");
const analyzeBtn = document.getElementById("analyze-btn");
const resetBtn = document.getElementById("reset-btn");
const loading = document.getElementById("loading");
const errorBanner = document.getElementById("error-banner");
const results = document.getElementById("results");

let selectedFile = null;

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.hidden = false;
}

function clearError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Por favor selecciona un archivo de imagen válido.");
    return;
  }
  clearError();
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.hidden = false;
    uploadInner.hidden = true;
  };
  reader.readAsDataURL(file);

  analyzeBtn.disabled = false;
  results.hidden = true;
}

uploadZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) setFile(e.target.files[0]);
});

["dragenter", "dragover"].forEach((evt) =>
  uploadZone.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  })
);

["dragleave", "drop"].forEach((evt) =>
  uploadZone.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
  })
);

uploadZone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});

resetBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  previewImg.hidden = true;
  uploadInner.hidden = false;
  analyzeBtn.disabled = true;
  resetBtn.hidden = true;
  results.hidden = true;
  clearError();
});

function renderDetections(container, detections, kind) {
  container.innerHTML = "";

  if (!detections.length) {
    const p = document.createElement("p");
    p.className = "empty-note";
    p.textContent = "No se detectó ningún vehículo en la imagen.";
    container.appendChild(p);
    return;
  }

  detections.forEach((d, idx) => {
    const row = document.createElement("div");
    row.className = "detection-row" + (idx === 0 ? " top" : "");

    const label = document.createElement("div");
    label.className = "det-label";
    label.textContent = d.class;

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${d.confidence * 100}%`;
    track.appendChild(fill);

    const value = document.createElement("div");
    value.className = "conf-value";
    value.textContent = `${(d.confidence * 100).toFixed(1)}%`;

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    container.appendChild(row);
  });
}

function renderProbs(container, predictions) {
  container.innerHTML = "";
  predictions.forEach((p, idx) => {
    const row = document.createElement("div");
    row.className = "prob-row" + (idx === 0 ? " top" : "");

    const label = document.createElement("div");
    label.className = "prob-label";
    label.textContent = p.class;

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${p.confidence * 100}%`;
    track.appendChild(fill);

    const value = document.createElement("div");
    value.className = "conf-value";
    value.textContent = `${(p.confidence * 100).toFixed(1)}%`;

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    container.appendChild(row);
  });
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  clearError();
  results.hidden = true;
  loading.hidden = false;
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Error desconocido al analizar la imagen.");
    }

    // --- YOLO ---
    const yoloImage = document.getElementById("yolo-image");
    const yoloDetections = document.getElementById("yolo-detections");
    const yoloTag = document.getElementById("yolo-count-tag");

    if (data.yolo.available) {
      yoloImage.src = data.yolo.annotated_image;
      yoloImage.hidden = false;
      yoloTag.textContent = `${data.yolo.num_detections} detección(es)`;
      renderDetections(yoloDetections, data.yolo.detections, "yolo");
    } else {
      yoloImage.hidden = true;
      yoloTag.textContent = "no disponible";
      yoloDetections.innerHTML = `<p class="empty-note">${data.yolo.error}</p>`;
    }

    // --- ViT ---
    const vitImage = document.getElementById("vit-image");
    const vitProbs = document.getElementById("vit-probs");
    const vitTag = document.getElementById("vit-top-tag");

    if (data.vit.available) {
      vitImage.src = data.original_image;
      vitImage.hidden = false;
      vitTag.textContent = `${data.vit.top_class} (${(data.vit.top_confidence * 100).toFixed(1)}%)`;
      renderProbs(vitProbs, data.vit.predictions);
    } else {
      vitImage.hidden = true;
      vitTag.textContent = "no disponible";
      vitProbs.innerHTML = `<p class="empty-note">${data.vit.error}</p>`;
    }

    results.hidden = false;
    resetBtn.hidden = false;
  } catch (err) {
    showError(err.message || "No se pudo completar el análisis.");
    analyzeBtn.disabled = false;
  } finally {
    loading.hidden = true;
  }
});
