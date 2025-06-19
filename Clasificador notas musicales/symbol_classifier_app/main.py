from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil, os, joblib
from deepv1 import predecir_y_comparar
from tensorflow.keras.models import load_model

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

#model_cnn = load_model("models/symbol_classifier_cnn.h5")
#model_rf = joblib.load("models/symbol_classifier_rf.pkl")
#encoder = joblib.load("models/label_encoder.pkl")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Para pruebas, usa predicciones simuladas
    pred_cnn = "Ejemplo_CNN"
    pred_rf = "Ejemplo_RF"
    logs = """
    Precisión CNN: 0.88<br>
    Precisión Random Forest: 0.85<br>
    Diferencia: 0.03<br>
    """

    return templates.TemplateResponse("resultados.html", {
        "request": request,
        "pred_cnn": pred_cnn,
        "pred_rf": pred_rf,
        "img_path": f"/uploads/{file.filename}",
        "logs": logs
    })


app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
@app.get("/uploads/{filename}", response_class=HTMLResponse)
def get_uploaded_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return HTMLResponse(content=f'<img src="/uploads/{filename}" alt="{filename}">')
    else:
        return HTMLResponse(content="File not found", status_code=404)          