from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, shutil
import cv2

# Configurações de upload
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"]
MAX_FILE_SIZE_MB = 5  # limite de 5MB para versão inicial

app = FastAPI()

# CORS para desenvolvimento local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Validação de extensão
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Arquivo inválido. Use PNG ou JPG.")

    # Verificação de tamanho
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Arquivo muito grande (máx. {MAX_FILE_SIZE_MB}MB).")

    # Salvar arquivo de entrada
    input_filename = f"input_{uuid.uuid4()}{ext}"
    with open(input_filename, "wb") as f:
        f.write(contents)

    # Geração de mapa de saliência (heatmap) com OpenCV
    image = cv2.imread(input_filename)
    if image is None:
        os.remove(input_filename)
        raise HTTPException(status_code=500, detail="Erro ao ler a imagem.")

    saliency_obj = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliencyMap = saliency_obj.computeSaliency(image)
    if not success:
        os.remove(input_filename)
        raise HTTPException(status_code=500, detail="Falha ao computar saliency map.")

    # Normalizar e aplicar colormap
    saliencyMap = (saliencyMap * 255).astype("uint8")
    heatmap = cv2.applyColorMap(saliencyMap, cv2.COLORMAP_JET)

    output_filename = "heatmap_output.png"
    cv2.imwrite(output_filename, heatmap)

    # Cleanup temporário
    os.remove(input_filename)

    # Retornar heatmap gerado
    return FileResponse(output_filename, media_type="image/png")
