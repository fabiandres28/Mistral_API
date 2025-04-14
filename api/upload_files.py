from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from typing import List
import uvicorn
import os
import base64
import json
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from mistralai import Mistral, DocumentURLChunk, ImageURLChunk, TextChunk
from mistralai.models import OCRResponse
from pydantic import BaseModel


load_dotenv()
api_key = os.getenv('api_key')
client = Mistral(api_key=api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """
    Reemplaza referencias como ![img-0.jpeg](img-0.jpeg)
    con data:image/jpeg;base64,....
    """

    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
        )

    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """
    pdf_response: objeto devuelto por el OCR de Mistral
    que contiene pages[i].markdown y pages[i].images.
    """

    markdowns: list[str] = []
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
            print(f"Imagen ID: {img.id} - Base64: {img.image_base64[:30]}...")  # Solo muestra los primeros 30 caracteres
        # Replace image placeholders with actual images
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))

    # Unimos todas las páginas con saltos de línea dobles
    return "\n\n".join(markdowns)


@app.post("/upload_files")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Procesa una lista de archivos subidos:
      - Si es PDF, utiliza la lógica de DocumentURLChunk (OCR de PDF).
      - Si es imagen (jpg/jpeg/png), utiliza la lógica de ImageURLChunk.
      - Devuelve un JSON con las respuestas individuales.
    """
    results = []

    for file in files:
        file_content = await file.read()        
        extension = os.path.splitext(file.filename)[1].lower()

        if extension == ".pdf":
            uploaded_file = client.files.upload(
                file={
                    "file_name": file.filename,
                    "content": file_content, 
                },
                purpose="ocr",
            )
            signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
            pdf_response = client.ocr.process(
                document=DocumentURLChunk(document_url=str(signed_url.url)),
                model="mistral-ocr-latest",
                include_image_base64=True
            )
            merged_markdown = get_combined_markdown(pdf_response)
            pdf_dict = json.loads(pdf_response.model_dump_json())
            response_dict = json.loads(pdf_response.model_dump_json())
            json_string = json.dumps(response_dict, indent=4, ensure_ascii=False)
            results.append({
                "file_type": "pdf",
                "ocr_markdown": merged_markdown
                
            })

        elif extension in [".jpg", ".jpeg", ".png"]:
            encoded = base64.b64encode(file_content).decode()
            base64_data_url = f"data:image/{extension.replace('.', '')};base64,{encoded}"
            image_response = client.ocr.process(
                document=ImageURLChunk(image_url=base64_data_url),
                model="mistral-ocr-latest"
            )
            Image_ocr_markdown = image_response.pages[0].markdown
            chat_response = client.chat.complete(
                model="pixtral-large-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            ImageURLChunk(image_url=base64_data_url),
                            TextChunk(
                                text=(
                                    "This is image's OCR in markdown:\n"
                                    "<BEGIN_IMAGE_OCR<\n"
                                    f"{Image_ocr_markdown}\n"
                                    "<END_IMAGE_OCR>\n"
                                    "Convert this into a sensible structured JSON response. "
                                    "The output should be strictly json, no other text, no explanation, no markdown."
                                )
                            )
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            response_dict = json.loads(chat_response.choices[0].message.content)
            json_stringI = json.dumps(response_dict, indent=4)
            results.append({
                "filename": file.filename,
                "file_type": "image",
                "ocr_result": response_dict
            })

        else:
            raise HTTPException(status_code=400, detail=f"Formato no soportado: {extension}")

    print(results)
    return JSONResponse(content={"archivos_procesados": results})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
