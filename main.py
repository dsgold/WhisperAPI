import whisper
from fastapi import FastAPI, UploadFile, File
from starlette.middleware.cors import CORSMiddleware
import torch

if torch.cuda.is_available():
    print("CUDA доступен. Можно использовать GPU.")
else:
    print("CUDA не доступен. Будет использоваться CPU.")

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:5173",
    "http://192.168.31.140:5173",
    "http://37.110.44.172:65"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def textFromVoice(path):
    model = whisper.load_model('medium', device='cuda')
    result = model.transcribe(path, fp16=False, language='russian')
    print(result['text'])
    return result['text']


@app.post("/process_voice_message")
async def process_voice_message(audio: UploadFile = File(...)):
    try:
        file_path = f"voice_messages/{audio.filename}"

        with open(file_path, "wb") as file:
            file.write(audio.file.read())
            audio.file.seek(0)
        result = textFromVoice(path=file_path)
        return result
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
