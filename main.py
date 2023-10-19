import os
from transformers import pipeline
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import httpx
import torch

# Set your OpenAI API key
model_ckpt = "openai/whisper-base"
lang = "en"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    task="automatic-speech-recognition",
    model=model_ckpt,
    chunk_length_s=30,
    device=device,
)
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")

# Define the audio transcription function
def transcribe_audio(audio_url):

    # Download the audio file from the provided URL
    with httpx.Client() as client:
        response = client.get(audio_url)
        audio_data = response.content

    # Transcribe audio using the model
    transcript = pipe(audio_data)["text"]

    return transcript

app = FastAPI()

@app.post("/transcribe/")
async def transcribe(audio_url: str):
    try:
        # Transcribe audio from the provided URL
        transcript = transcribe_audio(audio_url)

        return JSONResponse(content={"transcript": transcript}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
