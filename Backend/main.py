#import modules
from fastapi import FastAPI
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel

#Run modules
load_dotenv()   
app = FastAPI()
client = genai.Client()

#post route goes in here
# @app.get("/ai")
# def get_ai_explanation():
#     response = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents="Explain how AI works in a few words",
#     )
#     return {"text": response.text}
# Define what the POST request body should look like
class PromptRequest(BaseModel):
    prompt: str

@app.post("/")
def generate_ai_text(request: PromptRequest):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=request.prompt
    )
    return {"text": response.text}

