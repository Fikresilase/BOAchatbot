#imprt statments
from google import genai
from dotenv import load_dotenv


#load function
load_dotenv()
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="who are u"
)
print(response.text)