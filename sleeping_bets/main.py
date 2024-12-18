from fastapi import FastAPI
import solara.server.fastapi

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.mount("/", app=solara.server.fastapi.app)
