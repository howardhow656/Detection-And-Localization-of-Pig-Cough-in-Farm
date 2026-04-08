from fastapi import FastAPI, Form, APIRouter
import tensorflow as tf
from router import model


app = FastAPI()


app.include_router(model.router, prefix="/model", tags=['Model'])




@app.post('/')
async def login(username: str = Form(), password: str = Form()):
    return {"username": username}
