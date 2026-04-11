import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, Form, APIRouter
import tensorflow as tf
from router import process, storage


app = FastAPI()




app.include_router(process.router, prefix="/process", tags=['Model'])
app.include_router(storage.router, prefix="/storage", tags=['Storage'])



@app.post('/')
async def login(username: str = Form(), password: str = Form()):
    return {"username": username}
