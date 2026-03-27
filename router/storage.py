import pymongo
from fastapi import Depends, APIRouter, UploadFile, File, Form
from motor.motor_asyncio import AsyncIOMotorClient
from typing import AsyncGenerator
from pydantic import BaseModel
import pandas as pd
import os

router = APIRouter()

class MongoDB():
    def __init__(self):
        self.client = AsyncIOMotorClient("mongodb://localhost:27017")
        self.db = self.client['audio']


database = MongoDB()
async def get_db() -> AsyncGenerator:
    yield database.db


@router.post('/db_test')
async def test_connection(db = Depends(get_db)):
    existing = await db.test.find_one({"msg": "hello"})

    if not existing:
        await db.test.insert_one({"msg": "hello"})
    return {"inserted_id": str(existing.inserted_id)}



@router.post('/insert_raw_data')
async def insert_raw_data(
    db = Depends(get_db),
    audio: UploadFile = File(...),
    merge: UploadFile = File(...),
    Numofmic: int | None = Form(None)
):
    df = pd.read_csv(merge.file)

    save_path = f"data/audio/{audio.filename}"
    with open(save_path, "wb") as f:
        f.write(await audio.read())

    events = df.to_dict(orient="records")

    result = await db.raw_data.insert_one({
        "AudioPath": save_path,
        "Microphone": Numofmic,
        "Event": events
    })

    return {"id": str(result.inserted_id)}


