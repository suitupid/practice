#!/usr/bin/env python3
# -*- coding: utf8 -*-

from fastapi import FastAPI, Request

from inference import Inference

tool = Inference()
app = FastAPI()

@app.post('/predict')
async def get_result(request: Request):
    body = await request.json()
    result = tool.predict(body.get('text'))
    return {'result': result}