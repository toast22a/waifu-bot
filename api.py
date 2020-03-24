import asyncio
import logging

from asyncio import BoundedSemaphore
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from gpt2_lm_generator import GPT2LMGenerator

logger = logging.getLogger(__name__)

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')

bot = GPT2LMGenerator('models/waifu-bot-4')
senti_analyzer = SentimentIntensityAnalyzer()

padding = [
        {'author': 'user', 'text': "Hey. What's your name?"},
        {'author': 'bot', 'text': "Hi! My name is <|botname|>. What's yours?"},
        {'author': 'user', 'text': "Nice to meet you. My name is <|username|>."},
        {'author': 'bot', 'text': "Nice to meet you too! :)"},
]

def pad_history(history):
    return padding[-(10-len(history)):] + history

def format_history(history):
    return ''.join([f'<|{message["author"]}|>{message["text"]}' for message in
        history[-10:]]) + '<|bot|>'

def format_reply(reply):
    return reply.strip()

@app.get("/")
async def index():
    with open('www/index.html') as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data['label'] == 'chat':
                history = format_history(pad_history(data['history']))
                reply = format_reply(bot.sample(
                    history,
                    max_context=128,
                    min_generation=8,
                    max_generation=64,
                    temperature=0.6,
                    repetition_penalty=2.0,
                    stop_tokens=['<|user|>', '<|bot|>', '<|endoftext|>']
                ))
                await websocket.send_json({
                    'label': 'chat',
                    'text': reply,
                    'sentiment': senti_analyzer.polarity_scores(reply)['compound']
                })
    except WebSocketDisconnect:
        pass
