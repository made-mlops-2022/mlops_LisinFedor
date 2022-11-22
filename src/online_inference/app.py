from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, HTTPException
from http import HTTPStatus


app = FastAPI()


@app.get("/")
async def docs_redirect():
    return RedirectResponse(url="/docs")