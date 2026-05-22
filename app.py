import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket,WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from inference import run_style_transfer
app = FastAPI(title="Neural Art Studio")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/style")
async def style_transfer_ws(websocket:WebSocket):

    await websocket.accept()
    try :
        content_bytes = await websocket.receive_bytes()
        style_bytes = await websocket.receive_bytes()
        alpha = float(await websocket.receive_text())


        await run_style_transfer(content_bytes,style_bytes,alpha,websocket)
        await websocket.close()
    except WebSocketDisconnect :
        print("client disconnected early")
    except Exception as e:
        print(f"websocket error: {e}")
        await websocket.close()

app.mount("/", StaticFiles(directory="static",html=True), name="static")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)