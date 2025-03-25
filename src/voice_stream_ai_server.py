import ray
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ray import serve

import websockets
import uuid
import json
import asyncio
import logging

from audio_utils import save_audio_to_file
from client import Client
from asr.faster_whisper_asr import FasterWhisperASR
from vad.pyannote_vad import SileroVAD

logger = logging.getLogger("ray.serve")
logger.setLevel(logging.DEBUG)
fastapi_app = FastAPI()
from ray.serve.handle import DeploymentHandle

@serve.deployment
@serve.ingress(fastapi_app)
class TranscriptionServer:
    """
    Represents the WebSocket server for handling real-time audio transcription.
    """

    def __init__(self, asr_handle: DeploymentHandle, vad_handle=DeploymentHandle, sampling_rate=16000, samples_width=2):
        logger.info("Initializing TranscriptionServer")
        print("[DEBUG] Initializing TranscriptionServer")
        
        self.sampling_rate = sampling_rate
        self.samples_width = samples_width
        self.connected_clients = {}
        self.asr_handle = asr_handle
        self.vad_handle = vad_handle

        from asr.asr_factory import ASRFactory
        from vad.vad_factory import VADFactory
        
        logger.info("Creating VAD and ASR pipelines")
        print("[DEBUG] Creating VAD and ASR pipelines")
        self.vad_pipeline = VADFactory.create_vad_pipeline("pyannote")
        self.asr_pipeline = ASRFactory.create_asr_pipeline("faster_whisper")
        logger.info("VAD and ASR pipelines created successfully")
        print("[DEBUG] VAD and ASR pipelines created successfully")

    async def handle_audio(self, client: Client, websocket: WebSocket):
        logger.info(f"Handling audio for client {client.client_id}")
        print(f"[DEBUG] Handling audio for client {client.client_id}")
        
        while True:
            message = await websocket.receive()
            print(f"[DEBUG] Received message from client {client.client_id}: {message}")
            
            if "bytes" in message.keys():
                logger.info(f"Received audio data from client {client.client_id}")
                print(f"[DEBUG] Received audio data from client {client.client_id}")
                client.append_audio_data(message['bytes'])
            elif "text" in message.keys():
                import json
                
                config = json.loads(message['text'])
                if config.get('type') == 'config':
                    logger.info(f"Received config update from client {client.client_id}")
                    print(f"[DEBUG] Received config update from client {client.client_id}: {config}")
                    client.update_config(config['data'])
                    continue
            elif message["type"] == "websocket.disconnect":
                logger.info(f"Client {client.client_id} disconnected")
                print(f"[DEBUG] Client {client.client_id} disconnected")
                raise WebSocketDisconnect
            else:
                import json
                keys_list = list(message.keys())
                logger.debug(f"{type(message)} is not a valid message type. Type is {message['type']}; keys: {json.dumps(keys_list)}")
                logger.error(f"Unexpected message type from {client.client_id}")
                print(f"[ERROR] Unexpected message type from {client.client_id}: {message}")
            
            logger.info(f"Processing audio for client {client.client_id}")
            print(f"[DEBUG] Processing audio for client {client.client_id}")
            client.process_audio(websocket, self.vad_handle, self.asr_handle)
    
    @fastapi_app.websocket("/")
    async def handle_websocket(self, websocket: WebSocket):
        logger.info("New WebSocket connection initiated")
        print("[DEBUG] New WebSocket connection initiated")
        
        await websocket.accept()

        client_id = str(uuid.uuid4())
        client = Client(client_id, self.sampling_rate, self.samples_width)
        self.connected_clients[client_id] = client

        logger.info(f"Client {client_id} connected")
        print(f"[DEBUG] Client {client_id} connected")
        
        try:
            await self.handle_audio(client, websocket)
        except WebSocketDisconnect as e:
            logger.warning(f"Connection with {client_id} closed: {e}")
            print(f"[WARNING] Connection with {client_id} closed: {e}")
        finally:
            logger.info(f"Removing client {client_id} from connected clients")
            print(f"[DEBUG] Removing client {client_id} from connected clients")
            del self.connected_clients[client_id]

logger.info("Starting TranscriptionServer deployment")
print("[DEBUG] Starting TranscriptionServer deployment")
entrypoint = TranscriptionServer.bind(FasterWhisperASR.bind(), SileroVAD.bind())
serve.run(entrypoint)
logger.info("TranscriptionServer is running")
print("[DEBUG] TranscriptionServer is running")
#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