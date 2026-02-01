import os
import json
import base64
import asyncio
import websockets
import uuid
import time
import io
import traceback
import hashlib
from fastapi import FastAPI, WebSocket, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from datetime import datetime as dt, timedelta, timezone
import jwt
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream, Parameter
from dotenv import load_dotenv
from pydub import AudioSegment
import audioop
from contextlib import suppress
from prompts import function_call_tools, build_system_message
from utils import *
import httpx
from call_log_apis import *
from customer_card_tools import (
    verify_customer_by_cnic,
    confirm_physical_custody,
    verify_tpin,
    verify_card_details,
    activate_card,
    update_customer_tpin,
    transfer_to_ivr_for_pin,
    transfer_to_agent,
    get_customer_status,
    reset_verification_attempts
)
from rag_tools import search_knowledge_base

from src.utils.audio_transcription import transcribe_audio, analyze_call_with_llm

# Gemini Live API imports
from gemini_live import (
    GeminiLiveClient, 
    GeminiLiveConfig, 
    GeminiResponse,
    get_gemini_voice,
    GEMINI_RECEIVE_SAMPLE_RATE
)
from audio_utils import convert_browser_to_gemini, convert_gemini_to_browser, reset_audio_states

load_dotenv(override=True)

PORT = 6089  # Different port for UBL Digital

VOICE = 'echo'

LOG_EVENT_TYPES = [
    'response.content.done', 'input_audio_buffer.committed',
    'session.created', 'conversation.item.deleted', 'conversation.item.created'
]

WARNING_EVENT_TYPES = [
    'error', 'rate_limits.updated'
]

SHOW_TIMING_MATH = False
call_recordings = {}

app = FastAPI()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "ubl-digital-ai-call-center-secret-key-2024")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

USERS_DB = {
    "admin": {
        "username": "admin",
        "password": "admin1234",
        "full_name": "Administrator"
    },
    "demo": {
        "username": "demouser",
        "password": "demouser1234",
        "full_name": "Demo User"
    },
    "ubldigital": {
        "username": "ubldigital",
        "password": "ubldigital1234",
        "full_name": "UBL Digital Team"
    }
}

from fastapi.staticfiles import StaticFiles
app.mount("/client", StaticFiles(directory="static", html=True), name="client")

CHANNELS = 1
RATE = 8000

call_metadata: dict[str, dict] = {}

@app.get("/", response_class=HTMLResponse)
async def index_page():
    with open("static/voice-client.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content

from fastapi import Body

AVAILABLE_VOICES = {
    'echo': {
        'name': 'Saad',
        'age': 'Young Male',
        'personality': 'Warm, Friendly and Engaging'
    }
}


@app.post("/start-browser-call")
async def start_browser_call(request: Request, payload: dict = Body(...)):
    token = get_token_from_request(request)
    user_data = verify_jwt_token(token)
    
    phone = payload.get("phone", "webclient")
    voice = payload.get("voice", "echo")
    temperature = payload.get("temperature", 0.8)
    speed = payload.get("speed", 1.05)
    
    if voice not in AVAILABLE_VOICES:
        voice = "echo"
    
    temperature = max(0.0, min(1.2, float(temperature)))
    speed = max(0.5, min(2.0, float(speed)))
        
    print(f"🎙️ Voice selected: {voice} ({AVAILABLE_VOICES[voice]['name']})")
    print(f"🌡️ Temperature: {temperature}")
    print(f"⚡ Speed: {speed}x")
    
    call_id = await register_call(phone)
    call_id = str(call_id)
    call_recordings[call_id] = {"incoming": [], "outgoing": [], "start_time": time.time()}
    call_metadata[call_id] = {
        "phone": phone, 
        "language_id": payload.get("language_id", 1),
        "voice": voice,
        "temperature": temperature,
        "speed": speed
    }
    await update_call_status(int(call_id), "pick")
    return {
        "call_id": call_id, 
        "voice": voice,
        "temperature": temperature,
        "speed": speed
    }


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    form = await request.form()
    caller_number = form.get("From")
    print("Call is coming from", caller_number)  
    call_id = await register_call(caller_number)
    call_id = str(call_id)
    print("call id received is", call_id, type(call_id))

    call_recordings[call_id] = {"incoming": [], "outgoing": [], "start_time": time.time()}
    
    call_metadata[call_id] = {
        "phone": caller_number, 
        "language_id": 1,
        "voice": "echo",
        "temperature": 0.8,
        "speed": 1.05
    }
    
    response = VoiceResponse()
    response.say("This call may be recorded for quality purposes.", voice='Polly.Danielle-Generative', language='en-US')
    response.pause(length=1)
    host = request.url.hostname

    connect = Connect()
    stream = Stream(url=f"wss://{host}/media-stream")
    stream.parameter(name="call_id", value=call_id)
    connect.append(stream)
    response.append(connect)

    return HTMLResponse(content=str(response), media_type="application/xml")

    

import wave
import audioop
import io
import base64
import websockets as ws_client
from fastapi import WebSocket

USER_AUDIO_DIR = "recordings/user"
AGENT_AUDIO_DIR = "recordings/agent"
os.makedirs(USER_AUDIO_DIR, exist_ok=True)
os.makedirs(AGENT_AUDIO_DIR, exist_ok=True)
import struct
import wave
import struct


last_agent_response_time = None

def generate_silence(duration_sec, sample_rate=8000):
    num_samples = int(duration_sec * sample_rate)
    silence_pcm = b'\x00\x00' * num_samples
    return silence_pcm


async def execute_function_call(func_name: str, func_args: dict) -> dict:
    try:
        if func_name == "search_knowledge_base":
            return await search_knowledge_base(query=func_args.get("query", ""))
        
        elif func_name == "verify_customer_by_cnic":
            return await verify_customer_by_cnic(cnic=func_args.get("cnic", ""))
        
        elif func_name == "confirm_physical_custody":
            return await confirm_physical_custody(
                cnic=func_args.get("cnic", ""),
                has_card=func_args.get("has_card", False)
            )
        
        elif func_name == "verify_tpin":
            return await verify_tpin(
                cnic=func_args.get("cnic", ""),
                tpin=func_args.get("tpin", "")
            )
        
        elif func_name == "verify_card_details":
            return await verify_card_details(
                cnic=func_args.get("cnic", ""),
                last_four_digits=func_args.get("last_four_digits", ""),
                expiry_date=func_args.get("expiry_date", "")
            )
        
        elif func_name == "activate_card":
            return await activate_card(cnic=func_args.get("cnic", ""))
        
        elif func_name == "update_customer_tpin":
            return await update_customer_tpin(
                cnic=func_args.get("cnic", ""),
                new_tpin=func_args.get("new_tpin", "")
            )
        
        elif func_name == "transfer_to_ivr_for_pin":
            return await transfer_to_ivr_for_pin()
        
        elif func_name == "transfer_to_agent":
            return await transfer_to_agent(
                cnic=func_args.get("cnic", ""),
                reason=func_args.get("reason", "")
            )
        
        elif func_name == "get_customer_status":
            return await get_customer_status(cnic=func_args.get("cnic", ""))
        
        else:
            return {
                "success": False,
                "error": f"Unknown function: {func_name}",
                "message": "Function not found in the system."
            }
    
    except Exception as e:
        print(f"❌ Error executing function {func_name}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": f"An error occurred while executing {func_name}."
        }

@app.websocket("/media-stream-browser")
async def media_stream_browser(websocket: WebSocket):
    """
    WebSocket endpoint for browser-based voice calls using Gemini Live API.
    
    Handles:
    - Browser audio streaming (8kHz PCM) -> Gemini (16kHz PCM)
    - Gemini responses (24kHz PCM) -> Browser (8kHz PCM)
    - Function calling for RAG and customer verification
    """
    await websocket.accept()
    
    session_initialized = False
    call_id = None
    stream_sid = None
    gemini_client = None
    
    user_pcm_buffer = io.BytesIO()
    agent_pcm_buffer = io.BytesIO()
    
    function_call_completed_time = None
    FUNCTION_CALL_GRACE_PERIOD = 5.0
    
    try:
        # Wait for the start event with authentication
        start_msg = await websocket.receive_text()
        start_data = json.loads(start_msg)
        
        if start_data.get("event") != "start":
            print("❌ Expected 'start' event as first message")
            await websocket.close(code=1008, reason="Expected start event")
            return
        
        # Authenticate
        token = start_data["start"]["customParameters"].get("token")
        if not token:
            print("❌ No token provided in WebSocket connection")
            await websocket.close(code=1008, reason="Authentication required")
            return
        
        try:
            user_data = verify_jwt_token(token)
            print(f"✅ WebSocket authenticated for user: {user_data['username']}")
        except HTTPException as e:
            print(f"❌ Invalid token in WebSocket: {e.detail}")
            await websocket.close(code=1008, reason="Invalid or expired token")
            return
        
        call_id = start_data["start"]["customParameters"].get("call_id")
        stream_sid = start_data["start"].get("streamSid", "browser-stream")
        meta = call_metadata.get(call_id, {})
        
        # Build Gemini configuration
        instructions = meta.get("instructions", "")
        caller = meta.get("phone", "")
        openai_voice = meta.get("voice", "echo")
        temperature = meta.get("temperature", 0.8)
        
        # Map OpenAI voice to Gemini voice
        gemini_voice = get_gemini_voice(openai_voice)
        
        SYSTEM_MESSAGE = build_system_message(
            instructions=instructions,
            caller=caller,
            voice=openai_voice  # Keep for gendered pronouns in system message
        )
        
        print(f"🔧 Initializing Gemini session with voice: {gemini_voice}, temp: {temperature}")
        
        config = GeminiLiveConfig(
            system_instruction=SYSTEM_MESSAGE,
            tools=function_call_tools,
            voice=gemini_voice,
            temperature=temperature
        )
        
        # Connect to Gemini Live API
        gemini_client = GeminiLiveClient(config)
        
        # Reset audio conversion states for clean session
        reset_audio_states()
        
        await gemini_client.connect()
        session_initialized = True
        
        # Trigger initial greeting - send text to make agent speak first
        print("🎤 Triggering initial greeting...")
        await gemini_client.send_text("Start the conversation by greeting the customer warmly.")
        
        async def receive_from_browser():
            """Receive audio from browser and send to Gemini."""
            nonlocal session_initialized
            try:
                async for msg in websocket.iter_text():
                    try:
                        data = json.loads(msg)
                        
                        if data.get("event") == "media" and session_initialized:
                            # Browser now sends 16kHz PCM (Gemini's native input format)
                            payload_b64 = data["media"]["payload"]
                            pcm_data = base64.b64decode(payload_b64)
                            user_pcm_buffer.write(pcm_data)
                            
                            # Passthrough to Gemini (16kHz -> 16kHz, no conversion needed)
                            # This eliminates resampling overhead for lower latency
                            pcm_16khz = convert_browser_to_gemini(pcm_data, input_rate=16000)
                            
                            # Send to Gemini immediately
                            await gemini_client.send_audio(pcm_16khz)
                        
                        elif data.get("event") == "stop":
                            print(f"🛑 Browser sent stop event for call {call_id}")
                            break
                    
                    except json.JSONDecodeError as je:
                        print(f"⚠️ Failed to parse browser message: {je}")
                        continue
                    except Exception as inner_e:
                        print(f"⚠️ Error processing browser message: {inner_e}")
                        traceback.print_exc()
                        continue
                
                print(f"🔚 Browser WebSocket stream ended normally for call {call_id}")
                
            except WebSocketDisconnect:
                print(f"🔌 Browser WebSocket disconnected for call {call_id}")
            except Exception as e:
                print(f"❌ Unexpected error in browser receive loop: {e}")
                traceback.print_exc()
        
        async def receive_from_gemini_and_forward():
            """Receive responses from Gemini and forward to browser."""
            nonlocal function_call_completed_time
            
            try:
                async for response in gemini_client.receive():
                    try:
                        if response.type == 'audio':
                            # Clear function call flag when audio starts playing
                            if function_call_completed_time is not None:
                                function_call_completed_time = None
                            
                            # Gemini outputs 24kHz PCM - send directly to browser without conversion
                            # The browser AudioContext is now set to 24kHz to match
                            pcm_24khz = response.audio_data
                            
                            # Save 24kHz audio for recording (we'll convert when saving)
                            agent_pcm_buffer.write(pcm_24khz)
                            
                            # Send 24kHz directly to browser
                            pcm_b64 = base64.b64encode(pcm_24khz).decode('utf-8')
                            out = {
                                "event": "media",
                                "media": {
                                    "payload": pcm_b64,
                                    "format": "raw_pcm",
                                    "sampleRate": 24000,  # Native Gemini output rate
                                    "channels": 1,
                                    "bitDepth": 16
                                }
                            }
                            await websocket.send_json(out)
                        
                        elif response.type == 'tool_call':
                            # Handle function calls from Gemini
                            for tool_call in response.tool_calls:
                                func_name = tool_call.get("name")
                                func_id = tool_call.get("id")
                                func_args = tool_call.get("arguments", {})
                                
                                print(f"🔧 Function call: {func_name} with args: {func_args}")
                                
                                try:
                                    result = await asyncio.wait_for(
                                        execute_function_call(func_name, func_args),
                                        timeout=30.0
                                    )
                                except asyncio.TimeoutError:
                                    print(f"⚠️ Function call {func_name} timed out after 30 seconds")
                                    result = {
                                        "success": False,
                                        "error": "timeout",
                                        "message": f"The operation timed out. Please try again."
                                    }
                                
                                print(f"✅ Function result: {result}")
                                
                                # Send response back to Gemini
                                await gemini_client.send_tool_response([{
                                    "id": func_id,
                                    "name": func_name,
                                    "response": result
                                }])
                                
                                function_call_completed_time = time.time()
                                print(f"⏱️ Function call completed at {function_call_completed_time}, grace period: {FUNCTION_CALL_GRACE_PERIOD}s")
                                
                                # Notify browser about function result
                                outgoing_func_result = {
                                    "event": "function_result",
                                    "name": func_name,
                                    "arguments": json.dumps(func_args),
                                    "result": result
                                }
                                await websocket.send_json(outgoing_func_result)
                        
                        elif response.type == 'interrupted':
                            # User interrupted, clear audio buffer
                            current_time = time.time()
                            if function_call_completed_time is not None:
                                time_since_function_call = current_time - function_call_completed_time
                                if time_since_function_call < FUNCTION_CALL_GRACE_PERIOD:
                                    print(f"⚠️ Ignoring interruption {time_since_function_call:.2f}s after function call")
                                    continue
                            
                            await websocket.send_json({"event": "clear"})
                        
                        elif response.type == 'turn_complete':
                            print(f"📋 Gemini turn complete")
                            if function_call_completed_time is not None:
                                print(f"✅ Response completed, clearing function call flag")
                                function_call_completed_time = None
                        
                        elif response.type == 'input_transcription':
                            print(f"🎤 User said: {response.transcription}")
                        
                        elif response.type == 'output_transcription':
                            print(f"🔊 Agent said: {response.transcription}")
                        
                        elif response.type == 'tool_call_cancelled':
                            print(f"⚠️ Tool calls cancelled")
                            continue
                    
                    except Exception as inner_e:
                        print(f"⚠️ Error processing Gemini message: {inner_e}")
                        traceback.print_exc()
                        continue
            
            except Exception as e:
                print(f"❌ Unexpected error in Gemini receive loop: {e}")
                traceback.print_exc()
                try:
                    await websocket.send_json({
                        "event": "error",
                        "message": "An unexpected error occurred. Please try again."
                    })
                except:
                    pass
        
        # Run both tasks concurrently
        recv_task = asyncio.create_task(receive_from_browser())
        send_task = asyncio.create_task(receive_from_gemini_and_forward())
        
        try:
            done, pending = await asyncio.wait(
                [recv_task, send_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                if task == recv_task:
                    print(f"🔚 Browser receive task completed for call {call_id}")
                elif task == send_task:
                    print(f"🔚 Gemini send task completed for call {call_id}")
                
                if task.exception():
                    print(f"❌ Task exception: {task.exception()}")
            
            for task in pending:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        
        except Exception as e:
            print(f"❌ Error in main task loop: {e}")
            traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Error during WebSocket setup: {e}")
        traceback.print_exc()
    
    finally:
        # Close Gemini connection
        if gemini_client:
            await gemini_client.close()
        
        # Save recordings
        if call_id:
            print(f"💾 Saving recordings for call {call_id}...")
            
            user_file_path = f"recordings/user/{call_id}_user.wav"
            agent_file_path = f"recordings/agent/{call_id}_agent.wav"
            
            def save_wav_file(path: str, pcm_data: bytes, sample_rate: int = 8000):
                with wave.open(path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(pcm_data)
            
            # User audio at 16kHz (browser mic rate), agent at 24kHz (Gemini output rate)
            save_wav_file(user_file_path, user_pcm_buffer.getvalue(), sample_rate=16000)
            save_wav_file(agent_file_path, agent_pcm_buffer.getvalue(), sample_rate=24000)
            
            print(f"✅ Saved user audio: {user_file_path}")
            print(f"✅ Saved agent audio: {agent_file_path}")
            
            try:
                user_transcript = await transcribe_audio(user_file_path)
            except Exception as e:
                print(f"⚠️ Could not transcribe user audio: {e}")
                user_transcript = ""
            
            try:
                agent_transcript = await transcribe_audio(agent_file_path)
            except Exception as e:
                print(f"⚠️ Could not transcribe agent audio: {e}")
                agent_transcript = ""
            
            transcripts_output = {
                "call_id": call_id,
                "user_transcript": user_transcript,
                "agent_transcript": agent_transcript
            }
            
            print(f"📝 Transcripts saved for call {call_id}")
            
            analysis_result = await analyze_call_with_llm(call_id, user_transcript, agent_transcript)
            print(f"📊 Call analysis complete: {analysis_result}")
            
            with open(f"recordings/{call_id}_transcript.json", "w", encoding="utf-8") as f:
                json.dump(transcripts_output, f, ensure_ascii=False, indent=2)
        
        try:
            await websocket.close()
        except:
            pass



@app.get("/call-analysis/{call_id}")
async def get_call_analysis(call_id: str, request: Request):
    token = get_token_from_request(request)
    user_data = verify_jwt_token(token)
    
    analysis_file_path = f"recordings/analysis/{call_id}_analysis.json"
    
    if not os.path.exists(analysis_file_path):
        raise HTTPException(status_code=404, detail=f"Analysis not found for call_id: {call_id}")
    
    try:
        with open(analysis_file_path, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)
        return analysis_data
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error reading analysis file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving analysis: {str(e)}")




@app.get("/available-voices")
async def get_available_voices(request: Request):
    token = get_token_from_request(request)
    user_data = verify_jwt_token(token)
    
    return {
        "voices": AVAILABLE_VOICES
    }


def create_jwt_token(username: str, full_name: str) -> str:
    now = dt.now(timezone.utc)
    payload = {
        "username": username,
        "full_name": full_name,
        "exp": now + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": now
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


def verify_jwt_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_token_from_request(request: Request) -> str:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    return auth_header.replace("Bearer ", "")


@app.post("/auth/login")
async def login(credentials: dict = Body(...)):
    username = credentials.get("username", "").strip()
    password = credentials.get("password", "")
    
    if username in USERS_DB:
        user = USERS_DB[username]
        if user["password"] == password:
            token = create_jwt_token(username, user["full_name"])
            
            return {
                "success": True,
                "message": "Login successful",
                "token": token,
                "user": {
                    "username": username,
                    "full_name": user["full_name"]
                }
            }
    
    raise HTTPException(status_code=401, detail="Invalid username or password")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
