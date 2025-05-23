from fastapi import FastAPI, HTTPException, File, UploadFile
from databases import Database
from pydantic import BaseModel, constr, EmailStr
import os
import io
import json
import asyncio
from typing import Optional, List
from datetime import datetime
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
import aiosmtplib
from email.message import EmailMessage
import time
import asyncio
import hashlib
from aiocache import cached
from deepgram import DeepgramClient, PrerecordedOptions



import logging
MAX_FILE_SIZE_MB = 25
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # Load OPENAI_API_KEY from .env
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/tasks_db")
database = Database(DATABASE_URL)

# Reduce semaphore to 1 to be more conservative with rate limits
transcription_semaphore = asyncio.Semaphore(1)


# Initialize OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
# print(f"API Key: {api_key}")
if not api_key:
    raise ValueError("Missing OpenAI API key in environment variables")

client = OpenAI(api_key=api_key)
# Initialize Deepgram API key
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
# print(f"Deepgram API Key: {deepgram_api_key}")
if not deepgram_api_key:
    raise ValueError("Missing Deepgram API key in environment variables")

deepgram_client = DeepgramClient(deepgram_api_key)



# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Agentic AI Service", version="1.0")

# Database file path (SQLite)
DB_PATH = "tasks.db"



# Pydantic model for incoming requests
class AgentRequest(BaseModel):
    instruction: constr(min_length=3)
    email: Optional[EmailStr] = None

# Pydantic model for API response
class AgentResponse(BaseModel):
    message: str
    task_id: Optional[int] = None
    notification_sent: Optional[bool] = None

# Create the database table on startup
@app.on_event("startup")
async def startup():
    await database.connect()
    await database.execute("""
        CREATE TABLE IF NOT EXISTS task_log (
            id SERIAL PRIMARY KEY,
            description TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    
@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

async def create_task(description: str) -> int:
    """
    Insert a new task into the database and return its ID.
    """
    query = "INSERT INTO task_log (description, timestamp) VALUES (:description, :timestamp) RETURNING id"
    values = {"description": description, "timestamp": datetime.utcnow().isoformat()}
    return await database.execute(query=query, values=values)

async def send_notification(message: str, to_email: Optional[str] = None) -> bool:
    """
    Simulate sending a notification (e.g., logging or mock email).
    """
    # Here we just print the notification as a placeholder for a real side effect.
    # print(f"[Notification] {message}")
    logger.info(f"Notification: {message}")
    logger.info(f"to_email: {to_email}")

    """
    Send an email notification using SMTP.
    """

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("FROM_EMAIL")

    if not all([smtp_host, smtp_user, smtp_password, from_email]):
        logger.error("SMTP environment variables not set correctly.")
        return False
    
    email = EmailMessage()
    email["From"] = from_email
    email["To"] = to_email
    email["Subject"] = "Agent Notification"
    email.set_content(message)
    if to_email is not None:
        try:
            await aiosmtplib.send(
                email,
                hostname=smtp_host,
                port=smtp_port,
                username=smtp_user,
                password=smtp_password,
                start_tls=True,
            )
            logger.info(f"Email sent to {to_email}")
            return True
        except Exception as e:
            logger.exception(f"Failed to send email: {e}")
            return False
    else:
        logger.info("No email address provided, skipping email sending.")
        return False

async def call_openai(messages: List[dict], functions: Optional[List[dict]] = None) -> object:
    """
    Asynchronously call the OpenAI chat completion API (wrapping in a thread).
    """

    return await asyncio.to_thread(lambda: client.chat.completions.create(
        model="gpt-4o-mini", # model with function-calling support
        messages=messages,
        functions=functions,
        function_call="auto" # let the model choose
    ))

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/agent", response_model=AgentResponse, tags=["AI Agent"])
async def agent_endpoint(req: AgentRequest):
    """
    HTTP endpoint to trigger the AI agent. Parses instructions and
    performs actions (DB insert, send notification) as decided by the model.
    """
    instruction = req.instruction

    # System prompt describing capabilities and available functions
    system_message = {
        "role": "system",
        "content": (
            "You are an automated assistant. "
            "You can create tasks by name and send notifications. "
            "Available actions: create_task(description), send_notification(message). "
            "Decide which action(s) to take based on the user instruction."
        )
    }
    user_message = {"role": "user", "content": instruction}

    # Define function signatures for the model
    functions = [
        {
            "name": "create_task",
            "description": "Create a new task with a description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "The description of the task."
                    }
                },
                "required": ["description"]
            }
        },
        {
            "name": "send_notification",
            "description": "Send a notification with a message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message content to notify."
                    }
                },
                "required": ["message"]
            }
        }
    ]

    # Initialize conversation
    messages = [system_message, user_message]

    task_id: Optional[int] = None
    notification_sent: Optional[bool] = None
    final_reply = ""

    # Loop to handle multiple function calls if the model decides so
    for _ in range(3):  # limit to 3 iterations to avoid infinite loops
        response = await call_openai(messages, functions)
        choice = response.choices[0]
        msg = choice.message


        # Check if the model wants to call a function
        if choice.finish_reason == "function_call":
            fn_name = msg.function_call.name
            args = json.loads(msg.function_call.arguments or "{}")


            # Execute the corresponding function
            if fn_name == "create_task":
                desc = args.get("description", "")
                task_id = await create_task(desc)
                result_content = json.dumps({"task_id": task_id})
            elif fn_name == "send_notification":
                message_content = args.get("message", "")
                notification_sent = await send_notification(message_content, req.email)
                result_content = json.dumps({"status": "sent"})
            else:
                raise HTTPException(status_code=400, detail=f"Unknown function {fn_name}")

            # Add function call and response into the conversation
            messages.append(msg.model_dump()) # the function call message from model
            messages.append({
                "role": "function",
                "name": fn_name,
                "content": result_content
            })
            # Continue to let the model produce a final answer or next action
            continue

        # If no function call, treat it as final answer
        final_reply = msg.content or ""
        break

    if not final_reply:
        final_reply = "No response from the AI."

    # Return structured JSON
    return AgentResponse(
        message=final_reply.strip(),
        task_id=task_id,
        notification_sent=notification_sent
    )

@app.post("/agent/voice", response_model=AgentResponse, tags=["AI Agent"])
async def agent_voice_command(email: Optional[EmailStr] = None, file: UploadFile = File(...)):
    """
    Accepts an audio file, transcribes it to text using Deepgram API,
    and passes the transcribed instruction to the AI agent.
    """

    if not file.filename.endswith((".mp3", ".wav", ".m4a", ".webm", ".mpeg", ".mpga")):
        raise HTTPException(status_code=400, detail="Unsupported file format, only mp3, wav, m4a, webm, mpeg, mpga are supported.")

    try:
        # Read the file into memory
        contents = await file.read()
        
        # File size check (25MB limit)
        if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Limit is 25MB.")
        
        
        try:
             # Prepare transcription options
            options = PrerecordedOptions(
                punctuate=True,
                model="general",
                language="en"
            )
            # Transcribe using Deepgram
            response = deepgram_client.listen.prerecorded.v("1").transcribe_file(
                {"buffer": contents},
                options
            )
            instruction_text = response['results']['channels'][0]['alternatives'][0]['transcript']
        except Exception as e:
            logger.exception(f"Deepgram transcription failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to transcribe audio")

        if not instruction_text.strip():
            raise HTTPException(status_code=400, detail="No speech detected in audio")


        logger.info(f"Transcribed voice to text: {instruction_text}")

        # Reuse the agent logic
        agent_request = AgentRequest(instruction=instruction_text, email=email if email is not None else None)
        response = await agent_endpoint(agent_request)
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in voice command processing")
        raise HTTPException(status_code=500, detail="Failed to process voice command.")

