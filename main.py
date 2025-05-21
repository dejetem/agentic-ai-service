from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, constr, EmailStr
import os
import json
import asyncio
from typing import Optional, List
from datetime import datetime
import aiosqlite
from openai import OpenAI
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # Load OPENAI_API_KEY from .env



# Initialize OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OpenAI API key in environment variables")

client = OpenAI(api_key=api_key)

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
    """
    Create the task_log table if it doesn't exist.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS task_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        await db.commit()

async def create_task(description: str) -> int:
    """
    Insert a new task into the database and return its ID.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO task_log (description, timestamp) VALUES (?, ?)",
            (description, datetime.utcnow().isoformat())
        )
        await db.commit()
        return cursor.lastrowid

async def send_notification(message: str, to_email: Optional[str] = None) -> bool:
    """
    Simulate sending a notification (e.g., logging or mock email).
    """
    # Here we just print the notification as a placeholder for a real side effect.
    # print(f"[Notification] {message}")
    logger.info(f"Notification: {message}")
    return True

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
