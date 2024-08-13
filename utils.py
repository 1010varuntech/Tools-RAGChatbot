from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import ObjectId
import os
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME").split('=')[-1]

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# MongoDB operations for tools
def save_tool(tool):
    db.tools.update_one({"_id": tool["_id"]}, {"$set": tool}, upsert=True)

def load_tool(tool_id):
    return db.tools.find_one({"_id": ObjectId(tool_id)})

def delete_tool(tool_id):
    result = db.tools.delete_one({"_id": ObjectId(tool_id)})
    if result.deleted_count == 0:
        raise Exception(f"Tool with id {tool_id} not found")

def list_tools():
    return list(db.tools.find())

# MongoDB operations for sessions
def save_session(user_id, session_id, session_data, new_session=False):
    if new_session:
        session_data["created_at"] = datetime.now()
    session_data["updated_at"] = datetime.now()
    db.sessions.update_one({"user_id": user_id, "session_id": session_id}, {"$set": session_data}, upsert=True)

def load_session(user_id, session_id):
    return db.sessions.find_one({"user_id": user_id, "session_id": session_id})

def load_sessions(user_id):
    return list(db.sessions.find({"user_id": user_id}))

def delete_session(user_id, session_id):
    result = db.sessions.delete_one({"user_id": user_id, "session_id": session_id})
    if result.deleted_count == 0:
        raise Exception(f"Session with id {session_id} for user {user_id} not found")
    
def save_history(user_id, session_id, history, chat_name):
    update_data = {"history": history, "updated_at": datetime.now()}
    if chat_name != "oldChat":
        update_data["chat_name"] = chat_name
    db.sessions.update_one(
        {"user_id": user_id, "session_id": session_id},
        {"$set": update_data},
        upsert=True
    )

def load_history(user_id, session_id):
    session = load_session(user_id, session_id)
    return session.get("history", []) if session else []

from motor.motor_asyncio import AsyncIOMotorClient

async def list_sessions(user_id: str, page: int, limit: int, sort_by: str):
    print("list_sessions", user_id, page, limit, sort_by)
    query = {"user_id": user_id}
    print("query", query)
    
    if sort_by == "ascending":
        sort_order = [("session_id", ASCENDING)]
    elif sort_by == "descending":
        sort_order = [("session_id", DESCENDING)]
    elif sort_by == "newest":
        sort_order = [("$natural", DESCENDING)]  # Sort by natural order, which in absence of a timestamp approximates newest
    elif sort_by == "oldest":
        sort_order = [("$natural", ASCENDING)]  # Sort by natural order, which in absence of a timestamp approximates oldest
    else:
        sort_order = [("session_id", ASCENDING)]
    print("sort_order", sort_order)
    db = AsyncIOMotorClient(MONGODB_URI)[MONGODB_DB_NAME]
    print("db", db)
    cursor = db['sessions'].find(query).sort(sort_order).skip((page - 1) * limit).limit(limit)
    sessions = []
    async for session in cursor:
        sessions.append(session)
    print("cursor result", sessions)
    return sessions

from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

def aiChatName(query, history, user_id, session_id):
    print("aiChatName called")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("No OPENAI_API_KEY found in environment variables")
    client = OpenAI(api_key=openai_api_key)
    summary = load_session(user_id, session_id).get("summary") or None
    if summary is None:
        print("Generating summary as no summary found in the database for aiChatName")
        summary = generateSummary(history)
        saveSummary(user_id, session_id, summary)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "return a descriptive name for a chat according to the given context, it must be a chat name of 20 characters or less"},
            {"role": "system", "content": "context: "+summary},
            {"role": "user", "content": "user's query"+query},
        ],
    )
    return ((response.choices[0].message.content).replace('"', "")).replace("\n", "")

from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import os

def generateSummary(messages):
    try:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("No OPENAI_API_KEY found in environment variables")
        client = OpenAI(api_key=openai_api_key)

        # Initialize an empty list to hold all messages formatted correctly
        formatted_msgs = [{"role": "system", "content": "generate a detailed summary of the conversation generate the summary as in you are going to use it for your own context:"}]
        
        # Flatten the nested list of messages
        flattened_messages = [msg for sublist in messages for msg in sublist]

        # Add user and assistant messages alternatingly
        for msg in flattened_messages:
            formatted_msgs.append({"role": "user", "content": msg['humanReq']})
            formatted_msgs.append({"role": "assistant", "content": msg['aiRes']})

        # Generate a summary using the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=formatted_msgs,
            max_tokens=1500
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        print(f"An error occurred while generating the summary: {e}")
        return None

def saveSummary(user_id, session_id, summary):
    db.sessions.update_one(
        {"user_id": user_id, "session_id": session_id},
        {"$set": {"summary": summary}},
        upsert=True
    )

def generate_summary_new_msg(messages, user_id, session_id):
    try:
        print(messages, "this is the messages")
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("No OPENAI_API_KEY found in environment variables")
        client = OpenAI(api_key=openai_api_key)
        summary = db.sessions.find_one({"user_id": user_id, "session_id": session_id})["summary"]
        
        # Initialize an empty list to hold all messages formatted correctly
        formatted_msgs = [{"role": "system", "content": "generate a detailed summary of the conversation generate the summary as in you are going to use it for your own context:"}]
        
        # Check if messages is a list of dictionaries
        if isinstance(messages, list):
            # Flatten the nested list of messages if necessary
            flattened_messages = [msg for sublist in messages for msg in sublist] if isinstance(messages[0], list) else messages
        else:
            # If messages is a single dictionary, wrap it in a list
            flattened_messages = [messages]

        # Add user and assistant messages alternatingly
        for msg in flattened_messages:
            formatted_msgs.append({"role": "user", "content": msg['humanReq']})
            formatted_msgs.append({"role": "assistant", "content": msg['aiRes']})

        # Generate a summary using the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=formatted_msgs,
            max_tokens=1500
        )
        summary = response.choices[0].message.content
        saveSummary(user_id, session_id, summary)
        return summary
    except Exception as e:
        print(f"An error occurred while generating the summary: {e}")
        return None