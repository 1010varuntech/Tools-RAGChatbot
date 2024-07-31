from pymongo import MongoClient
from bson import ObjectId
import os

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
def save_session(user_id, session_id, session_data):
    db.sessions.update_one({"user_id": user_id, "session_id": session_id}, {"$set": session_data}, upsert=True)

def load_session(user_id, session_id):
    return db.sessions.find_one({"user_id": user_id, "session_id": session_id})

def load_sessions(user_id):
    return list(db.sessions.find({"user_id": user_id}))

def delete_session(user_id, session_id):
    result = db.sessions.delete_one({"user_id": user_id, "session_id": session_id})
    if result.deleted_count == 0:
        raise Exception(f"Session with id {session_id} for user {user_id} not found")
    
def save_history(user_id, session_id, history):
    db.sessions.update_one(
        {"user_id": user_id, "session_id": session_id},
        {"$set": {"history": history}},
        upsert=True
    )

def load_history(user_id, session_id):
    session = load_session(user_id, session_id)
    return session.get("history", []) if session else []
