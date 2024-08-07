import os
from pydantic import Field
from fastapi import FastAPI, HTTPException, APIRouter, Depends, Query, Body
from pydantic import BaseModel
import asyncio
import httpx
from utils import save_history, load_history, save_session, load_sessions, delete_session, list_tools, save_tool, load_tool, delete_tool
from openai import OpenAI
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import ChatOpenAI
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from supertokens_python import init, InputAppInfo, SupertokensConfig
from supertokens_python.recipe import thirdparty, emailpassword, session
from supertokens_python.framework.fastapi import get_middleware
from supertokens_python import get_all_cors_headers
from supertokens_python.recipe.session.framework.fastapi import verify_session
from supertokens_python.recipe.session import SessionContainer
from fastapi import Request
from contextlib import asynccontextmanager
from bson import ObjectId 
import pinecone
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Read the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("No OPENAI_API_KEY found in environment variables")

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

pc = pinecone.Pinecone(
    api_key=pinecone_api_key,
    environment=pinecone_environment
)

# Check if the index exists, create it if it doesn't
if pinecone_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=pinecone_index_name,
        dimension=1536,  # Update this dimension as per your requirements
        metric='cosine',
        spec=pinecone.ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(pinecone_index_name)

client = OpenAI(api_key=openai_api_key)

app = FastAPI()

init(
    app_info=InputAppInfo(
        app_name="Chatbot",
        api_domain="http://localhost:8000",
        website_domain="http://localhost:3000",
        api_base_path="/auth",
        website_base_path="/auth"
    ),
    supertokens_config=SupertokensConfig(
        connection_uri=os.getenv("SUPERTOKENS_CONNECTION_URI"),
        api_key=os.getenv("SUPERTOKENS_API_KEY")
    ),
    framework='fastapi',
    recipe_list=[
        session.init(),  # initializes session features
        thirdparty.init(
            sign_in_and_up_feature=thirdparty.SignInAndUpFeature(
                providers=[
                    # Google OAuth provider
                    thirdparty.ProviderInput(
                        config=thirdparty.ProviderConfig(
                            third_party_id="google",
                            clients=[
                                thirdparty.ProviderClientConfig(
                                    client_id=os.getenv("GOOGLE_CLIENT_ID"),
                                    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
                                ),
                            ],
                        ),
                    ),
                ]
            )
        ),
        emailpassword.init()
    ],
    mode='asgi'
)

app.add_middleware(get_middleware())
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "PUT", "POST", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Content-Type"] + get_all_cors_headers(),
)

# Define a simple class to represent a document
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

async def load_and_process_documents(namespace: str):
    tools = list_tools()  # Retrieve tool information from MongoDB
    documents = [Document(page_content=tool["prompt"], metadata={"source": tool["name"]}) for tool in tools]

    # Splitting the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Embeddings and storing the texts
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Use Pinecone for vector storage
    vectordb = Pinecone.from_documents(documents=texts, embedding=embedding, index_name=pinecone_index_name, namespace=namespace)

    return vectordb

async def clear_and_reload_vectordb(namespace: str):
    global vectordb, retriever, qa_chain
    if vectordb:
        index.delete(delete_all=True, namespace=namespace)  # Clear existing database by deleting the index
    vectordb = await load_and_process_documents(namespace)  # Recreate the vector database
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


# Setting up the turbo LLM
turbo_llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', api_key=openai_api_key)
vectordb = None
retriever = None
qa_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectordb, retriever, qa_chain
    default_namespace = "default"  # Define a default namespace
    try:
        vectordb = await load_and_process_documents(default_namespace)
        retriever = vectordb.as_retriever(search_kwargs={"k": 2})
        qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        yield
    except httpx.HTTPStatusError as e:
        print(f"Error during startup: {e.response.json()}")
    except httpx.RequestError as e:
        print(f"Request error during startup: {str(e)}")


app = FastAPI(lifespan=lifespan)

# Defining Pydantic models
class Query(BaseModel):
    query: str
    source_filter: str = "all"

class Tool(BaseModel):
    _id: str  # Added to include _id for updating tool
    name: str
    description: str
    logo: str
    prompt: str
    userId: str

class ToolQuery(BaseModel):
    toolId: str
    userId: str
    
class ToolUpdate(BaseModel):
    toolId: str
    name: str
    description: str
    logo: str
    prompt: str
    userId: str

    def to_mongo(self):
        return {
            "_id": ObjectId(self.toolId),
            "name": self.name,
            "description": self.description,
            "logo": self.logo,
            "prompt": self.prompt,
            "userId": self.userId,
        }

# Session management
router = APIRouter()

@router.post("/create_session/{user_id}")
async def create_session(user_id: str):
    session_id = str(uuid.uuid4())
    save_session(user_id, session_id, {"history": []}, new_session=True)
    return {"message": "Session created", "user_id": user_id, "session_id": session_id}

from utils import list_sessions
from fastapi import Query as fApiQry
@router.get("/list_sessions/{user_id}")
async def list_sessions_endpoint(
    user_id: str,
    page: int = fApiQry(1, description="Page number"),
    limit: int = fApiQry(10, description="Number of items per page"),
    sort_by: str = fApiQry("ascending", description="Sort order: 'ascending', 'descending', 'newest', 'oldest'")
):
    sessions = await list_sessions(user_id, page, limit, sort_by)
    print("sessions", sessions)
    if not sessions:
        raise HTTPException(status_code=404, detail="User not found or no sessions available")
    return {"sessions": [
        {
            "session_id": session["session_id"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "chat_name": session["chat_name"]
        } for session in sessions
    ]}# return all the sessions

@router.delete("/delete_session/{user_id}/{session_id}")
async def delete_session_endpoint(user_id: str, session_id: str):
    try:
        delete_session(user_id, session_id)
        return {"message": "Session deleted", "user_id": user_id, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def convert_objectid_to_str(data):
    if isinstance(data, list):
        return [convert_objectid_to_str(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_objectid_to_str(value) for key, value in data.items()}
    elif isinstance(data, ObjectId):
        return str(data)
    else:
        return data
    

from utils import load_session
@router.get("/history/{user_id}/{session_id}")
async def get_history(user_id: str, session_id: str):
    history = convert_objectid_to_str(load_session(user_id, session_id))
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")
    print("history", history)
    return history

@router.delete("/history/{user_id}/{session_id}")
async def clear_history(user_id: str, session_id: str):
    try:
        save_history(user_id, session_id, [])
        return {"message": "Session history cleared", "user_id": user_id, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class userQuery(BaseModel):
    query: str
    source_filter: str = "all"
    namespace: str

@app.post("/query/{user_id}/{session_id}")
async def ask_query(user_id: str, session_id: str, query: userQuery):
    namespace = query.namespace
    existing_sessions = load_sessions(user_id)
    if not any(session["session_id"] == session_id for session in existing_sessions):
        raise HTTPException(status_code=404, detail="Session not found")

    source_choice = query.source_filter.lower()
    query_text = query.query

    if source_choice != "all":
        retrieved_docs = retriever.get_relevant_documents(query_text, filters={"source": source_choice}, namespace=namespace)
    else:
        retrieved_docs = retriever.get_relevant_documents(query_text, namespace=namespace)

    # Modify the query text to instruct the model to use specific tags
    formatted_query = f"Respond to the following query using only the tags h1, h2, h3, h4, h5, h6, ul, ol, li, and p: {query_text}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": formatted_query}]
    )
    answer = response.choices[0].message.content
    
    # Remove newline characters
    answer = answer.replace('\n', '')

    formatted_answer = f"<h1>Response</h1>{answer}"

    # Save to user-specific and session-specific history
    history = load_history(user_id, session_id)
    from utils import aiChatName
    chatName = "oldChat"
    if history == []:
        chatName = aiChatName(query_text)
    history.append([
            {
                "humanReq": query_text,
                "aiRes": formatted_answer,
                "timestamp": datetime.now().isoformat()
            }
        ]
    )
    save_history(user_id, session_id, history, chatName)

    return {"answer": formatted_answer}

@app.get("/chats/user/{user_id}")
async def get_user_chats(user_id: str):
    sessions = load_sessions(user_id)
    if not sessions:
        raise HTTPException(status_code=404, detail="User not found")
    user_chats = []
    for session in sessions:
        session_id = session["session_id"]
        for history_item in session.get("history", []):
            for message in history_item.get("messages", []):
                user_chats.append({
                    "session_id": session_id,
                    "query": message["humanReq"],
                    "answer": message["aiRes"],
                })
    return user_chats


from bson import ObjectId
import uuid

def convert_object_id(document):
    if isinstance(document, list):
        return [convert_object_id(item) for item in document]
    elif isinstance(document, dict):
        for key, value in document.items():
            if isinstance(value, ObjectId):
                document[key] = str(value)
            elif isinstance(value, (list, dict)):
                document[key] = convert_object_id(value)
    return document

@app.post("/tools/get")
async def get_tool(query: ToolQuery):
    try:
        tool = load_tool(query.toolId)
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        tool = convert_object_id(tool)  # Convert ObjectId fields to strings
        return tool
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/create")
async def add_tool(tool: Tool):
    try:
        tool_data = tool.dict()
        tool_data["_id"] = ObjectId()
        save_tool(tool_data)
        await clear_and_reload_vectordb(namespace="default")  # Clear and reload vector database with default namespace
        return {"message": "Tool created successfully", "toolId": str(tool_data["_id"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/update")
async def update_tool(tool: ToolUpdate):
    try:
        tool_data = tool.to_mongo()
        save_tool(tool_data)
        await clear_and_reload_vectordb(namespace="default")  # Clear and reload vector database with default namespace
        return {"message": "Tool updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/delete")
async def delete_tool_endpoint(query: ToolQuery):
    try:
        delete_tool(query.toolId)
        await clear_and_reload_vectordb(namespace="default")  # Clear and reload vector database with default namespace
        return {"message": "Tool deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools")
async def list_tools_endpoint():
    try:
        tools = list_tools()
        tools = [convert_object_id(tool) for tool in tools]  # Convert ObjectId fields to strings
        return tools
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Document management endpoints
@app.get("/documents")
async def get_documents():
    documents = list_tools()
    return [{"content": doc["page_content"], "metadata": doc["metadata"]} for doc in documents]

app.include_router(router, prefix="/session")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
