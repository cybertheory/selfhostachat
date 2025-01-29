from fastapi import FastAPI, HTTPException, Depends, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.security.api_key import APIKey
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import ARRAY
from datetime import datetime, timedelta
import httpx
import asyncio
import json
import numpy as np
import os
import hashlib
import secrets
from typing import List, Optional
from pydantic import BaseModel

# Environment variables
DATABASE_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}/{os.getenv('POSTGRES_DB')}"
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
ALLOWED_ORIGIN = os.getenv('ALLOWED_ORIGIN')
MODEL_NAME = os.getenv('MODEL_NAME', 'mistral')
ADMIN_PIN = os.getenv('ADMIN_PIN')
JWT_SECRET = os.getenv('JWT_SECRET', secrets.token_urlsafe(32))
TOKEN_EXPIRY_HOURS = int(os.getenv('TOKEN_EXPIRY_HOURS', '24'))

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    pin_hash = Column(String)
    role = Column(String)  # 'admin' or 'user'
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    is_active = Column(Boolean, default=True)

class ApiToken(Base):
    __tablename__ = "api_tokens"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    token = Column(String, unique=True, index=True)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime)
    is_revoked = Column(Boolean, default=False)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)
    content = Column(Text)
    embedding = Column(ARRAY(Float))
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class UserCreate(BaseModel):
    username: str
    pin: str

class PinValidation(BaseModel):
    username: str
    pin: str

class ConversationCreate(BaseModel):
    title: str

class MessageCreate(BaseModel):
    content: str
    conversation_id: Optional[int]
    role: str = "user"

class TokenResponse(BaseModel):
    token: str
    expires_at: datetime
    username: str
    role: str

# Security setup
api_key_header = APIKeyHeader(name="X-API-Key")

# Helper functions
def get_password_hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return get_password_hash(plain_password) == hashed_password

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(
    api_key: str = Security(api_key_header),
    db: Session = Depends(get_db)
) -> User:
    token = db.query(ApiToken).filter(
        ApiToken.token == api_key,
        ApiToken.expires_at > datetime.utcnow(),
        ApiToken.is_revoked == False
    ).first()
    
    if not token:
        raise HTTPException(status_code=401, detail="Invalid or expired API key")
    
    user = db.query(User).filter(User.id == token.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    token.last_used_at = datetime.utcnow()
    db.commit()
    
    return user

def check_admin_role(user: User = Depends(get_current_user)):
    if user.role != 'admin':
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user

async def get_embedding(text: str) -> List[float]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": MODEL_NAME, "prompt": text}
        )
        return response.json()["embedding"]

async def get_similar_messages(db: Session, current_message: str, limit: int = 5) -> List[Message]:
    current_embedding = await get_embedding(current_message)
    current_embedding_array = np.array(current_embedding)
    
    similar_messages = db.query(Message).order_by(
        Message.embedding.cosine_distance(current_embedding_array)
    ).limit(limit).all()
    
    return similar_messages

async def stream_ollama_response(prompt: str, context: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": f"Context: {context}\n\nUser: {prompt}\n\nAssistant:",
                "stream": True
            },
            stream=True
        )
        
        async for line in response.aiter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "response" in data:
                        yield f"data: {json.dumps({'content': data['response']})}\n\n"
                except json.JSONDecodeError:
                    continue

def init_admin(db: Session):
    if not ADMIN_PIN:
        print("Warning: ADMIN_PIN not set in environment variables")
        return
        
    admin = db.query(User).filter(User.role == 'admin').first()
    if not admin:
        admin = User(
            username="admin",
            pin_hash=get_password_hash(ADMIN_PIN),
            role="admin"
        )
        db.add(admin)
        db.commit()
        print("Admin user created successfully")

# FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize admin user on startup
@app.on_event("startup")
async def startup_event():
    db = SessionLocal()
    init_admin(db)
    db.close()

# User management endpoints
@app.post("/users", response_model=dict)
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(check_admin_role),
    db: Session = Depends(get_db)
):
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    new_user = User(
        username=user_data.username,
        pin_hash=get_password_hash(user_data.pin),
        role="user",
        created_by_id=current_user.id
    )
    db.add(new_user)
    db.commit()
    
    return {"message": "User created successfully", "username": new_user.username}

@app.delete("/users/{username}")
async def delete_user(
    username: str,
    current_user: User = Depends(check_admin_role),
    db: Session = Depends(get_db)
):
    if username == "admin":
        raise HTTPException(status_code=400, detail="Cannot delete admin user")
    
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.is_active = False
    db.commit()
    return {"message": "User deactivated successfully"}

# Authentication endpoints
@app.post("/auth/token", response_model=TokenResponse)
async def get_token(
    credentials: PinValidation,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(
        User.username == credentials.username,
        User.is_active == True
    ).first()
    
    if not user or not verify_password(credentials.pin, user.pin_hash):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or PIN"
        )
    
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS)
    
    db_token = ApiToken(
        user_id=user.id,
        token=token,
        expires_at=expires_at,
        last_used_at=datetime.utcnow()
    )
    db.add(db_token)
    db.commit()
    
    return TokenResponse(
        token=token,
        expires_at=expires_at,
        username=user.username,
        role=user.role
    )

@app.post("/auth/revoke")
async def revoke_token(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    token = db.query(ApiToken).filter(
        ApiToken.user_id == current_user.id,
        ApiToken.is_revoked == False
    ).first()
    
    if token:
        token.is_revoked = True
        db.commit()
    
    return {"message": "Token revoked successfully"}

# Conversation endpoints
@app.post("/conversations")
async def create_conversation(
    conversation: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_conversation = Conversation(
        title=conversation.title,
        user_id=current_user.id
    )
    db.add(db_conversation)
    db.commit()
    db.refresh(db_conversation)
    return db_conversation

@app.get("/conversations")
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).all()

# Message endpoints
@app.post("/messages")
async def create_message(
    message: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if message.conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == message.conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conversation = Conversation(
            title="New Conversation",
            user_id=current_user.id
        )
        db.add(conversation)
        db.commit()
        message.conversation_id = conversation.id

    # Get similar messages for context
    similar_messages = await get_similar_messages(db, message.content)
    context = "\n".join([f"{msg.role}: {msg.content}" for msg in similar_messages])

    # Create user message with embedding
    embedding = await get_embedding(message.content)
    db_message = Message(
        conversation_id=message.conversation_id,
        role=message.role,
        content=message.content,
        embedding=embedding
    )
    db.add(db_message)
    db.commit()

    # Return streaming response
    return StreamingResponse(
        stream_ollama_response(message.content, context),
        media_type="text/event-stream"
    )

@app.get("/messages/{conversation_id}")
async def list_messages(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Verify conversation belongs to user
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).all()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)