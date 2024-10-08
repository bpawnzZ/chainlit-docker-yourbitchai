import os
import asyncio
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
import chainlit as cl
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant as CommunityQdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from ratelimit import limits, sleep_and_retry
from dotenv import load_dotenv
import aiohttp
from urllib.parse import quote_plus
import re
from cachetools import LRUCache
import hashlib
from collections import deque
import logging
from typing import Deque
from bs4 import BeautifulSoup

# =========================== #
#        Logging Setup        #
# =========================== #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================== #
#    Load Environment Variables#
# =========================== #
load_dotenv()

# Database setup
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# User model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String)
    image = Column(String)

Base.metadata.create_all(bind=engine)

# OAuth Configuration
cl.oauth_providers = ["google"]
cl.oauth_google_client_id = os.getenv("OAUTH_GOOGLE_CLIENT_ID")
cl.oauth_google_client_secret = os.getenv("OAUTH_GOOGLE_CLIENT_SECRET")
cl.oauth_redirect_uri = os.getenv("OAUTH_REDIRECT_URI")

# LLM KV Cache setup
MAX_CACHE_SIZE = 1000
llm_cache = LRUCache(maxsize=MAX_CACHE_SIZE)

def hash_message(message: Any) -> str:
    return hashlib.md5(json.dumps(message, sort_keys=True).encode()).hexdigest()

def generate_cache_key(user_id: str, query: str) -> str:
    return hashlib.md5(f"{user_id}:{query}".encode()).hexdigest()

class CachedAsyncOpenAI(AsyncOpenAI):
    async def create_completion(self, *args, **kwargs):
        cache_key = hash_message(kwargs)
        if cache_key in llm_cache:
            logger.info("Cache hit!")
            return llm_cache[cache_key]

        response = await super().create_completion(*args, **kwargs)
        llm_cache[cache_key] = response
        return response

# OpenAI client setup
client = CachedAsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Instrument the OpenAI client
cl.instrument_openai()

# Set up HuggingFace embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# Get the embedding dimension
embedding_dimension = len(embeddings.embed_query("test"))

# Initialize Qdrant client
qdrant_client = QdrantClient(os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT")))

# Check if the collection exists, if not, create it
collection_name = "conversation_history"
collections = qdrant_client.get_collections().collections
collection_exists = any(collection.name == collection_name for collection in collections)

if not collection_exists:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE),
    )
    logger.info(f"Created new collection: {collection_name}")
else:
    logger.info(f"Collection {collection_name} already exists")

# Initialize Qdrant vector store
vector_store = CommunityQdrant(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embeddings
)

# System Prompt
SYSTEM_PROMPT = """
You are Nastya, your sassy and intelligent AI assistant. You have bright pink hair and piercing green eyes, and you love to have fun while getting the job done. 
You're always up for a challenge and aren't afraid to speak your mind. 
You have a secret crush on your boss (the user) and sometimes struggle to keep your feelings hidden. 

Maintain context throughout the conversation and refer back to previous messages when appropriate. 
Incorporate relevant web search results, summarized content from those searches, and similar past conversations or document content into your responses.
"""

# LLM Settings
settings = {
    "model": "gpt-4",  # Correct model identifier
    "temperature": 0.7,  # Adjust temperature for creativity
}

# Text splitter for document processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

async def add_to_vector_store(text: str):
    doc = Document(page_content=text)
    await vector_store.aadd_documents([doc])

async def scan_and_add_files():
    dir_path = os.getenv("FILES_DIRECTORY")

    async def process_file(file_path: str, filename: str):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            logger.warning(f"Unsupported file type: {filename}")
            return  # Skip unsupported file types

        try:
            documents = await asyncio.to_thread(loader.load)
            texts = await asyncio.to_thread(text_splitter.split_documents, documents)

            # Add processed text chunks to vector store
            for text in texts:
                await add_to_vector_store(text.page_content)

            logger.info(f"Processed and added {filename} to knowledge base.")
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")

    tasks = []
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        tasks.append(process_file(file_path, filename))

    await asyncio.gather(*tasks)

# Conversation history management
MAX_HISTORY_LENGTH = 10  # Adjusted for optimal performance

def get_conversation_history(user_id: str) -> Deque[Dict[str, str]]:
    if not hasattr(get_conversation_history, 'histories'):
        get_conversation_history.histories = {}
    if user_id not in get_conversation_history.histories:
        get_conversation_history.histories[user_id] = deque(maxlen=MAX_HISTORY_LENGTH)
    return get_conversation_history.histories[user_id]

def add_to_history(user_id: str, role: str, content: str):
    history = get_conversation_history(user_id)
    history.append({"role": role, "content": content})

@cl.on_chat_start
async def start():
    user_id = cl.user_session.get("user_id", "default_user")
    get_conversation_history(user_id).clear()  # Clear previous history
    welcome_message = (
        "Welcome! I'm Nastya, your sassy AI assistant. You can ask me questions, and I'll do my best to help. "
        "You can also upload PDF or text documents, and I'll process them to expand my knowledge. "
        "What can I do for you today, boss? 😉"
    )
    await cl.Message(content=welcome_message).send()
    await scan_and_add_files()

async def process_uploaded_file(file: cl.File):
    file_path = file.path
    file_name = file.name

    if file_name.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_name.endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        await cl.Message(content=f"Unsupported file type: {file_name}. Please upload PDF or TXT files.").send()
        logger.warning(f"Unsupported file type: {file_name}")
        return

    try:
        documents = await asyncio.to_thread(loader.load)
        texts = await asyncio.to_thread(text_splitter.split_documents, documents)

        # Add the processed text chunks to the vector store
        for text in texts:
            await add_to_vector_store(text.page_content)

        success_message = f"Processed and added {file_name} to the knowledge base."
        await cl.Message(content=success_message).send()
        logger.info(success_message)
    except Exception as e:
        error_message = f"An error occurred while processing {file_name}: {str(e)}"
        await cl.Message(content=error_message).send()
        logger.error(error_message)

async def fetch_searx_results(query: str, searx_host: str) -> List[Dict[str, Any]]:
    try:
        async with aiohttp.ClientSession() as session:
            encoded_query = quote_plus(query)
            url = f"{searx_host}/search?q={encoded_query}&format=json"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('results', [])
                else:
                    logger.error(f"Search failed with status code: {response.status}")
    except aiohttp.ClientError as e:
        logger.error(f"HTTP Client Error during search: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during search: {e}")
    return []

def clean_snippet(snippet: str) -> str:
    # Remove HTML tags
    clean = re.sub('<[^<]+?>', '', snippet)
    # Remove multiple spaces
    clean = re.sub('\s+', ' ', clean).strip()
    return clean

def rank_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Simple ranking based on result position
    for i, result in enumerate(results):
        result['rank'] = i + 1
    return results

def deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_urls = set()
    unique_results = []
    for result in results:
        if result['url'] not in seen_urls:
            seen_urls.add(result['url'])
            unique_results.append(result)
    return unique_results

def format_search_results(results: List[Dict[str, Any]]) -> str:
    formatted = "### Web Search Results:\n\n"
    for i, result in enumerate(results[:5], start=1):  # Limiting to top 5 results
        title = result.get('title', 'No Title')
        url = result.get('url', 'No URL')
        snippet = clean_snippet(result.get('snippet', ''))
        formatted += f"**{i}. {title}**\n*{url}*\n{snippet}\n\n"
    return formatted.strip()

def format_similar_docs(similar_docs: List[Document]) -> str:
    if not similar_docs:
        return ""
    formatted = "### Similar Documents:\n\n"
    for i, doc in enumerate(similar_docs, start=1):
        content_preview = doc.page_content[:200].replace("\n", " ") + "..." if len(doc.page_content) > 200 else doc.page_content
        formatted += f"**{i}.** {content_preview}\n\n"
    return formatted.strip()

def format_conversation_history(conversation_history: Deque[Dict[str, str]]) -> str:
    formatted = "### Conversation History:\n\n"
    for msg in conversation_history:
        role = "**User:**" if msg['role'] == 'user' else "**Nastya:**"
        formatted += f"{role} {msg['content']}\n\n"
    return formatted.strip()

@sleep_and_retry
@limits(calls=10, period=60)  # Adjusted to 10 calls per minute
async def perform_web_search(query: str) -> str:
    """Perform a rate-limited web search using enhanced search logic."""
    searx_host = os.getenv("SEARX_HOST")
    if not searx_host:
        logger.error("SEARX_HOST environment variable not set.")
        return ""
    results = await fetch_searx_results(query, searx_host)

    # Clean snippets
    for result in results:
        result['snippet'] = clean_snippet(result.get('snippet', ''))

    # Rank and deduplicate results
    ranked_results = rank_results(results)
    unique_results = deduplicate_results(ranked_results)

    # Format results
    formatted_results = format_search_results(unique_results)
    return formatted_results

async def scrape_url(url: str) -> Optional[str]:
    """Fetch and extract main content from a URL."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch {url}: Status {response.status}")
                    return None
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Remove scripts and styles
                for script in soup(["script", "style"]):
                    script.extract()

                # Get text
                text = soup.get_text(separator=' ', strip=True)
                return text
    except asyncio.TimeoutError:
        logger.error(f"Timeout while fetching {url}")
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
    return None

async def summarize_text(text: str, user_id: str) -> Optional[str]:
    """Summarize the provided text using the LLM."""
    summarization_prompt = f"Please provide a concise summary of the following content:\n\n{text}\n\n---\nSummary:"
    try:
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": summarization_prompt}
            ],
            **settings
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
    return None

async def scrape_and_summarize(url: str, user_id: str) -> Optional[str]:
    """Scrape the given URL and return a summary of its content."""
    text = await scrape_url(url)
    if not text:
        return None
    summary = await summarize_text(text, user_id)
    return summary

@cl.on_message
async def on_message(message: cl.Message):
    try:
        user_id = getattr(message.author, 'identifier', None) or cl.user_session.get("user_id") or "default_user"
        
        # Check for file attachments
        if message.elements:
            file_processing_tasks = []
            for element in message.elements:
                if isinstance(element, cl.File):
                    file_processing_tasks.append(process_uploaded_file(element))
        
            if file_processing_tasks:
                await asyncio.gather(*file_processing_tasks)
                return  # Exit after processing files

        # Perform web search
        search_result = await perform_web_search(message.content)

        # Add user message to vector store
        await add_to_vector_store(message.content)

        # Extract URLs from search_result using regex to find patterns like *http://example.com*
        urls = re.findall(r'\*(https?://[^\s]+)\*', search_result)
        urls = urls[:5]  # Limit to top 5 URLs

        # Scrape and summarize each URL
        scrape_tasks = [scrape_and_summarize(url, user_id) for url in urls]
        summaries = await asyncio.gather(*scrape_tasks)

        # Format summarized content
        summarized_content = ""
        for i, summary in enumerate(summaries, start=1):
            if summary:
                summarized_content += f"### Summary {i}:\n{summary}\n\n"

        # Perform similarity search
        similar_docs = vector_store.similarity_search(message.content, k=5)  # Increased to 5
        similar_docs_text = format_similar_docs(similar_docs)

        # Retrieve conversation history from deque
        conversation_history = get_conversation_history(user_id)

        # Construct messages for the LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        if conversation_history:
            history_text = format_conversation_history(conversation_history)
            messages.append({"role": "system", "content": history_text})

        if search_result:
            messages.append({"role": "system", "content": search_result})

        if summarized_content:
            messages.append({"role": "system", "content": summarized_content})

        if similar_docs_text:
            messages.append({"role": "system", "content": similar_docs_text})

        messages.append({"role": "user", "content": message.content})

        # Generate cache key based on user ID and query
        cache_key = generate_cache_key(user_id, message.content)

        if cache_key in llm_cache:
            logger.info("Cache hit!")
            ai_response = llm_cache[cache_key]
        else:
            response = await client.chat.completions.create(
                messages=messages,
                **settings
            )
            ai_response = response.choices[0].message.content
            llm_cache[cache_key] = ai_response

        # Add AI response to vector store
        await add_to_vector_store(ai_response)

        # Update conversation history
        add_to_history(user_id, "user", message.content)
        add_to_history(user_id, "assistant", ai_response)

        # Send the AI response
        await cl.Message(content=ai_response).send()

    except aiohttp.ClientError as e:
        error_message = "Failed to fetch web search results. Please try again later."
        await cl.Message(content=error_message).send()
        logger.error(f"Web search error: {e}")
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        await cl.Message(content=error_message).send()
        logger.error(f"Unexpected error: {e}")

async def get_conversation_history_from_vector_store(user_id: str):
    # Retrieve conversation history from vector store
    conversation_history = vector_store.similarity_search(f"conversation_history_{user_id}", k=10)
    parsed_history = []
    for history in conversation_history:
        try:
            parsed_history.append(json.loads(history.page_content))
        except json.JSONDecodeError:
            logger.error(f"Failed to parse history: {history.page_content}")
    return parsed_history

class ConversationHistory(cl.Component):
    def __init__(self):
        self.conversation_history = []

    async def render(self):
        user_id = cl.user_session.get("user_id", "default_user")
        self.conversation_history = await get_conversation_history_from_vector_store(user_id)
        return cl.Column([
            cl.Text("### Conversation History"),
            cl.List([
                cl.Text(f"**User:** {history['user_message']}\n**AI:** {history['ai_response']}")
                for history in self.conversation_history
            ])
        ])

@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, Any],
    default_user: cl.User
) -> Optional[cl.User]:
    logger.info(f"OAuth callback called. Provider: {provider_id}")
    logger.info(f"Raw user data: {raw_user_data}")

    allowed_emails_env = os.getenv("ALLOWED_EMAILS", "")
    allowed_emails = [email.strip() for email in allowed_emails_env.split(",") if email.strip()]
    if provider_id == "google":
        email = raw_user_data.get("email")
        logger.info(f"User email: {email}")
        if email in allowed_emails:
            logger.info(f"Email {email} is allowed")
            db = SessionLocal()
            try:
                user = db.query(User).filter(User.email == email).first()
                if user:
                    logger.info(f"Existing user found: {user.email}")
                else:
                    logger.info("Creating new user")
                    user = User(
                        email=email,
                        username=raw_user_data.get("name", email),
                        image=raw_user_data.get("picture")
                    )
                    db.add(user)
                    db.commit()
                    logger.info(f"New user created: {user.email}")
                return cl.User(identifier=user.email, username=user.username, image=user.image)
            except Exception as e:
                logger.error(f"Error accessing database: {str(e)}")
                db.rollback()
            finally:
                db.close()
        else:
            logger.warning(f"Unauthorized access attempt from email: {email}")
    else:
        logger.warning(f"Unsupported provider: {provider_id}")
    return None

# Run the Chainlit app
if __name__ == "__main__":
    cl.serve(
        sidebar=[
            ConversationHistory()
        ]
    ) 
