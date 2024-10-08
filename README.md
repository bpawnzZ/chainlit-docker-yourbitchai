# Chainlit AI Assistant

This repository contains a Chainlit-based AI assistant that uses OpenAI's GPT model, Qdrant for vector storage, and PostgreSQL for user management. The assistant can perform web searches, process uploaded documents, and maintain conversation history.

## Features

- OAuth authentication with Google
- Web search functionality using SearX
- Document processing (PDF and TXT files)
- Conversation history management
- Vector storage for similarity search
- Rate limiting for API calls
- Caching for LLM responses

## Prerequisites

- Docker and Docker Compose
- OpenAI API key
- Google OAuth credentials
- SearX instance

## Setup

1. Clone this repository:
git clone https://github.com/your-username/your-repo-name.git cd your-repo-name



Copy code


2. Create a `.env` file in the root directory with the following variables:
DB_USER=your_db_user DB_PASSWORD=your_db_password DB_HOST=db DB_PORT=5432 DB_NAME=your_db_name OAUTH_GOOGLE_CLIENT_ID=your_google_client_id OAUTH_GOOGLE_CLIENT_SECRET=your_google_client_secret OAUTH_REDIRECT_URI=your_redirect_uri OPENAI_API_KEY=your_openai_api_key OPENAI_BASE_URL=https://api.openai.com/v1 QDRANT_HOST=qdrant QDRANT_PORT=6333 FILES_DIRECTORY=/app/files SEARX_HOST=your_searx_host ALLOWED_EMAILS=email1@example.com,email2@example.com



Copy code


3. Create a `files` directory in the project root for document storage.

## Running the Application

To run the application, use Docker Compose:

docker-compose up --build



Copy code


The application will be available at `http://localhost:8501`.

## Usage

1. Open the application in your web browser.
2. Log in using your Google account (must be in the allowed emails list).
3. Start chatting with the AI assistant.
4. You can upload PDF or TXT files for processing.
5. The assistant will perform web searches and use its knowledge base to answer your questions.

## Development

The main application logic is in `app.py`. To modify the assistant's behavior, edit this file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Insert your chosen license here]

