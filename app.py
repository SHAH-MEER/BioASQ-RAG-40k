import os
import openai
import gradio as gr
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Constants
DB_NAME = 'vector_db'
MODEL = "gpt-3.5-turbo"

def initialize_components():
    """Initialize all components with error handling"""
    try:
        # Check if vector database exists
        if not os.path.exists(DB_NAME):
            logger.error(f"Vector database '{DB_NAME}' not found")
            raise FileNotFoundError(f"Vector database '{DB_NAME}' not found in the current directory")
        
        # Load vector database
        logger.info("Loading vector database...")
        db = FAISS.load_local(DB_NAME, OpenAIEmbeddings(openai_api_key=openai_api_key))
        
        # Initialize LLM
        logger.info("Initializing LLM...")
        llm = ChatOpenAI(
            model_name=MODEL, 
            temperature=0, 
            api_key=openai_api_key
        )
        
        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        # Create retriever
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        # Create chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever, 
            memory=memory,
            return_source_documents=True
        )
        
        logger.info("All components initialized successfully")
        return chain
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

# Initialize components
try:
    chain = initialize_components()
except Exception as e:
    logger.error(f"Failed to initialize application: {str(e)}")
    chain = None

def chat(message, history):
    """Chat function for Gradio interface"""
    if chain is None:
        return "❌ Error: Application not properly initialized. Please check the logs."
    
    try:
        # Get response from chain
        response = chain({"question": message})
        answer = response['answer']
        
        # Optionally include source information
        if 'source_documents' in response and response['source_documents']:
            sources = "\n\n📚 Sources:"
            for i, doc in enumerate(response['source_documents'][:2]):  # Limit to 2 sources
                source_text = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                sources += f"\n{i+1}. {source_text}"
            answer += sources
        
        return answer
        
    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}")
        return f"❌ Sorry, I encountered an error: {str(e)}"

def clear_memory():
    """Clear conversation memory"""
    if chain and hasattr(chain, 'memory'):
        chain.memory.clear()
    return "🔄 Conversation history cleared!"

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="RAG Chat Assistant") as demo:
    gr.Markdown("# 🤖 RAG Chat Assistant")
    gr.Markdown("Ask questions and I'll search through the knowledge base to provide accurate answers!")
    
    chatbot = gr.Chatbot(height=500, show_label=False)
    msg = gr.Textbox(
        placeholder="Type your question here...", 
        show_label=False,
        container=False
    )
    
    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear History", variant="secondary")
    
    # Event handlers
    def respond(message, chat_history):
        if not message.strip():
            return chat_history, ""
        
        bot_message = chat(message, chat_history)
        chat_history.append((message, bot_message))
        return chat_history, ""
    
    def clear_history():
        clear_memory()
        return []
    
    # Bind events
    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
    clear_btn.click(clear_history, outputs=[chatbot])

# Launch the app
if __name__ == "__main__":
    demo.launch()