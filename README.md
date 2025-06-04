# 🧬  BioASQ-40k RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot built on a comprehensive biomedical corpus from Hugging Face. This system combines state-of-the-art dense retrieval with FAISS indexing and semantic embeddings to enable fast, accurate question answering over 40,000+ curated scientific passages covering diverse biomedical domains.

---

## 🚀 Features

- ✅ **Comprehensive Biomedical Corpus**: 40K+ professionally curated passages spanning medical literature, research papers, and clinical documentation
- ✅ **Advanced Embedding Pipeline**: Utilizes `intfloat/e5-base` model for high-quality semantic embeddings with GPU acceleration support
- ✅ **Optimized Vector Search**: FAISS-powered vector database enabling sub-second retrieval across massive document collections
- ✅ **Flexible LLM Integration**: Compatible with OpenAI GPT models and local Hugging Face transformers for generation
- ✅ **Interactive Web Interface**: Clean, responsive Gradio UI for seamless user interaction and real-time Q&A
- ✅ **Cloud-Ready Deployment**: One-click deployment to Hugging Face Spaces, Google Colab, or custom cloud infrastructure
- ✅ **Intelligent Document Chunking**: Smart text segmentation preserving semantic coherence and context
- ✅ **Retrieval Quality Metrics**: Built-in evaluation tools for assessing answer relevance and accuracy

---

## 🏗️ Architecture

### System Overview
```
User Query → Embedding Model → FAISS Retrieval → Context Ranking → LLM Generation → Response
```

The system follows a sophisticated RAG pipeline:

1. **Query Processing**: User questions are processed and embedded using the same model as the corpus
2. **Semantic Retrieval**: FAISS performs approximate nearest neighbor search across embedded passages
3. **Context Selection**: Top-k most relevant passages are retrieved and ranked
4. **Answer Generation**: Selected context is fed to the language model for response synthesis
5. **Response Delivery**: Generated answers are returned with source attribution

### Key Components

- **Embedding Model**: `intfloat/e5-base` - A robust multilingual embedding model optimized for retrieval tasks
- **Vector Database**: FAISS (Facebook AI Similarity Search) for efficient similarity search and clustering
- **Generation Models**: Support for OpenAI GPT-3.5/4 or local models like Llama, Mistral, or BioBERT
- **Interface**: Gradio-based web application with customizable themes and layouts

---

## 📊 Dataset

The biomedical corpus includes:

- **Medical Literature**: Peer-reviewed research papers and clinical studies
- **Drug Information**: Pharmaceutical data, interactions, and mechanisms
- **Disease Descriptions**: Comprehensive pathology and symptom information
- **Treatment Protocols**: Evidence-based therapeutic guidelines
- **Anatomical References**: Detailed human anatomy and physiology content
- **Diagnostic Procedures**: Medical imaging, laboratory tests, and clinical assessments

**Data Sources**: rag-datasets/rag-mini-bioasq.

---

## 🔧 Usage

### Basic Query Examples

```python
# Medical condition inquiry
query = "What are the symptoms of Type 2 diabetes?"

# Drug interaction check
query = "What are the side effects of combining metformin and insulin?"

# Treatment information
query = "What are the latest treatment options for rheumatoid arthritis?"

# Diagnostic procedures
query = "How is Alzheimer's disease diagnosed?"
```
