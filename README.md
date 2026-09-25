# 🔎 Multimodal Search System

A cross-modal image retrieval system that enables users to search an image dataset using natural-language text queries, images, negative prompts, and hybrid multimodal inputs.

The system uses OpenAI CLIP to generate shared text-image embeddings and FAISS for efficient vector similarity search. An interactive Streamlit interface provides search, visualization, evaluation, and structured query-agent functionality.

---

## 🚀 Overview

Traditional search systems generally rely on keywords or manually assigned metadata. This project uses multimodal embeddings to understand the semantic relationship between text and images.

For example:

    red sports car

can retrieve visually relevant car images even when the exact words do not appear in the image metadata.

The system supports:

- Text-to-Image Search
- Image-to-Image Search
- Hybrid Text + Image Search
- Negative-Prompt Search
- FAISS Vector Similarity Search
- Structured Query Agent
- Retrieval Evaluation
- Embedding Visualization
- Incremental Image Ingestion
- Automated Unit Tests
- Failure Analysis

---

## ✨ Key Features

### 1. Text-to-Image Search

Users can enter natural-language queries such as:

    red sports car

The query is converted into a CLIP text embedding and compared against image embeddings stored in the FAISS index.

The system returns the most semantically similar images.

---

### 2. Image-to-Image Search

Users can upload an image and search for visually similar images.

The uploaded image is converted into a CLIP image embedding and compared against the indexed image vectors.

Example workflow:

    Upload Image
          ↓
    CLIP Image Encoder
          ↓
    512-Dimensional Embedding
          ↓
    FAISS Similarity Search
          ↓
    Top-K Similar Images

---

### 3. Hybrid Multimodal Search

The system can combine information from both an image and a text query.

For example:

    Image: Sports car
    Text: Red

The image and text embeddings are combined to create a hybrid query representation.

Conceptually:

    v_hybrid = α(v_text) + (1 - α)(v_image)

The resulting vector is normalized before being passed to FAISS.

This allows users to refine visual searches using natural-language descriptions.

---

### 4. Negative-Prompt Search

The system supports excluding unwanted concepts from a search.

Example:

    red sports car without vintage styling

The query representation can be modified using vector arithmetic:

    v_query = normalize(v_positive - β(v_negative))

where:

- v_positive = embedding of the desired query
- v_negative = embedding of the unwanted concept
- β = negative-prompt strength

This provides a semantic mechanism for reducing unwanted concepts in the retrieved results.

---

### 5. Structured Query Agent

The project includes a structured query agent that converts natural-language requests into a validated retrieval plan.

Example:

    Find red sports cars without vintage styling

can be converted into a structured plan such as:

    {
        "query": "Find red sports cars",
        "mode": "text",
        "negative_prompt": "vintage styling",
        "top_k": 8
    }

The agent then selects the appropriate retrieval operation.

The system records structured JSONL traces containing:

- Request ID
- Query plan
- Selected tool
- Retrieval latency
- Result count
- Timestamp

Image bytes are not stored in the trace logs.

---

## 🧠 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning and model execution |
| Torchvision | Image processing utilities |
| OpenAI CLIP | Multimodal text/image embeddings |
| Hugging Face Transformers | CLIP model implementation |
| FAISS | Vector similarity search |
| NumPy | Numerical computation |
| Pandas | Dataset and metadata processing |
| Pillow | Image loading and processing |
| Streamlit | Interactive web application |
| Scikit-learn | Evaluation and dimensionality reduction |
| Matplotlib | Visualization |
| Seaborn | Evaluation visualization |
| PyArrow | Parquet metadata storage |
| PyTest | Unit testing |
| tqdm | Progress tracking |

---

## 🏗️ System Architecture

    Dataset Images
          |
          v
    CLIP Image Encoder
          |
          v
    Image Embeddings
          |
          v
    L2 Normalization
          |
          v
    FAISS IndexFlatIP
          |
          v
    Vector Index
          ^
          |
    +-----+----------------+
    |                      |
    |                      |
    v                      v
    Text Query        Uploaded Image
    |                      |
    v                      v
    CLIP Text          CLIP Image
    Encoder             Encoder
    |                      |
    +----------+-----------+
               |
               v
      Query Embedding
               |
               v
    Hybrid / Negative Prompt
               |
               v
       FAISS Similarity Search
               |
               v
        Top-K Results
               |
               v
       Streamlit Interface

---

## 🔬 How It Works

### Step 1 — Image Encoding

Every image in the dataset is passed through the CLIP image encoder.

The model used is:

    openai/clip-vit-base-patch32

The resulting image representation contains 512 dimensions.

---

### Step 2 — L2 Normalization

The embeddings are normalized using L2 normalization:

    v_normalized = v / ||v||₂

This converts every vector to unit length.

---

### Step 3 — FAISS Indexing

The normalized vectors are stored in:

    FAISS IndexFlatIP

Because the vectors are L2-normalized, inner-product similarity corresponds to cosine similarity.

    cosine_similarity = vector_1 · vector_2

This allows the system to retrieve the nearest image vectors.

---

### Step 4 — Query Processing

For a text query such as:

    red sports car

CLIP generates a text embedding.

The text embedding is then compared against the indexed image embeddings.

---

### Step 5 — Top-K Retrieval

FAISS returns the highest-similarity vectors.

The corresponding image metadata is retrieved and displayed through the Streamlit interface.

---

## 📁 Project Structure

    Multimodal Search System/
    │
    ├── app.py
    ├── build_index.py
    ├── evaluate.py
    ├── prepare_dataset.py
    ├── requirements.txt
    ├── README.md
    │
    ├── dataset/
    │   ├── images/
    │   ├── metadata.csv
    │   └── test_queries.json
    │
    ├── index/
    │   ├── embeddings.npy
    │   ├── faiss_index.bin
    │   └── metadata.parquet
    │
    ├── eval_results/
    │   ├── evaluation_report.json
    │   ├── failure_analysis.json
    │   └── precision_at_k.png
    │
    ├── src/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── dataset_loader.py
    │   ├── evaluator.py
    │   ├── indexer.py
    │   ├── models.py
    │   ├── query_agent.py
    │   └── visualization.py
    │
    ├── tests/
    │   └── test_query_agent.py
    │
    └── .github/
        └── workflows/
            └── ci.yml

---

## ⚙️ Installation

### 1. Clone the Repository

    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

Move into the project directory:

    cd "Multimodal Search System"

---

### 2. Create a Virtual Environment

#### Windows

    python -m venv venv

Activate the environment:

    venv\Scripts\activate

#### Linux / macOS

    python3 -m venv venv

Activate the environment:

    source venv/bin/activate

---

### 3. Install Dependencies

    pip install -r requirements.txt

---

## ▶️ Running the Project

If the repository already contains the generated FAISS index and metadata, start the Streamlit application using:

    streamlit run app.py

The application will normally be available at:

    http://localhost:8501

---

## 🏗️ Preparing the Dataset

To prepare the dataset:

    python prepare_dataset.py

This prepares the dataset metadata and test-query information used by the retrieval and evaluation pipeline.

---

## 🧮 Building the FAISS Index

To generate image embeddings and construct the FAISS index:

    python build_index.py

The generated files include:

    index/faiss_index.bin
    index/metadata.parquet
    index/embeddings.npy

---

## 📊 Running Evaluation

Run:

    python evaluate.py

The evaluation pipeline measures retrieval performance using metrics such as:

- Precision@K
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)
- Search latency

Evaluation results are stored inside:

    eval_results/

---

## 🧪 Running Tests

Run the test suite using:

    python -m pytest tests/test_query_agent.py -q

The tests validate:

- Structured query parsing
- Negative-prompt detection
- Top-K constraints
- Query tracing
- Protection against storing image payload bytes

---

## 📈 Evaluation

The project contains an evaluation pipeline for measuring retrieval quality.

### Precision@K

Measures the fraction of retrieved results that are relevant among the top K results.

### Mean Reciprocal Rank (MRR)

Measures how early the first relevant result appears in the ranked retrieval results.

### Mean Average Precision (MAP)

Measures ranking quality across multiple relevant results.

### Search Latency

Measures the time required to process a retrieval query.

Example evaluation artifacts include:

    evaluation_report.json
    failure_analysis.json
    precision_at_k.png

---

## 🔍 Failure Analysis

Multimodal retrieval models can sometimes confuse visually similar concepts.

For example, a query describing:

    yellow sports shoes

may retrieve visually similar yellow objects or visually similar footwear.

Fine-grained categories can also be difficult when differences between classes are subtle.

The project includes failure-analysis functionality to document and analyze retrieval errors.

---

## 📊 Embedding Visualization

The application includes embedding-space visualization functionality.

Embeddings can be projected into two dimensions using dimensionality-reduction techniques such as:

- PCA
- t-SNE

This allows users to inspect relationships between image embeddings and query neighborhoods.

---

## 🔐 Privacy Considerations

The structured query agent is designed to operate locally.

The query tracing system records structured metadata such as:

- Request ID
- Query plan
- Tool name
- Result count
- Latency
- Timestamp

Raw image bytes are not stored in the trace logs.

---

## ⚡ Hardware Support

The system automatically detects the available PyTorch device.

Supported execution environments include:

- CUDA
- Apple MPS
- CPU

For systems without a compatible GPU, the application can fall back to CPU execution.

---

## 📌 Important Files

### app.py

Main Streamlit application containing the interactive search interface.

### build_index.py

Builds the FAISS vector index from the image dataset.

### evaluate.py

Runs the retrieval evaluation pipeline.

### prepare_dataset.py

Prepares dataset metadata and evaluation queries.

### src/models.py

Handles CLIP model loading and embedding generation.

### src/indexer.py

Manages FAISS indexing and similarity retrieval.

### src/query_agent.py

Implements structured query planning, tool selection, and trace logging.

### src/evaluator.py

Contains retrieval evaluation metrics and benchmarking functionality.

### src/visualization.py

Provides embedding and query-neighborhood visualization.

---

## 🧩 Example Queries

The system can handle queries such as:

    red sports car

    black leather boots

    yellow city bike

    blue luxury car

Negative-prompt examples:

    red sports cars without vintage styling

    black shoes excluding boots

---

## 💡 Example Workflow

    User
      |
      v
    Enter Text / Upload Image
      |
      v
    CLIP Encoder
      |
      v
    Multimodal Embedding
      |
      v
    Optional Query Processing
      |
      +---- Hybrid Search
      |
      +---- Negative Prompt
      |
      v
    FAISS IndexFlatIP
      |
      v
    Top-K Similar Results
      |
      v
    Streamlit Interface

---

## 🛠️ Future Improvements

Potential future improvements include:

- Approximate nearest-neighbor FAISS indexes for larger datasets
- GPU-accelerated FAISS search
- Larger CLIP or newer multimodal embedding models
- Metadata-aware filtering
- Improved fine-grained retrieval
- Multilingual text queries
- Persistent incremental indexing
- Larger benchmark datasets
- FastAPI deployment
- Docker-based deployment
- Authentication and user management
- Production-scale vector database integration

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Build a cross-modal retrieval system using multimodal embeddings.
2. Map text and images into a shared semantic vector space.
3. Implement efficient similarity search using FAISS.
4. Support text, image, and hybrid retrieval.
5. Implement semantic negative prompting.
6. Develop an interactive search interface.
7. Evaluate retrieval quality quantitatively.
8. Analyze retrieval failure cases.
9. Provide structured and auditable query execution.

---

## 👨‍💻 Author

**Monish C**

Artificial Intelligence & Machine Learning

PES University, Bengaluru

---

## 📄 License

This project is intended for educational, academic, and portfolio purposes.

If you plan to distribute the project publicly, add an appropriate open-source license to the repository.

---

## ⭐ Acknowledgements

This project uses open-source technologies and research in multimodal representation learning, image retrieval, vector databases, and semantic search.

Key technologies include:

- OpenAI CLIP
- PyTorch
- Hugging Face Transformers
- FAISS
- Streamlit
- Scikit-learn
- NumPy
- Pandas
- Pillow

---

## ⭐ If You Find This Project Useful

Consider giving the repository a star ⭐ and exploring the project to learn more about multimodal search, CLIP embeddings, and vector similarity retrieval.
