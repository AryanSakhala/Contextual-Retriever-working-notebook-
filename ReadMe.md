**Contextual Retrieval using VLLM Notebook using llamaIndex**

This notebook provides a comprehensive guide for implementing **Contextual Retrieval** using **VLLM**. To get started, follow these essential steps:

### Setting up VLLM

* Replace `<YOUR_API_URI>` with the actual address where **VLLM** is hosted.
* Ensure the correct **model** and **max_tokens** are set for your specific use case.
* Adjust or set the **`prompt_template`** to align with your project's requirements.

### Milvus Vector DB Setup

* Set up **Milvus Vector DB** and verify it's running correctly.

**Best of Luck with Your Project!**




<span style="color:white;">Skip to content
logo
LlamaIndex
Contextual Retrieval

Home
Learn
Use Cases
Examples
Component Guides
Advanced Topics
API Reference
Open-Source Community
LlamaCloud
Examples
Agents
Chat Engines
Cookbooks
GraphRAG Implementation with LlamaIndex
GraphRAG Implementation with LlamaIndex - V2
AirtrainAI Cookbook
Anthropic Haiku Cookbook
Trustworthy RAG with the Trustworthy Language Model
Codestral from MistralAI Cookbook
Cohere init8 and binary Embeddings Retrieval Evaluation
Contextual Retrieval
CrewAI + LlamaIndex Cookbook
Llama3 Cookbook
Llama3 Cookbook with Groq
Llama3 Cookbook with Ollama and Replicate
MistralAI Cookbook
mixedbread Rerank Cookbook
Optimizing for relevance using MongoDB and LlamaIndex
Oracle AI Vector Search with Document Processing
Components Of LlamaIndex
Evaluating RAG Systems
Ingestion Pipeline
Metadata Extraction
Observability
Agents
Router QueryEngine and SubQuestion QueryEngine
Multi-Modal RAG System
Advanced RAG with LlamaParse
Prometheus-2 Cookbook
Sales Prospecting Workflow with Toolhouse
Customization
Data Connectors
Discover LlamaIndex
Docstores
Embeddings
Evaluation
Finetuning
Ingestion
LLMs
Llama Datasets
Llama Hub
Low Level
Managed Indexes
Metadata Extractors
Multi-Modal
Multi-Tenancy
Node Parsers & Text Splitters
Node Postprocessors
Object Stores
Observability
Output Parsers
Param Optimizer
Prompts
Property Graph
Query Engines
Query Pipeline
Query Transformations
Response Synthesizers
Retrievers
Tools
Transforms
Use Cases
Vector Stores
Workflow
Table of contents
Installation
Setup API Keys
Setup LLM and Embedding model
Download Data
Load Data
Prompts for creating context for each chunk
Utils
Create Nodes
Set node ids
Create contextual nodes
Set
Set
Create retrievers.
Create retrievers using contextual nodes.
Create Synthetic query dataset
Evaluate retrievers with and without contextual nodes
Display results
Without Context
With Context
Observation:
Open In Colab

Contextual Retrieval
In this notebook we will demonstrate how you can implement Anthropic's Contextual Retrieval using LlamaIndex abstractions.

We will use:

Paul Graham Essay dataset.
Anthropic LLM for context creation for each chunk.
OpenAI LLM for Synthetic query generation and embedding model.
CohereAI Reranker.
Installation
!pip install -U llama-index llama-index-llms-anthropic llama-index-postprocessor-cohere-rerank llama-index-retrievers-bm25 stemmer
import nest_asyncio

nest_asyncio.apply()
Setup API Keys
import os

# For creating context for each chunk
os.environ["ANTHROPIC_API_KEY"] = "<YOUR ANTHROPIC API KEY>"

# For creating synthetic dataset and embedding model
os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI API KEY>"

# For reranker
os.environ["COHERE_API_KEY"] = "<YOUR COHEREAI API KEY>"
Setup LLM and Embedding model
from llama_index.llms.anthropic import Anthropic

llm_anthropic = Anthropic(model="claude-3-5-sonnet-20240620")
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

Settings.embed_model = OpenAIEmbedding()
Download Data
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O './paul_graham_essay.txt'
--2024-10-01 13:00:06--  https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 75042 (73K) [text/plain]
Saving to: ‘./paul_graham_essay.txt’

./paul_graham_essay 100%[===================>]  73.28K  --.-KB/s    in 0.08s   

2024-10-01 13:00:06 (921 KB/s) - ‘./paul_graham_essay.txt’ saved [75042/75042]

Load Data
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./paul_graham_essay.txt"],
).load_data()

WHOLE_DOCUMENT = documents[0].text
Prompts for creating context for each chunk
We will utilize anthropic prompt caching for creating context for each chunk. If you haven’t explored our integration yet, please take a moment to review it here.

prompt_document = """<document>
{WHOLE_DOCUMENT}
</document>"""

prompt_chunk = """Here is the chunk we want to situate within the whole document
<chunk>
{CHUNK_CONTENT}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
Utils
create_contextual_nodes - Function to create contextual nodes for a list of nodes.

create_embedding_retriever - Function to create an embedding retriever for a list of nodes.

create_bm25_retriever - Function to create a bm25 retriever for a list of nodes.

EmbeddingBM25RerankerRetriever - Custom retriever that uses both embedding and bm25 retrievers and reranker.

create_eval_dataset - Function to create a evaluation dataset from a list of nodes.

set_node_ids - Function to set node ids for a list of nodes.

retrieval_results - Function to get retrieval results for a retriever and evaluation dataset.

display_results - Function to display results from retrieval_results

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    RetrieverEvaluator,
)
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.llms import ChatMessage

import pandas as pd
import copy
import Stemmer

from typing import List


def create_contextual_nodes(nodes_):
    """Function to create contextual nodes for a list of nodes"""
    nodes_modified = []
    for node in nodes_:
        new_node = copy.deepcopy(node)
        messages = [
            ChatMessage(role="system", content="You are helpful AI Assitant."),
            ChatMessage(
                role="user",
                content=[
                    {
                        "text": prompt_document.format(
                            WHOLE_DOCUMENT=WHOLE_DOCUMENT
                        ),
                        "type": "text",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "text": prompt_chunk.format(CHUNK_CONTENT=node.text),
                        "type": "text",
                    },
                ],
            ),
        ]
        new_node.metadata["context"] = str(
            llm_anthropic.chat(
                messages,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            )
        )
        nodes_modified.append(new_node)

    return nodes_modified


def create_embedding_retriever(nodes_, similarity_top_k=2):
    """Function to create an embedding retriever for a list of nodes"""
    vector_index = VectorStoreIndex(nodes_)
    retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)
    return retriever


def create_bm25_retriever(nodes_, similarity_top_k=2):
    """Function to create a bm25 retriever for a list of nodes"""
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes_,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    return bm25_retriever


def create_eval_dataset(nodes_, llm, num_questions_per_chunk=2):
    """Function to create a evaluation dataset for a list of nodes"""
    qa_dataset = generate_question_context_pairs(
        nodes_, llm=llm, num_questions_per_chunk=num_questions_per_chunk
    )
    return qa_dataset


def set_node_ids(nodes_):
    """Function to set node ids for a list of nodes"""

    # by default, the node ids are set to random uuids. To ensure same id's per run, we manually set them.
    for index, node in enumerate(nodes_):
        node.id_ = f"node_{index}"

    return nodes_


async def retrieval_results(retriever, eval_dataset):
    """Function to get retrieval results for a retriever and evaluation dataset"""

    metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]

    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        metrics, retriever=retriever
    )

    eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)

    return eval_results


def display_results(name, eval_results):
    """Display results from evaluate."""

    metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    columns = {
        "retrievers": [name],
        **{k: [full_df[k].mean()] for k in metrics},
    }

    metric_df = pd.DataFrame(columns)

    return metric_df


class EmbeddingBM25RerankerRetriever(BaseRetriever):
    """Custom retriever that uses both embedding and bm25 retrievers and reranker"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        bm25_retriever: BM25Retriever,
        reranker: CohereRerank,
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker

        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)

        vector_nodes.extend(bm25_nodes)

        retrieved_nodes = self.reranker.postprocess_nodes(
            vector_nodes, query_bundle
        )

        return retrieved_nodes
Create Nodes
from llama_index.core.node_parser import SentenceSplitter

node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)
Set node ids
Useful to have consistent result comparison for nodes with and without contextual text.

# set node ids

nodes = set_node_ids(nodes)
nodes[0].metadata
{'file_path': 'paul_graham_essay.txt',
 'file_name': 'paul_graham_essay.txt',
 'file_type': 'text/plain',
 'file_size': 75042,
 'creation_date': '2024-10-01',
 'last_modified_date': '2024-10-01'}
Create contextual nodes
nodes_contextual = create_contextual_nodes(nodes)
nodes[0].metadata, nodes_contextual[0].metadata
({'file_path': 'paul_graham_essay.txt',
  'file_name': 'paul_graham_essay.txt',
  'file_type': 'text/plain',
  'file_size': 75042,
  'creation_date': '2024-10-01',
  'last_modified_date': '2024-10-01'},
 {'file_path': 'paul_graham_essay.txt',
  'file_name': 'paul_graham_essay.txt',
  'file_type': 'text/plain',
  'file_size': 75042,
  'creation_date': '2024-10-01',
  'last_modified_date': '2024-10-01',
  'context': 'assistant: This chunk is the opening section of Paul Graham\'s essay "What I Worked On," describing his early experiences with programming and writing as a teenager, his initial interest in philosophy in college, and his subsequent shift to studying artificial intelligence in the mid-1980s.'})
Set similarity_top_k
similarity_top_k = 3
Set CohereReranker
from llama_index.postprocessor.cohere_rerank import CohereRerank

cohere_rerank = CohereRerank(
    api_key=os.environ["COHERE_API_KEY"], top_n=similarity_top_k
)
Create retrievers.
Embedding based retriever.
BM25 based retriever.
Embedding + BM25 + Cohere reranker retriever.
embedding_retriever = create_embedding_retriever(
    nodes, similarity_top_k=similarity_top_k
)
bm25_retriever = create_bm25_retriever(
    nodes, similarity_top_k=similarity_top_k
)
embedding_bm25_retriever_rerank = EmbeddingBM25RerankerRetriever(
    embedding_retriever, bm25_retriever, reranker=cohere_rerank
)
DEBUG:bm25s:Building index from IDs objects
Create retrievers using contextual nodes.
contextual_embedding_retriever = create_embedding_retriever(
    nodes_contextual, similarity_top_k=similarity_top_k
)
contextual_bm25_retriever = create_bm25_retriever(
    nodes_contextual, similarity_top_k=similarity_top_k
)
contextual_embedding_bm25_retriever_rerank = EmbeddingBM25RerankerRetriever(
    contextual_embedding_retriever,
    contextual_bm25_retriever,
    reranker=cohere_rerank,
)
DEBUG:bm25s:Building index from IDs objects
Create Synthetic query dataset
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4")

qa_dataset = create_eval_dataset(nodes, llm=llm, num_questions_per_chunk=2)
100%|██████████| 21/21 [02:59<00:00,  8.53s/it]
list(qa_dataset.queries.values())[1]
"The author initially intended to study philosophy in college but later switched to AI. Discuss the reasons behind this shift in interest and how specific influences like Heinlein's novel and Winograd's SHRDLU contributed to his decision."
Evaluate retrievers with and without contextual nodes
embedding_retriever_results = await retrieval_results(
    embedding_retriever, qa_dataset
)
bm25_retriever_results = await retrieval_results(bm25_retriever, qa_dataset)
embedding_bm25_retriever_rerank_results = await retrieval_results(
    embedding_bm25_retriever_rerank, qa_dataset
)
contextual_embedding_retriever_results = await retrieval_results(
    contextual_embedding_retriever, qa_dataset
)
contextual_bm25_retriever_results = await retrieval_results(
    contextual_bm25_retriever, qa_dataset
)
contextual_embedding_bm25_retriever_rerank_results = await retrieval_results(
    contextual_embedding_bm25_retriever_rerank, qa_dataset
)
Display results
Without Context
pd.concat(
    [
        display_results("Embedding Retriever", embedding_retriever_results),
        display_results("BM25 Retriever", bm25_retriever_results),
        display_results(
            "Embedding + BM25 Retriever + Reranker",
            embedding_bm25_retriever_rerank_results,
        ),
    ],
    ignore_index=True,
    axis=0,
)
retrievers	hit_rate	mrr	precision	recall	ap	ndcg
0	Embedding Retriever	0.857143	0.726190	0.285714	0.857143	0.726190	0.356613
1	BM25 Retriever	0.904762	0.777778	0.301587	0.904762	0.777778	0.380157
2	Embedding + BM25 Retriever + Reranker	0.952381	0.865079	0.456349	0.952381	0.865079	0.530172
With Context
pd.concat(
    [
        display_results(
            "Contextual Embedding Retriever",
            contextual_embedding_retriever_results,
        ),
        display_results(
            "Contextual BM25 Retriever", contextual_bm25_retriever_results
        ),
        display_results(
            "Contextual Embedding + Contextual BM25 Retriever + Reranker",
            contextual_embedding_bm25_retriever_rerank_results,
        ),
    ],
    ignore_index=True,
    axis=0,
)
retrievers	hit_rate	mrr	precision	recall	ap	ndcg
0	Contextual Embedding Retriever	0.928571	0.746032	0.309524	0.928571	0.746032	0.372175
1	Contextual BM25 Retriever	0.952381	0.829365	0.317460	0.952381	0.829365	0.403967
2	Contextual Embedding + Contextual BM25 Retriev...	0.976190	0.900794	0.476190	0.976190	0.900794	0.555746
Observation:
We observed improved metrics with contextual retrieval; however, our experiments showed that much depends on the queries, chunk size, chunk overlap, and other variables. Therefore, it’s essential to experiment to optimize the benefits of this technique.

 Back to top
Previous
Cohere init8 and binary Embeddings Retrieval Evaluation
Next
CrewAI + LlamaIndex Cookbook
Read the Docs
 stable

Hi, how can I help you?

🦙</span>
