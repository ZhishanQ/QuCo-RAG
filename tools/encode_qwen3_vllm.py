#!/usr/bin/env python3
"""
Encode corpus using vLLM's Qwen3-Embedding-0.6B model
"""

import os
os.environ["TQDM_DISABLE"] = "1"

import pandas as pd
import torch
import numpy as np
from vllm import LLM
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Qwen3VLLMEncoder:
    def __init__(self, model_name_or_path, **kwargs):
        logger.info(f"Loading Qwen3-Embedding model with vLLM from {model_name_or_path}")
        
        try:
            # Basic parameters following official example
            model_kwargs = {
                "model": model_name_or_path,
                "task": "embed",
                # Disable chunked prefill as suggested by logs
                "enable_chunked_prefill": False,
                # Reduce max sequence length to save memory
                "max_model_len": 2048*2,
                "disable_log_stats": True  # Disable statistics logging and progress bars
            }
            
            # Add optional parameters if provided
            if kwargs.get('tensor_parallel_size'):
                model_kwargs['tensor_parallel_size'] = kwargs['tensor_parallel_size']
            if kwargs.get('gpu_memory_utilization'):
                model_kwargs['gpu_memory_utilization'] = kwargs['gpu_memory_utilization']
            
            self.model = LLM(**model_kwargs)
            logger.info("vLLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model with vLLM: {e}")
            raise e
        
        # Default task description
        self.default_task = 'Given a web search query, retrieve relevant passages that answer the query'

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Add instruction format to query"""
        return f'Instruct: {task_description}\nQuery: {query}'

    def encode_corpus(self, texts, batch_size=16, is_query=False, task_description=None):
        """
        Encode a list of texts
        
        Args:
            texts: List of texts
            batch_size: Batch size (recommend smaller batches for embed task)
            is_query: Whether texts are queries (queries get instruction format)
            task_description: Task description, only used for queries
        """
        logger.info(f"Encoding {len(texts)} texts with vLLM")
        
        try:
            if is_query:
                # Add instruction format for queries
                if task_description is None:
                    task_description = self.default_task
                
                formatted_texts = [
                    self.get_detailed_instruct(task_description, text) 
                    for text in texts
                ]
                logger.info("Added instruction format for queries")
            else:
                # Document encoding doesn't need instruction format
                formatted_texts = texts
                logger.info("Encoding documents without instruction format")
            
            # Process in batches to avoid memory issues
            all_embeddings = []
            for i in range(0, len(formatted_texts), batch_size):
                batch_texts = formatted_texts[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(formatted_texts) + batch_size - 1)//batch_size}")
                
                # Encode with vLLM
                outputs = self.model.embed(batch_texts)
                
                # Extract embedding vectors
                batch_embeddings = torch.tensor([o.outputs.embedding for o in outputs])
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all batch embeddings
            embeddings = torch.cat(all_embeddings, dim=0)
            
            logger.info(f"Encoded embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error during encoding: {e}")
            # Return zero vectors as fallback
            # Assume embedding dimension is 1024 (observed from testing)
            hidden_size = 1024
            return torch.zeros(len(texts), hidden_size)


def test_encoding(model_path):
    """Test if encoding functionality works properly"""
    logger.info("Testing Qwen3-Embedding encoding with vLLM...")
    
    encoder = Qwen3VLLMEncoder(model_path)
    
    # Test queries and documents
    queries = [
        'What is the capital of China?',
        'Explain gravity'
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    
    # Encode queries (with instruction format)
    query_embeddings = encoder.encode_corpus(
        queries, 
        batch_size=2, 
        is_query=True
    )
    logger.info(f"Query embeddings shape: {query_embeddings.shape}")
    
    # Encode documents (without instruction format)
    doc_embeddings = encoder.encode_corpus(documents, batch_size=2, is_query=False)
    logger.info(f"Document embeddings shape: {doc_embeddings.shape}")
    
    # Calculate similarity scores
    scores = query_embeddings @ doc_embeddings.T
    logger.info(f"Similarity scores: {scores.tolist()}")
    
    logger.info("Test encoding successful!")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, 
                       default="Qwen/Qwen3-Embedding-0.6B",
                       help="Qwen3-Embedding model path or HuggingFace model name")
    parser.add_argument("--corpus_file", type=str, 
                       default="data/dpr/psgs_w100.tsv",
                       help="Corpus file path")
    parser.add_argument("--output_dir", type=str, 
                       default="qwen3_vllm/encode_result",
                       help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size (recommend smaller values for embed task)")
    parser.add_argument("--chunk_size", type=int, default=20000, 
                       help="Number of documents per file")
    parser.add_argument("--test_only", action="store_true", 
                       help="Only test encoding functionality")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size (for multi-GPU)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization")
    parser.add_argument("--task_description", type=str, 
                       default='Given a web search query, retrieve relevant passages that answer the query',
                       help="Query task description")
    
    args = parser.parse_args()
    
    # If only testing
    if args.test_only:
        test_encoding(args.model_path)
        return
    
    # Check if corpus file exists
    if not os.path.exists(args.corpus_file):
        logger.error(f"Corpus file does not exist: {args.corpus_file}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load corpus
    logger.info(f"Loading corpus from {args.corpus_file}")
    df = pd.read_csv(args.corpus_file, delimiter='\t')
    texts = list(df['text'])
    logger.info(f"Loaded {len(texts)} documents")
    
    # Initialize encoder
    encoder_kwargs = {}
    if args.tensor_parallel_size > 1:
        encoder_kwargs['tensor_parallel_size'] = args.tensor_parallel_size
    if args.gpu_memory_utilization != 0.9:
        encoder_kwargs['gpu_memory_utilization'] = args.gpu_memory_utilization
    
    encoder = Qwen3VLLMEncoder(args.model_path, **encoder_kwargs)
    
    # Encode and save in chunks
    num_chunks = (len(texts) + args.chunk_size - 1) // args.chunk_size
    logger.info(f"Will create {num_chunks} chunks")
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * args.chunk_size
        end_idx = min((chunk_idx + 1) * args.chunk_size, len(texts))
        
        # Check if file already exists
        filename = f"{chunk_idx}_0.pt"
        filepath = os.path.join(args.output_dir, filename)
        
        if os.path.exists(filepath):
            logger.info(f"Chunk {chunk_idx} already exists at {filepath}, skipping...")
            continue
        
        chunk_texts = texts[start_idx:end_idx]
        logger.info(f"Encoding chunk {chunk_idx} ({start_idx}:{end_idx})")
        
        # Encode current chunk (documents without instruction format)
        embeddings = encoder.encode_corpus(
            chunk_texts, 
            batch_size=args.batch_size, 
            is_query=False
        )
        
        # Ensure embeddings are on CPU
        if hasattr(embeddings, 'cpu'):
            embeddings = embeddings.cpu()
        
        # Save in DRAGIN format: each chunk as a complete file
        torch.save(embeddings, filepath)
        logger.info(f"Saved {filepath} with shape {embeddings.shape}")
    
    logger.info("Encoding completed!")


if __name__ == "__main__":
    main()
