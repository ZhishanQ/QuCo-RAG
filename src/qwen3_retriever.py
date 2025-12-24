"""
Qwen3-Embedding Retriever
For DRAGIN project
"""

import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import logging
from typing import List

logger = logging.getLogger(__name__)

class Qwen3Retriever:
    def __init__(self, model_name_or_path, qwen3_encode_file_path, passage_file):
        """
        Initialize Qwen3 retriever
        
        Args:
            model_name_or_path: Qwen3 model path
            qwen3_encode_file_path: Pre-encoded vector file directory
            passage_file: Document file path
        """
        logger.info(f"Loading Qwen3-Embedding model from {model_name_or_path}")
        
        # Set tokenizer to use left padding
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
        
        # Fix Qwen3 model on GPU 0
        try:
            self.model = AutoModel.from_pretrained(
                model_name_or_path, 
                device_map="cuda:0",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16
            )
            logger.info("Loaded model with flash_attention_2 and float16")
        except Exception as e:
            logger.warning(f"Failed to load with flash_attention_2: {e}")
            self.model = AutoModel.from_pretrained(model_name_or_path, device_map="cuda:0")
        
        self.model.eval()
        logger.info(f"Qwen3 model loaded on device: {next(self.model.parameters()).device}")
        self.max_length = 8192
        
        # Default task description
        self.task_description = "Given a web search query, retrieve relevant passages that answer the query"
        
        logger.info(f"Building Qwen3 indexes")
        
        # Load pre-encoded vectors
        self.p_reps = []
        
        encode_file_path = qwen3_encode_file_path
        if not os.path.exists(encode_file_path):
            raise FileNotFoundError(f"Qwen3 encode file path does not exist: {encode_file_path}")
        
        dir_names = sorted(os.listdir(encode_file_path))
        logger.info(f"Found {len(dir_names)} files in encode directory")
        if len(dir_names) == 0:
            raise ValueError(f"No files found in encode directory: {encode_file_path}")
        
        # Calculate number of chunks
        split_parts = 0
        for d in dir_names:
            if d.endswith('_0.pt'):
                split_parts += 1
        
        logger.info(f"Found {split_parts} split parts")
        if split_parts == 0:
            raise ValueError(f"No valid split parts found. Files should be named as '0_0.pt', '1_0.pt', etc.")

        # Load all embedding files
        pbar = tqdm(range(split_parts), desc="Loading embeddings")
        for i in pbar:
            filename = f"{i}_0.pt"
            file_path = os.path.join(encode_file_path, filename)
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            pbar.set_description(f"Loading {filename}")
            embeddings = torch.load(file_path)
            
            if i <= 10:
                logger.info(f"Embedding dtype for {filename}: {embeddings.dtype}, loading to GPU 0.")

            if embeddings.dtype != torch.float16:
                embeddings = embeddings.half()
                logger.info(f"Converted embeddings to float16 for {filename}")
                logger.info(f"New dtype: {embeddings.dtype}")

            # FP16 embeddings are already normalized during generation, skip check to improve loading speed
            # If verification is needed, use the check_embedding_normalization.py script
            
            # Split in half for storage (improve retrieval efficiency), fixed on GPU 0
            sz = embeddings.shape[0] // 2
            emb1 = embeddings[:sz, :].cuda(0)
            emb2 = embeddings[sz:, :].cuda(0)
            
            self.p_reps.append(emb1)
            self.p_reps.append(emb2)
        
        pbar.set_description("Embedding loading completed")
        
        # Load documents
        docs_file = passage_file
        df = pd.read_csv(docs_file, delimiter='\t')
        self.docs = list(df['text'])
        
        logger.info(f"Loaded {len(self.p_reps)} embedding chunks and {len(self.docs)} documents")

    def last_token_pool(self, last_hidden_states, attention_mask):
        """Last token pooling"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description, query):
        """Add instruction to query"""
        return f'Instruct: {task_description}\nQuery:{query}'

    def encode_query(self, queries: List[str]):
        """Encode queries"""
        # Add instruction
        processed_queries = [self.get_detailed_instruct(self.task_description, query) for query in queries]
        
        # Tokenize
        batch_dict = self.tokenizer(
            processed_queries,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Move to model device
        device = next(self.model.parameters()).device
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        
        # Last token pooling
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

    def retrieve(self, queries: List[str], topk: int = 1):
        """Retrieve relevant documents"""
        if len(self.p_reps) == 0:
            raise RuntimeError("No passage representations loaded. Check if Qwen3 index files exist and are properly formatted.")
        
        # Encode queries
        q_reps = self.encode_query(queries)
        q_reps.requires_grad_(False)
        # Convert to float16 to match p_rep dtype
        q_reps = q_reps.half()
        q_reps_trans = torch.transpose(q_reps, 0, 1)

        topk_values_list = []
        topk_indices_list = []
        prev_count = 0
        
        for p_rep in self.p_reps:
            # Compute similarity (cosine similarity)
            sim = p_rep @ q_reps_trans.to(p_rep.device)
            
            topk_values, topk_indices = torch.topk(sim, k=topk, dim=0)
            topk_values_list.append(topk_values.to('cpu'))
            topk_indices_list.append(topk_indices.to('cpu') + prev_count)
            prev_count += p_rep.shape[0]

        # Merge all results
        all_topk_values = torch.cat(topk_values_list, dim=0)
        global_topk_values, global_topk_indices = torch.topk(all_topk_values, k=topk, dim=0)

        # Get corresponding documents
        psgs = []
        for qid in range(q_reps.shape[0]):
            ret = []
            for j in range(topk):
                idx = global_topk_indices[j][qid].item()
                fid, rk = idx // topk, idx % topk
                psg = self.docs[topk_indices_list[fid][rk][qid]]
                ret.append(psg)
            psgs.append(ret)
        
        return psgs
