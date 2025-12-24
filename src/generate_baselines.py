import numpy as np
import logging
import spacy
import torch
import json
import os
import time
import requests
import string
from math import exp
from scipy.special import softmax
from retriever import BM25, SGPT
from qwen3_retriever import Qwen3Retriever
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.generation.utils import GenerateDecoderOnlyOutput
from gpt_api_client import create_gpt_client
from typing import Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

# ==================== ETC Related Classes ====================
@dataclass
class Block:
    """Block data structure for ETC method"""
    text: str = None
    tokens: List[str] = None 
    range_: List[Tuple[int, int]] = None 
    
    @property
    def len_tokens(self):
        return len(self.tokens)
    
    @property
    def len_words(self):
        return len(self.range_)

def merge_blocks(blocks: List[Block]) -> Block:
    """Merge multiple Blocks into one"""
    text = "".join([block.text for block in blocks])
    tokens = sum([block.tokens for block in blocks], [])
    range_ = []
    st = 0
    for block in blocks:
        if block.range_:
            for l, r in block.range_:
                range_.append((st+l, st+r))
            st = range_[-1][1]
    return Block(text=text, tokens=tokens, range_=range_)

def join_if_nonempty(*li, sep=" "):
    """Join non-empty strings"""
    return sep.join([s for s in li if len(s) > 0])

def match(word: str, real_words): 
    """Check if word contains any word in real_words"""
    for real_word in real_words:
        if real_word in word: 
            return True
    return False

@dataclass
class CheckerOutput:
    """ETC hallucination detection output"""
    hallucination: bool 
    curr_st: int = None  # Start position of hallucinated sentence
    curr_en: int = None  # End position of hallucinated sentence
    curr_thres: List[bool] = None

@dataclass
class GeneratorOutput:
    """ETC Generator output"""
    ended: bool
    empty: bool
    blocks: List[Block] = None
    merged_blocks: Block = None
    atten: torch.Tensor = None
    max_atten: torch.Tensor = None
    entropies: torch.Tensor = None
    entropies_s1: List = None
    entropies_s2: List = None
    smooth_s2: List = None
    mt_s2: List = None
    fun_word: List = None
    
    @property
    def new_text(self):
        return self.blocks[-1].text if self.blocks else ""
    
    @property
    def len_new_words(self):
        return self.blocks[-1].len_words if self.blocks else 0

# ==================== End of ETC Related Classes ====================

def get_infini_gram_count(query, index='v4_rpj_llama_s4', query_type='count', max_diff_tokens=1000, max_clause_freq=500000):
    """
    Sends a request to the Infini-gram API and returns the JSON response.

    :param query: The query string.
    :param index: The index to search, defaults to 'v4_rpj_llama_s4'.
    :param query_type: The type of query, defaults to 'count'.
    :param max_diff_tokens: Max token distance for AND co-occurrences, defaults to 1000.
    :param max_clause_freq: Threshold for approximate counts, defaults to 500000.
    :return: JSON data from the API, or None if an error occurs.
    """
    payload = {
        'index': index,
        'query_type': query_type,
        'query': query
    }
    
    # Only add CNF parameters if query contains AND/OR operators
    if 'AND' in query or 'OR' in query:
        payload['max_diff_tokens'] = max_diff_tokens
        payload['max_clause_freq'] = max_clause_freq
    
    response = requests.post('https://api.infini-gram.io/', json=payload)
    result = response.json()
    if 'error' in result:
        print(f"An error occurred: {result['error']}")
        return None
    return result


class BasicGenerator:
    def __init__(
        self,
        model_name_or_path,
        device_map="auto",
        use_api=False,
        api_base_url=None,
        tokenizer_name_or_path: Optional[str] = None,
        use_web_search: bool = False,
    ):
        self.use_api = use_api
        self.model_name_or_path = model_name_or_path
        self.api_base_url = api_base_url or "https://api.openai.com/v1"
        self.model = None
        self.model_config = None
        self.api_client = None
        self.tokenizer = None
        self.space_token = " "
        self.use_web_search = use_web_search  # Store web search flag
        
        # Track whether this is an API model for token usage tracking
        self.is_api_model = "gpt-" in model_name_or_path.lower() if model_name_or_path else False
        self.last_usage = None  # Store last API call usage

        if self.use_api:
            logger.info(
                f"Initializing API-based generator for model '{self.model_name_or_path}' via {self.api_base_url}"
            )
            try:
                self.api_client = create_gpt_client(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=self.api_base_url,
                )
            except Exception as exc:
                logger.error(f"Failed to initialize LLM API client: {exc}")
                raise

            # Load tokenizer for accurate token counting
            tokenizer_source = tokenizer_name_or_path
            if tokenizer_source:
                logger.info(f"Loading tokenizer from '{tokenizer_source}' for API-based model statistics")
                self._load_tokenizer(tokenizer_source, allow_failure=True)
                if self.tokenizer:
                    logger.info(f"Successfully loaded tokenizer from '{tokenizer_source}'")
                else:
                    logger.warning(f"Failed to load tokenizer from '{tokenizer_source}'. Token statistics will be approximate.")
            else:
                logger.info("No tokenizer specified for API-based model. Token statistics will be approximate (word-count based).")

            if self.tokenizer and hasattr(self.tokenizer, "tokenize"):
                tokens = self.tokenizer.tokenize(" ")
                if tokens:
                    self.space_token = tokens[0]
        else:
            logger.info(f"Loading model from {model_name_or_path}")
            tokenizer_source = tokenizer_name_or_path or self.model_name_or_path
            self._load_tokenizer(tokenizer_source, allow_failure=False)

            self.model_config = AutoConfig.from_pretrained(model_name_or_path)
            
            # All methods in generate_save_more_info.py require attention weights
            # Must set to eager to support output_attentions=True
            self.model_config._attn_implementation = "eager"
            logger.info("Using eager attention implementation for output_attentions support")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                device_map=device_map, 
                config=self.model_config
            )
            
            # Set pad_token
            if self.tokenizer and self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Explicitly set model's pad_token_id to avoid warnings during generation
            if self.model.config.pad_token_id is None and self.tokenizer is not None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
            if self.model_config.model_type == "llama":
                self.space_token = "▁"
            elif self.model_config.model_type == "olmo2":
                # OLMo-2 models use GPT2Tokenizer, space is handled differently
                tokens = self.tokenizer.tokenize(' ') if self.tokenizer else []
                self.space_token = tokens[0] if tokens else " "
            else:
                tokens = self.tokenizer.tokenize(' ') if self.tokenizer else []
                self.space_token = tokens[0] if tokens else " "

    def _load_tokenizer(self, source: str, allow_failure: bool) -> None:
        """
        Load tokenizer from a given source path or model name.
        
        Args:
            source: Path to local model directory or HuggingFace model name (e.g., 'meta-llama/Llama-2-7b-chat-hf')
            allow_failure: If True, log warning on failure; if False, raise exception
        """
        try:
            logger.info(f"Attempting to load tokenizer from: {source}")
            tokenizer = AutoTokenizer.from_pretrained(source)
            if tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            self.tokenizer = tokenizer
            logger.info(f"Tokenizer loaded successfully from: {source}")
        except Exception as exc:
            if allow_failure:
                logger.warning(
                    f"Unable to load tokenizer from '{source}'. Generated text statistics may be approximate. Error: {exc}"
                )
                self.tokenizer = None
            else:
                logger.error(f"Failed to load required tokenizer from '{source}': {exc}")
                raise

    def generate(self, input_text, max_length, return_logprobs=False):
        if self.use_api:
            # API mode: simple text generation
            if return_logprobs:
                logger.warning("API mode does not support return_logprobs. Returning text only.")
                return_logprobs = False
            
            # For API models, request usage information
            if self.is_api_model or self.use_web_search:
                response, usage = self.api_client.simple_completion(
                    prompt=input_text,
                    model=self.model_name_or_path,
                    return_usage=True,
                    use_web_search=self.use_web_search,
                )
                self.last_usage = usage
                return response, None, None
            else:
                response = self.api_client.simple_completion(
                    prompt=input_text,
                    model=self.model_name_or_path,
                    use_web_search=self.use_web_search,
                )
                self.last_usage = None
                return response, None, None
        
        # Local model mode
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        if return_logprobs:
            outputs = self.model.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                max_new_tokens = max_length, 
                return_dict_in_generate = True, 
                output_scores = True,
                pad_token_id = self.tokenizer.pad_token_id,
            )
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_tokens = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0]) # text = "".join(tokens)
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            return text, tokens, logprobs
        
        else:
            outputs = self.model.generate(
                input_ids = input_ids, 
                max_new_tokens = max_length, 
                attention_mask = attention_mask,
                pad_token_id = self.tokenizer.pad_token_id,
            )
            generated_tokens = outputs[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0])
            self.last_usage = None
            return text, None, None

    def generate_attn(self, input_text, max_length, solver="max", use_entropy = False, use_logprob = False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            max_new_tokens = max_length, 
            return_dict_in_generate = True, 
            output_scores = True,
            pad_token_id = self.tokenizer.pad_token_id,
        )
        generated_tokens = outputs.sequences[:, input_length:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
        text = self.tokenizer.decode(generated_tokens[0])

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i] == 13 or tokens[i-1] == self.tokenizer.eos_token:
                range_.append([i, i])
            else:
                range_[-1][-1] += 1

        # attention
        model_outputs = self.model(generated_tokens, output_attentions=True)
        if model_outputs.attentions is None or len(model_outputs.attentions) == 0:
            raise RuntimeError("Failed to get attention weights. Try setting attention_type to 'eager'")
        atten = model_outputs.attentions[-1][0]
        if solver == "max": 
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        elif solver == "avg":
            mean_atten = torch.sum(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
            for i in range(mean_atten.shape[0]):
                mean_atten[i] /= (mean_atten.shape[0] - i)
        elif solver == "last_token":
            mean_atten = torch.mean(atten[:, -1], dim=0)
        else:
            raise NotImplementedError
        if mean_atten.shape[0] > 1 and tokens[0] == self.tokenizer.eos_token:
            mean_atten = mean_atten / sum(mean_atten[1:]).item()
        # mean_atten = mean_atten[tl:tr]
            
        # regular tokens
        seqlist = []
        attns = []
        for r in range_:
            tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(self.space_token, "")
            value = sum(mean_atten[r[0]: r[1]+1]).item()
            seqlist.append(tokenseq)
            attns.append(value)

        # -log prob
        if use_logprob:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            seqlogprobs = []
            for r in range_:
                logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqlogprobs.append(logprobseq)
        else:
            seqlogprobs = None

        # entropy
        if use_entropy:
            tmp = []
            for v in outputs.scores:
                tmp.append(v.cpu())
            softmax_probs = softmax(tmp, axis=-1)
            entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
            entropies = [v[0] for v in entropies]
            seqentropies = []
            for r in range_:
                entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqentropies.append(entropyseq) 
        else:
            seqentropies = None 

        return text, seqlist, attns, seqlogprobs, seqentropies


class Counter:
    def __init__(self, debug=False):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0
        self.debug = debug
        # Track API token usage for API-based models
        self.api_input_tokens = 0
        self.api_output_tokens = 0
        self.api_total_tokens = 0
        self.web_search_calls = 0  # Track web search calls

    def add_generate(self, text, tokenizer=None, api_usage=None):
        self.generate += 1
        
        # Track API token usage if available (for API-based models)
        if api_usage is not None:
            self.api_input_tokens += api_usage.get("input_tokens", 0)
            self.api_output_tokens += api_usage.get("output_tokens", 0)
            self.api_total_tokens += api_usage.get("total_tokens", 0)
            # Track web search calls if present
            self.web_search_calls += api_usage.get("web_search_calls", 0)
        
        if tokenizer is not None:
            ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
            self.token += len(ids)
        else:
            # Fallback to word count if no tokenizer available
            self.token += len(text.split())
            if self.debug:
                logger.info(f"Tokenizer not available, using word count as replacement. ")
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        result = {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }
        
        # Add API token usage if there's any difference
        api_input_diff = self.api_input_tokens - other_counter.api_input_tokens
        api_output_diff = self.api_output_tokens - other_counter.api_output_tokens
        api_total_diff = self.api_total_tokens - other_counter.api_total_tokens
        web_search_diff = self.web_search_calls - other_counter.web_search_calls
        
        if api_input_diff > 0 or api_output_diff > 0 or api_total_diff > 0 or web_search_diff > 0:
            result["api_input_tokens"] = api_input_diff
            result["api_output_tokens"] = api_output_diff
            result["api_total_tokens"] = api_total_diff
            result["web_search_calls"] = web_search_diff
        
        return result
         

class BasicRAG:
    def __init__(self, args):
        args = args.__dict__ 
        for k, v in args.items():
            setattr(self, k, v)
        
        # Read API-related parameters from config
        use_llm_api = getattr(self, "use_llm_api", False)
        api_base_url = getattr(self, "api_base_url", None)
        tokenizer_name_or_path = getattr(self, "tokenizer_name_or_path", None)
        use_web_search = getattr(self, "use_web_search", False)  # Read from config
        
        # Save use_web_search as instance attribute
        self.use_web_search = use_web_search
        
        # Read device_map from config, use "auto" if not specified
        model_device_map = getattr(self, "model_device_map", "auto")
        
        if use_llm_api:
            logger.info(f"Initializing with API mode for model: {self.model_name_or_path}")
            if use_web_search:
                logger.info("Web search tool is enabled for API calls")
        else:
            logger.info(f"Loading model with device_map: {model_device_map}")
        
        # Initialize generator (supports API and local mode)
        self.generator = BasicGenerator(
            self.model_name_or_path,
            device_map=model_device_map,
            use_api=use_llm_api,
            api_base_url=api_base_url,
            tokenizer_name_or_path=tokenizer_name_or_path,
            use_web_search=use_web_search,
        )
        
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                # For API mode, if no tokenizer, BM25 will handle it
                self.retriever = BM25(
                    tokenizer = self.generator.tokenizer, 
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name, 
                    engine = "elasticsearch",
                )
            elif self.retriever_type == "SGPT":
                self.retriever = SGPT(
                    model_name_or_path = self.sgpt_model_name_or_path, 
                    sgpt_encode_file_path = self.sgpt_encode_file_path,
                    passage_file = self.passage_file
                )
            elif self.retriever_type == "Qwen3":
                self.retriever = Qwen3Retriever(
                    model_name_or_path = self.qwen3_model_name_or_path, 
                    qwen3_encode_file_path = self.qwen3_encode_file_path,
                    passage_file = self.passage_file
                )
            else:
                raise NotImplementedError
        
        debug = getattr(self, "debug", False)
        self.counter = Counter(debug=debug)
        
        # Initialize document saving related attributes
        self.retrieved_docs = []
        self.docs_output_file = None
        if hasattr(self, 'output_dir'):
            docs_file_path = os.path.join(self.output_dir, "retrieved_docs.json")
            self.docs_output_file = docs_file_path
        
        # Load prompt template
        self.prompt_template_path = args.get('prompt_template_path', "prompt_templates.json")
        self.prompt_template_key = args.get('prompt_template_key', "llama-2")
        self.prompt_templates = self.load_prompt_templates(self.prompt_template_path)
        
        # If user-specified key not found, use default 'llama-2' template
        if self.prompt_template_key not in self.prompt_templates:
            logger.warning(
                "Prompt template key '%s' not found. Using default 'llama-2' template.",
                self.prompt_template_key
            )
            self.prompt_template_key = "llama-2"
        
        self.prompt_template = self.prompt_templates.get(
            self.prompt_template_key,
            self.prompt_templates.get("llama-2", {})
        )
        
        logger.info(
            "Using prompt template key: %s (template path: %s)",
            self.prompt_template_key,
            self.prompt_template_path
        )

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _docs_ids, docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk, 
                max_query_length = max_query_length,
            )
            return docs[0]
        elif self.retriever_type == "SGPT":
            docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk,
            )
            return docs[0] 
        elif self.retriever_type == "Qwen3":
            docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk,
            )
            return docs[0] 
        else:
            raise NotImplementedError
    
    def load_prompt_templates(self, template_path=None):
        """Load prompt template"""
        if template_path is None:
            template_path = os.path.join(os.path.dirname(__file__), "prompt_templates.json")
        
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Prompt template file not found: {template_path}. Please provide a valid prompt_templates.json file.")
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                templates = json.load(f)
            logger.info(f"Loaded prompt templates from {template_path}")
            return templates
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt template from {template_path}: {e}")
    
    def _get_prompt_section(self, section_name):
        """Get specified prompt section (initial or hallucination)"""
        return self.prompt_template.get(section_name, {})
    
    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else ""
    
    def _make_json_serializable(self, obj):
        """
        Recursively convert objects to JSON-serializable format
        Handle numpy arrays, numpy scalars, and other special types
        """
        
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            # For other types, try to convert to string
            return str(obj)
    
    def save_retrieval_info(self, qid, question, query, retrieved_docs, retrieval_round, 
                           current_text="", hallucinated_content=None, additional_info=None):
        """
        Generic method to save retrieval information
        
        Args:
            qid: Question ID
            question: Original question
            query: Query for this retrieval
            retrieved_docs: Retrieved documents
            retrieval_round: Which retrieval round this is
            current_text: Currently generated text
            hallucinated_content: Detected hallucination content (dict format with details)
            additional_info: Other additional info (dict format)
        """
        # Normalize retrieved results to pure Python list[str] for JSON serialization
        try:
            docs_list = retrieved_docs.tolist() if hasattr(retrieved_docs, 'tolist') else list(retrieved_docs)
            # Ensure all are strings for safety
            docs_list = ["" if d is None else str(d) for d in docs_list]
        except Exception:
            # Fall back to single-element list if not iterable or conversion fails
            docs_list = [str(retrieved_docs)]
        
        doc_entry = {
            "qid": qid,
            "question": question,
            "query": query,
            "retrieved_docs": docs_list,
            "retrieval_round": retrieval_round,
            "topk": self.retrieve_topk,
            "current_text": current_text,
            "timestamp": __import__('time').time()
        }
        
        # Add hallucination-related info and ensure all content is JSON-serializable
        if hallucinated_content is not None:
            serializable_content = self._make_json_serializable(hallucinated_content)
            doc_entry["hallucinated_content"] = serializable_content
        
        # Add other additional info
        if additional_info is not None:
            serializable_info = self._make_json_serializable(additional_info)
            doc_entry.update(serializable_info)
        
        self.retrieved_docs.append(doc_entry)
        
        # If output file path exists, save to file immediately
        if self.docs_output_file:
            self._save_docs_to_file()
    
    def _save_docs_to_file(self):
        """Save retrieved documents to JSON file"""
        if self.docs_output_file and self.retrieved_docs:
            try:
                with open(self.docs_output_file, 'w', encoding='utf-8') as f:
                    json.dump(self.retrieved_docs, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save retrieved docs to {self.docs_output_file}: {e}") 


    def inference(self, question, demo, case):
        # non-retrieval
        assert self.query_formulation == "direct"
        prompt = "".join([d["case"]+"\n" for d in demo])
        
        # Use instruction from prompt template
        initial_section = self._get_prompt_section("initial")
        instruction = initial_section["instruction"]
        
        if self.use_web_search:
            prompt += "\nPlease search the web when you think it is necessary to find the answer.\n"
        
        prompt += instruction
        prompt += case

        if self.debug:
            logger.info(f"[BasicRAG] Generating without retrieval")
            logger.info(f"[BasicRAG] Prompt:\n{prompt}")
       
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        
        if self.debug:
            logger.info(f"[BasicRAG] Generated text: {text}")
            
        if self.use_counter == True:
            self.counter.add_generate(text, tokenizer=self.generator.tokenizer, api_usage=self.generator.last_usage)
        return text


class SingleRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
        if self.debug:
            logger.info("[SingleRAG] Initialized")
    
    def inference(self, question, demo, case, qid=None):
        assert self.query_formulation == "direct"
        
        if self.debug:
            logger.info(f"[SingleRAG] Starting inference for question: {question}")
            logger.info(f"[SingleRAG] Retrieving top-{self.retrieve_topk} documents")
        
        docs = self.retrieve(question, topk=self.retrieve_topk)
        
        # Save retrieved documents
        if qid is not None:
            self.save_retrieval_info(
                qid=qid,
                question=question,
                query=question,  # SingleRAG uses question directly as query
                retrieved_docs=docs,
                retrieval_round=1,  # SingleRAG retrieves only once
                current_text="",  # Initially no generated text
                additional_info={"method": "SingleRAG", "query_formulation": self.query_formulation}
            )
        
        # Normalize retrieved results to pure Python list[str] for later processing
        try:
            docs_list = docs.tolist() if hasattr(docs, 'tolist') else list(docs)
            # Ensure all are strings for safety
            docs_list = ["" if d is None else str(d) for d in docs_list]
        except Exception:
            # Fall back to single-element list if not iterable or conversion fails
            docs_list = [str(docs)]
        
        # Generate prompt for topk passages
        prompt = "".join([d["case"]+"\n" for d in demo])
        
        # Use prompt template
        initial_section = self._get_prompt_section("initial")
        context_header = initial_section["context_header"]
        doc_format = initial_section["doc_format"]
        post_doc = initial_section["post_doc"]
        instruction = initial_section["instruction"]
        
        prompt += context_header
        for i, doc in enumerate(docs_list):
            prompt += doc_format.format(index=i+1, doc=doc)
        prompt += post_doc
        prompt += instruction
        prompt += case

        if self.debug:
            logger.info(f"[SingleRAG] Retrieved {len(docs_list)} documents")
            logger.info(f"[SingleRAG] Prompt:\n{prompt}")

        # print('='*20)
        # print("prompt:", prompt)
        # print('='*20)

        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        
        if self.debug:
            logger.info(f"[SingleRAG] Generated text: {text}")
        
        if self.use_counter == True:
            self.counter.add_generate(text, tokenizer=self.generator.tokenizer, api_usage=self.generator.last_usage)
        return text



class FixLengthRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
        if self.debug:
            logger.info("[FixLengthRAG] Initialized")
    
    def inference(self, question, demo, case, qid=None):
        assert self.query_formulation == "direct"
        text = ""
        retrieve_question = question
        retrieval_round = 0
        
        if self.debug:
            logger.info(f"[FixLengthRAG] Starting inference for question: {question}")
        
        while True:
            old_len = len(text)
            retrieval_round += 1
            docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
            
            if self.debug:
                logger.info(f"[FixLengthRAG] Retrieval round {retrieval_round}")
                logger.info(f"[FixLengthRAG] Query: {retrieve_question}")
                logger.info(f"[FixLengthRAG] Retrieved {len(docs)} documents")
            
            # Save retrieval info
            if qid is not None:
                self.save_retrieval_info(
                    qid=qid,
                    question=question,
                    query=retrieve_question,
                    retrieved_docs=docs,
                    retrieval_round=retrieval_round,
                    current_text=text,
                    additional_info={
                        "method": "FixLengthRAG", 
                        "query_formulation": self.query_formulation,
                        "fix_method": self.method,
                        "fix_length": getattr(self, 'fix_length', None)
                    }
                )
            
            prompt = "".join([d["case"]+"\n" for d in demo])
            
            # Use prompt template
            hallucination_section = self._get_prompt_section("hallucination")
            context_header = hallucination_section["context_header"]
            doc_format = hallucination_section["doc_format"]
            post_doc = hallucination_section["post_doc"]
            instruction = hallucination_section["instruction"]
            
            prompt += context_header
            for i, doc in enumerate(docs):
                prompt += doc_format.format(index=i+1, doc=doc)
            prompt += post_doc
            prompt += instruction
            prompt += case + " " + text
            if self.method == "fix-length-retrieval":
                if self.debug:
                    logger.info(f"[FixLengthRAG] Using fix-length-retrieval with length {self.fix_length}")
                    logger.info(f"[FixLengthRAG] Prompt:\n{prompt}")
                
                new_text, _, _ = self.generator.generate(prompt, self.fix_length)
                
                if self.debug:
                    logger.info(f"[FixLengthRAG] Generated text: {new_text}")
                
                if self.use_counter == True:
                    self.counter.add_generate(new_text, tokenizer=self.generator.tokenizer, api_usage=self.generator.last_usage)
                
                # For API models, remove duplicate content from the beginning of new_text
                # This handles the case where some API models repeat the prompt content
                if self.generator.is_api_model and text:
                    new_text_stripped = new_text.strip()
                    text_stripped = text.strip()
                    
                    # Check if new_text starts with text
                    if new_text_stripped.startswith(text_stripped):
                        new_text = new_text_stripped[len(text_stripped):].strip()
                        if self.debug:
                            logger.info("API model detected: Removed duplicate text from new_text beginning")
                            logger.info(f"Removed content: {text_stripped}")
                            logger.info(f"Original new_text length: {len(new_text_stripped)}")
                            logger.info(f"After removing duplicate: {len(new_text)}")
                
                text = text.strip() + " " + new_text.strip()
                retrieve_question = new_text.strip()
            else:
                # fix sentence
                if self.debug:
                    logger.info(f"[FixLengthRAG] Using fix-sentence-retrieval")
                    logger.info(f"[FixLengthRAG] Prompt:\n{prompt}")
                
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                
                if self.debug:
                    logger.info(f"[FixLengthRAG] Generated text: {new_text}")
                
                if self.use_counter == True:
                    self.counter.add_generate(new_text, tokenizer=self.generator.tokenizer, api_usage=self.generator.last_usage)
                
                # For API models, remove duplicate content from the beginning of new_text
                # This handles the case where some API models repeat the prompt content
                if self.generator.is_api_model and text:
                    new_text_stripped = new_text.strip()
                    text_stripped = text.strip()
                    
                    # Check if new_text starts with text
                    if new_text_stripped.startswith(text_stripped):
                        new_text = new_text_stripped[len(text_stripped):].strip()
                        if self.debug:
                            logger.info("API model detected: Removed duplicate text from new_text beginning")
                            logger.info(f"Removed content: {text_stripped}")
                            logger.info(f"Original new_text length: {len(new_text_stripped)}")
                            logger.info(f"After removing duplicate: {len(new_text)}")
                
                new_text = new_text.strip()
                sentences = list(nlp(new_text).sents)
                sentences = [str(sent).strip() for sent in sentences]
                if len(sentences) == 0:
                    break
                text = text.strip() + " " + str(sentences[0])
                retrieve_question = sentences[0]
                
                if self.debug:
                    logger.info(f"[FixLengthRAG] Next query sentence: {retrieve_question}")
            
            # Judge if token count is less than generate_max_length
            if self.generator.tokenizer is not None:
                tokens_count = len(self.generator.tokenizer.encode(text))
            else:
                # Fallback to word count for API mode without tokenizer
                tokens_count = len(text.split())
            
            if self.debug:
                logger.info(f"[FixLengthRAG] Current text length: {tokens_count} tokens, current text: {text}")
            
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text or "Question:" in text:
                if self.debug:
                    logger.info(f"[FixLengthRAG] Stopping generation. Reason: tokens={tokens_count}, old_len={old_len}, has_answer={'the answer is' in text}, has_question={'Question:' in text}")
                break
        return text


class TokenRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
        if self.debug:
            logger.info(f"[TokenRAG] Initialized with threshold: {self.hallucination_threshold}")

    def modifier(self, text, tokens, logprobs):
        
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        if self.debug:
            logger.info(f"[TokenRAG] Modifier analyzing {len(sentences)} sentences")

        tid = 0
        for sid, sent in enumerate(sentences):
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr])
                if apr == -1:
                    break
                pos = apr + len(tokens[tr])
                tr += 1
            probs = [1 - exp(v) for v in logprobs[tid:tr+1]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                if self.debug:
                    logger.info(f"[TokenRAG] Hallucination detected in sentence {sid}: {sent}")
                    logger.info(f"[TokenRAG] Hallucination score: {p:.4f}, threshold: {self.hallucination_threshold}")
                
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                # # This was changed to replace only the maximum one, not all of them
                # max_prob = 0
                # for prob, tok in zip(probs, tokens[tid:tr+1]):
                #     max_prob = max(prob, max_prob)
                for prob, tok in zip(probs, tokens[tid:tr+1]):
                    apr = curr[pos:].find(tok) + pos
                    if prob > self.hallucination_threshold:
                    # if prob == max_prob:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True
            tid = tr + 1
        
        # No hallucination
        return text, None, False
    
    def inference(self, question, demo, case, qid=None):
        # assert self.query_formulation == "direct"
        text = ""
        retrieval_round = 0
        
        if self.debug:
            logger.info(f"[TokenRAG] Starting inference for question: {question}")
        
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += case + " " + text
            
            if self.debug:
                logger.info(f"[TokenRAG] Generation round, current text length: {len(text)}")
                logger.info(f"[TokenRAG] Prompt:\n{prompt}")
            
            new_text, tokens, logprobs = self.generator.generate(
                prompt, 
                self.generate_max_length, 
                return_logprobs=True
            )
            
            if self.debug:
                logger.info(f"[TokenRAG] Generated text: {new_text}")
            
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer, api_usage=self.generator.last_usage)
            ptext, curr, hallucination = self.modifier(new_text, tokens, logprobs)
            if not hallucination:
                if self.debug:
                    logger.info(f"[TokenRAG] No hallucination detected, accepting generated text")
                text = text.strip() + " " + new_text.strip()
            else:
                retrieval_round += 1
                if self.query_formulation == "direct":
                    retrieve_question = curr.replace("[xxx]", "")
                elif self.query_formulation == "forward_all":
                    tmp_all = [question, text, ptext]
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented

                if self.debug:
                    logger.info(f"[TokenRAG] Retrieval round {retrieval_round}")
                    logger.info(f"[TokenRAG] Hallucinated sentence: {curr}")
                    logger.info(f"[TokenRAG] Query: {retrieve_question}")

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                
                if self.debug:
                    logger.info(f"[TokenRAG] Retrieved {len(docs)} documents")
                
                # Save retrieval info with hallucination detection details
                if qid is not None:
                    hallucinated_content = {
                        "original_tokens": tokens,
                        "log_probs": logprobs,
                        "partial_text": ptext,
                        "current_sentence": curr,
                        "replaced_tokens": curr.replace("[xxx]", ""),
                        "threshold_used": self.hallucination_threshold,
                        "sentence_solver": self.sentence_solver
                    }
                    
                    self.save_retrieval_info(
                        qid=qid,
                        question=question,
                        query=retrieve_question,
                        retrieved_docs=docs,
                        retrieval_round=retrieval_round,
                        current_text=text,
                        hallucinated_content=hallucinated_content,
                        additional_info={
                            "method": "TokenRAG", 
                            "query_formulation": self.query_formulation,
                            "hallucination_detected": True
                        }
                    )
                
                prompt = "".join([d["case"]+"\n" for d in demo])
                
                # Use prompt template
                hallucination_section = self._get_prompt_section("hallucination")
                context_header = hallucination_section["context_header"]
                doc_format = hallucination_section["doc_format"]
                post_doc = hallucination_section["post_doc"]
                instruction = hallucination_section["instruction"]
                
                prompt += context_header
                for i, doc in enumerate(docs):
                    prompt += doc_format.format(index=i+1, doc=doc)
                prompt += post_doc
                prompt += instruction
                prompt += case + " " + text + " " + ptext.strip()
                
                if self.debug:
                    logger.info(f"[TokenRAG] Regeneration prompt:\n{prompt}")
                
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                
                if self.debug:
                    logger.info(f"[TokenRAG] Regenerated text: {new_text}")
                
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer, api_usage=self.generator.last_usage)
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
            
            # Judge if the token count is less than generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            
            if self.debug:
                logger.info(f"[TokenRAG] Current text length: {tokens_count} tokens")
            
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text or "Question:" in text:
                if self.debug:
                    logger.info(f"[TokenRAG] Stopping generation. Reason: tokens={tokens_count}, old_len={old_len}, has_answer={'the answer is' in text}, has_question={'Question:' in text}")
                break
        return text


class TokenRAG_fixed(BasicRAG):
    """
    Fixed version of TokenRAG, solving repeated content issues caused by token replacement
    Main fixes:
    1. Improved position mapping logic from token to text
    2. Use backward replacement strategy to avoid position shift
    3. Handle tokenizer space prefix issue
    """
    def __init__(self, args):
        super().__init__(args)
        if self.debug:
            logger.info(f"[TokenRAG_fixed] Initialized with threshold: {self.hallucination_threshold}")

    def modifier(self, text, tokens, logprobs):
        
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        if self.debug:
            logger.info(f"[TokenRAG_fixed] Modifier analyzing {len(sentences)} sentences")

        tid = 0
        for sid, sent in enumerate(sentences):
            pos = 0
            tr = tid
            while tr < len(tokens):
                # Use lstrip() to handle tokenizer's possible space prefix
                clean_tok = tokens[tr].lstrip()
                apr = sent[pos:].find(clean_tok)
                if apr == -1:
                    break
                pos = apr + len(clean_tok)
                tr += 1
            
            probs = [1 - exp(v) for v in logprobs[tid:tr+1]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            
            if p > self.hallucination_threshold: # hallucination
                if self.debug:
                    logger.info(f"[TokenRAG_fixed] Hallucination detected in sentence {sid}: {sent}")
                    logger.info(f"[TokenRAG_fixed] Hallucination score: {p:.4f}, threshold: {self.hallucination_threshold}")
                
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                
                # Create token position mapping, backward replacement to avoid position shift
                curr = sentences[sid] 
                token_replacements = []
                pos = 0
                
                for i, (prob, tok) in enumerate(zip(probs, tokens[tid:tr+1])):
                    clean_tok = tok.lstrip()  # Remove leading space to match original text
                    apr = curr[pos:].find(clean_tok)
                    if apr != -1:
                        apr += pos
                        if prob > self.hallucination_threshold:
                            token_replacements.append((apr, apr + len(clean_tok), clean_tok))
                        pos = apr + len(clean_tok)
                    else:
                        # If token not found, skip but continue with next
                        continue
                
                # Replace backward to avoid position shift
                for start, end, original_tok in reversed(token_replacements):
                    curr = curr[:start] + "[xxx]" + curr[end:]
                
                return prev, curr, True
            tid = tr + 1
        
        # No hallucination
        return text, None, False
    
    def inference(self, question, demo, case, qid=None):
        # assert self.query_formulation == "direct"
        text = ""
        retrieval_round = 0
        
        if self.debug:
            logger.info(f"[TokenRAG_fixed] Starting inference for question: {question}")
        
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += case + " " + text
            
            if self.debug:
                logger.info(f"[TokenRAG_fixed] Generation round, current text length: {len(text)}")
                logger.info(f"[TokenRAG_fixed] Prompt:\n{prompt}")
            
            new_text, tokens, logprobs = self.generator.generate(
                prompt, 
                self.generate_max_length, 
                return_logprobs=True
            )
            
            if self.debug:
                logger.info(f"[TokenRAG_fixed] Generated text: {new_text}")
            
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer, api_usage=self.generator.last_usage)
            ptext, curr, hallucination = self.modifier(new_text, tokens, logprobs)
            if not hallucination:
                if self.debug:
                    logger.info(f"[TokenRAG_fixed] No hallucination detected, accepting generated text")
                text += new_text
            else:
                # hallucination detected
                retrieval_round += 1
                text = ptext # Keep text before hallucination
                retrieve_question = curr.replace("[xxx]", "")
                
                if self.debug:
                    logger.info(f"[TokenRAG_fixed] Retrieval round {retrieval_round}")
                    logger.info(f"[TokenRAG_fixed] Hallucinated sentence: {curr}")
                    logger.info(f"[TokenRAG_fixed] Query: {retrieve_question}")
                
                if qid is not None:
                    # Save retrieval info with hallucination detection details
                    hallucinated_info = {
                        "detected_sentence": curr,
                        "original_sentence": curr.replace("[xxx]", ""),
                        "retrieval_reason": "hallucination_detected",
                        "threshold": self.hallucination_threshold
                    }
                    
                    self.save_retrieval_info(
                        qid=qid,
                        question=question,
                        query=retrieve_question,
                        retrieved_docs=[],  # Save empty first, another record will follow when retrieval completes
                        retrieval_round=retrieval_round,
                        current_text=text,
                        hallucinated_content=hallucinated_info,
                        additional_info={"method": "TokenRAG_fixed", "query_formulation": self.query_formulation}
                    )
                
                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                
                if self.debug:
                    logger.info(f"[TokenRAG_fixed] Retrieved {len(docs)} documents")
                
                # Save retrieval results
                if qid is not None:
                    self.save_retrieval_info(
                        qid=qid,
                        question=question,
                        query=retrieve_question,
                        retrieved_docs=docs,
                        retrieval_round=retrieval_round,
                        current_text=text,
                        additional_info={"method": "TokenRAG_fixed", "retrieval_completed": True}
                    )
                
                # Normalize retrieved results to pure Python list[str] for later processing
                try:
                    docs_list = docs.tolist() if hasattr(docs, 'tolist') else list(docs)
                    # Ensure all are strings for safety
                    docs_list = ["" if d is None else str(d) for d in docs_list]
                except Exception:
                    # Fall back to single-element list if not iterable or conversion fails
                    docs_list = [str(docs)]
                
                prompt = "".join([d["case"]+"\n" for d in demo])
                
                # Use prompt template
                hallucination_section = self._get_prompt_section("hallucination")
                context_header = hallucination_section["context_header"]
                doc_format = hallucination_section["doc_format"]
                post_doc = hallucination_section["post_doc"]
                instruction = hallucination_section["instruction"]
                
                prompt += context_header
                for i, doc in enumerate(docs_list):
                    prompt += doc_format.format(index=i+1, doc=doc)
                prompt += post_doc
                prompt += instruction
                prompt += case + " " + text
                
                if self.debug:
                    logger.info(f"[TokenRAG_fixed] Regeneration prompt:\n{prompt}")
                
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                
                if self.debug:
                    logger.info(f"[TokenRAG_fixed] Regenerated text: {new_text}")
                
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer, api_usage=self.generator.last_usage)
                text += new_text
            
            # Judge if token count is less than generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            
            if self.debug:
                logger.info(f"[TokenRAG_fixed] Current text length: {tokens_count} tokens")
            
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text or "Question:" in text:
                if self.debug:
                    logger.info(f"[TokenRAG_fixed] Stopping generation. Reason: tokens={tokens_count}, old_len={old_len}, has_answer={'the answer is' in text}, has_question={'Question:' in text}")
                break
        return text


class EntityRAG(TokenRAG):
    def __init__(self, args):
        super().__init__(args)
        if self.debug:
            logger.info(f"[EntityRAG] Initialized with threshold: {self.hallucination_threshold}")
    
    def modifier(self, text, tokens, logprobs):
        if self.debug:
            logger.info(f"[EntityRAG] Analyzing entities in text")
        
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        entity = []
        for sent in sentences:
            doc = nlp(sent)
            li = [ent.text for ent in doc.ents]
            entity.append(li)
        
        if self.debug:
            logger.info(f"[EntityRAG] Found entities: {entity}")
        
        belonging = [-1] * len(text)
        pos = 0
        for tid, tok in enumerate(tokens):
            apr = text[pos:].find(tok) + pos
            assert apr != -1
            for j in range(pos, apr+len(tok)):
                belonging[j] = tid
            pos = apr + len(tok)
        
        entity_intv = []
        for sid, sent in enumerate(sentences):
            tmp = []
            pos = text.find(sent)
            for ent in entity[sid]:
                apr = text[pos:].find(ent) + pos
                el = belonging[apr]
                er = belonging[apr + len(ent) - 1]
                tmp.append((el, er))
                pos = apr + len(ent)
            entity_intv.append(tmp)

        entity_prob = []
        for ent_itv_per_sent in entity_intv:
            tmp = []
            for itv in ent_itv_per_sent:
                probs = np.array(logprobs[itv[0]:itv[1]+1])
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "first": lambda x: x[0] if len(x) > 0 else 0
                }.get(self.entity_solver, lambda x: 0)(probs)
                tmp.append(p)
            entity_prob.append(tmp)

        for sid in range(len(sentences)):
            if len(entity_prob[sid]) == 0:
                continue
            probs = [1 - exp(v) for v in entity_prob[sid]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                if self.debug:
                    logger.info(f"[EntityRAG] Hallucination detected in sentence {sid}: {sentences[sid]}")
                    logger.info(f"[EntityRAG] Hallucination score: {p:.4f}, threshold: {self.hallucination_threshold}")
                    logger.info(f"[EntityRAG] Hallucinated entities: {[ent for prob, ent in zip(probs, entity[sid]) if prob > self.hallucination_threshold]}")
                
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated entities in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, ent in zip(probs, entity[sid]):
                    apr = curr[pos:].find(ent) + pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(ent):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(ent)
                return prev, curr, True
        # No hallucination
        if self.debug:
            logger.info(f"[EntityRAG] No hallucination detected")
        return text, None, False

    def inference(self, question, demo, case, qid=None):
        # EntityRAG inherits TokenRAG's logic but needs to override modifier method to handle entities
        text = ""
        retrieval_round = 0
        
        if self.debug:
            logger.info(f"[EntityRAG] Starting inference for question: {question}")
        
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += case + " " + text
            
            if self.debug:
                logger.info(f"[EntityRAG] Generation round, current text length: {len(text)}")
            
            new_text, tokens, logprobs = self.generator.generate(
                prompt, 
                self.generate_max_length, 
                return_logprobs=True
            )
            
            if self.debug:
                logger.info(f"[EntityRAG] Generated text: {new_text}")
            
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer, api_usage=self.generator.last_usage)
            ptext, curr, hallucination = self.modifier(new_text, tokens, logprobs)
            if not hallucination:
                if self.debug:
                    logger.info(f"[EntityRAG] No hallucination detected, accepting generated text")
                text = text.strip() + " " + new_text.strip()
            else:
                retrieval_round += 1
                if self.query_formulation == "direct":
                    retrieve_question = curr.replace("[xxx]", "")
                elif self.query_formulation == "forward_all":
                    tmp_all = [question, text, ptext]
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented

                if self.debug:
                    logger.info(f"[EntityRAG] Retrieval round {retrieval_round}")
                    logger.info(f"[EntityRAG] Hallucinated sentence: {curr}")
                    logger.info(f"[EntityRAG] Query: {retrieve_question}")

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                
                if self.debug:
                    logger.info(f"[EntityRAG] Retrieved {len(docs)} documents")
                
                # Save more detailed entity information for EntityRAG
                if qid is not None:
                    # Extract entity information
                    sentences = [sent.text.strip() for sent in nlp(new_text).sents]
                    sentences = [sent for sent in sentences if len(sent) > 0]
                    
                    entity_info = []
                    for sent in sentences:
                        doc = nlp(sent)
                        entities = [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char} 
                                  for ent in doc.ents]
                        entity_info.append({"sentence": sent, "entities": entities})
                    
                    hallucinated_content = {
                        "original_tokens": tokens,
                        "log_probs": logprobs,
                        "partial_text": ptext,
                        "current_sentence": curr,
                        "replaced_tokens": curr.replace("[xxx]", ""),
                        "entity_info": entity_info,
                        "threshold_used": self.hallucination_threshold,
                        "entity_solver": self.entity_solver,
                        "sentence_solver": self.sentence_solver
                    }
                    
                    self.save_retrieval_info(
                        qid=qid,
                        question=question,
                        query=retrieve_question,
                        retrieved_docs=docs,
                        retrieval_round=retrieval_round,
                        current_text=text,
                        hallucinated_content=hallucinated_content,
                        additional_info={
                            "method": "EntityRAG", 
                            "query_formulation": self.query_formulation,
                            "hallucination_detected": True
                        }
                    )
                
                prompt = "".join([d["case"]+"\n" for d in demo])
                
                # Use prompt template
                hallucination_section = self._get_prompt_section("hallucination")
                context_header = hallucination_section["context_header"]
                doc_format = hallucination_section["doc_format"]
                post_doc = hallucination_section["post_doc"]
                instruction = hallucination_section["instruction"]
                
                prompt += context_header
                for i, doc in enumerate(docs):
                    prompt += doc_format.format(index=i+1, doc=doc)
                prompt += post_doc
                prompt += instruction
                prompt += case + " " + text + " " + ptext.strip()
                
                if self.debug:
                    logger.info(f"[EntityRAG] Regeneration prompt:\n{prompt}")
                
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                
                if self.debug:
                    logger.info(f"[EntityRAG] Regenerated text: {new_text}")
                
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer, api_usage=self.generator.last_usage)
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
            
            # Judge if token count is less than generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            
            if self.debug:
                logger.info(f"[EntityRAG] Current text length: {tokens_count} tokens")
            
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text or "Question:" in text:
                if self.debug:
                    logger.info(f"[EntityRAG] Stopping generation. Reason: tokens={tokens_count}, old_len={old_len}, has_answer={'the answer is' in text}, has_question={'Question:' in text}")
                break
        return text


class ETCGenerator:
    """ETC generator class for ETC method"""
    def __init__(self, model_name_or_path: str, device_map="auto"):
        logger.info(f"[ETC] Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Must use eager attention to support output_attentions=True
        # SDPA does not support attention weight output
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map=device_map,
            attn_implementation="eager"
        )
        logger.info(f"[ETC] device = {self.model.device}")
        logger.info("[ETC] Using eager attention implementation for output_attentions support")
        
        # Identify space token (llama3 and llama2 are different)
        self.space_token = "Ġ" if "llama-3" in model_name_or_path.lower() else "▁"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set of tokens that cannot be merged
        self.tokens_cannot_merged = {
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode("0" + ch)[-1:])[0]
            for ch in string.whitespace + string.punctuation
        } | {self.space_token, self.tokenizer.bos_token, self.tokenizer.eos_token}
    
    def simply_generate(self, input_text: str, max_length: int) -> Tuple[bool, str]:
        """Simple generation based on retrieved documents (used for regeneration after hallucination detected)"""
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device)
        input_length = input_ids.shape[1]
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
            stop_strings="\n",
            tokenizer=self.tokenizer
        )[0, input_length:]
        
        if output_ids.shape[0] == 0:
            logger.info("[ETC] generate '' in simply_generate()!")
            return True, ""
        
        if output_ids[0] == self.tokenizer.bos_token_id:
            output_ids = output_ids[1:]
        
        if output_ids[-1] == self.tokenizer.eos_token_id:
            return True, self.tokenizer.decode(output_ids[:-1])
        
        return False, self.tokenizer.decode(output_ids)
    
    def tokenize(self, text: str, is_start: bool = False):
        """Convert text to tokens"""
        ids = self.tokenizer.encode(text)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        if not is_start and tokens and tokens[0] == self.tokenizer.bos_token:
            tokens = tokens[1:]
        return tokens
    
    def merge_tokens(self, tokens) -> List[Tuple[int, int]]:
        """Merge tokens into words"""
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) \
                or tokens[i] in self.tokens_cannot_merged \
                or tokens[i-1] in self.tokens_cannot_merged:
                range_.append([i, i+1])
            else:
                range_[-1][1] += 1
        return range_
    
    def build_block(self, text: str, is_start: bool = False) -> Block:
        """Build Block"""
        tokens = self.tokenize(text, is_start=is_start)
        range_ = self.merge_tokens(tokens)
        return Block(text=text, tokens=tokens, range_=range_)
    
    def generate(self, input_texts: List[str], max_length: int) -> GeneratorOutput:
        """Generate text and compute attention, entropy and other metrics"""
        blocks = []
        for text in input_texts:
            blocks.append(self.build_block(text, is_start=not blocks))
        
        input_tokens = sum([block.tokens for block in blocks], [])
        input_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(input_tokens)], 
            device=self.model.device
        )
        input_len_tokens = len(input_tokens)
        
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
            return_dict_in_generate=True,
            output_scores=True,
            stop_strings="\n",
            tokenizer=self.tokenizer,
        )
        
        tokens = self.tokenizer.convert_ids_to_tokens(
            outputs.sequences[0, input_len_tokens:]
        )
        
        # If generated tokens too few, return empty
        if len(tokens) <= 1:
            return GeneratorOutput(empty=True, ended=True)
        
        ended = (tokens[-1] == self.tokenizer.eos_token)
        if ended:
            tokens = tokens[:-1]
        
        text = self.tokenizer.convert_tokens_to_string(tokens)
        range_ = self.merge_tokens(tokens)
        new_block = Block(text=text, tokens=tokens, range_=range_)
        
        blocks.append(new_block)
        merged_blocks = merge_blocks(blocks)
        
        # Compute attention after merging
        atten = self.model(
            outputs.sequences, output_attentions=True
        ).attentions[-1][0][:, -new_block.len_tokens:, :]
        atten = atten.mean(dim=0)
        atten = torch.stack(
            [atten[:, l:r].sum(dim=-1) for l, r in merged_blocks.range_], dim=-1
        )
        atten = torch.stack(
            [atten[l:r, :].mean(dim=-2) for l, r in range_], dim=-2
        )
        
        atten_to_new = atten[:, -new_block.len_words:]
        atten_to_new /= atten.sum(dim=-1, keepdim=True) + 1e-10
        max_atten, _ = atten_to_new.max(dim=1)
        
        # Compute entropy
        probs = torch.stack(outputs.scores).softmax(dim=-1)
        entropies = (-probs * torch.log(probs + 1e-10)).sum(dim=-1)
        entropies = torch.stack([entropies[l:r, 0].max() for l, r in range_])
        
        # Identify function words
        func_words = []
        doc = nlp(new_block.text)
        real_words = set(
            token.text for token in doc 
            if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM']
        )
        
        for i in range(new_block.len_words):
            tl, tr = new_block.range_[i]
            word = self.tokenizer.convert_tokens_to_string(new_block.tokens[tl:tr])
            if not match(word, real_words):
                func_words.append(i)
        
        # Compute first and second order difference entropy
        entropies_s1 = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} 
                        for i in range(len(range_))]
        entropies_s2 = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} 
                        for i in range(len(range_))]
        mt_s2 = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} 
                 for i in range(len(range_))]
        fun_word = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} 
                    for i in range(len(range_))]
        
        # Mark real words
        for i in range(len(range_)):
            if i not in func_words:
                fun_word[i]['val'] = torch.tensor(1, dtype=torch.float64)
        
        # First order difference entropy
        for i in range(1, len(range_)):
            if i not in func_words:
                j = i - 1
                while j >= 0:
                    if j not in func_words:
                        s1 = entropies[i].to(torch.float64) - entropies[j].to(torch.float64)
                        entropies_s1[i]['val'] = s1
                        break
                    j -= 1
        
        # Second order difference entropy
        for i in range(2, len(range_)):
            if i not in func_words:
                j = i - 1
                while j >= 1:
                    if entropies_s1[j]['val'].item() != 0:
                        s2 = (entropies_s1[i]['val'].to(torch.float64) - 
                              entropies_s1[j]['val'].to(torch.float64))
                        entropies_s2[i]['val'] = s2
                        break
                    j -= 1
        
        # Dynamic smoothing of second order difference entropy
        count_fun = 0
        sum_s2 = 0
        Mt_1 = torch.tensor(0, dtype=torch.float64)
        
        for i in range(2, len(range_)):
            if entropies_s2[i]['val'] != 0:
                count_fun += 1
                sum_s2 += entropies_s2[i]['val'].item()
                s2_mean = sum_s2 / count_fun
                w = torch.abs(Mt_1 - s2_mean) / (
                    torch.abs(entropies_s2[i]['val'] - s2_mean) + 
                    torch.abs(Mt_1 - s2_mean)
                )
                α = 0.9 + 0.1 * w
                Mt = α * entropies_s2[i]['val'] + (1 - α) * Mt_1
                mt_s2[i]['val'] = Mt
                Mt_1 = entropies_s2[i]['val']
        
        return GeneratorOutput(
            empty=False,
            ended=ended,
            blocks=blocks,
            merged_blocks=merged_blocks,
            atten=atten,
            max_atten=max_atten,
            entropies=entropies,
            entropies_s1=entropies_s1,
            entropies_s2=entropies_s2,
            smooth_s2=None,
            mt_s2=mt_s2,
            fun_word=fun_word,
        )


class ETCRAG(BasicRAG):
    """ETC (Entropy-based Token-level Confidence) RAG method"""
    def __init__(self, args):
        super().__init__(args)
        
        # Create ETC-specific generator
        model_device_map = getattr(self, "model_device_map", "auto")
        self.etc_generator = ETCGenerator(
            self.model_name_or_path, 
            device_map=model_device_map
        )
        
        # ETC-specific parameters
        self.hallucination_threshold = getattr(self, "hallucination_threshold", 0.5)
        self.thres_abs = getattr(self, "thres_abs", True)
        
        if self.debug:
            logger.info(f"[ETCRAG] Initialized with threshold={self.hallucination_threshold}")
    
    def get_top_sentence(self, text):
        """Get first sentence"""
        prev = ""
        for sent in nlp(text).sents:
            prev += sent.text
            sent = sent.text.strip()
            if len(sent) > 0:
                return prev
        return ""
    
    def hallucination_check(self, outputs: GeneratorOutput) -> CheckerOutput:
        """Detect hallucinations"""
        if self.debug:
            logger.info("[ETC] Start detecting hallucinations")
        
        new_block = outputs.blocks[-1]
        sentences = [sent.text.strip() for sent in nlp(new_block.text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        
        if self.debug:
            logger.info(f"[ETC] Found {len(sentences)} sentences")
        
        wid = 0
        word_counts = [0] * len(sentences)
        
        for sid, sent in enumerate(sentences):
            wl, wr = wid, wid
            if wid == new_block.len_words:
                break
            
            # Find word range for current sentence
            while wr < new_block.len_words and sent not in self.etc_generator.tokenizer.convert_tokens_to_string(
                new_block.tokens[new_block.range_[wl][0]:new_block.range_[wr][1]]
            ):
                wr += 1
            
            if wr < new_block.len_words:
                wr += 1
            
            wid = wr
            len_sent = wid
            
            if wl == wr:
                continue
            
            if sid == 0:
                word_counts[sid] = wid
            else:
                for t in range(0, sid):
                    len_sent -= word_counts[t]
                word_counts[sid] = len_sent
            
            # Compute hallucination score for sentence
            max_atten_sent = outputs.max_atten[wl:wr]
            max_atten_sent = max_atten_sent * (wr - wl) / (max_atten_sent.sum() + 1e-10)
            
            # Final metric: max_atten * mt_s2
            value = max_atten_sent * torch.tensor(
                [entry['val'] for entry in outputs.mt_s2[wl:wr]]
            ).to(max_atten_sent.device)
            
            # Judge if threshold exceeded
            if self.thres_abs:
                thres = (torch.abs(value) > self.hallucination_threshold)
            else:
                thres = (value > self.hallucination_threshold)
            
            # If hallucination detected
            if True in thres:
                for i in range(wl, wr):
                    if thres[i - wl].item() == True:
                        # Find first two real words
                        count_k_2 = 0
                        j = i - 1
                        while count_k_2 < 2 and j >= 0:
                            if outputs.fun_word[j]['val'].item() != 0:
                                count_k_2 += 1
                            if count_k_2 == 2:
                                break
                            j -= 1
                        
                        return CheckerOutput(
                            hallucination=True, 
                            curr_st=i, 
                            curr_en=wr, 
                            curr_thres=thres[i-wl:wr]
                        )
        
        return CheckerOutput(hallucination=False)
    
    def generate_retrieve_qry(self, outputs: GeneratorOutput, check_info: CheckerOutput):
        """Generate retrieval query"""
        ques_st = outputs.blocks[0].len_words + outputs.blocks[1].len_words
        ques_en = ques_st + outputs.blocks[2].len_words
        
        text_st = ques_en + outputs.blocks[3].len_words
        text_en = text_st + outputs.blocks[4].len_words + check_info.curr_st
        
        # Compute attention of question and text
        ques_atten = outputs.atten[check_info.curr_st:check_info.curr_en, ques_st:ques_en]
        text_atten = outputs.atten[check_info.curr_st:check_info.curr_en, text_st:text_en]
        
        ques_atten = ques_atten[check_info.curr_thres, :].sum(dim=0)
        text_atten = text_atten[check_info.curr_thres, :].sum(dim=0)
        
        # Extract real words
        doc = nlp(outputs.merged_blocks.text)
        real_words = set(
            token.text for token in doc 
            if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM']
        )
        
        real_pairs = []
        for i in range(ques_st, ques_en):
            a = ques_atten[i - ques_st]
            tl, tr = outputs.merged_blocks.range_[i]
            word = self.etc_generator.tokenizer.convert_tokens_to_string(
                outputs.merged_blocks.tokens[tl:tr]
            )
            if match(word, real_words):
                real_pairs.append((a, word, i))
        
        for i in range(text_st, text_en):
            a = text_atten[i - text_st]
            tl, tr = outputs.merged_blocks.range_[i]
            word = self.etc_generator.tokenizer.convert_tokens_to_string(
                outputs.merged_blocks.tokens[tl:tr]
            )
            if match(word, real_words):
                real_pairs.append((a, word, i))
        
        # Select top-k real words
        if hasattr(self, "retrieve_keep_top_k"):
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif hasattr(self, "retrieve_keep_ratio"):
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)
        else:
            top_k = len(real_pairs)
        
        real_pairs.sort(key=lambda x: -x[0])
        real_pairs = real_pairs[:top_k]
        real_pairs.sort(key=lambda x: x[2])
        
        return " ".join([x[1] for x in real_pairs])
    
    def inference(self, question, demo, case, qid=None):
        """ETC inference process"""
        text = ""
        demo_text = "\n".join([d["case"] for d in demo])
        retrieval_round = 0
        
        if self.debug:
            logger.info("[ETC] Begin reasoning")
        
        while True:
            old_len = len(text)
            
            # Generate new text
            outputs = self.etc_generator.generate(
                input_texts=[demo_text, "\nQuestion:", question, "\nAnswer:", text],
                max_length=self.generate_max_length,
            )
            
            if self.debug and not outputs.empty:
                logger.info(f"[ETC] Generated: {outputs.new_text}")
            
            if self.use_counter and not outputs.empty:
                self.counter.add_generate(
                    outputs.new_text, 
                    self.etc_generator.tokenizer
                )
            
            # If generation is empty, break
            if outputs.empty:
                if self.debug:
                    logger.info("[ETC] Empty generation, breaking")
                break
            
            # Detect hallucination
            check_info = self.hallucination_check(outputs)
            
            if not check_info.hallucination:
                # No hallucination, accumulate text
                if self.debug:
                    logger.info("[ETC] No hallucination")
                
                text = join_if_nonempty(text, outputs.new_text.strip())
                
                if outputs.ended or outputs.merged_blocks.len_tokens > self.generate_max_length:
                    if self.debug:
                        logger.info("[ETC] Ending generation")
                    break
            else:
                # Hallucination detected, perform retrieval
                if self.debug:
                    logger.info("[ETC] Hallucination detected, retrieving")
                
                retrieval_round += 1
                retrieve_qry = self.generate_retrieve_qry(outputs, check_info)
                
                if self.debug:
                    logger.info(f"[ETC] Retrieve query: {retrieve_qry}")
                
                docs = self.retrieve(retrieve_qry, topk=self.retrieve_topk)
                
                # Save retrieval info
                if qid is not None:
                    hallucinated_content = {
                        "curr_st": check_info.curr_st,
                        "curr_en": check_info.curr_en,
                    }
                    self.save_retrieval_info(
                        qid=qid,
                        question=question,
                        query=retrieve_qry,
                        retrieved_docs=docs,
                        retrieval_round=retrieval_round,
                        current_text=text,
                        hallucinated_content=hallucinated_content,
                        additional_info={"method": "ETC"}
                    )
                
                # Normalize retrieved results
                try:
                    docs_list = docs.tolist() if hasattr(docs, 'tolist') else list(docs)
                    docs_list = ["" if d is None else str(d) for d in docs_list]
                except Exception:
                    docs_list = [str(docs)]
                
                # Build prompt with retrieved documents
                prompt = demo_text
                prompt += "\nContext:\n"
                for i, doc in enumerate(docs_list):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                
                # Add Question and Answer prefix
                for i in [1, 2, 3]:
                    prompt += outputs.blocks[i].text
                
                # Keep text before hallucination
                text = self.etc_generator.tokenizer.convert_tokens_to_string(
                    outputs.blocks[-2].tokens + 
                    outputs.blocks[-1].tokens[:outputs.blocks[-1].range_[check_info.curr_st][0]]
                )
                prompt += text
                
                # Regenerate
                ended, new_texts = self.etc_generator.simply_generate(
                    prompt, max_length=self.generate_max_length
                )
                
                if self.use_counter:
                    self.counter.add_generate(new_texts, self.etc_generator.tokenizer)
                    self.counter.hallucinated += 1
                
                new_text = self.get_top_sentence(new_texts)
                text = join_if_nonempty(text, new_text.strip())
                
                if self.debug:
                    logger.info(f"[ETC] Regenerated: {new_text}")
                
                if ended and len(new_text) >= len(new_texts.strip()):
                    if self.debug:
                        logger.info("[ETC] Ended after retrieval")
                    break
                
                if len(self.etc_generator.tokenizer.encode(text)) > self.generate_max_length:
                    if self.debug:
                        logger.info("[ETC] Max length reached")
                    break
            
            # Prevent infinite loop
            if old_len >= len(text):
                logger.info("[ETC] old_len >= len(text), breaking")
                break
        
        if self.debug:
            logger.info(f"[ETC] Finished: {text}")
        
        return text


class AttnWeightRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def modifier(self, text, tokens, attentions, weight):
        """
        Detect hallucination in generated text based on attention and weight.
        
        Returns:
            True: hallucination detected
            prev: clean text before hallucination (to keep correct part)
            tokens[tl:tr]: all tokens of the current sentence
            thres: indicator array, marks which tokens are hallucinated (1 for hallucinated, 0 for normal)
        If no hallucination:
            False, text, None, None
        """
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        tid = 0
        for sid, sent in enumerate(sentences):
            tl, tr = tid, tid
            if sid == len(sentences) - 1:
                tl, tr = tid, len(tokens)
            else:
                for i in range(tid + 1, len(tokens)):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq:
                        tr = i
                        break
                tid = tr
            # value = attention * (-log prob)
            attns = attentions[tl:tr]
            attns = np.array(attns) / sum(attns)
            value = [attns[i-tl] * weight[i] * (tr-tl) for i in range(tl, tr)] 
            thres = [1 if v > self.hallucination_threshold else 0 for v in value]
            if 1 in thres:
                # hallucinated
                if "check_real_words" in self.__dict__ and self.check_real_words:
                    doc = nlp(sent)
                    real_words = set(token.text for token in doc if token.pos_ in 
                        ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                    def match(tok):
                        for word in real_words:
                            if word in tok:
                                return True
                        return False
                    for i in range(len(thres)):
                        if not match(tokens[tl+i]):
                            thres[i] = 0                
                
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # curr = " ".join(
                #     [tokens[i] if thres[i] == 0 else "[xxx]" for i in range(len(thres))]
                # )
                return True, prev, tokens[tl:tr], thres
        return False, text, None, None

    def keep_real_words(self, prev_text, curr_tokens, curr_hit, all_tokens=None, all_attns=None):
        """
        Extract real words with high attention from the context for retrieval.
        
        When all_tokens and all_attns are provided (from generate_attn), we work directly
        with those to avoid re-encoding and tokenization mismatch issues.
        
        Args:
            prev_text: Text before the hallucinated part
            curr_tokens: Merged tokens of the current sentence
            curr_hit: Binary array indicating hallucinated tokens
            all_tokens: Full merged token list from generate_attn (if available)
            all_attns: Attention scalars for each token in all_tokens (if available)
        
        Returns:
            Query string built from high-attention real words
        """
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        
        # Strategy: If we have all_tokens from generate_attn, use them directly
        # This avoids re-encoding which causes tokenization differences
        if all_tokens is not None:
            tokens = all_tokens
            
            # Locate where curr_tokens appears in all_tokens
            def _find_curr_start(all_toks, sub_toks):
                n, m = len(all_toks), len(sub_toks)
                if m == 0:
                    return n
                # Prefer exact match at the end (most recent generation)
                for st in range(n - m, -1, -1):
                    if all_toks[st:st + m] == sub_toks:
                        return st
                # Fallback: longest suffix match
                for k in range(min(n, m), 0, -1):
                    if all_toks[n - k:n] == sub_toks[m - k:m]:
                        return n - k
                return max(0, n - m)
            
            curr_st = _find_curr_start(tokens, curr_tokens)
            
            # Check if curr_st is valid
            if curr_st + len(curr_tokens) > len(tokens):
                logger.warning(
                    "keep_real_words: curr_tokens alignment out of bounds (curr_st=%d, curr_len=%d, all_len=%d)",
                    curr_st, len(curr_tokens), len(tokens)
                )
                # Fallback to simple query
                try:
                    fallback = self.get_last_sentence(all_text)
                except Exception:
                    fallback = None
                if not fallback or len(fallback.strip()) == 0:
                    clean_toks = [tok for i, tok in enumerate(curr_tokens)
                                  if not (i < len(curr_hit) and curr_hit[i] == 1)]
                    fallback = " ".join(clean_toks if clean_toks else curr_tokens)
                return fallback
            
            # Build token-attention pairs from all_tokens (excluding hallucinated curr_tokens)
            token_att_pairs = []
            for i in range(len(tokens)):
                # Skip hallucinated tokens in curr region
                if curr_st <= i < curr_st + len(curr_tokens):
                    offset = i - curr_st
                    if offset < len(curr_hit) and curr_hit[offset] == 1:
                        continue
                
                tok = tokens[i]
                att = all_attns[i] if all_attns and i < len(all_attns) else 0.0
                token_att_pairs.append((att, tok, i))
            
            # Filter for real words (nouns, verbs, adjectives, etc.)
            doc = nlp(all_text)
            real_words = set(token.text for token in doc if token.pos_ in 
                          ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
            
            def match(token):
                for word in real_words:
                    if word in token:
                        return True
                return False
            
            real_pairs = [(att, tok, idx) for att, tok, idx in token_att_pairs if match(tok)]
            
            if len(real_pairs) == 0:
                logger.warning("keep_real_words: no real words found, using fallback")
                try:
                    fallback = self.get_last_sentence(all_text)
                except Exception:
                    fallback = None
                if not fallback or len(fallback.strip()) == 0:
                    clean_toks = [tok for i, tok in enumerate(curr_tokens)
                                  if not (i < len(curr_hit) and curr_hit[i] == 1)]
                    fallback = " ".join(clean_toks if clean_toks else curr_tokens)
                return fallback
            
            # Select top-k by attention
            if "retrieve_keep_top_k" in self.__dict__:
                top_k = min(self.retrieve_keep_top_k, len(real_pairs))
            elif "retrieve_keep_ratio" in self.__dict__:
                top_k = max(1, int(len(real_pairs) * self.retrieve_keep_ratio))
            else:
                top_k = len(real_pairs)
            
            real_pairs = sorted(real_pairs, key=lambda x: x[0], reverse=True)
            real_pairs = real_pairs[:top_k]
            real_pairs = sorted(real_pairs, key=lambda x: x[2])  # restore order
            
            return " ".join([tok for _, tok, _ in real_pairs])
        
        else:
            # Legacy fallback: re-encode (may have alignment issues)
            logger.warning("keep_real_words: all_tokens not provided, using re-encoding (may be unreliable)")
            try:
                fallback = self.get_last_sentence(all_text)
            except Exception:
                fallback = None
            if not fallback or len(fallback.strip()) == 0:
                clean_toks = [tok for i, tok in enumerate(curr_tokens)
                              if not (i < len(curr_hit) and curr_hit[i] == 1)]
                fallback = " ".join(clean_toks if clean_toks else curr_tokens)
            return fallback
        
    def inference(self, question, demo, case, qid=None):
        # assert self.query_formulation == "direct"
        # print(question)
        # print("#" * 20)
        text = ""
        retrieval_round = 0
        
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            tmp_li = [case, text]
            prompt += " ".join(s for s in tmp_li if len(s) > 0)
            
            new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt, 
                self.generate_max_length, 
                # self.attention_solver, 
                use_entropy = self.method == "dragin", 
                use_logprob = self.method == "attn_prob"
            )
            weight = entropies if self.method == "dragin" else [-v for v in logprobs]

            if self.debug:
                logging.info(f"Prompt for Generation:\n{prompt}\n")
                logging.info(f"Generated Text: {new_text}\n")
            
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer, api_usage=self.generator.last_usage)
            hallucination, ptext, curr_tokens, curr_hit =  self.modifier(new_text, tokens, attns, weight)
            
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                retrieval_round += 1
                forward_all = [question, text, ptext]
                forward_all = " ".join(s for s in forward_all if len(s) > 0)

                def fetch_last_n_tokens(text, num, tokenizer = self.generator.tokenizer):
                    tokens = tokenizer.tokenize(text)
                    if num >= len(tokens):
                        return text
                    last_n_tokens = tokens[-num:]
                    last_n_sentence = ' '.join(last_n_tokens)
                    return last_n_sentence

                if self.query_formulation == "current":
                    retrieve_question = " ".join(curr_tokens)

                elif self.query_formulation == "current_wo_wrong":
                    retrieve_question = " ".join(
                        list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens)))
                    )

                elif self.query_formulation == "forward_all":
                    retrieve_question = forward_all
                
                elif self.query_formulation == "last_sentence":
                    retrieve_question = self.get_last_sentence(forward_all)
                
                elif self.query_formulation == "last_n_tokens":
                    assert "retrieve_keep_top_k" in self.__dict__
                    retrieve_question = fetch_last_n_tokens(
                        forward_all, self.retrieve_keep_top_k)
                
                elif self.query_formulation == "real_words": 
                    retrieve_question = self.keep_real_words(
                        prev_text = question + " " + text + " " + ptext, 
                        curr_tokens = curr_tokens, 
                        curr_hit = curr_hit,
                        all_tokens = tokens,  # Pass full merged tokens from generate_attn
                        all_attns = attns,    # Pass full attention weights from generate_attn
                    ) 
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                
                # Save detailed attention weight information for AttnWeightRAG
                if qid is not None:
                    hallucinated_content = {
                        "original_tokens": tokens,
                        "attention_weights": [float(att) for att in attns],  # Convert to serializable format
                        "log_probs": logprobs if logprobs else [],
                        "entropies": entropies if entropies else [],
                        "weight_values": [float(w) for w in weight],
                        "partial_text": ptext,
                        "current_tokens": curr_tokens,
                        "hallucination_hits": curr_hit,
                        "threshold_used": self.hallucination_threshold,
                        "method_used": self.method,
                        "query_formulation_method": self.query_formulation,
                        "forward_all_context": forward_all
                    }
                    
                    additional_info = {
                        "method": "AttnWeightRAG", 
                        "query_formulation": self.query_formulation,
                        "hallucination_detected": True,
                        "attention_solver": getattr(self, 'attention_solver', 'unknown'),
                        "check_real_words": getattr(self, 'check_real_words', False)
                    }
                    
                    # If special retrieval keep strategy used, add related info
                    if hasattr(self, 'retrieve_keep_top_k'):
                        additional_info["retrieve_keep_top_k"] = self.retrieve_keep_top_k
                    if hasattr(self, 'retrieve_keep_ratio'):
                        additional_info["retrieve_keep_ratio"] = self.retrieve_keep_ratio
                    
                    self.save_retrieval_info(
                        qid=qid,
                        question=question,
                        query=retrieve_question,
                        retrieved_docs=docs,
                        retrieval_round=retrieval_round,
                        current_text=text,
                        hallucinated_content=hallucinated_content,
                        additional_info=additional_info
                    )

                if self.debug:
                    logging.info(f"Retrieval Round {retrieval_round}")
                    logging.info(f"Retrieve Question: {retrieve_question}")
                    logging.info(f"Retrieved Docs:")
                    for i, doc in enumerate(docs):
                        logging.info(f"  [{i+1}] {doc}")
                    logging.info(f"Partial Text before hallucination: {ptext}")
                    logging.info(f"Current Tokens with Hallucination Marks: ")
                    for i, (tok, hit) in enumerate(zip(curr_tokens, curr_hit)):
                        mark = "HALLUCINATED" if hit == 1 else "NORMAL"
                        logging.info(f"  {i}: '{tok}' -> {mark}")
                    logging.info("#" * 20)
                    logging.info(f"text so far: {text}\n")

                prompt = "".join([d["case"]+"\n" for d in demo])
                
                # Use prompt template
                hallucination_section = self._get_prompt_section("hallucination")
                context_header = hallucination_section["context_header"]
                doc_format = hallucination_section["doc_format"]
                post_doc = hallucination_section["post_doc"]
                instruction = hallucination_section["instruction"]
                
                prompt += context_header
                for i, doc in enumerate(docs):
                    prompt += doc_format.format(index=i+1, doc=doc)
                prompt += post_doc
                prompt += instruction
                tmp_li = [case, text, ptext.strip()]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)
                
                if self.debug:
                    logging.info("New Prompt for Generation:")
                    logging.info(prompt)
                
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer, api_usage=self.generator.last_usage)
                    self.counter.hallucinated += 1

                if self.debug:
                    logging.info(f"New Generated Text: {new_text}")

                new_text = self.get_top_sentence(new_text)

                if self.debug:
                    logging.info(f"Top Sentence Extracted: {new_text}")

                tmp_li = [text.strip(), ptext.strip(), new_text.strip()]
                text = " ".join(s for s in tmp_li if len(s) > 0)
                
                if self.debug:
                    logging.info(f"Updated Full Text: {text}\n")
                    
            
            # Judge if token count is less than generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text or "Question:" in text:
                if self.debug:
                    logging.info(f"Stopping criteria met. Tokens count: {tokens_count > self.generate_max_length}, Length change: {len(text) <= old_len}, 'the answer is' in text: {'the answer is' in text}, 'Question:' in text: {'Question:' in text}")

                break
        
        return text

