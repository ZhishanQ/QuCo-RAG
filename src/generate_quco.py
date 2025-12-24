import numpy as np
import logging
import spacy
import torch
import json
import os
import time
import requests
import re
from typing import Optional
from contextlib import contextmanager

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None
from copy import deepcopy
from retriever import BM25, SGPT
from qwen3_retriever import Qwen3Retriever
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from gpt_api_client import create_gpt_client

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

_file_lock_warning_emitted = False

system_prompt_for_llama2 = "You are a helpful assistant."

def get_infini_gram_count(query, index='v4_rpj_llama_s4', query_type='count', max_diff_tokens=1000, max_clause_freq=500000):
    """
    Sends a request to the Infini-gram API and returns the count and engine latency.

    :param query: The query string.
    :param index: The index to search, defaults to 'v4_rpj_llama_s4'.
    :param query_type: The type of query, defaults to 'count'.
    :param max_diff_tokens: Max token distance for AND co-occurrences, defaults to 1000.
    :param max_clause_freq: Threshold for approximate counts, defaults to 500000.
    :return: Tuple of (count, engine_latency_in_seconds) or (None, None) if an error occurs.
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
    
    try:
        response = requests.post('https://api.infini-gram.io/', json=payload)
        response.raise_for_status()  # check HTTP errors
        result = response.json()
        if 'error' in result:
            logger.error(f"An error occurred: {result['error']}")
            return None, None
        
        # Extract count
        count = None
        if 'count' in result:
            count = result['count']
        elif isinstance(result, (int, float)):
            count = result
        else:
            logger.warning(f"Unexpected API response format: {result}")
            return None, None
        
        # Extract engine latency (convert from milliseconds to seconds)
        engine_latency = None
        if 'latency' in result:
            engine_latency = result['latency'] / 1000.0  # Convert ms to seconds
        
        return count, engine_latency
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request failed: {e}")
        return None, None
    except ValueError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        return None, None


class BasicGenerator:
    def __init__(
        self,
        model_name_or_path,
        device_map="auto",
        use_api=False,
        api_base_url=None,
        tokenizer_name_or_path: Optional[str] = None,
        use_web_search: bool = False,
        use_chat_template: bool = False,
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
        self.use_chat_template = use_chat_template  # Store chat template flag
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
            logger.info(f"Loading model from {self.model_name_or_path}")
            tokenizer_source = tokenizer_name_or_path or self.model_name_or_path
            self._load_tokenizer(tokenizer_source, allow_failure=False)

            self.model_config = AutoConfig.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map=device_map,
                config=self.model_config,
            )

            if self.tokenizer and self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if self.model.config.pad_token_id is None and self.tokenizer is not None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            if self.model_config.model_type == "llama":
                self.space_token = "▁"
            elif self.model_config.model_type == "olmo2":
                tokens = self.tokenizer.tokenize(" ") if self.tokenizer else []
                self.space_token = tokens[0] if tokens else " "
            else:
                tokens = self.tokenizer.tokenize(" ") if self.tokenizer else []
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

    def generate(self, input_text, max_length):
        # if self.debug:
        #     logger.info("=" * 50)
        #     logger.info("Generating text with local model...")
        #     logger.info(f"Input text: {input_text}")
        #     logger.info(f"Max length: {max_length}")
        #     logger.info("=" * 50)

        if self.use_api:
            response = self.api_client.simple_completion(
                prompt=input_text,
                model=self.model_name_or_path,
                use_web_search=self.use_web_search,
            )
            self.last_usage = None
            return response

        # Wrap with chat template if enabled (use model's default template)
        if self.use_chat_template:
            messages = [
                {"role": "user", "content": input_text}
            ]
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        # if self.debug:
        #     logger.info(f"Final input text for generation: {input_text}")

        # Tokenize (works for both chat template and direct input)
        model_inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)
        input_length = model_inputs.input_ids.shape[1]

        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Extract only the newly generated tokens
        generated_tokens = outputs[:, input_length:]
        text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        self.last_usage = None
        return text


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

    def add_generate(self, text, tokenizer, api_usage=None):
        self.generate += 1
        
        # Track API token usage if available (for API-based models)
        if api_usage is not None:
            self.api_input_tokens += api_usage.get("input_tokens", 0)
            self.api_output_tokens += api_usage.get("output_tokens", 0)
            self.api_total_tokens += api_usage.get("total_tokens", 0)
            # Track web search calls if present
            self.web_search_calls += api_usage.get("web_search_calls", 0)
        
        token_count = None
        if tokenizer is not None:
            try:
                ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
                token_count = len(ids)
            except Exception as exc:
                logger.debug(f"Failed to tokenize generated text for counting: {exc}")
        if token_count is None:
            token_count = len(text.split())
            if self.debug:
                logger.debug("Using approximate token count based on word splitting.")
                
        self.token += token_count
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
         

def parse_case(case_str: str):
    case_str = case_str.strip()
    if not case_str:
        return []

    stack = []
    current_token = []
    in_quotes = False
    escape_next = False
    expected_quote = None
    root = None

    quote_pairs = {
        '"': '"',
        '“': '”',
    }
    closing_quotes = {closing: opening for opening, closing in quote_pairs.items()}

    for ch in case_str:
        if in_quotes:
            if escape_next:
                current_token.append(ch)
                escape_next = False
                continue

            if ch == '\\':
                escape_next = True
                continue

            if expected_quote and ch == expected_quote:
                in_quotes = False
                expected_quote = None
                continue

            current_token.append(ch)
            continue

        if ch in quote_pairs:
            in_quotes = True
            expected_quote = quote_pairs[ch]
            continue

        if ch in closing_quotes:
            current_token.append(ch)
            continue

        if ch == '[':
            new_list = []
            if stack:
                stack[-1].append(new_list)
            stack.append(new_list)
            if root is None:
                root = stack[0]
            continue

        if ch == ']':
            token = ''.join(current_token).strip()
            if token:
                stack[-1].append(token)
            current_token = []
            if stack:
                finished = stack.pop()
                if not stack:
                    root = finished
            continue

        if ch == ',':
            if stack:
                current_list = stack[-1]
                if (
                    len(stack) >= 2
                    and len(current_list) >= 2
                    and current_token
                ):
                    current_token.append(ch)
                    continue

            token = ''.join(current_token).strip()
            if token and stack:
                stack[-1].append(token)
            current_token = []
            continue

        if ch.isspace():
            if current_token:
                current_token.append(' ')
            continue

        current_token.append(ch)

    if stack:
        token = ''.join(current_token).strip()
        if token:
            stack[-1].append(token)

    return root or []


class BasicRAG:
    def __init__(self, args):
        args = args.__dict__ 
        for k, v in args.items():
            setattr(self, k, v)

        # Ensure debug attribute exists with default value
        self.debug = self.debug
        
        self.use_llm_api = getattr(self, "use_llm_api", False)
        model_device_map = getattr(self, "model_device_map", "auto")
        api_base_url = getattr(self, "api_base_url", "https://api.openai.com/v1")
        tokenizer_name_or_path = getattr(self, "tokenizer_name_or_path", None)
        use_web_search = getattr(self, "use_web_search", False)  # Read from config
        use_chat_template = getattr(self, "chat_template", False)  # Read chat_template from config

        if self.use_llm_api:
            logger.info(f"Using remote LLM API model '{self.model_name_or_path}' via {api_base_url}")
            if tokenizer_name_or_path:
                logger.info(f"Will use tokenizer '{tokenizer_name_or_path}' for accurate token counting")
            else:
                logger.info("No tokenizer specified. Token statistics will be approximate.")
            if use_web_search:
                logger.info("Web search tool is enabled for API calls")
        else:
            logger.info(f"Loading model '{self.model_name_or_path}' with device_map: {model_device_map}")
        
        if use_chat_template:
            logger.info("Chat template wrapping is enabled")
        
        self.generator = BasicGenerator(
            self.model_name_or_path,
            device_map=model_device_map,
            use_api=self.use_llm_api,
            api_base_url=api_base_url,
            tokenizer_name_or_path=tokenizer_name_or_path,
            use_web_search=use_web_search,
            use_chat_template=use_chat_template,
        )
        setattr(self.generator, "debug", self.debug)
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
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
        
        self.retrieved_docs = []
        self.docs_output_file = None
        if hasattr(self, 'output_dir'):
            docs_file_path = os.path.join(self.output_dir, "retrieved_docs.json")
            self.docs_output_file = docs_file_path

        self.enable_cache = getattr(self, "enable_cache", False)
        self._cache_dir = None
        self._cache_prefix = None
        self._retrieval_cache = {}
        self._retrieval_cache_dirty = False
        self._retrieval_cache_hits = 0
        if self.enable_cache:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cache"))
            os.makedirs(base_dir, exist_ok=True)
            dataset_name = getattr(self, "dataset", "default")
            safe_dataset = re.sub(r"[^A-Za-z0-9_.-]", "_", str(dataset_name))
            # Base prefix uses only dataset (for entity_extraction, infini_gram)
            self._cache_prefix = safe_dataset
            self._cache_dir = base_dir
            
            # For retrieval cache, separately add es_index_name to distinguish different indices
            es_index = getattr(self, "es_index_name", None)
            if es_index:
                safe_index = re.sub(r"[^A-Za-z0-9_.-]", "_", str(es_index))
                retrieval_prefix = f"{safe_dataset}_{safe_index}"
            else:
                retrieval_prefix = safe_dataset
            
            # Use special prefix to generate retrieval cache path
            retrieval_filename = f"{retrieval_prefix}_retrieval.json"
            self.retrieval_cache_path = os.path.join(self._cache_dir, retrieval_filename)
            self._retrieval_cache = self._load_cache_file(self.retrieval_cache_path)
            self._ensure_cache_file(self.retrieval_cache_path)
        else:
            self.retrieval_cache_path = None

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1

        if getattr(self, 'retriever_type', None) == "Qwen3":
            task = 'Given a web search query, retrieve relevant passages that answer the query'
            query = f'Instruct: {task}\nQuery:{query}'
            if self.debug:
                logger.info(f"Qwen3 retriever, using instructed query: {query}")

        cache_key = None
        if self.enable_cache:
            cache_key = self._make_retrieval_cache_key(query, topk, max_query_length)
            cached_docs = self._retrieval_cache.get(cache_key)
            if cached_docs is not None:
                if self.debug:
                    logger.info(f"Retrieval cache hit for query='{query}' topk={topk} max_query_length={max_query_length}")
                self._retrieval_cache_hits += 1
                return list(cached_docs)

        if getattr(self, 'retriever_type', None) == "BM25":
            _docs_ids, docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk, 
                max_query_length = max_query_length,
            )
            result = docs[0]
        elif getattr(self, 'retriever_type', None) == "SGPT":
            docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk,
            )
            result = docs[0]
        elif getattr(self, 'retriever_type', None) == "Qwen3":
            docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk,
            )
            result = docs[0]
        else:
            raise NotImplementedError

        normalized_docs = self._normalize_docs(result)
        if self.enable_cache and cache_key is not None:
            self._retrieval_cache[cache_key] = list(normalized_docs)
            self._retrieval_cache_dirty = True
            if self.debug:
                logger.info(f"Retrieval result is now cached for query='{query}' topk={topk} max_query_length={max_query_length}")

        return normalized_docs
    
    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else ""
    
    def filter_invalid_sentences(self, text):
        """
        Filter out invalid sentences that contain common assistant-style phrases.
        
        :param text: The text to filter
        :return: Filtered text with invalid sentences removed
        """
        # Define invalid patterns (case-insensitive)
        invalid_patterns = [
            "happy to help",
            "here's the",
            "i'd be happy",
            "i'm happy to",
            "sure,",
            "certainly",
            "of course",
            "let me help",
            "i can help",
            "here are the",
            "here is the",
            "let me provide",
            "i'll help",
            "glad to help",
        ]
        
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        filtered_sentences = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            # Check if sentence contains any invalid pattern
            is_invalid = any(pattern in sent_lower for pattern in invalid_patterns)
            
            if not is_invalid:
                filtered_sentences.append(sent)
            elif self.debug:
                logger.info(f"Filtered out invalid sentence: {sent}")
        
        # Join filtered sentences back together
        result = " ".join(filtered_sentences)
        
        if self.debug and result != text:
            logger.info(f"Text filtering applied. Original length: {len(text)}, Filtered length: {len(result)}")
        
        return result
    
    def save_retrieval_info(self, qid, question, query, retrieved_docs, retrieval_round, 
                           current_text="", hallucinated_content=None, additional_info=None):
        try:
            docs_list = retrieved_docs.tolist() if hasattr(retrieved_docs, 'tolist') else list(retrieved_docs)
            docs_list = ["" if d is None else str(d) for d in docs_list]
        except Exception:
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
        
        if hallucinated_content is not None:
            doc_entry["hallucinated_content"] = hallucinated_content
        
        if additional_info is not None:
            doc_entry.update(additional_info)
        
        self.retrieved_docs.append(doc_entry)
        
        if self.docs_output_file:
            self._save_docs_to_file()
    
    def _save_docs_to_file(self):
        if self.docs_output_file and self.retrieved_docs:
            try:
                with self._cache_lock(self.docs_output_file, exclusive=True):
                    with open(self.docs_output_file, 'w', encoding='utf-8') as f:
                        json.dump(self.retrieved_docs, f, ensure_ascii=False, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
            except Exception as e:
                logger.warning(f"Failed to save retrieved docs to {self.docs_output_file}: {e}") 

    @contextmanager
    def _cache_lock(self, path, exclusive=True):
        if not path or fcntl is None:
            global _file_lock_warning_emitted
            if fcntl is None and not _file_lock_warning_emitted:
                logger.warning("fcntl module unavailable; cache operations will not be locked across processes.")
                _file_lock_warning_emitted = True
            yield
            return

        lock_path = f"{path}.lock"
        lock_dir = os.path.dirname(lock_path)
        if lock_dir and not os.path.exists(lock_dir):
            os.makedirs(lock_dir, exist_ok=True)

        lock_mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        with open(lock_path, 'a+') as lock_file:
            fcntl.flock(lock_file, lock_mode)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _get_cache_path(self, suffix):
        if not self.enable_cache or self._cache_dir is None or self._cache_prefix is None:
            return None
        filename = f"{self._cache_prefix}_{suffix}.json"
        return os.path.join(self._cache_dir, filename)

    def _load_cache_file(self, path):
        if not path:
            return {}
        with self._cache_lock(path, exclusive=False):
            if not os.path.exists(path):
                return {}
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as e:
                logger.warning(f"Failed to load cache file {path}: {e}")
            return {}

    def _ensure_cache_file(self, path):
        if not path:
            return
        with self._cache_lock(path, exclusive=True):
            if not os.path.exists(path):
                try:
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump({}, f)
                        f.flush()
                        os.fsync(f.fileno())
                except Exception as e:
                    logger.warning(f"Failed to initialize cache file {path}: {e}")

    def _persist_cache(self, path, data):
        if not path:
            return
        with self._cache_lock(path, exclusive=True):
            tmp_path = f"{path}.tmp"
            try:
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
            except Exception as e:
                logger.warning(f"Failed to save cache file {path}: {e}")
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    def _make_retrieval_cache_key(self, query, topk, max_query_length):
        payload = {
            "retriever": getattr(self, "retriever_type", "unknown"),
            "query": query,
            "topk": topk,
            "max_query_length": max_query_length
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _normalize_docs(self, docs):
        if isinstance(docs, np.ndarray):
            docs_list = docs.tolist()
        elif isinstance(docs, (list, tuple)):
            docs_list = list(docs)
        else:
            docs_list = [docs]

        normalized = []
        for item in docs_list:
            if isinstance(item, str):
                normalized.append(item)
            elif item is None:
                normalized.append("")
            else:
                normalized.append(str(item))
        return normalized

    def save_cache(self):
        if not self.enable_cache:
            return
        if getattr(self, "_retrieval_cache_dirty", False):
            logger.info(f"Persisting retrieval cache to {self.retrieval_cache_path}")
            self._persist_cache(self.retrieval_cache_path, self._retrieval_cache)
            self._retrieval_cache_dirty = False
    
    def _generate_text(self, prompt, max_length):
        """Generate text using the model directly without chat template."""
        if self.debug:
            logger.info("Generating text without chat template")
        return self.generator.generate(prompt, max_length)

    def inference(self, question, demo, case):
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += case
 
        prediction = self._generate_text(prompt, self.generate_max_length)
            
        if self.use_counter == True:
            self.counter.add_generate(prediction, self.generator.tokenizer, self.generator.last_usage)
        
        # Filter out invalid sentences before returning
        prediction = self.filter_invalid_sentences(prediction)
        
        return prediction


class QuCo_RAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
        
        # Ensure use_web_search attribute is set (inherited from parent class or set default value)
        if not hasattr(self, 'use_web_search'):
            self.use_web_search = False
        
        # ==== Determine entity extraction method ====
        self.gpt_model = getattr(self, "gpt_model", "gpt-4o-mini")
        self.use_local_entity_model = "gpt" not in self.gpt_model.lower()
        
        # ==== Setup entity extraction (local model or GPT API) ====
        self.entity_extractor = None
        self.gpt_client = None
        
        if self.use_local_entity_model:
            # Use local model for entity extraction
            logger.info(f"Using local entity extraction model: {self.gpt_model}")
            entity_device_map = getattr(self, "entity_model_device_map", "auto")
            self.entity_extractor = BasicGenerator(
                model_name_or_path=self.gpt_model,
                device_map=entity_device_map,
                use_api=False,
                use_chat_template=True,  # Use chat template for entity extraction
            )
            # setattr(self.entity_extractor, "debug", self.debug)
            logger.info(f"Local entity extraction model loaded successfully")
        else:
            # Use GPT API for entity extraction
            logger.info(f"Using GPT API for entity extraction: {self.gpt_model}")
            API_KEY = os.getenv("OPENAI_API_KEY")
            BASE_URL = "https://api.openai.com/v1"
            self.gpt_client = create_gpt_client(api_key=API_KEY, base_url=BASE_URL, max_retries=3)
        
        
        self.infini_gram_index_name = args.infini_gram_index_name
        self.retrieval_query_num = args.retrieval_query_num
        self.enable_time_stats = getattr(self, "enable_time_stats", False)
        self._entity_cache_hits = 0
        self._infini_cache_hits = 0
        self._question_analysis_cache_hits = 0

        if self.enable_cache:
            self.entity_cache_path = self._get_cache_path("entity_extraction")
            self._entity_cache = self._load_cache_file(self.entity_cache_path)
            self._ensure_cache_file(self.entity_cache_path)
            self._entity_cache_dirty = False

            self.infini_cache_path = self._get_cache_path("infini_gram")
            self._infini_cache = self._load_cache_file(self.infini_cache_path)
            self._ensure_cache_file(self.infini_cache_path)
            self._infini_cache_dirty = False

            self.question_analysis_cache_path = self._get_cache_path("question_analysis")
            self._question_analysis_cache = self._load_cache_file(self.question_analysis_cache_path)
            self._ensure_cache_file(self.question_analysis_cache_path)
            self._question_analysis_cache_dirty = False
        else:
            self.entity_cache_path = None
            self._entity_cache = {}
            self._entity_cache_dirty = False
            self.infini_cache_path = None
            self._infini_cache = {}
            self._infini_cache_dirty = False
            self.question_analysis_cache_path = None
            self._question_analysis_cache = {}
            self._question_analysis_cache_dirty = False

        retrieval_cache_size = len(getattr(self, "_retrieval_cache", {}))
        entity_cache_size = len(getattr(self, "_entity_cache", {}))
        infini_cache_size = len(getattr(self, "_infini_cache", {}))
        question_analysis_cache_size = len(getattr(self, "_question_analysis_cache", {}))
        logger.info(
            "Cache summary on startup: retrieval=%d, entity=%d, infini_gram=%d, question_analysis=%d",
            retrieval_cache_size,
            entity_cache_size,
            infini_cache_size,
            question_analysis_cache_size,
        )
        logger.info(
            "Cache hit counters on startup: retrieval=%d, entity=%d, infini_gram=%d, question_analysis=%d",
            getattr(self, "_retrieval_cache_hits", 0),
            self._entity_cache_hits,
            self._infini_cache_hits,
            self._question_analysis_cache_hits,
        )

        # ==== QuCo_RAG: question-level threshold and prompts ====
        if not hasattr(args, 'ngram_threshold_question'):
            raise ValueError("ngram_threshold_question is required in args but not found")
        self.ngram_threshold_question = args.ngram_threshold_question
        
        if not hasattr(args, 'question_query_formulation'):
            raise ValueError("question_query_formulation is required in args but not found")
        self.question_query_formulation = args.question_query_formulation
        
        # ==== Hallucination detection thresholds ====
        # Threshold for ternary tuple (3-element): entity pair count < threshold → hallucination
        self.ternary_tuple_threshold = getattr(args, 'ternary_tuple_threshold', 1)
        # Threshold for binary tuple (2-element): entity count < threshold → hallucination
        # Set to -1 to disable binary tuple check, or use a large value to effectively disable
        self.binary_tuple_threshold = getattr(args, 'binary_tuple_threshold', 20)
        # Sentence-level query formulation strategy: 'head_relation', 'full_triplet', 'original_sentence'
        self.sentence_query_formulation = getattr(args, 'sentence_query_formulation', 'head_relation')
        self.prompt_template_path = getattr(args, 'prompt_template_path', None)
        self.prompt_template_key = getattr(args, 'prompt_template_key', None)
        self.prompt_templates = self.load_prompt_templates(self.prompt_template_path)

        self.entity_extraction_prompt = self.prompt_templates.get("entity_extraction_prompt", "")
        self.entity_extraction_prompt_for_question = self.prompt_templates.get("entity_extraction_prompt_for_question", "")
        self.query_rewrite_prompt = self.prompt_templates.get("query_rewrite_prompt", "")

        if not self.entity_extraction_prompt:
            raise ValueError("entity_extraction_prompt not found in prompt_templates.json")
        if not self.entity_extraction_prompt_for_question:
            raise ValueError("entity_extraction_prompt_for_question not found in prompt_templates.json")
        if not self.query_rewrite_prompt:
            raise ValueError("query_rewrite_prompt not found in prompt_templates.json")

        if self.prompt_template_key and self.prompt_template_key not in self.prompt_templates:
            logger.warning(
                "Prompt template key '%s' not found. Falling back to automatic selection.",
                self.prompt_template_key
            )
            self.prompt_template_key = None

        if not self.prompt_template_key:
            self.prompt_template_key = self._resolve_prompt_template_key()

        self.prompt_template = self.prompt_templates.get(
            self.prompt_template_key,
            self.prompt_templates.get("default", {})
        )

        logger.info(
            "Using prompt template key: %s (template path: %s)",
            self.prompt_template_key,
            self.prompt_template_path or os.path.join(os.path.dirname(__file__), "prompt_templates.json")
        )
        logger.info(f"QuCo_RAG initialized with question threshold: {self.ngram_threshold_question}")
        if self.binary_tuple_threshold < 0:
            logger.info(f"Hallucination detection threshold - Binary tuple: DISABLED (threshold < 0)")
        else:
            logger.info(f"Hallucination detection threshold - Binary tuple: < {self.binary_tuple_threshold}")
        if self.ternary_tuple_threshold < 0:
            logger.info(f"Hallucination detection threshold - Ternary tuple: DISABLED (threshold < 0)")
        else:
            logger.info(f"Hallucination detection threshold - Ternary tuple: < {self.ternary_tuple_threshold}")
        logger.info(f"Sentence-level query formulation strategy: {self.sentence_query_formulation}")

        if self.enable_time_stats:
            self.time_stats = {
                "retrieval": {"total_time": 0.0, "count": 0},
                "entity_extraction": {"total_time": 0.0, "count": 0},
                "infini_gram": {"total_time": 0.0, "engine_time": 0.0, "count": 0},
                "generation": {"total_time": 0.0, "count": 0},
            }
            self.time_stats_file = os.path.join(self.output_dir, "time_stats.json") if hasattr(self, "output_dir") else None
        else:
            self.time_stats = None
            self.time_stats_file = None

    def _extract_entities_local(self, prompt, max_new_tokens=256):
        """
        Use local model to extract entities.
        
        Args:
            prompt: The prompt for entity extraction
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            str: The model's response
        """
        if self.entity_extractor is None:
            raise RuntimeError("Local entity extractor not initialized")
        
        try:
            # Generate with the local model
            # logger.info("Using local model for entity extraction")
            # logger.info(f"Entity extraction prompt: {prompt}\n")

            response = self.entity_extractor.generate(prompt, max_new_tokens)
            return response
        except Exception as e:
            logger.error(f"Local entity extraction failed: {e}")
            return None
    
    def _record_time(self, key, duration):
        if not self.enable_time_stats or self.time_stats is None or key not in self.time_stats:
            return
        stat = self.time_stats[key]
        stat["total_time"] += duration
        stat["count"] += 1

    def _generate_with_timing(self, prompt, max_length):
        """Generate text with timing statistics if enabled."""
        if not self.enable_time_stats:
            return self._generate_text(prompt, max_length)
        
        start = time.perf_counter()
        try:
            return self._generate_text(prompt, max_length)
        finally:
            self._record_time("generation", time.perf_counter() - start)

    def _timed_infini_gram_count(self, *args, **kwargs):
        cache_key = None
        if self.enable_cache:
            cache_key = self._make_infini_cache_key(*args, **kwargs)
            if cache_key in self._infini_cache:
                if self.debug:
                    logger.info("Infini-gram cache hit for params: %s", json.dumps({"args": list(args), "kwargs": kwargs}, ensure_ascii=False))
                self._infini_cache_hits += 1
                return self._infini_cache[cache_key]

        start = time.perf_counter() if self.enable_time_stats else None
        count, engine_latency = get_infini_gram_count(*args, **kwargs)
        
        if self.enable_time_stats and start is not None:
            total_time = time.perf_counter() - start
            # Record total time (including network overhead)
            self._record_time("infini_gram", total_time)
            # Record engine latency separately if available
            if engine_latency is not None:
                stat = self.time_stats.get("infini_gram")
                if stat:
                    stat["engine_time"] += engine_latency

        if self.enable_cache and cache_key is not None:
            self._infini_cache[cache_key] = count
            self._infini_cache_dirty = True
            if self.debug:
                logger.info("Infini-gram result is now cached for cache_key: %s", cache_key)

        return count

    def retrieve(self, *args, **kwargs):
        if not self.enable_time_stats:
            return super().retrieve(*args, **kwargs)
        start = time.perf_counter()
        try:
            return super().retrieve(*args, **kwargs)
        finally:
            self._record_time("retrieval", time.perf_counter() - start)

    def save_time_stats(self):
        if not self.enable_time_stats or self.time_stats is None:
            return
        summary = {}
        for key, values in self.time_stats.items():
            total = values["total_time"]
            count = values["count"]
            average = total / count if count else 0.0
            summary[key] = {
                "total_time": self._format_time_value(total),
                "call_count": count,
                "average_time": self._format_time_value(average)
            }
            
            # Add engine_time statistics for infini_gram
            if key == "infini_gram" and "engine_time" in values:
                engine_time = values["engine_time"]
                engine_average = engine_time / count if count else 0.0
                summary[key]["engine_time"] = self._format_time_value(engine_time)
                summary[key]["engine_average_time"] = self._format_time_value(engine_average)

        if summary:
            logger.info("Time statistics summary:\n%s", json.dumps(summary, ensure_ascii=False, indent=2))

        if self.time_stats_file and summary:
            try:
                with open(self.time_stats_file, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write time statistics to {self.time_stats_file}: {e}")

    def _format_time_value(self, value):
        if value is None:
            return "0.00000"
        return f"{float(value):.5f}"

    def _make_infini_cache_key(self, *args, **kwargs):
        payload = {
            "args": list(args),
            "kwargs": {k: kwargs[k] for k in sorted(kwargs)}
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _make_question_analysis_cache_key(self, question, qid=None):
        payload = {
            "question": question,
            "qid": qid,
            "model": self.gpt_model,
            "index": self.infini_gram_index_name,
            "threshold": self.ngram_threshold_question,
            "query_formulation": self.question_query_formulation,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _make_entity_extraction_cache_key(self, sentence):
        """
        Generate cache key for entity extraction from a sentence.
        Includes the model name to differentiate results from different models.
        """
        payload = {
            "sentence": sentence,
            "model": self.gpt_model,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def save_cache(self):
        retrieval_cache_size = len(getattr(self, "_retrieval_cache", {}))
        entity_cache_size = len(getattr(self, "_entity_cache", {}))
        infini_cache_size = len(getattr(self, "_infini_cache", {}))
        question_analysis_cache_size = len(getattr(self, "_question_analysis_cache", {}))
        logger.info(
            "Cache summary on shutdown: retrieval=%d, entity=%d, infini_gram=%d, question_analysis=%d",
            retrieval_cache_size,
            entity_cache_size,
            infini_cache_size,
            question_analysis_cache_size,
        )
        logger.info(
            "Cache hits during run: retrieval=%d, entity=%d, infini_gram=%d, question_analysis=%d",
            getattr(self, "_retrieval_cache_hits", 0),
            getattr(self, "_entity_cache_hits", 0),
            getattr(self, "_infini_cache_hits", 0),
            getattr(self, "_question_analysis_cache_hits", 0),
        )

        super().save_cache()
        if not self.enable_cache:
            return
        if getattr(self, "_entity_cache_dirty", False):
            logger.info(f"Persisting entity extraction cache to {self.entity_cache_path}")
            self._persist_cache(self.entity_cache_path, self._entity_cache)
            self._entity_cache_dirty = False
        if getattr(self, "_infini_cache_dirty", False):
            logger.info(f"Persisting infini-gram cache to {self.infini_cache_path}")
            self._persist_cache(self.infini_cache_path, self._infini_cache)
            self._infini_cache_dirty = False
        if getattr(self, "_question_analysis_cache_dirty", False):
            logger.info(f"Persisting question analysis cache to {self.question_analysis_cache_path}")
            self._persist_cache(self.question_analysis_cache_path, self._question_analysis_cache)
            self._question_analysis_cache_dirty = False


    def modifier(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        not_strings = ["does not", "not provide", "not mention", "not available", "not include", "not specif", "not explicitly", 
        "not publicly", "No information"]

        if self.debug:
            logger.info("=======Analyzing Generated Text for Hallucinations=======")
            logger.info(f"Total sentences: {len(sentences)}")
            for sid, sent in enumerate(sentences):
                logger.info(f"Sentence {sid}: {sent}")
            logger.info("")

        flag_of_hallucination = False
        question_to_retrieve = []
        clear_text = ""
        hallucinated_sentence = None  # Store the original sentence with hallucination
        hallucinated_triplets = []  # Store full triplets that triggered hallucination
        
        for sid, sent in enumerate(sentences):
            if self.debug:
                logger.info(f"Analyzing sentence {sid}: {sent}")
            
            sentence_has_hallucination = False

            if any(ns in sent for ns in not_strings):
                if self.debug:
                    logger.info(f"Sentence {sid} contains negation phrase, skipping hallucination check.")
                break

            if 'the answer is' in sent:
                if self.debug:
                    logger.info(f"Sentence {sid} contains 'the answer is', skipping hallucination check.")
                if len(clear_text) == 0:
                    clear_text = sent
                else:
                    clear_text += " " + sent
                if self.debug:
                    logger.info(f"Sentence contains 'the answer is', added to clear text and breaking out of loop.")
                break

            cache_key = self._make_entity_extraction_cache_key(sent) if self.enable_cache else None
            extraction_result = None
            cache_hit = False
            if self.enable_cache and cache_key is not None and cache_key in getattr(self, '_entity_cache', {}):
                extraction_result = deepcopy(self._entity_cache[cache_key])
                cache_hit = True
                self._entity_cache_hits += 1
                if self.debug:
                    logger.info(f"Sentence {sid} entity extraction loaded from cache.")
                    logger.info(f"Cached extraction result: {extraction_result}\n")
            else:
                input_for_extraction = self.entity_extraction_prompt.format(sent)

                extraction_start = time.perf_counter() if self.enable_time_stats else None
                try:
                    # Use local model or GPT API based on configuration
                    if self.use_local_entity_model:
                        extraction_result_raw = self._extract_entities_local(input_for_extraction)
                    else:
                        extraction_result_raw = self.gpt_client.simple_completion(
                            prompt=input_for_extraction,
                            model=self.gpt_model,
                        )
                except Exception as e:
                    if self.enable_time_stats and extraction_start is not None:
                        self._record_time("entity_extraction", time.perf_counter() - extraction_start)
                    if self.debug:
                        logger.warning(f"Entity extraction failed for sentence {sid}: {e}")
                    extraction_result = None
                else:
                    if self.enable_time_stats and extraction_start is not None:
                        self._record_time("entity_extraction", time.perf_counter() - extraction_start)

                    if not isinstance(extraction_result_raw, str):
                        if self.debug:
                            logger.warning(f"Unexpected GPT extraction result type for sentence {sid}: {type(extraction_result_raw)}")
                        extraction_result = None
                    else:
                        extraction_result_str = extraction_result_raw.strip()

                        if self.debug:
                            logger.info(f"GPT extraction result for sentence {sid}: '{extraction_result_str}'")

                        if extraction_result_str.startswith('entities:'):
                            extraction_result_str = extraction_result_str.replace('entities:', '', 1).strip()

                        extraction_result = None

                        try:
                            extraction_result = parse_case(extraction_result_str)
                            if self.debug:
                                logger.info(f"GPT extraction result for sentence {sid} (parsed): {extraction_result}")
                                logger.info(f"Type: {type(extraction_result)}, Length: {len(extraction_result) if isinstance(extraction_result, list) else 'N/A'}")
                        except Exception as e:
                            if self.debug:
                                logger.warning(f"Failed to parse GPT extraction result for sentence {sid}: {e}")
                            extraction_result = None

                if self.enable_cache and cache_key is not None and extraction_result is not None:
                    self._entity_cache[cache_key] = deepcopy(extraction_result)
                    self._entity_cache_dirty = True
                if cache_hit is False and self.debug and self.enable_cache and extraction_result is not None:
                    logger.info(f"Sentence {sid} entity extraction is now cached.")
            

            if extraction_result is None or len(extraction_result) == 0:
                if self.debug:
                    logger.info(f"No entities extracted for sentence {sid}, adding to clear text.") 
                if len(clear_text) == 0:
                    clear_text = sent
                else:
                    clear_text += " " + sent
                continue

            for trp in extraction_result:
                if not isinstance(trp, list):
                    logger.warning(f"Unexpected entity format in sentence {sid} (not a list): {trp}")
                    continue
                
                if len(trp) == 2:
                    ent_1, relation_1 = trp
                    if not isinstance(ent_1, str) or not isinstance(relation_1, str):
                        logger.warning(f"Unexpected entity format in sentence {sid}: ent_1={ent_1} (type={type(ent_1)}), relation_1={relation_1} (type={type(relation_1)})")
                        if self.enable_cache and cache_key is not None:
                            if hasattr(self, "_entity_cache") and cache_key in self._entity_cache:
                                del self._entity_cache[cache_key]
                                self._entity_cache_dirty = True
                        continue
                    
                    count_query = ent_1
                    count_result = self._timed_infini_gram_count(count_query, index=self.infini_gram_index_name) 
                    
                    if self.debug:
                        logger.info(f"Query: '{count_query}' -> Count result: {count_result}")
                    
                    if self.binary_tuple_threshold >= 0 and count_result is not None and count_result < self.binary_tuple_threshold:
                        sentence_has_hallucination = True
                        flag_of_hallucination = True
                        question_to_retrieve.append(ent_1 + " " + relation_1)
                        if hallucinated_sentence is None:
                            hallucinated_sentence = sent
                        hallucinated_triplets.append(trp)  # Store full triplet [ent_1, relation_1]
                        # Log binary tuple hallucination detection only in debug mode
                        if self.debug:
                            logger.info(f"[Binary Tuple Hallucination] Sentence {sid}: entity '{ent_1}' has count {count_result} < {self.binary_tuple_threshold}")
                            logger.info(f"  -> Extracted tuple: {trp}")
                            logger.info(f"  -> Sentence: {sent}")
                        
                    elif count_result is None:
                        if self.debug:
                            logger.info(f"Warning: Failed to get count for query '{count_query}'")

                elif len(trp) == 3:
                    ent_1, relation_1, ent_2 = trp
                    if not isinstance(ent_1, str) or not isinstance(relation_1, str) or not isinstance(ent_2, str):
                        logger.warning(f"Unexpected entity format in sentence {sid}: ent_1={ent_1} (type={type(ent_1)}), relation_1={relation_1} (type={type(relation_1)}), ent_2={ent_2} (type={type(ent_2)})")
                        if self.enable_cache and cache_key is not None:
                            if hasattr(self, "_entity_cache") and cache_key in self._entity_cache:
                                del self._entity_cache[cache_key]
                                self._entity_cache_dirty = True
                        continue
                    
                    count_query = ent_1 + " AND " + ent_2
                    count_result = self._timed_infini_gram_count(count_query, index=self.infini_gram_index_name)
                    
                    if self.debug:
                        logger.info(f"Query: '{count_query}' -> Count result: {count_result}")
                    
                    if self.ternary_tuple_threshold >= 0 and count_result is not None and count_result < self.ternary_tuple_threshold:
                        sentence_has_hallucination = True
                        flag_of_hallucination = True
                        question_to_retrieve.append(ent_1 + " " + relation_1)
                        if hallucinated_sentence is None:
                            hallucinated_sentence = sent
                        hallucinated_triplets.append(trp)  # Store full triplet [ent_1, relation_1, ent_2]
                        if self.debug:
                            logger.info(f"[Ternary Tuple Hallucination] Sentence {sid}: entity pair '{ent_1}' AND '{ent_2}' has count {count_result} < {self.ternary_tuple_threshold}")
                            logger.info(f"  -> Extracted tuple: {trp}")
                            logger.info(f"  -> Sentence: {sent}")
                    elif count_result is None:
                        if self.debug:
                            logger.info(f"Warning: Failed to get count for query '{count_query}'")

                else:
                    logger.warning(f"Unexpected entity format in sentence {sid}: {trp}")
                    if self.enable_cache and cache_key is not None:
                        if hasattr(self, "_entity_cache") and cache_key in self._entity_cache:
                            del self._entity_cache[cache_key]
                            self._entity_cache_dirty = True
                    continue

            if sentence_has_hallucination:
                if self.debug:
                    logger.info(f"Hallucination detected in sentence {sid}. Stopping processing. Clear text contains sentences 0 - [{sid-1}]")
                break
            else:
                if len(clear_text) == 0:
                    clear_text = sent
                else:
                    clear_text += " " + sent
                if self.debug:
                    logger.info(f"Sentence {sid} added to clear text (no hallucination detected)")
            
        if self.debug:
            logger.info("="*20)
            logger.info("Hallucination detection completed.")
            logger.info(f"Flag of hallucination: {flag_of_hallucination}")
            logger.info(f"Questions to retrieve: {question_to_retrieve}")
            logger.info(f"Clear text: {clear_text}")
            logger.info(f"Hallucinated sentence: {hallucinated_sentence}")
            logger.info(f"Hallucinated triplets: {hallucinated_triplets}")
            logger.info("="*20)

        # Return a dictionary with all information needed for different query formulation strategies
        return {
            'clear_text': clear_text,
            'question_to_retrieve': question_to_retrieve,  # head+relation format
            'hallucination': flag_of_hallucination,
            'hallucinated_sentence': hallucinated_sentence,  # Original sentence
            'hallucinated_triplets': hallucinated_triplets  # Full triplets
        }
        
    # ===== QuCo-RAG specific methods =====
    def load_prompt_templates(self, template_path=None):
        resolved_path = template_path or os.path.join(os.path.dirname(__file__), "prompt_templates.json")

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"Prompt template file not found: {resolved_path}. "
                f"Please ensure the prompt_templates.json file exists."
            )

        try:
            with open(resolved_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(
                    f"Prompt template file {resolved_path} must contain a dictionary, "
                    f"but got {type(data).__name__}"
                )
            logger.info(f"Successfully loaded prompt templates from {resolved_path}")
            return data
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load prompt templates from {resolved_path}: {exc}"
            ) from exc

    def _resolve_prompt_template_key(self):
        if self.prompt_template_key and self.prompt_template_key in self.prompt_templates:
            return self.prompt_template_key

        model_name = getattr(self, 'model_name_or_path', '')
        model_lower = model_name.lower()

        for key in self.prompt_templates.keys():
            if key == "default":
                continue
            if key.lower() in model_lower:
                return key

        return "default"

    def _get_prompt_section(self, section_name):
        section = {}
        if isinstance(self.prompt_template, dict):
            section = self.prompt_template.get(section_name, {})

        if section:
            return section

        default_template = self.prompt_templates.get("default", {})
        return default_template.get(section_name, {})

    @staticmethod
    def _format_doc_line(template_str, index, doc):
        if not template_str:
            return f"{doc}\n"
        safe_doc = str(doc).replace("{", "{{").replace("}", "}}")
        try:
            return template_str.format(index=index, doc=safe_doc)
        except Exception:
            logger.debug("Failed to format doc line with template '%s'. Falling back to raw doc.", template_str)
            return f"{safe_doc}\n"
    
    def analyze_question_for_retrieval(self, question, qid=None):
        result = {
            'needs_retrieval': False,
            'entities': [],
            'query': question,
            'avg_count': None
        }
        
        # Check cache first
        cache_key = None
        if self.enable_cache:
            cache_key = self._make_question_analysis_cache_key(question, qid)
            if cache_key in self._question_analysis_cache:
                cached_result = deepcopy(self._question_analysis_cache[cache_key])
                self._question_analysis_cache_hits += 1
                if self.debug:
                    logger.info(f"Question analysis cache hit for qid={qid}")
                    logger.info(f"Cached result: {cached_result}")
                return cached_result

        if self.debug:
            logger.info(f"use_local_entity_model: {self.use_local_entity_model}")

        try:
            # Use local model or GPT API for entity extraction from question
            input_for_extraction = self.entity_extraction_prompt_for_question.format(question)
            
            extraction_start = time.perf_counter() if self.enable_time_stats else None
            try:
                # Use local model or GPT API based on configuration
                if self.use_local_entity_model:
                    extraction_result_raw = self._extract_entities_local(input_for_extraction)
                else:
                    extraction_result_raw = self.gpt_client.simple_completion(
                        prompt=input_for_extraction,
                        model=self.gpt_model,
                    )
            except Exception as e:
                if self.enable_time_stats and extraction_start is not None:
                    self._record_time("entity_extraction", time.perf_counter() - extraction_start)
                if self.debug:
                    logger.warning(f"Entity extraction failed for question analysis: {e}")
                # Don't cache on failure
                return result
            else:
                if self.enable_time_stats and extraction_start is not None:
                    self._record_time("entity_extraction", time.perf_counter() - extraction_start)
            
            extraction_result_str = extraction_result_raw.strip()
            if extraction_result_str.startswith('entities:'):
                extraction_result_str = extraction_result_str.replace('entities:', '', 1).strip()
            
            extraction_result = None
            try:
                extraction_result = parse_case(extraction_result_str)
            except Exception as e:
                if self.debug:
                    logger.warning(f"Failed to parse GPT extraction result: {e}")
                # Don't cache on parsing failure
                return result
            
            if self.debug:
                logger.info(f"GPT extraction result: {extraction_result}")
            
            if not extraction_result or len(extraction_result) == 0:
                if self.debug:
                    logger.info(f"Empty extraction result for question")
                # Don't cache empty results
                return result
            
            entities = []
            for trp in extraction_result:
                entities.append(trp[0])
                
            entities = list(dict.fromkeys(entities))
            result['entities'] = entities
            
            if not entities:
                if self.debug:
                    logger.info(f"No entities after deduplication")
                # Don't cache when no entities
                return result
            
            # Query count for each entity individually
            entity_counts = []
            for entity in entities:
                count = self._timed_infini_gram_count(entity, index=self.infini_gram_index_name)
                if count is not None:
                    entity_counts.append(count)
                    if self.debug:
                        logger.info(f"Question analysis - Entity: '{entity}', Count: {count}")
                else:
                    if self.debug:
                        logger.info(f"Question analysis - Entity: '{entity}', Count: None (failed)")
            
            # Calculate average count
            if entity_counts:
                avg_count = sum(entity_counts) / len(entity_counts)
                result['avg_count'] = avg_count
                
                if self.debug:
                    logger.info(f"Question analysis - Total entities: {len(entities)}, Valid counts: {len(entity_counts)}, Average count: {avg_count}")
                
                if avg_count < self.ngram_threshold_question:
                    result['needs_retrieval'] = True
                    if self.question_query_formulation == 'key_terms':
                        result['query'] = " ".join(entities)
                    elif self.question_query_formulation == 'direct':
                        result['query'] = question
                    else:
                        raise NotImplementedError(f"Question query formulation '{self.question_query_formulation}' is not implemented. Supported: 'direct', 'key_terms'.")
                    if self.debug:
                        logger.info(f"Question needs retrieval: avg_count={avg_count} < threshold={self.ngram_threshold_question}")
                        logger.info(f"Query: {result['query']}")
                        
            else:
                result['avg_count'] = 0
                if self.debug:
                    logger.info(f"Question analysis - No valid entity counts obtained")
            
            # Cache the successful result (only when we have entities and counts)
            if self.enable_cache and cache_key is not None:
                self._question_analysis_cache[cache_key] = deepcopy(result)
                self._question_analysis_cache_dirty = True
                if self.debug:
                    logger.info(f"Question analysis result cached for qid={qid}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in question analysis: {e}")
            # Don't cache on exception
            return result
    
    def inference(self, question, demo, case, qid=None):
        retrieval_round = 0
        prediction = ""
        new_text = ""
         
        if self.debug:
            logger.info("="*20)
            logger.info("Starting pre-generation question analysis")
            logger.info("="*20)
        
        question_analysis = self.analyze_question_for_retrieval(question, qid=qid)
        initial_docs = []
        
        if question_analysis.get('needs_retrieval'):
            retrieval_round += 1
            retrieve_query = question_analysis['query']
            
            if self.debug:
                logger.info(f"Question-level retrieval triggered")
                logger.info(f"Retrieval query: {retrieve_query}")
                logger.info(f"Extracted entities: {question_analysis.get('entities')}")
                logger.info(f"Average count: {question_analysis.get('avg_count')}")
            
            retrieved_docs = self.retrieve(retrieve_query, topk=self.retrieve_topk)
            
            if isinstance(retrieved_docs, np.ndarray):
                initial_docs = retrieved_docs.tolist()
            elif isinstance(retrieved_docs, list):
                initial_docs = retrieved_docs
            else:
                initial_docs = [retrieved_docs]
            
            initial_docs = list(dict.fromkeys(initial_docs))
            
            if self.debug:
                logger.info(f"Retrieved {len(initial_docs)} documents for question")
                for i, doc in enumerate(initial_docs):
                    logger.info(f"[{i+1}] {doc}")
            
            if qid is not None:
                self.save_retrieval_info(
                    qid=qid,
                    question=question,
                    query=retrieve_query,
                    retrieved_docs=initial_docs,
                    retrieval_round=retrieval_round,
                    current_text="",
                    hallucinated_content=None,
                    additional_info={
                        "method": "QuCo-RAG",
                        "retrieval_type": "question_level",
                        "question_analysis": question_analysis,
                    }
                )
        
        while True:
            old_len = len(prediction)

            if len(new_text) == 0:
                prompt = "Examples:\n"
                prompt += "".join([d["case"]+"\n" for d in demo])
                
                initial_template = self._get_prompt_section("initial")
                if initial_docs and len(prediction) == 0:
                    context_header = initial_template.get("context_header", "")
                    if context_header:
                        prompt += context_header

                    doc_format = initial_template.get("doc_format", "[{index}] {doc}\n")
                    for i, doc in enumerate(initial_docs):
                        prompt += self._format_doc_line(doc_format, i + 1, doc)

                    post_doc = initial_template.get("post_doc", "")
                    if post_doc:
                        prompt += post_doc
                 
                if self.use_web_search:
                    prompt += "\nPlease search the web when you think it is necessary to find the answer. Please do not regenerate the content that is already in the Answer part."

                instruction = initial_template.get("instruction", "")
                if instruction:
                    prompt += instruction
                
                tmp_li = [case, prediction]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)
                
                if self.debug:
                    logger.info("="*20)
                    logger.info("Current prompt:")
                    logger.info(prompt)
                    logger.info("\n")

                new_text = self._generate_with_timing(
                    prompt, 
                    self.generate_max_length
                )
                
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer, self.generator.last_usage)

                # Filter out invalid sentences immediately after generation
                new_text = self.filter_invalid_sentences(new_text)
                
                if self.debug:
                    logger.info("After filtering invalid sentences from new_text:")
                    logger.info(new_text)
                
                if initial_docs:
                    initial_docs = []

            else:
                if self.debug:
                    logger.info("="*20)
                    logger.info("Using remaining new_text for hallucination check:")
                    logger.info(new_text)
                    logger.info("\n")

            modifier_result = self.modifier(new_text)
            clean_text = modifier_result['clear_text']
            question_to_retrieve = modifier_result['question_to_retrieve']
            hallucination = modifier_result['hallucination']
            hallucinated_sentence = modifier_result.get('hallucinated_sentence', None)
            hallucinated_triplets = modifier_result.get('hallucinated_triplets', [])

            if not hallucination:
                prediction = prediction.strip() + " " + clean_text.strip()
                new_text = ""
            else:
                retrieval_round += 1
                
                # Determine query formulation strategy for sentence-level hallucination
                sentence_query_formulation = getattr(self, 'sentence_query_formulation', 'head_relation')
                
                if sentence_query_formulation == "head_relation":
                    # Strategy 1: head+relation (default)
                    if self.debug:
                        logger.info("Using head+relation query formulation for retrieval.")
                    retrieve_inputs = question_to_retrieve
                    
                elif sentence_query_formulation == "full_triplet":
                    # Strategy 2: full triplet
                    if self.debug:
                        logger.info("Using full triplet query formulation for retrieval.")
                    retrieve_inputs = []
                    for trp in hallucinated_triplets:
                        if len(trp) == 2:
                            # Binary tuple: [ent_1, relation_1]
                            query = f"{trp[0]} {trp[1]}"
                        elif len(trp) == 3:
                            # Ternary tuple: [ent_1, relation_1, ent_2]
                            query = f"{trp[0]} {trp[1]} {trp[2]}"
                        else:
                            continue
                        retrieve_inputs.append(query)
                    if self.debug:
                        logger.info(f"Full triplet queries: {retrieve_inputs}")
                    
                elif sentence_query_formulation == "original_sentence":
                    # Strategy 3: original sentence
                    if self.debug:
                        logger.info("Using original sentence query formulation for retrieval.")
                    if hallucinated_sentence:
                        retrieve_inputs = [hallucinated_sentence]
                    else:
                        # Fallback to head+relation if sentence not available
                        retrieve_inputs = question_to_retrieve
                        if self.debug:
                            logger.warning("Hallucinated sentence not available, falling back to head+relation")
                    
                else:
                    raise NotImplementedError(f"Sentence query formulation '{sentence_query_formulation}' is not implemented. Supported: 'head_relation', 'full_triplet', 'original_sentence'.")
                
                if self.debug:
                    logger.info("="*20)
                    logger.info(f"Retrieval round {retrieval_round}")
                    logger.info(f"Retrieve question: {retrieve_inputs}")
                    logger.info("")

                docs = []
                if self.retrieval_query_num > 0 and len(retrieve_inputs) > self.retrieval_query_num:
                    retrieve_inputs = retrieve_inputs[:self.retrieval_query_num]
                    if self.debug:
                        logger.info(f"Limiting to first {self.retrieval_query_num} retrieval queries.")
                        logger.info(f"Trimmed retrieve question: {retrieve_inputs}")
                        logger.info("")
                
                for rq in retrieve_inputs:
                    if self.debug:
                        logger.info(f"Retrieving for input: {rq}")
                    retrieved_docs = self.retrieve(rq, topk=self.retrieve_topk)

                    if isinstance(retrieved_docs, np.ndarray):
                        for doc in retrieved_docs:
                            docs.append(doc)
                    elif isinstance(retrieved_docs, list):
                        docs.extend(retrieved_docs)
                    else:
                        docs.append(retrieved_docs)

                docs = list(dict.fromkeys(docs))

                if self.debug:
                    logger.info(f"Retrieved {len(docs)} documents:")
                    for i, doc in enumerate(docs):
                        logger.info(f"[{i+1}] {doc}")
                    logger.info("\n")

                if qid is not None:
                    hallucinated_content = {
                        "generated_text": new_text,
                        "clean_text": clean_text,
                        "question_to_retrieve": question_to_retrieve,
                        "gpt_model_used": self.gpt_model,
                        "infini_gram_index": self.infini_gram_index_name,
                        "detection_method": "entity_extraction_with_infini_gram"
                    }
                    
                    self.save_retrieval_info(
                        qid=qid,
                        question=question,
                        query=retrieve_inputs,
                        retrieved_docs=docs,
                        retrieval_round=retrieval_round,
                        current_text=prediction,
                        hallucinated_content=hallucinated_content,
                        additional_info={
                            "method": "QuCo-RAG",
                            "retrieval_type": "answer_level",
                            "sentence_query_formulation": self.sentence_query_formulation,
                        }
                    )
                
                prompt = "Examples:\n"
                prompt += "".join([d["case"]+"\n" for d in demo])

                hallucination_template = self._get_prompt_section("hallucination")

                context_header = hallucination_template.get("context_header", "")
                if context_header:
                    prompt += context_header

                doc_format = hallucination_template.get("doc_format", "[{index}] {doc}\n")
                for i, doc in enumerate(docs):
                    prompt += self._format_doc_line(doc_format, i + 1, doc)

                post_doc = hallucination_template.get("post_doc", "")
                if post_doc:
                    prompt += post_doc

                # Add web search instruction if enabled
                if self.use_web_search:
                    prompt += "\nPlease search the web when you think it is necessary to find the answer."

                additional_instruction = hallucination_template.get("instruction", "")
                if additional_instruction:
                    prompt += additional_instruction
                
                tmp_li = [case, prediction, clean_text.strip()]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)

                if self.debug:
                    logger.info("Prompt for regeneration:")
                    logger.info(prompt)
                    logger.info("")
                 
                new_text = self._generate_with_timing(prompt, self.generate_max_length)
                
                if self.debug:
                    logger.info("Regenerated text:")
                    logger.info(new_text)
                    logger.info("")

                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer, self.generator.last_usage)
                    self.counter.hallucinated += 1

                # Filter out invalid sentences immediately after generation
                new_text = self.filter_invalid_sentences(new_text)
                
                if self.debug:
                    logger.info("After filtering invalid sentences from regenerated text:")
                    logger.info(new_text)

                first_sentence = self.get_top_sentence(new_text)
                tmp_text = new_text.replace(first_sentence, "", 1).strip()
                second_sentence = self.get_top_sentence(tmp_text)

                if len(second_sentence) > 0 and second_sentence[0].isalpha() and 'the answer is' in second_sentence:
                    first_sentence += " " + second_sentence
                    if self.debug:
                        logger.info("Merged second sentence into first sentence for answer completeness.")
                        logger.info(f"Second sentence: {second_sentence}")

                tmp_li = [prediction.strip(), clean_text.strip(), first_sentence.strip()]
                prediction = " ".join(s for s in tmp_li if len(s) > 0)

                new_text = new_text.replace(first_sentence, "", 1).strip()

                if self.debug:
                    logger.info("Updated full prediction:")
                    logger.info(prediction)
                    logger.info("first_sentence added:")
                    logger.info(first_sentence)
                    logger.info("Remaining new_text for next iteration:")
                    logger.info(new_text)
                    logger.info("\n")

            tokenizer = getattr(self.generator, "tokenizer", None)
            if tokenizer is not None:
                try:
                    tokens_count = len(tokenizer.encode(prediction))
                except Exception as exc:
                    logger.debug(f"Failed to compute token count using tokenizer.encode: {exc}")
                    tokens_count = len(prediction.split())
            else:
                tokens_count = len(prediction.split())
                if self.debug:
                    logger.debug("Tokenizer not found, using word count as token count approximation.")

            if tokens_count > self.generate_max_length or len(prediction) <= old_len or "the answer is" in prediction or "Question:" in prediction:
                if self.debug:
                    logger.info("Stopping criteria met.")
                    logger.info(f"Token count: {tokens_count}, Now length: {len(prediction)}, Old length: {old_len}, 'the answer is' in prediction: {'the answer is' in prediction}, 'Question:' in prediction: {'Question:' in prediction}")
                    logger.info("Final generated prediction:")
                    logger.info(prediction)
                    logger.info("="*20)        
                break
            
            eos_token = None
            if tokenizer is not None:
                eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token and eos_token in prediction:
                if self.debug:
                    logger.info("EOS token detected in prediction, stopping generation.")
                    logger.info("Final generated prediction:")
                    logger.info(prediction)
                    logger.info("="*20)        
                break

        # Return the final prediction
        return prediction
