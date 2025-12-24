import os
# Disable HuggingFace tokenizer parallelism to avoid deadlock warnings after fork
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import json
import argparse
from tqdm import tqdm
from copy import copy
import logging
from data import WikiMultiHopQA, HotpotQA
from generate_quco import BasicRAG, QuCo_RAG


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path, "r") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.config_path = config_path
    if "shuffle" not in args:
        args.shuffle = False 
    if "use_counter" not in args:
        args.use_counter = True
    
    if "debug" not in args:
        args.debug = False
    
    # Resume functionality
    if "resume" not in args:
        args.resume = False
    if "resume_output_path" not in args:
        args.resume_output_path = None
        
    return args


def count_completed_samples(output_path):
    """Count how many samples have been completed in a previous run."""
    output_file_path = os.path.join(output_path, "output.txt")
    if not os.path.exists(output_file_path):
        logger.warning(f"Output file not found: {output_file_path}")
        return 0
    
    completed = 0
    try:
        with open(output_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # Count non-empty lines
                    completed += 1
        logger.info(f"Found {completed} completed samples in {output_file_path}")
    except Exception as e:
        logger.error(f"Error reading output file: {e}")
        return 0
    
    return completed


def get_completed_qids(output_path):
    """Get the set of qids that have been completed."""
    output_file_path = os.path.join(output_path, "output.txt")
    completed_qids = set()
    
    if not os.path.exists(output_file_path):
        return completed_qids
    
    try:
        with open(output_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if "qid" in data:
                            completed_qids.add(data["qid"])
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Found {len(completed_qids)} unique completed qids")
    except Exception as e:
        logger.error(f"Error reading completed qids: {e}")
    
    return completed_qids


def main():
    args = get_args()
    logger.info(f"{args}")

    # Resume mode
    resume_mode = args.resume and args.resume_output_path
    completed_qids = set()
    start_index = 0
    
    if resume_mode:
        if not os.path.exists(args.resume_output_path):
            logger.error(f"Resume output path does not exist: {args.resume_output_path}")
            raise ValueError(f"Resume output path not found: {args.resume_output_path}")
        
        logger.info("="*20+" RESUME MODE "+"="*20)
        logger.info(f"Resuming from: {args.resume_output_path}")
        
        # Count completed samples and get completed qids
        completed_count = count_completed_samples(args.resume_output_path)
        completed_qids = get_completed_qids(args.resume_output_path)
        start_index = completed_count
        
        logger.info(f"Completed samples: {completed_count}")
        logger.info(f"Will skip first {start_index} samples or samples with completed qids")
        
        # Use the same output directory
        args.output_dir = args.resume_output_path
        logger.info(f"Using existing output directory: {args.output_dir}")
    else:
        # output dir - original logic
        if os.path.exists(args.output_dir) is False:
            os.makedirs(args.output_dir, exist_ok=True)
        dir_name = os.listdir(args.output_dir)
        
        # print("dir_name:", dir_name)

        for i in range(10000):
            if str(i) not in dir_name:
                temp_output_dir = os.path.join(args.output_dir, str(i))
                try:
                    os.makedirs(temp_output_dir)
                    args.output_dir = temp_output_dir
                    break
                except FileExistsError:
                    continue

    logger.info("="*20+" config "+"*"*20)
    logger.info(f"config path: {args.config_path}")
    logger.info("="*20+" output dir "+"*"*20)
    logger.info(f"output dir: {args.output_dir}")

    # save config
    config_save_path = os.path.join(args.output_dir, "config.json")
    if not resume_mode or not os.path.exists(config_save_path):
        with open(config_save_path, "w") as f:
            json.dump(args.__dict__, f, indent=4)
    
    # create output file (append mode for resume)
    output_file_mode = "a" if resume_mode else "w"
    output_file = open(os.path.join(args.output_dir, "output.txt"), output_file_mode)

    debug_flag = True if args.debug else False
    logger.info(f"debug_flag: {debug_flag}")

    # load data
    if args.dataset == "2wikimultihopqa":
        data_split = getattr(args, 'data_split', 'dev')
        data = WikiMultiHopQA(args.data_path, data_split=data_split)
    elif args.dataset == "hotpotqa":
        data = HotpotQA(args.data_path)
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not supported. Supported: '2wikimultihopqa', 'hotpotqa'")

    data.format(fewshot=args.fewshot)
    data = data.dataset
    if args.shuffle:
        data = data.shuffle()
    if args.sample != -1:
        samples = min(len(data), args.sample)
        # if debug_flag:
        #     # select last samples for debug
        #     logger.info(f"sample {samples} data points from the end for debug")
        #     data = data.select(range(len(data)-samples, len(data)))
        # else:
        logger.info(f"sample {samples} data points from the beginning")
        data = data.select(range(samples))

    logger.info(f"data size: {len(data)}")


    # Initialize model based on method
    if args.method == "non-retrieval":
        model = BasicRAG(args)
    elif args.method == "QuCo-RAG":
        model = QuCo_RAG(args)
    else:
        raise NotImplementedError(
            f"method={args.method} is not supported in main_quco.py. Supported: 'non-retrieval', 'QuCo-RAG'. Use main_baseline.py for other methods."
        )
    
    # For retrieval-based models, pass qid to save retrieved documents
    logger.info(f"model type: {type(model)}")

    logger.info("start inference")

    processed_count = 0
    skipped_count = 0
    
    for i in tqdm(range(len(data))):
        last_counter = copy(model.counter)
        batch = data[i]
        
        # Skip already completed samples in resume mode
        if resume_mode:
            if batch["qid"] in completed_qids:
                skipped_count += 1
                if skipped_count <= 10:  # Log first 10 skips
                    logger.info(f"Skipping already completed qid: {batch['qid']}")
                continue
        
        if debug_flag:
            logger.info("="*20)
            # logger.info(f"demo:\n{batch['demo']}")
            logger.info(f"case:\n{batch['case']}")
            logger.info('-'*20)
            # logger.info(f"question:\n{batch['question']}")
            # logger.info('-'*20)


        if hasattr(model, 'inference') and 'qid' in batch:
            # Check if the model is a RAG model that requires retrieval (except BasicRAG's non-retrieval method)
            if i == 0:
                logger.info("hasattr(model, 'inference') and 'qid' in batch")
            
            if type(model) is BasicRAG:
                if i == 0:
                    logger.info("type(model) is BasicRAG == True")
                result = model.inference(batch["question"], batch["demo"], batch["case"])
            else:
                if i == 0:
                    logger.info("type(model) is BasicRAG == False")
                result = model.inference(batch["question"], batch["demo"], batch["case"], qid=batch["qid"])
        else:
            if i == 0:
                logger.info("not hasattr(model, 'inference') or 'qid' not in batch")
            result = model.inference(batch["question"], batch["demo"], batch["case"])
        
        # inference returns a single prediction string
        pred = result.strip()

        if debug_flag:
            logger.info('-'*20)
            logger.info(f"prediction: {pred}")
            logger.info("="*20)
            logger.info("")


        ret = {
            "qid": batch["qid"], 
            "prediction": pred,
        }
        
        if args.use_counter:
            ret.update(model.counter.calc(last_counter))
        
        output_file.write(json.dumps(ret)+"\n")
        output_file.flush()
        
        processed_count += 1
        
        # Save cache every 1200 samples
        if getattr(model, 'enable_cache', False) and hasattr(model, 'save_cache'):
            if processed_count > 0 and processed_count % 1200 == 0:
                logger.info(f"Saving cache at {processed_count} processed samples...")
                model.save_cache()

        if processed_count >= 20:
            # close all debug info
            if debug_flag:
                debug_flag = False
            
            if model.debug:
                model.debug = False
                if hasattr(model, 'generator') and hasattr(model.generator, 'debug'):
                    model.generator.debug = False
                logger.info("debugged 20 cases, turn off debug info for the rest cases")


    output_file.close()
    
    if resume_mode:
        logger.info(f"Resume completed! Skipped: {skipped_count}, Processed: {processed_count}")
    
    logger.info("inference done!")

    # Save additional statistics if available
    if hasattr(model, 'save_token_stats'):
        model.save_token_stats()
    if getattr(model, 'enable_time_stats', False) and hasattr(model, 'save_time_stats'):
        model.save_time_stats()
    if getattr(model, 'enable_cache', False) and hasattr(model, 'save_cache'):
        logger.info("Saving cache...")
        model.save_cache()
    

if __name__ == "__main__":
    main()