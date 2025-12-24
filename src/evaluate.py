import os
import sys
import json
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from data import WikiMultiHopQA, HotpotQA
from generate_quco import BasicGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    tmp = parser.parse_args()
    with open(os.path.join(tmp.dir, "config.json"), "r") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.output_dir = tmp.dir
    return args


def regenerate_answer(cot, generator, case, demo):
    if generator is None:
        logger.warning("Generator is None, skipping answer regeneration.")
        return cot
    # print("##### origin #####")
    # print(cot)
    split_words = ["Question:", "#10000000", "Note:"]
    # split_words = ["Question:", "#10000000", "\n"]
    for word in split_words:
        pos = cot.find(word)
        if pos != -1 and pos > 0:
            cot = cot[:pos]
    if "answer is" in cot:
        return cot 

    cot += " So the answer is "
    prompt = "".join([d["case"]+"\n" for d in demo])
    prompt += case + " " + cot
    
    text = generator.generate(prompt, max_length=128)
    
    text = cot + text.strip()
    for word in split_words:
        pos = text.find(word)
        if pos != -1:
            text = text[:pos] 
    # print("##### prompt #####")
    # print(prompt)
    # print("##### output #####")
    # print(text)
    # print("##### pred #####")
    return text


def main():
    args = get_args()
    # logger.info(f"{args}")
    print("\n\n\n***** Evaluate *****", file=sys.stderr)
    print(f"model_name_or_path: {args.model_name_or_path}", file=sys.stderr)
    print(f"output_dir: {args.output_dir}", file=sys.stderr)
    print(f"args:{args}", file=sys.stderr)

    logger.info(f"{args}")

    if args.dataset == '2wikimultihopqa':
        data = WikiMultiHopQA(args.data_path)
    elif args.dataset == 'hotpotqa':
        data = HotpotQA(args.data_path)
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not supported. Supported: '2wikimultihopqa', 'hotpotqa'")
    data.format(fewshot=args.fewshot)

    dataset = {}
    for i in range(len(data.dataset)):
        t = data.dataset[i]
        ground_truth = t["answer"]
        answer_id = t.get("answer_id")
        dataset[t["qid"]] = [
            ground_truth, 
            answer_id,
            t["case"] if "case" in t else None
        ]

    output_file = os.path.join(args.output_dir, "output.txt")
    if not os.path.exists(output_file):
        logger.error(f"Output file not found: {output_file}")
        print(f"Error: Output file not found: {output_file}", file=sys.stderr)
        sys.exit(1)
    
    with open(output_file, "r") as fin:
        lines = fin.readlines()
    
    if len(lines) == 0:
        logger.error(f"Output file is empty: {output_file}")
        print(f"Error: Output file is empty: {output_file}", file=sys.stderr)
        print(f"Please ensure the inference task completed successfully.", file=sys.stderr)
        sys.exit(1)
    
    logger.info(f"Found {len(lines)} predictions to evaluate")
    
    # Use EM/F1 for evaluation
    metrics = ["EM", "F1", "Precision", "Recall"]
    if "use_counter" not in args or args.use_counter:
        count_list = ["retrieve_count", "generate_count", "hallucinated_count", "token_count", "sentence_count"]
        metrics += count_list
        
        # Add API token usage metrics for API-based models or when web search is enabled
        model_name = args.model_name_or_path.lower() if hasattr(args, 'model_name_or_path') else ""
        use_web_search = getattr(args, 'use_web_search', False)
        if "gpt-" in model_name or use_web_search:
            api_metrics = ["api_input_tokens", "api_output_tokens", "api_total_tokens"]
            metrics += api_metrics
            count_list += api_metrics
            
            # Only add web_search_calls metric when web search is actually enabled
            if use_web_search:
                metrics.append("web_search_calls")
                count_list.append("web_search_calls")
    
    value = [[] for _ in range(len(metrics))]
    
    need_generate = args.dataset in ['2wikimultihopqa', "hotpotqa"] 
    generator = None
    if need_generate:
        logger.info("Answer regeneration is enabled for this dataset. Initializing generator...")
        # Initialize BasicGenerator, which will handle both local and API models based on config
        generator = BasicGenerator(
            model_name_or_path=args.model_name_or_path,
            device_map="auto",
            use_api=getattr(args, "use_llm_api", False),
            api_base_url=getattr(args, "api_base_url", None),
            tokenizer_name_or_path=getattr(args, "tokenizer_name_or_path", None)
        )
        demo = data.dataset[0]["demo"]

    pred_out = open(f"{args.output_dir}/details.txt", "w")
    
    for line in tqdm(lines):
        rd = json.loads(line)
        qid = rd["qid"]
        pred = rd["prediction"]
        ground_truth, ground_truth_id, case = dataset[qid]
        if need_generate:
            pred = regenerate_answer(pred, generator, case, demo) 
        pred = data.get_real_prediction(pred)

        em_ret = data.exact_match_score(
            pred, 
            ground_truth, 
            ground_truth_id
        )
        f1_ret = data.f1_score(
            pred, 
            ground_truth, 
            ground_truth_id
        )
        value[0].append(em_ret["correct"])
        for i, k in enumerate(f1_ret.keys()):
            value[i+1].append(f1_ret[k])
        if "use_counter" not in args or args.use_counter:
            for i, k in enumerate(count_list):
                # Use 0 as default if the key doesn't exist in rd (for backwards compatibility)
                value[i+4].append(rd.get(k, 0))
        detail = {
            "qid": qid, 
            "final_pred": pred,
            "EM": str(em_ret["correct"]), 
            "F1": str(f1_ret["f1"]) 
        }
        pred_out.write(json.dumps(detail)+"\n")

    ret = []
    for i, metric in enumerate(metrics):
        val = np.array(value[i])
        if len(val) == 0:
            logger.warning(f"No values for metric {metric}, using 0.0")
            mean_str = "0.0000"
        else:
            mean_val = val.mean()
            try:
                mean_str = f"{float(mean_val):.4f}"
            except Exception:
                mean_str = str(mean_val)
        
        # Use more descriptive names for API token metrics
        display_metric = metric
        if metric == "api_input_tokens":
            display_metric = "#API_Input"
        elif metric == "api_output_tokens":
            display_metric = "#API_Output"
        elif metric == "api_total_tokens":
            display_metric = "#API_Total"
        elif metric == "web_search_calls":
            display_metric = "#WebSearch"
        
        ret.append([display_metric, mean_str])
    df = pd.DataFrame(ret)
    df.to_csv(f"{args.output_dir}/result.tsv", index=False, header=False)

    pred_out.close()
    
    # Output evaluation results and configuration information
    print("\n" + "="*80, file=sys.stderr)
    print("***** Evaluation Results (Final Prediction) *****", file=sys.stderr)
    print("="*80, file=sys.stderr)
    for metric, score in ret:
        print(f"{metric}: {score}", file=sys.stderr)

    print("\n" + "="*60, file=sys.stderr)
    print("***** Configuration *****", file=sys.stderr)
    print("="*60, file=sys.stderr)
    
    # Automatically output all parameters from config file
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}", file=sys.stderr)
    
    print("="*60, file=sys.stderr)
    print("***** Evaluate End *****\n\n", file=sys.stderr)


if __name__ == "__main__":
    main()