import asyncio
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm.asyncio import tqdm
from .code_util import evaluate_code
from .math_util import evaluate_math
# from code_util import evaluate_code
# from math_util import evaluate_math

def process_completion(completion, task, reference):
    if task == "code":
        return evaluate_code(completion, reference)
    elif task == "math":
        return evaluate_math(completion, str(reference))
    else:
        raise NotImplementedError

async def process_row(row, executor):
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            executor,
            partial(process_completion, completion, row["task"], reference=row["reference"])
        )
        for completion in row["completions"]
    ]
    return await asyncio.gather(*tasks)

async def parallel_evaluate_async(df, num_processes):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = [
            process_row(row, executor)
            for _, row in df.iterrows()
        ]
        results = await tqdm.gather(*tasks, desc="Processing rows")
    return results

def parallel_eval(df, num_processes):

    from copy import deepcopy
    df_for_eval = deepcopy(df).to_dict("records")
    flattened_df = []
    index_steps_to_inst = []
    index_steps_to_comp = []
    step_nums = []
    rollout_nums = []
    for inst_idx, d in enumerate(df_for_eval):
        for comp_idx, completion_steps in enumerate(d["steps"][:len(d["rollouts"].values())]):
            for step_num in range(1, len(d["steps"][comp_idx])):
                partial_response = "".join(completion_steps[:step_num])
                flattened_df.append({
                    "task": d["task"],
                    "reference": d["reference"],
                    "completions": [partial_response + rollout for rollout in d["rollouts"][f"completion {comp_idx}"][f"continuation {step_num}"]] # coding requires prefixes to run
                })
                index_steps_to_inst.append(inst_idx)
                index_steps_to_comp.append(comp_idx)
                step_nums.append(step_num)

    flattened_df = pd.DataFrame(flattened_df)
    flattened_results = asyncio.run(parallel_evaluate_async(flattened_df, num_processes))

    results = []
    acc = []
    next_step_num = -1
    for idx, step_results in enumerate(flattened_results):
        if len(results) < index_steps_to_inst[idx] + 1:
            results.append({})
            acc.append({})
            assert len(results) == index_steps_to_inst[idx] + 1

        if f"completion {index_steps_to_comp[idx]}" not in results[index_steps_to_inst[idx]].keys():
            results[index_steps_to_inst[idx]][f"completion {index_steps_to_comp[idx]}"] = {}
            acc[index_steps_to_inst[idx]][f"completion {index_steps_to_comp[idx]}"] = {}

        results[index_steps_to_inst[idx]][f"completion {index_steps_to_comp[idx]}"][f"continuation {step_nums[idx]}"] = step_results
        
        acc[index_steps_to_inst[idx]][f"completion {index_steps_to_comp[idx]}"][f"continuation {step_nums[idx]}"] = sum([r[0] for r in step_results])/len(step_results)

    assert len(results) == len(df)
    df["rollout_correctness"] = results
    df["rollout_acc"] = acc
    return df


def parallel_eval_one_step(df, num_processes, step_num, num_resp_per_inst=8):

    flattened_df = []
    index_steps_to_inst = []
    index_steps_to_comp = []
    rollout_nums = []
    for inst_idx, d in enumerate(df):
        # print(d["steps"])
        for comp_idx, completion_steps in enumerate(d["steps"][:num_resp_per_inst]):
            partial_response = "".join(completion_steps[:step_num])
            
            try:
                if f"completion {comp_idx}" not in d["rollouts"].keys():
                    # d["rollouts"][f"completion {comp_idx}"] = {}
                    continue
            except:
                print(d["rollouts"])
                continue

            if f"continuation {step_num}" in d["rollouts"][f"completion {comp_idx}"].keys():
                flattened_df.append({
                    "task": d["task"],
                    "reference": d["reference"],
                    "completions": [partial_response + rollout for rollout in d["rollouts"][f"completion {comp_idx}"][f"continuation {step_num}"]] # coding requires prefixes to run
                })
                index_steps_to_inst.append(inst_idx)
                index_steps_to_comp.append(comp_idx)

    flattened_df = pd.DataFrame(flattened_df)
    # flattened_results = asyncio.run(parallel_evaluate_async(flattened_df, num_processes))
    flattened_results = flattened_df.apply(lambda row: [process_completion(completion, row["task"], row["reference"]) 
                                    for completion in row["completions"]], axis=1)
    # print(type(df))
    
    for idx, step_results in enumerate(flattened_results):
        d = df[index_steps_to_inst[idx]]
        if "rollout_correctness" not in d.keys():
            d["rollout_correctness"] = {}
            d["rollout_acc"] = {}

        if f"completion {index_steps_to_comp[idx]}" not in d["rollout_correctness"].keys():
            d["rollout_correctness"][f"completion {index_steps_to_comp[idx]}"] = {}
            d["rollout_acc"][f"completion {index_steps_to_comp[idx]}"] = {}

        d["rollout_correctness"][f"completion {index_steps_to_comp[idx]}"][f"continuation {step_num}"] = step_results
        d["rollout_acc"][f"completion {index_steps_to_comp[idx]}"][f"continuation {step_num}"] = sum([r[0] for r in step_results])/len(step_results)

    return df


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_json("/home/test/test05/ylf/prm/rollout_data_debug/onpolicy-inst/code/0-1000.json")
    df["task"] = ["code"] * len(df)
    df = parallel_eval(df, 20)
    print(df["acc"])