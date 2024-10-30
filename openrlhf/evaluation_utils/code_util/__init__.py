from .utils import check_correctness as apps_check_correctness
import json
import re
from tqdm import tqdm
import os

from typing import Mapping
import re
import signal
from contextlib import contextmanager
from typing import Any
import subprocess
from tqdm import tqdm
from ..math_util import math_equal
import uuid

import os


class PythonREPL():
    def __init__(self, timeout=5, tmp_file="cache/tmp"):
        self.timeout = timeout
        self.tmp_file = tmp_file
        os.makedirs("/".join(self.tmp_file.split("/")[:-1]), exist_ok=True)
        os.system(f"touch {self.tmp_file}.py")

    @contextmanager
    def time_limit(self, seconds):
        def signal_handler(signum, frame):
            raise TimeoutError(f"Timed out after {seconds} seconds.")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)  # Disable the alarm

    def __call__(self, query: str) -> tuple:
        query = query.strip().split("\n")
        if "print(" not in query[-1]:
            query[-1] = "print(" + query[-1] + ")"
        query = "\n".join(query)

        with open(f'{self.tmp_file}.py', "w") as f:
            f.write(query)

        with self.time_limit(self.timeout):
            result = subprocess.run(
                ['python3', f'{self.tmp_file}.py'], capture_output=True, check=False, text=True, timeout=self.timeout)

            if result.returncode == 0:
                output = result.stdout
                return True, output.strip()
            else:
                error_msg = result.stderr.strip()
                msgs = error_msg.split("\n")
                new_msgs = []
                want_next = False
                for m in msgs:
                    if "Traceback" in m:
                        new_msgs.append(m)
                    elif m == msgs[-1]:
                        new_msgs.append(m)
                    elif self.tmp_file in m:
                        st = m.index('"/') + 1 if '"/' in m else 0
                        ed = m.index(f'/{self.tmp_file}.py') + 1 if f'/{self.tmp_file}.py' in m else None
                        clr = m[st:ed] if not ed else m[st:]
                        m = m.replace(clr, "")
                        new_msgs.append(m)
                        want_next = True
                    elif want_next:
                        new_msgs.append(m)
                        want_next = False
                error_msg = "\n".join(new_msgs)
                return False, error_msg.strip()

def postprocess_completion(executor, completion):
    executions = ["!" + code for code in re.findall(r"```bash(.*?)```", completion, re.DOTALL) if "!" not in code]
    executions.extend(re.findall(r"```python(.*?)```", completion, re.DOTALL))

    if len(executions) == 0:  # directly return cot result
        return completion
    else:
        ### Python
        execution_outputs = []
        for code in executions:
            try:
                success, output = executor(code)
            except TimeoutError:
                print("time out")
                # success = False
                output = ""
            else:
                output = output if success else ""
            execution_outputs.append(output)
        extracted_outputs = execution_outputs

        for index in range(1, len(extracted_outputs) + 1):
            extracted_solution = str(extracted_outputs[-index]).strip()
            break

        return extracted_solution


def postprocess_completions(completion_list):
    executor = PythonREPL()

    solution_list = []
    for completion in completion_list:
        solution_list.append(postprocess_completion(executor, completion))

    del executor

    return solution_list

def evaluate_repl(solution, gt):
    executor = PythonREPL(tmp_file='cache/tmp-'+uuid.uuid5(uuid.NAMESPACE_URL, solution).__str__())
    result = postprocess_completion(executor, solution)
    try:
        return (math_equal(result, gt), result)
    except Exception as e:
        return (False, result)



def evaluate_code(solution, test_cases):
    try:
        try:
            if not isinstance(test_cases, dict):
                test_cases = json.loads(test_cases)
        except Exception as e:
            print(f"Error:{e}")
        

        # 先检查正确性，如果正确，则再one by one 检查test case
        try:
            res, metadata = apps_check_correctness(
                in_outs=test_cases,
                generation=solution,
                timeout=10,
                # debug=False
                debug=False
                )
            metadata = dict(enumerate(metadata))[0]
            success = all(map(lambda x: x == True, res))
            if success:
                return success, metadata
        except Exception as e:
            pass

        test_cases_list = []
        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]
        for i in range(len(inputs)):
            test_cases_list.append({
                "inputs": [inputs[i]],
                "outputs": [outputs[i]]
            })

        metadata_list = []
        res_list = []
        for test_case in test_cases_list:
            res, metadata = apps_check_correctness(
                in_outs=test_case,
                generation=solution,
                timeout=10,
                # debug=False
                debug=False
            )
            metadata = dict(enumerate(metadata))[0]
            metadata["test_case"] = {}
            metadata["test_case"]["input"] = str(test_case["inputs"][0])
            metadata["test_case"]["output"] = str(test_case["outputs"][0])
            metadata["test_case"]["res"] = str(res)
            metadata_list.append(metadata)
            res_list.extend(res)
            tmp_success = all(map(lambda x: x == True, res))
            if not tmp_success:
                return tmp_success, metadata_list
            # print("*" * 50)
            # print("res:", res)
            # print("metadata:", metadata.keys())
            # print("metadata:", metadata)
            # print("*" * 50)
        success = all(map(lambda x: x == True, res_list))
    except Exception as e:
        print(e)
        success = False
        metadata_list = None
    return success, metadata_list
    





# from .utils import check_correctness as apps_check_correctness
# import json
# import re
# from tqdm import tqdm
# import os

# def evaluate_code(solution, test_cases):
#     try:
#         try:
#             if not isinstance(test_cases, dict):
#                 test_cases = json.loads(test_cases)
#         except Exception as e:
#             print(f"Error:{e}")
#         res, metadata = apps_check_correctness(
#             in_outs=test_cases,
#             generation=solution,
#             timeout=10,
#             # debug=False
#             debug=False
#         )
#         metadata = dict(enumerate(metadata))[0]
#         print("*" * 50)
#         print("res:", res)
#         print("metadata:", metadata.keys())
#         print("metadata:", metadata)
#         print("*" * 50)
#         success = all(map(lambda x: x == True, res))
#     except Exception as e:
#         print(e)
#         success = False
#         metadata = None
#     return success, metadata
    
if __name__ == "__main__":
    code = """
Step 1: First, let's calculate the total number of eggs laid by Janet's ducks in a day.
Step 2: Next, let's calculate the number of eggs Janet eats for breakfast each day.
Step 3: Then, let's calculate the number of eggs Janet bakes for her friends each day.
Step 4: Finally, let's calculate the number of eggs Janet sells at the farmers' market each day.
Step 5: To find the total amount of money Janet makes each day at the farmers' market, we can multiply the number of eggs she sells by the price per egg.
```python
# Step 6: Calculate the total number of eggs laid by Janet's ducks in a day.
total_eggs_per_day = 16
# Step 7: Calculate the number of eggs Janet eats for breakfast each day.
eggs_eaten_per_day = 3
# Step 8: Calculate the number of eggs Janet bakes for her friends each day.
eggs_baked_per_day = 4
# Step 9: Calculate the number of eggs Janet sells at the farmers' market each day.
eggs_sold_per_day = total_eggs_per_day - eggs_eaten_per_day - eggs_baked_per_day
# Step 10: Calculate the total amount of money Janet makes each day at the farmers' market.
price_per_egg = 2
total_money_per_day = eggs_sold_per_day * price_per_egg
total_money_per_day
```
Answer:
12

"""
    executor = PythonREPL()
    result = postprocess_completion(executor, code)
    print(result)