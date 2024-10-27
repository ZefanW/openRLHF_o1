from .utils import check_correctness as apps_check_correctness
import json
import re
from tqdm import tqdm
import os

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
    
