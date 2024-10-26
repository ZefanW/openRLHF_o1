import time
import ray
import requests
import torch

# from openrlhf.utils.logging_utils import init_logger

# logger = init_logger(__name__)


def request_api_wrapper(url, data, score_key="rewards", try_max_times=10):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            assert score_key in response, f"{score_key} not in {response}"
            return response.get(score_key)
        except requests.RequestException as e:
            print(f"Request error, please check: {e}")
            #logger.info(f"Request error, please check: {e}")
        except Exception as e:
            print(f"Unexpected error, please check: {e}")
            #logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


def remote_rm_fn(api_url, queries, prompts, score_key="rewards"):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    queries: query+response with the template
    design is made optional.
    score_key: RM score key
    """
    scores = request_api_wrapper(api_url, prompts, score_key)
    #scores = request_api_wrapper(api_url, {"query": queries, "prompts": prompts}, score_key)
    return torch.tensor(scores)


@ray.remote
def remote_rm_fn_ray(api_url, queries, prompts, score_key="rewards"):
    return remote_rm_fn(api_url, queries, prompts, score_key)


if __name__ == "__main__":
    # test utils
    url = "http://localhost:5000/get_reward"
    sol = '''
def can_glue(block1, block2):
    faces1 = [(block1[i], block1[j]) for i in range(3) for j in range(i+1, 3)]
    faces2 = [(block2[i], block2[j]) for i in range(3) for j in range(i+1, 3)]
    
    for f1 in faces1:
        for f2 in faces2:
            if (f1[0] == f2[0] and f1[1] == f2[1]) or (f1[0] == f2[1] and f1[1] == f2[0]):
                return True
    return False

def get_combined_block(block1, block2):
    max_radius = 0
    best_dims = None
    
    faces1 = [(block1[i], block1[j], block1[3-i-j]) for i in range(3) for j in range(i+1, 3)]
    faces2 = [(block2[i], block2[j], block2[3-i-j]) for i in range(3) for j in range(i+1, 3)]
    
    for f1 in faces1:
        for f2 in faces2:
            if (f1[0] == f2[0] and f1[1] == f2[1]) or (f1[0] == f2[1] and f1[1] == f2[0]):
                dims = [f1[0], f1[1], f1[2] + f2[2]]
                radius = min(dims) / 2
                if radius > max_radius:
                    max_radius = radius
                    best_dims = dims
    
    return best_dims, max_radius

def solve(n, blocks):
    max_radius = 0
    result = []
    
    for i in range(n):
        radius = min(blocks[i]) / 2
        if radius > max_radius:
            max_radius = radius
            result = [i + 1]
    
    for i in range(n):
        for j in range(i + 1, n):
            if can_glue(blocks[i], blocks[j]):
                _, radius = get_combined_block(blocks[i], blocks[j])
                if radius > max_radius:
                    max_radius = radius
                    result = [i + 1, j + 1]
    
    return len(result), result

def main():
    n = int(input())
    blocks = []
    for _ in range(n):
        a, b, c = map(int, input().split())
        blocks.append([a, b, c])
    
    k, result = solve(n, blocks)
    print(k)
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()
'''
    prompts = [{"prompt": "There were sweets on the table. Jack came and took half of all the candies and 4 more candies. Then Paul came and took the remaining 7 sweets. How many sweets were on the table at first?", "reference": "22", "task": "math", "completions": ["22"]}, {"prompt": """Problem description. Vipul is a hardworking super-hero who maintains the bracket ratio of all the strings in the world. Recently he indulged himself in saving the string population so much that he lost his ability for checking brackets (luckily, not permanently ).Being his super-hero friend help him in his time of hardship. Input The first line of the input contains an integer T denoting the number of test cases. The description of T test cases follows. The first line of each test case contains a single string S denoting the string to be checked. Output For each test case, output a single line printing "YES" or "NO" (without " " and in uppercase only) , denoting if the brackets in the given string is balanced or not . Constraints 1 ≤ T ≤ 10 1 ≤ length of S ≤ 60 Example Input: 3 ((())) (())() ()(() Output: YES YES NO   Explanation Example is self-explanatory.""", "reference": """{ "inputs": [ "3\n((()))\n(())()\n()(()" ], "outputs": [ "YES\nYES\nNO" ] }""", "task": "code", "completions": ["```\nfor _ in range(input()):\n try:\n eval(raw_input())\n print 'YES'\n except TypeError:\n print 'YES'\n except:\n print 'NO'\n```"]}]
    queries = ["There were sweets on the table. Jack came and took half of all the candies and 4 more candies. Then Paul came and took the remaining 7 sweets. How many sweets were on the table at first?"]

    score = remote_rm_fn(url, queries, prompts)
    print(score)
