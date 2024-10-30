# 将已有可做evaluation的数据合并
# 需要先尝试一遍校验，chosen结果校验通过的才允许加入结果
# 由于评价gt的时间开销问题，这里应该尽量让正式预测时，rm和gt并行工作。

import msgspec
import json
import glob
import os
from openrlhf.evaluation_utils.parallel_evaluation import process_completion,parallel_eval
from tqdm import tqdm
import pandas as pd

if __name__=='__main__':
    input_path='/home/wangzefan/data/OpenRLHF/datasets/UltraInteract_0911'
    files=glob.glob(input_path+'/*.json')
    combined_file_name='have_reference.jsonl'
    with open(os.path.join(input_path,combined_file_name),'wb') as f_write:

        for filename in files:
            if 'logic-pair' in filename:
                continue
            # if 'math-withtool' not in filename:
            #     continue
            task_name_set=set()
            with open(filename,'r') as f:
                cur_file=json.load(f)
                row_dicts = []
                for id,data_entry in tqdm(enumerate(cur_file)):
                    reference=None
                    task_name_set.add(data_entry['task'])

                    # 只保留有用的列，防止加载数据集出各种问题
                    new_data_entry={}
                    new_data_entry['trajectory'] = data_entry['trajectory']
                    new_data_entry['dataset']= data_entry['dataset']
                    new_data_entry['task']= data_entry['task']
                    new_data_entry['chosen']=data_entry['chosen']
                    new_data_entry['rejected']=data_entry['rejected']
                    new_data_entry['reference']=data_entry['reference'] if 'reference' in data_entry else data_entry['gt']

                    # 一般load dataset要求每个column的数据类型一致
                    if type(new_data_entry['reference']) is not str:
                        new_data_entry['reference']=json.dumps(new_data_entry['reference'],ensure_ascii=False)

                    f_write.write(msgspec.json.encode(new_data_entry) + b'\n')
                    # test the first 10 samples to check if our evaluations are correctly implemented

                    if id<64:
                        if 'code-pair' in filename:
                            reference=data_entry['reference']
                            code_response=data_entry['chosen'].split('```python')[-1].split('```')[0]
                            # res=process_completion(code_response,'code',reference)
                            # print(res[0])

                            row_dicts.append({
                                'task':'code',
                                'completions':code_response,
                                'reference':json.dumps(reference,ensure_ascii=False)
                            })
                        elif 'math-notool' in filename:
                            reference=data_entry['gt']
                            # res=process_completion(data_entry['chosen'],'math',reference)
                            # print(res[0])

                            row_dicts.append({
                                'task':'math',
                                'completions':data_entry['chosen'],
                                'reference':str(reference)
                            })

                        elif 'math-withtool' in filename: # 不需要提取字符串
                            reference=data_entry['gt']
                            # code_response = data_entry['chosen'].split('```python')[-1].split('```')[0]
                            code_response=data_entry['chosen']
                            # res=process_completion(code_response,'repl',reference)
                            # print(res[0])

                            row_dicts.append({
                                'task':'repl',
                                'completions':data_entry['chosen'],
                                'reference':str(reference)
                            })
                        # 测试parallel evaluation
                        # if id==8:
                        #     result=parallel_eval(pd.DataFrame(row_dicts),4)
                        #     print(result)
                            # 比单进程还慢...并且code部分跑不通。

            print(filename,task_name_set)

