import ray
from ray import job_submission

# 初始化 Ray JobSubmissionClient，连接到 Ray 的 HTTP 地址
client = job_submission.JobSubmissionClient("http://localhost:8265")  # 请确保你的Ray Dashboard运行在该端口

# 获取所有作业信息
jobs = client.get_all_jobs()

# 遍历所有作业，筛选出失败的作业，并停止它们
for job in jobs:
    if job.status == job_submission.JobStatus.FAILED:
        try:
            client.stop_job(job.job_id)
            print(f"Stopped failed job: {job.job_id}")
        except Exception as e:
            print(f"Error stopping job {job.job_id}: {str(e)}")

print("All failed jobs have been processed.")
