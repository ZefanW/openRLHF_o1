from parallel_evaluation import parallel_eval
from flask import Flask, request, jsonify
import logging
import pandas as pd
import torch
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def calculate_reward():
    # Replace this with your actual reward calculation logic
    reward_score = 100
    return reward_score

@app.route('/get_reward', methods=['POST'])
def get_reward():
    data = request.get_json(force=True)
    logger.info(f"Received data: {data}")
    # Calculate reward (you can use the received data if needed)
    reward = parallel_eval(pd.DataFrame(data), num_processes=50)
    
    # Return in the format expected by the wrapper
    logger.info(f"reward: {reward}")
    return {'rewards': reward}
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)