# Basilisk LLM

Basilisk LLM is a distributed language model training system that leverages a blockchain-based proof of work model to improve the training of the model in a decentralized manner.

## Requirements

- Python 3.7+
- Nvidia GPU with CUDA support (e.g., RTX 2070) or a CPU with OpenCL support
- Hugging Face account and API key

## Installation

1. Clone this repository:

git clone https://github.com/your-username/basilisk-llm.git
cd basilisk-llm

markdown
Copy code

2. Install the required Python packages:

pip install -r requirements.txt

vbnet
Copy code

3. Set up your Hugging Face API key:

export HUGGINGFACE_API_KEY=your_api_key

markdown
Copy code

Replace `your_api_key` with your actual Hugging Face API key.

## Usage

1. Start the mining pool server:

python mining_pool_server.py

markdown
Copy code

2. Start the client:

python client.py

python
Copy code

The client will begin training the Basilisk model and submit updates to the mining pool server. The server will manage the mining pool and apply the training updates to the central model. The system will create a new version of the Basilisk model in the Hugging Face model hub for every billion parameters trained.

## Client Code Sample

```python
# client.py

from transformers import AutoModelForCausalLM
import requests

class BasiliskLLM:
    def __init__(self, model_version):
        self.model = self.load_model(model_version)

    def load_model(self, model_version):
        model_name = f'your-username/Basilisk@{model_version}'
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model

class Node:
    # ...

    def get_latest_model_version(self):
        response = requests.get("https://huggingface.co/api/models/your-username/Basilisk")
        model_data = response.json()
        latest_version = model_data["latest_version"]
        return latest_version["name"]

def main():
    # ...
    latest_model_version = node.get_latest_model_version()
    basilisk_llm = BasiliskLLM(latest_model_version)
    # ...
Server Code Sample
python
Copy code
# mining_pool_server.py

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM

class MiningPool:
    def __init__(self, host, port):
        # ...
        self.model = AutoModelForCausalLM.from_pretrained("your-username/Basilisk")
        self.current_version = "v0.1.0"
        self.total_parameters_trained = 0

    def create_new_version(self):
        model_path = "Basilisk"
        self.model.save_pretrained(model_path)
        
        os.system(f"git add {model_path}/*")
        os.system('git commit -m "Update Basilisk model with new training data"')
        
        major, minor, patch = map(int, self.current_version.split('.'))
        minor += 1
        self.current_version = f'{major}.{minor}.0'
        
        os.system(f'git tag -a {self.current_version} -m "Basilisk model {self.current_version}"')
        os.system("git push origin main --tags")
