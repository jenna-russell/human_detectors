import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn.functional as F
import time
import random
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from anthropic import AnthropicVertex
import re

torch.manual_seed(42)
random.seed(42)

class Detector():
    def __init__(self, llm="gpt-4o-2024-08-06", explain=True, path=None, key_file="", api_key="", project_id=None, region=None, guide=True):
        self.llm = LLM(llm, path, key_file, api_key, project_id, region)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.explain = explain
        self.guide = guide
        with open(f"prompts/detection_guide.txt", 'r') as f:
            self.detection_guide = f.read()
        for filename in os.listdir('prompts/detector_prompts'):
            file_name_without_extension = os.path.splitext(filename)[0]
            with open(os.path.join('prompts/detector_prompts', filename), 'r') as f:
                file_content = f.read()
                setattr(self, file_name_without_extension, file_content)

    def detect(self, text):
        if self.guide:
            if self.explain:
                prompt = self.classify_zero_shot_cot_guide.format(self.detection_guide, text)
            else:
                prompt = self.classify_zero_shot_guide.format(self.detection_guide, text)
        else:
            if self.explain:
                prompt = self.classify_zero_shot_cot.format(text)
            else:
                prompt = self.classify_zero_shot.format(text)

        response_text = self.llm.generate(prompt, sample=False)

        explanation_pattern = r"<description>(.*?)</description>"
        answer_pattern = r"<answer>(.*?)</answer>"

        explanation_match = re.search(explanation_pattern, response_text, re.DOTALL)
        answer_match = re.search(answer_pattern, response_text, re.DOTALL)

        explanation = explanation_match.group(1).strip() if explanation_match else None
        answer = answer_match.group(1).strip() if answer_match else None

        return answer, explanation

class Evader():
    def __init__(self, llm="gpt-4o-2024-08-06", path=None, key_file="", api_key="", project_id=None, region=None):
        self.llm = LLM(llm, path, key_file, api_key, project_id, region)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(f"prompts/detection_guide.txt", 'r') as f:
            self.detection_guide = f.read()
        for filename in os.listdir('prompts/humanizer_prompts'):
            file_name_without_extension = os.path.splitext(filename)[0]
            with open(os.path.join('prompts/humanizer_prompts', filename), 'r') as f:
                file_content = f.read()
                setattr(self, file_name_without_extension, file_content)

    def evade(self, prompt, publication, examples=None):
        if examples:
            prompt = self.evade_fewshot.format(publication, self.detection_guide, examples, prompt)
        else:
            prompt = self.evade_prompt.format(publication, self.detection_guide, prompt)
        response = self.llm.generate(prompt, max_tokens=2024)
        return response

class LLM():
    def __init__(self, model, path=None, key_file="", api_key="", project_id=None, region=None):
        if "gpt" in model or "o1" in model:
            self.model = ChatGPT(model, key_file, api_key)
        elif "claude" in model:
            self.model = Claude(model, project_id=project_id, region=region)
        else:
            if not path:
                raise ValueError(f"Please enter a valid model name or path")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = AutoTokenizer.from_pretrained(path, device_map="auto")
            self.model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")
            self.model.eval()

    def generate(self, prompt, max_tokens=2024, temperature=1, sample=True):
        if sample == True:
            return self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        else:
            return self.model.generate(prompt, max_tokens=max_tokens, sample=False)

class ChatGPT():
    def __init__(self, llm, key_file="", api_key=""):
        if key_file != "":
            with open(key_file, "r") as f:
                key = f.read().strip()
        elif api_key != "":
            key = api_key
        if not key:
            raise ValueError(f"The key file {key_file} is empty. Please provide a valid API key.")
        self.llm = llm
        print(f"Loading {llm}...")
        self.client = OpenAI(api_key=key)

    def obtain_response(self, prompt, max_tokens, temperature, seed=42, sample=True):
        response = None
        num_attempts = 0
        messages = []
        messages.append({"role": "user", "content": prompt})
        while response is None:
            try:
                if 'o1' in self.llm:
                    response = self.client.chat.completions.create(
                        model=self.llm,
                        messages=messages,
                        seed=seed)
                else:
                    response = self.client.chat.completions.create(
                        model=self.llm,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed)
            except Exception as e:
                if num_attempts == 5:
                    print(f"Attempt {num_attempts} failed, breaking...")
                    return None
                print(e)
                num_attempts += 1
                print(f"Attempt {num_attempts} failed, trying again after 5 seconds...")
                time.sleep(5)
        return response.choices[0].message.content.strip()

    def generate(self, prompt, max_tokens, temperature=1.0, sample=False):
        if sample == False:
            return self.obtain_response(prompt, max_tokens=max_tokens, sample=False, temperature=0.0)
        return self.obtain_response(prompt, max_tokens=max_tokens, temperature=temperature)

class Claude():
    def __init__(self, llm, region=None, project_id=None):
        self.region = region
        if not region:
            raise ValueError(f"Please provide a valid region.")
        self.project_id = project_id
        if not project_id:
            raise ValueError(f"Please provide a valid project_id.")
        self.client = AnthropicVertex(region=region, project_id=project_id)
        self.llm = llm
        print(f"Loading {llm}...")

    def obtain_response(self, prompt, temperature=1, max_tokens=4000):
        response = None
        num_attempts = 0
        while response is None:
            try:
                response = self.client.messages.create(
                    model=self.llm,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )
            except Exception as e:
                if num_attempts == 5:
                    print(f"Attempt {num_attempts} failed, breaking...")
                    return None
                print(e)
                num_attempts += 1
                print(f"Attempt {num_attempts} failed, trying again after 5 seconds...")
                time.sleep(5)

        message_json_str = response.model_dump_json(indent=2)
        message_dict = json.loads(message_json_str)
        text_content = message_dict['content'][0]['text']
        return text_content.strip()

    def generate(self, prompt, max_tokens, temperature=1.0, sample=False):
        if sample == False:
            return self.obtain_response(prompt, max_tokens=max_tokens, temperature=0.0)
        return self.obtain_response(prompt, max_tokens=max_tokens, temperature=temperature) 