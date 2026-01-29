# llm_client.py

import os
import re
import json
import random
import time
from openai import OpenAI

    # "Player_Llama_4_Maverick": {"provider": "openai", "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"},
    # "Player_GLM_4_5": {"provider": "openai", "model": "glm-4.5"},
    # "Player_Qwen3_30B": {"provider": "siliconflow", "model": "Qwen/Qwen3-30B-A3B-Thinking-2507"},
special_model = [
    "Qwen/Qwen3-30B-A3B-Thinking-2507", 
    "glm-4.5", 
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    'kimi-k2-250711',
    "zai-org/GLM-4.5",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "qwen3-235b-a22b-thinking-2507"
    ]


API_CONFIG = {
    "all": {
        "base_url": "",
        "model_prefix": "",
        "keys": [
            ''
        ]
    },
}
RETRY_DELAY_SECONDS = 60

class LLMClient:
    def __init__(self, provider, model, max_retries=3, max_tokens=5000): 
        if provider not in API_CONFIG:
            raise ValueError(f"Provider '{provider}' not found in API_CONFIG.")
        # if model == 'gpt-4o' or model == 'gpt-4o-mini':
        #     max_tokens=16384
        self.max_tokens=max_tokens
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self.api_config = API_CONFIG[provider]
        self.model_prefix = self.api_config["model_prefix"]
        self.keys = self.api_config["keys"]
        self.current_key_index = random.randint(0, len(self.keys) - 1)
        self._initialize_client()

    def _initialize_client(self):
        """Initializes the OpenAI client with the current API key."""
        current_key = self.keys[self.current_key_index]
        self.client = OpenAI(
            api_key=current_key,
            base_url=self.api_config["base_url"],
        )
        print(f"[{self.provider}/{self.model}] Client initialized with key index {self.current_key_index}.")

    def _rotate_key_and_reinit(self):
        """Rotates to the next key in the pool and re-initializes the client."""
        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        print(f"  [Key Rotation] Rotating to key index {self.current_key_index}.")
        self._initialize_client()

    def make_request(self, system_prompt, user_prompt, temperature=1e-7):
        print('temperature:',temperature)
        return_data = {"action": None, "response_text": None}
        attempt = 0
        # print(user_prompt)
        # quit()
        while attempt < self.max_retries:
            # time.sleep(30)
            # quit()
            try:
                # time.sleep(60)
                if f"{self.model_prefix}{self.model}" in special_model:
                    response = self.client.chat.completions.create(
                        model=f"{self.model_prefix}{self.model}",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=temperature, # Very low temperature for deterministic output
                        # response_format={"type": "json_object"},
                        max_tokens=self.max_tokens
                    )
                elif 'qwen3-8b' in f"{self.model_prefix}{self.model}" or 'qwen3-14b' in f"{self.model_prefix}{self.model}" :
                    response = self.client.chat.completions.create(
                        model=f"{self.model_prefix}{self.model}",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=temperature, # Very low temperature for deterministic output
                        response_format={"type": "json_object"},
                        max_tokens=8192,
                        extra_body={'enable_thinking': False} # 添加此行来修复问题                  
                    )
                elif 'qwen' in f"{self.model_prefix}{self.model}":
                    response = self.client.chat.completions.create(
                        model=f"{self.model_prefix}{self.model}",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=temperature, # Very low temperature for deterministic output
                        response_format={"type": "json_object"},
                        max_tokens=self.max_tokens,
                        extra_body={'enable_thinking': False} # 添加此行来修复问题                  
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=f"{self.model_prefix}{self.model}",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=temperature, # Very low temperature for deterministic output
                        response_format={"type": "json_object"},
                        max_tokens=self.max_tokens
                    )
                print(response)
                # quit()
                full_response_text = response.choices[0].message.content.strip()
                return_data["response_text"] = full_response_text
                
                try:
                    action = json.loads(full_response_text)
                    return_data["action"] = action
                    return return_data
                except json.JSONDecodeError:
                    print(f"  [Warning] Failed to decode JSON from response: {full_response_text}")
                    json_match = re.search(r"\{[\s\S]*\}", full_response_text)
                    if json_match:
                        try:
                            action = json.loads(json_match.group(0))
                            return_data["action"] = action
                            print("  [Info] Successfully extracted JSON with regex.")
                            return return_data
                        except json.JSONDecodeError:
                             print(f"  [Warning] Regex extraction also failed.")
                
                # If JSON parsing fails even with regex, we count it as a failed attempt
                attempt += 1

            except Exception as e:
                print(f"  [Error] Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                error_str = str(e).lower()
                if any(err in error_str for err in ["rate limit", "quota", "insufficient", "limit exceeded", "key"]):
                    self._rotate_key_and_reinit()
                    print(f"  Waiting {RETRY_DELAY_SECONDS}s before retrying...")
                    time.sleep(RETRY_DELAY_SECONDS)
                    attempt += 1
                else:
                    print("  [FATAL Error] Unrecoverable API error.")
                    return_data["response_text"] = f"FATAL ERROR: {e}"
                    self._rotate_key_and_reinit()
                    attempt += 1
                    # return return_data
        
        print(f"\n[Error] All {self.max_retries} retry attempts failed for this request.")
        return_data["response_text"] = "Exceeded max retries."
        return return_data