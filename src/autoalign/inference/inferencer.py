import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
import multiprocessing as mp
from typing import List, Dict
from accelerate import Accelerator
from accelerate.utils import gather_object

import openai
from multiprocessing.pool import ThreadPool
from tenacity import (
    retry,
    stop_after_attempt,
    wait_chain,
    wait_fixed
)

from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

import multiprocessing as mp
import os

class MultiProcessHFInferencer:

    def __init__(self, 
                model_path:str, 
                do_sample:bool=False,
                num_beams:int=1,
                max_new_tokens:int=512,
                temperature:float=0,
                top_p:float=1,
                ):
        
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p

        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=64000))
        self.accelerator = Accelerator(kwargs_handlers=[kwargs])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # get tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # get model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            device_map={"": self.accelerator.process_index},
        )


    def inference(self,
                data,
                do_sample=None, 
                top_p=None, 
                temperature=None,
                num_beams=None,
                max_new_tokens=None,
                ):
        
        self.do_sample = do_sample if do_sample is not None else self.do_sample
        self.top_p = top_p if top_p is not None else self.top_p
        self.temperature = temperature if temperature is not None else self.temperature
        self.num_beams = num_beams if num_beams is not None else self.num_beams
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

        # for each process
        with self.accelerator.split_between_processes(data) as data_on_process:

            responses = []

            for d in tqdm(data_on_process):
                inputs = self.tokenizer(d, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    do_sample=self.do_sample,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens
                )

                outputs = outputs[:, len(inputs["input_ids"][0]):]

                responses.extend(self.tokenizer.batch_decode(
                    outputs
                ))

        # gather responses
        gathered_responses = gather_object(responses)

        return gathered_responses

    def get_tokenizer(self):
        return self.tokenizer
    
class OpenAIInferencer:

    def __init__(
        self,
        model:str,
        api_key:str,
        api_base:str=None,
        temperature:float=0,
        max_tokens:int=1024,
        request_timeout=120,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        openai.api_key = api_key
        if api_base is not None:
            openai.api_base = api_base

    @retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                        [wait_fixed(5) for i in range(2)] +
                        [wait_fixed(10)]))
    def inference(self, prompt, stop=None):
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop
        )
        return response['choices'][0]['text'].strip()

    @retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                        [wait_fixed(5) for i in range(2)] +
                        [wait_fixed(10)]))
    def chat_inference(self, messages: List[Dict]):
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                request_timeout=self.request_timeout,
            )
        except Exception as e:
            if str(e).startswith("Detected an error in the prompt. Please try again with a different prompt.") or \
                str(e).startswith("Sorry! We've encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt."):
                print(e)
                return "Prompt Error"
            print(e)
        return response['choices'][0]['message']['content'].strip()
    
    def multiprocess_chat_inference(self, 
                                    batch_of_messages: List[List[Dict]], 
                                    num_workers:int=8):

        pool = ThreadPool(num_workers)
        all_completions = list(
            tqdm(pool.imap(self.chat_inference, batch_of_messages), 
                 total=len(batch_of_messages))
        )

        return all_completions

    def encode(self, prompt:str) -> List:
        enc = tiktoken.encoding_for_model(self.model)
        return enc.encode(prompt)
    
    def decode(self, tokens:List) -> str:
        enc = tiktoken.encoding_for_model(self.model)
        return enc.decode(tokens)
    
class HFPipelineInferencer:

    def __init__(self, 
                model_path:str, 
                num_gpus:int, 
                do_sample:bool=False,
                num_beams:int=1,
                max_new_tokens:int=512,
                temperature:float=0,
                top_p:float=1,
                ):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            device_map="auto",
        )
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        
        self.pipe = pipeline(
            task="text-generation",
            model=self.model, 
            tokenizer=self.tokenizer, 
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens
        )

    def inference(self, 
                prompt:str,
                do_sample=None, 
                top_p=None, 
                temperature=None,
                num_beams=None,
                max_new_tokens=None):
        
        self.do_sample = do_sample if do_sample is not None else self.do_sample
        self.top_p = top_p if top_p is not None else self.top_p
        self.temperature = temperature if temperature is not None else self.temperature
        self.num_beams = num_beams if num_beams is not None else self.num_beams
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        
        outputs = self.pipe(
            prompt,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens
        )
        generated_text = outputs[0]["generated_text"]
        generated_text = generated_text.split(prompt)[-1]
        return generated_text

    def get_tokenizer(self):
        return self.tokenizer
    
class OpenAIInferencer:

    def __init__(
        self,
        model:str,
        api_key:str,
        api_base:str=None,
        temperature:float=0,
        max_tokens:int=1024,
        request_timeout=120,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        openai.api_key = api_key
        if api_base is not None:
            openai.api_base = api_base

    @retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                        [wait_fixed(5) for i in range(2)] +
                        [wait_fixed(10)]))
    def inference(self, prompt, stop=None):
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop
        )
        return response['choices'][0]['text'].strip()

    @retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                        [wait_fixed(5) for i in range(2)] +
                        [wait_fixed(10)]))
    def chat_inference(self, messages: List[Dict]):
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                request_timeout=self.request_timeout,
            )
        except Exception as e:
            if str(e).startswith("Detected an error in the prompt. Please try again with a different prompt.") or \
                str(e).startswith("Sorry! We've encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt."):
                print(e)
                return "Prompt Error"
            print(e)
        return response['choices'][0]['message']['content'].strip()
    
    def multiprocess_chat_inference(self, 
                                    batch_of_messages: List[List[Dict]], 
                                    num_workers:int=8):

        pool = ThreadPool(num_workers)
        all_completions = list(
            tqdm(pool.imap(self.chat_inference, batch_of_messages), 
                 total=len(batch_of_messages))
        )

        return all_completions

    def encode(self, prompt:str) -> List:
        enc = tiktoken.encoding_for_model(self.model)
        return enc.encode(prompt)
    
    def decode(self, tokens:List) -> str:
        enc = tiktoken.encoding_for_model(self.model)
        return enc.decode(tokens)

class MultiProcessVllmInferencer:

    def __init__(self, 
                model_path:str, 
                num_gpus:int,
                do_sample:bool=False,
                num_beams:int=1,
                max_new_tokens:int=512,
                temperature:float=0,
                top_p:float=1.0,
                top_k:int=-1,
                frequency_penalty=0.0,
                model_max_tokens=4096,
                ):

        self.num_gpus = num_gpus

        self.manager = mp.Manager()

        self.model_path = model_path
        self.model_max_tokens = model_max_tokens

        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            use_beam_search=(not do_sample) and (not num_beams==1),
            best_of=num_beams,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            length_penalty=1.0,
            frequency_penalty=frequency_penalty,
            early_stopping=False,
        )

    def inference_on_device(self, device_id, data, shared_dict, batch_size=32):

        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id) # hacking the visible devices to isolate gpus

        model = LLM(
            model=self.model_path, # for small models, tensor parallel doesn't accelerate much
            tensor_parallel_size=1,
            max_num_batched_tokens=self.model_max_tokens,
        )

        for i in tqdm(range(0, len(data), batch_size)):
            batched_inp = []
            for j in data[i: i + batch_size]:
                batched_inp.append(j)

            outputs = model.generate(
                batched_inp, 
                self.sampling_params,
                use_tqdm=False
            )
                
            for k in range(len(batched_inp)):
                shared_dict[(i + k) * self.num_gpus + device_id] = outputs[k].outputs[0].text

    def inference(self, data:List, batch_size=32):

        processes = []
        shared_dict = self.manager.dict()

        for device_id in range(self.num_gpus):
            data_on_device = data[device_id::self.num_gpus]
            p = mp.Process(
                target=self.inference_on_device, 
                args=(device_id, data_on_device, shared_dict, batch_size)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        sorted_results = [value for key, value in sorted(shared_dict.items(), key=lambda item: item[0])]

        return sorted_results