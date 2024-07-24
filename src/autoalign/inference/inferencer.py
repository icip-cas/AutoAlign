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
import math
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
import ray
import multiprocessing as mp
from vllm import LLM, SamplingParams

class MultiProcessHFInferencer:

    def __init__(self, 
                model_path:str, 
                do_sample:bool=False,
                num_beams:int=1,
                max_new_tokens:int=512,
                temperature:float=0,
                top_p:float=1,
                nccl_timeout:int=64000
                ):
        
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p

        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=nccl_timeout))
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
                # currently only support bs=1
                inputs = self.tokenizer(d, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    do_sample=self.do_sample,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
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
                num_gpus_per_model:int=1,
                do_sample:bool=False,
                num_beams:int=1,
                max_new_tokens:int=1024,
                temperature:float=0,
                top_p:float=1.0,
                top_k:int=-1,
                frequency_penalty=0.0,
                ):

        self.num_gpus_total = torch.cuda.device_count()
        self.num_gpus_per_model = num_gpus_per_model

        self.model_path = model_path

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

        self.use_ray = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.num_gpus_total // num_gpus_per_model > 1:
            self.use_ray = True
            ray.init(ignore_reinit_error=True)

    @staticmethod
    def single_process_inference(model_path, num_gpus_per_model,
                                *args, **kwargs):

        model = LLM(
            model=model_path,
            tensor_parallel_size=num_gpus_per_model,
        )

        return model.generate(*args, **kwargs)

    def inference(self, data:List[str], temperature:float=None, do_sample:bool=None):
        if self.use_ray:
            get_answers_func = ray.remote(num_gpus=self.num_gpus_per_model)(
                MultiProcessVllmInferencer.single_process_inference
            ).remote
        else:
            get_answers_func = MultiProcessVllmInferencer.single_process_inference

        num_processes = max(1, self.num_gpus_total // self.num_gpus_per_model)
        chunk_size = math.ceil(len(data) / num_processes)
        
        gathered_responses = []
        for i in range(0, len(data), chunk_size):
            tmp_sampling_params = self.sampling_params
            tmp_sampling_params.temperature = temperature if temperature is not None else self.sampling_params.temperature
            gathered_responses.append(
                get_answers_func(
                    self.model_path,
                    self.num_gpus_per_model,
                    data[i:i+chunk_size],
                    tmp_sampling_params
                )
            )
        if self.use_ray:
            gathered_responses = ray.get(gathered_responses)

        gathered_responses = [item for sublist in gathered_responses for item in sublist]
        gathered_responses = [response.outputs[0].text for response in gathered_responses]

        return gathered_responses
    
    def get_tokenizer(self):

        return self.tokenizer