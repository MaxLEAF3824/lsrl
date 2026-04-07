# vllm_workers.py
import multiprocessing as mp
import math
import traceback
import signal
import torch

def vllm_worker_loop(gpu_id, model_name, task_queue, result_queue):
    """运行在独立子进程中的 vLLM Worker"""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 【核心防御】：让子进程忽略 Jupyter 发出的中断信号 (KeyboardInterrupt)
    # 这样只有当主进程通过 Queue 发送 None 时，子进程才会退出
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    try:
        import torch
        from vllm import LLM, SamplingParams
        
        print(f"🚀 [Worker GPU {gpu_id}] 正在初始化 vLLM 引擎...")
        llm = LLM(
            model=model_name, 
            trust_remote_code=True, 
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.8, 
            dtype="float16",
            enable_prompt_embeds=True
        )
        print(f"✅ [Worker GPU {gpu_id}] 初始化完成，等待任务...")

        while True:
            task = task_queue.get()
            if task is None: 
                break
                
            task_idx, embeds_list, sp_dict = task
            sp = SamplingParams(**sp_dict)
            
            inputs = [{"prompt_embeds": torch.tensor(emb, dtype=torch.float16)} for emb in embeds_list]
            outputs = llm.generate(prompts=inputs, sampling_params=sp, use_tqdm=False)
            
            batch_res = []
            for out in outputs:
                req_res = []
                for comp in out.outputs:
                    req_res.append(list(comp.token_ids))
                batch_res.append(req_res)
                
            result_queue.put((task_idx, batch_res))
            
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"\n❌ [Worker GPU {gpu_id}] 发生致命错误:\n{error_msg}\n")
        result_queue.put((-1, error_msg))


class VLLMDPWorkerPool:
    def __init__(self, model_name, gpu_ids=[1, 2, 3]):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
        self.num_workers = len(gpu_ids)
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers = []
        
        for gpu_id in gpu_ids:
            p = mp.Process(
                target=vllm_worker_loop, 
                args=(gpu_id, model_name, self.task_queue, self.result_queue)
            )
            p.start()
            self.workers.append(p)
            
    def clear_queues(self):
        """【新增】：用于在主进程中断后，清理队列里上一轮残留的数据，防止新旧数据错乱"""
        while not self.task_queue.empty():
            try: self.task_queue.get_nowait()
            except: pass
        while not self.result_queue.empty():
            try: self.result_queue.get_nowait()
            except: pass

    def generate(self, inputs_embeds_list, sampling_params_dict):
        total_prompts = len(inputs_embeds_list)
        if total_prompts == 0:
            return []
            
        chunk_size = math.ceil(total_prompts / self.num_workers)
        chunks = [inputs_embeds_list[i:i + chunk_size] for i in range(0, total_prompts, chunk_size)]
        
        active_tasks = 0
        for idx, chunk in enumerate(chunks):
            if len(chunk) > 0:
                np_chunk = [emb.detach().cpu().to(torch.float16).numpy() for emb in chunk]
                self.task_queue.put((idx, np_chunk, sampling_params_dict))
                active_tasks += 1
                
        results = {}
        for _ in range(active_tasks):
            idx, res = self.result_queue.get()
            if idx == -1:
                raise RuntimeError(f"vLLM 子进程崩溃，错误信息：\n{res}")
            results[idx] = res
            
        final_batch_token_ids = []
        for i in range(active_tasks):
            final_batch_token_ids.extend(results[i])
            
        return final_batch_token_ids
        
    def close(self):
        for _ in self.workers:
            self.task_queue.put(None)
        for p in self.workers:
            p.join()