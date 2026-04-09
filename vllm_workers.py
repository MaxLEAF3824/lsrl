import multiprocessing as mp
import math
import traceback
import signal
import torch
import os

def vllm_worker_loop(gpu_id, model_name, task_queue, result_queue):
    """运行在独立子进程中的 vLLM Worker"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # [🔥 修复点 1] 强行禁用 V1 引擎的多进程隔离，确保向后兼容性
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    try:
        from vllm import LLM, SamplingParams
        
        print(f"🚀 [Worker GPU {gpu_id}] 正在初始化 vLLM 引擎...")
        llm = LLM(
            model=model_name, 
            trust_remote_code=True, 
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.9, 
            dtype="float16",
            enable_prompt_embeds=True,
            max_model_len=40960
        )
        print(f"✅ [Worker GPU {gpu_id}] 初始化完成，等待任务...")

        while True:
            task = task_queue.get()
            if task is None: 
                break
                
            task_type = task.get('type')
            
            # =============== 处理推理请求 ===============
            if task_type == 'GENERATE':
                task_idx = task['task_idx']
                sp_dict = task['sp_dict']
                sp = SamplingParams(**sp_dict)
                
                if 'embeds_list' in task and task['embeds_list']:
                    inputs = [{"prompt_embeds": torch.tensor(emb, dtype=torch.float16)} for emb in task['embeds_list']]
                elif 'prompts_list' in task and task['prompts_list']:
                    inputs = task['prompts_list'] 
                else:
                    inputs = []

                outputs = llm.generate(prompts=inputs, sampling_params=sp, use_tqdm=False)
                
                batch_res = []
                for out in outputs:
                    req_res = []
                    for comp in out.outputs:
                        req_res.append(list(comp.token_ids))
                    batch_res.append(req_res)
                    
                result_queue.put(('GENERATE_DONE', task_idx, batch_res))
                
            # =============== 处理权重更新请求 ===============
            elif task_type == 'UPDATE_WEIGHTS':
                payload = task['payload']
                print(f"🔄 [Worker GPU {gpu_id}] 开始热更新权重...")
                
                # [🔥 修复点 2] 兼容 vLLM 0.8.x+ 的 RPC 架构
                if hasattr(llm, 'collective_rpc'):
                    # 新版 vLLM 推荐使用 RPC 向底层 Worker 下发函数
                    def worker_load_weights(worker):
                        import torch
                        if isinstance(payload, str):
                            if payload.endswith('.safetensors'):
                                from safetensors.torch import load_file
                                state_dict = load_file(payload)
                            else:
                                state_dict = torch.load(payload, map_location="cpu")
                        else:
                            state_dict = payload
                        
                        # 此时运行在底层 worker 环境中，直接拿到 model 赋值
                        worker.model_runner.model.load_weights(state_dict.items())
                        del state_dict
                        torch.cuda.empty_cache()
                    
                    llm.collective_rpc(worker_load_weights)
                
                # 旧版 vLLM 兼容逻辑
                else:
                    if isinstance(payload, str):
                        if payload.endswith('.safetensors'):
                            from safetensors.torch import load_file
                            hf_state_dict = load_file(payload)
                        else:
                            hf_state_dict = torch.load(payload, map_location="cpu")
                    else:
                        hf_state_dict = payload
                    
                    # 因为前面注入了环境变量，这里保证能取到 model_executor
                    executor = llm.llm_engine.model_executor
                    if hasattr(executor, 'driver_worker'):
                        model = executor.driver_worker.model_runner.model
                    elif hasattr(executor, 'last_worker'):
                        model = executor.last_worker.model_runner.model
                    elif hasattr(executor, 'model'):
                        model = executor.model
                    else:
                        raise AttributeError("无法定位 vLLM 内部的 PyTorch 模型。")
                    
                    model.load_weights(hf_state_dict.items())
                    del hf_state_dict
                    torch.cuda.empty_cache()
                
                print(f"✅ [Worker GPU {gpu_id}] 权重更新完成！")
                result_queue.put(('UPDATE_DONE', gpu_id, "SUCCESS"))

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"\n❌ [Worker GPU {gpu_id}] 发生错误:\n{error_msg}\n")
        result_queue.put(('ERROR', gpu_id, error_msg))


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
        while not self.task_queue.empty():
            try: self.task_queue.get_nowait()
            except: pass
        while not self.result_queue.empty():
            try: self.result_queue.get_nowait()
            except: pass

    def generate(self, inputs_list, sampling_params_dict, input_type="embeds"):
        total_prompts = len(inputs_list)
        if total_prompts == 0:
            return []
            
        chunk_size = math.ceil(total_prompts / self.num_workers)
        chunks = [inputs_list[i:i + chunk_size] for i in range(0, total_prompts, chunk_size)]
        
        active_tasks = 0
        for idx, chunk in enumerate(chunks):
            if len(chunk) > 0:
                task_dict = {
                    'type': 'GENERATE',
                    'task_idx': idx,
                    'sp_dict': sampling_params_dict
                }
                if input_type == "embeds":
                    task_dict['embeds_list'] = [emb.detach().cpu().to(torch.float16).numpy() for emb in chunk]
                else:
                    task_dict['prompts_list'] = chunk
                    
                self.task_queue.put(task_dict)
                active_tasks += 1
                
        results = {}
        for _ in range(active_tasks):
            res_tuple = self.result_queue.get()
            if res_tuple[0] == 'ERROR':
                _, gpu_id, error_msg = res_tuple
                raise RuntimeError(f"vLLM Worker {gpu_id} 崩溃:\n{error_msg}")
            _, idx, res = res_tuple
            results[idx] = res
            
        final_batch_token_ids = []
        for i in range(active_tasks):
            final_batch_token_ids.extend(results[i])
            
        return final_batch_token_ids
        
    def update_weights(self, state_dict_or_path):
        for _ in range(self.num_workers):
            self.task_queue.put({'type': 'UPDATE_WEIGHTS', 'payload': state_dict_or_path})
            
        success_count = 0
        for _ in range(self.num_workers):
            res_tuple = self.result_queue.get()
            if res_tuple[0] == 'ERROR':
                _, gpu_id, error_msg = res_tuple
                raise RuntimeError(f"vLLM Worker {gpu_id} 权重更新失败:\n{error_msg}")
            res_type, gpu_id, status = res_tuple
            if res_type == 'UPDATE_DONE' and status == "SUCCESS":
                success_count += 1
        print(f"🌟 成功热更新了 {success_count} 个 Worker 的权重。")

    def close(self):
        for _ in self.workers:
            self.task_queue.put(None)
        for p in self.workers:
            p.join()