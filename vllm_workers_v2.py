import multiprocessing as mp
import math
import traceback
import signal
import torch
import os

def vllm_worker_loop(gpu_id, model_name, task_queue, result_queue):
    """运行在独立子进程中的 vLLM Worker"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 【核心防御】：让子进程忽略 Jupyter 发出的中断信号
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
                inputs_list = task['inputs_list'] # 变量名从 embeds_list 改为 inputs_list
                sp_dict = task['sp_dict']
                
                sp = SamplingParams(**sp_dict)
                inputs = []
                for item in inputs_list:
                    # 【兼容处理】：如果是字符串，说明是标准文本 Prompt；如果是数组/Tensor，说明是 Embeddings
                    if isinstance(item, str):
                        inputs.append({"prompt": item})
                    else:
                        inputs.append({"prompt_embeds": torch.tensor(item, dtype=torch.float16)})
                        
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
                
                # 1. 加载权重字典到 CPU (避免 GPU 显存峰值)
                if isinstance(payload, str):
                    # 如果传入的是文件路径
                    if payload.endswith('.safetensors'):
                        from safetensors.torch import load_file
                        hf_state_dict = load_file(payload)
                    else:
                        hf_state_dict = torch.load(payload, map_location="cpu")
                else:
                    # 如果直接传入的是 state_dict
                    hf_state_dict = payload
                
                # 2. 定位 vLLM 内部的 PyTorch 模型对象
                # (不同 vLLM 版本内部结构略有不同，做向下/向上兼容)
                executor = llm.llm_engine.model_executor
                if hasattr(executor, 'driver_worker'):
                    model = executor.driver_worker.model_runner.model
                elif hasattr(executor, 'last_worker'):
                    model = executor.last_worker.model_runner.model
                elif hasattr(executor, 'model'):
                    model = executor.model
                else:
                    raise AttributeError("无法定位 vLLM 内部的 PyTorch 模型。")
                
                # 3. 利用 vLLM 原生的 load_weights 方法覆盖权重
                # 它会自动处理 HuggingFace 权重名称到 vLLM fused 权重名称的映射 (如 QKV fusion)
                model.load_weights(hf_state_dict.items())
                
                # 4. 清理内存
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
                # 修改点：将任务封装为字典协议
                self.task_queue.put({
                    'type': 'GENERATE',
                    'task_idx': idx,
                    'embeds_list': np_chunk,
                    'sp_dict': sampling_params_dict
                })
                active_tasks += 1
                
        results = {}
        for _ in range(active_tasks):
            res_tuple = self.result_queue.get()
            # 捕获崩溃或错误
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
        """
        【新增】：在不断开服务的情况下动态更新 vLLM Worker 权重。
        支持传入 HuggingFace 格式的 state_dict 字典，或本地文件路径。
        """
        # 下发更新指令
        for _ in range(self.num_workers):
            self.task_queue.put({
                'type': 'UPDATE_WEIGHTS',
                'payload': state_dict_or_path
            })
            
        # 等待所有 Worker 返回更新结果，保持主进程同步阻塞
        success_count = 0
        for _ in range(self.num_workers):
            res_tuple = self.result_queue.get()
            
            if res_tuple[0] == 'ERROR':
                _, gpu_id, error_msg = res_tuple
                raise RuntimeError(f"vLLM Worker {gpu_id} 权重更新失败:\n{error_msg}")
                
            res_type, gpu_id, status = res_tuple
            if res_type == 'UPDATE_DONE' and status == "SUCCESS":
                success_count += 1
                
        print(f"🌟 全部就绪！成功热更新了 {success_count} 个 Worker 的权重。")

    def close(self):
        for _ in self.workers:
            self.task_queue.put(None)
        for p in self.workers:
            p.join()