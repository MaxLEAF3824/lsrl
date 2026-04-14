import math
import traceback
import signal
import torch
import os
import atexit

# 🚀 [核心修改 1] 将标准的 multiprocessing 替换为 torch 深度定制的版本
# 这允许我们在跨进程传递 Queue 时，自动使用共享内存传递 Tensor 指针，而非序列化数据
import torch.multiprocessing as mp

def vllm_worker_loop(gpu_id, model_name, task_queue, result_queue):
    """运行在独立子进程中的 vLLM Worker"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 强行禁用 V1 引擎的多进程隔离，确保向后兼容性
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
            gpu_memory_utilization=0.4, 
            dtype="bfloat16",
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
                    # 🚀 [核心修改 2] 拿到的直接是共享内存中的 PyTorch Tensor，零反序列化开销
                    # vLLM 会在内部自动将这些 CPU Tensor 搬运到当前 GPU
                    inputs = [{"prompt_embeds": emb} for emb in task['embeds_list']]
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
                
                # 手动清理引用，加速底层共享内存的回收
                del inputs, outputs
                
            # =============== 处理权重更新请求 ===============
            elif task_type == 'UPDATE_WEIGHTS':
                payload = task['payload']
                print(f"🔄 [Worker GPU {gpu_id}] 开始热更新权重...")
                
                if hasattr(llm, 'collective_rpc'):
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
                        
                        worker.model_runner.model.load_weights(state_dict.items())
                        del state_dict
                        torch.cuda.empty_cache()
                    
                    llm.collective_rpc(worker_load_weights)
                else:
                    if isinstance(payload, str):
                        if payload.endswith('.safetensors'):
                            from safetensors.torch import load_file
                            hf_state_dict = load_file(payload)
                        else:
                            hf_state_dict = torch.load(payload, map_location="cpu")
                    else:
                        hf_state_dict = payload
                    
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
    finally:
        # 🚀 [核心修复] 在进程退出前，优雅清理 PyTorch 环境
        print(f"🛑 [Worker GPU {gpu_id}] 正在清理底层分布式环境...")
        
        # 1. 显式删除 LLM 对象，帮助垃圾回收
        if 'llm' in locals():
            del llm 
            
        # 2. 显式销毁 PyTorch 分布式进程组，消除 NCCL 警告
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            
        # 3. 清空当前 Worker 的 CUDA 缓存
        import torch
        torch.cuda.empty_cache()
        print(f"✅ [Worker GPU {gpu_id}] 环境清理完毕，安全退出。")

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
        self._is_closed = False 

        for gpu_id in gpu_ids:
            p = mp.Process(
                target=vllm_worker_loop, 
                args=(gpu_id, model_name, self.task_queue, self.result_queue)
            )
            p.start()
            self.workers.append(p)
            
        atexit.register(self.close)
    
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
                    shared_embeds = []
                    for emb in chunk:
                        # 兜底：防止主代码误传了 numpy 进来
                        t = emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
                        shared_embeds.append(t.share_memory_())
                    task_dict['embeds_list'] = shared_embeds
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
        """安全关闭所有 Worker 进程"""
        if getattr(self, '_is_closed', True): 
            return
            
        self._is_closed = True
        print("\n🧹 接收到停止信号，正在安全关闭 vLLM Worker 进程并释放显存...")
        
        try:
            for _ in self.workers:
                self.task_queue.put(None)
                
            for p in self.workers:
                p.join(timeout=5)  
                if p.is_alive():
                    print(f"⚠️ Worker 进程 (PID: {p.pid}) 未响应，强制终止...")
                    p.terminate()
            
            print("✅ vLLM Worker 进程已全部安全清理完毕。")
        except Exception as e:
            print(f"⚠️ 清理 Worker 时发生异常: {e}")