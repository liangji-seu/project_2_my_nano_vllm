"""
单机多进程 demo: getattr 查表执行 —— RPC 远程调用的核心

模拟 vLLM Executor → Worker 的 RPC:
  客户端(Executor) 只知道"方法名 + 参数", 打包发给服务端(Worker)
  服务端用 getattr(self.worker, method) 把名字查成真实函数, 调用后回传结果

为什么用 getattr 查表, 而不是一大串 if/elif:
  if   method == "load_model":     worker.load_model(*args)
  elif method == "execute_model":  worker.execute_model(*args)
  elif ...
  每加一个新方法都要改这段分发表。
  getattr 让分发表"自动扩展": 只要 worker 上定义了方法, 名字对得上就能被远程调用。

getattr 三件套 (Python 反射):
  getattr(obj, "name")          取属性/方法, 不存在则抛 AttributeError
  getattr(obj, "name", default) 取不到时返回 default, 不抛异常
  hasattr(obj, "name")          判断是否存在, 返回 True/False

运行: python demo_getattr_rpc.py
"""

import multiprocessing


# ============================================================
# 服务端: Worker 真正实现这些方法 (类比 vLLM 的 Worker)
# ============================================================
class Worker:
    def __init__(self, model_name="llama-7b"):
        self.model_name = model_name
        self.loaded = False

    def load_model(self):
        self.loaded = True
        return f"模型 {self.model_name} 已加载"

    def execute_model(self, input_ids, temperature=1.0):
        # 模拟一次 forward, 输出假 logits
        logits = [t * 2 for t in input_ids]
        return {"logits": logits, "temperature": temperature}

    def get_cache_config(self):
        return {"block_size": 16, "num_blocks": 1024}

    def health_check(self):
        return self.loaded


# ============================================================
# 核心: 查表执行 (类比 vLLM 的 WorkerWrapperBase.execute_method)
# ============================================================
def execute_method(worker, method, *args, **kwargs):
    """把方法名字符串查成真实函数, 再调用它"""
    func = getattr(worker, method)  # 不存在 -> AttributeError
    return func(*args, **kwargs)


# ============================================================
# 服务端进程: 收 (方法名, args, kwargs) → 查表调用 → 回结果
# ============================================================
def worker_server(conn):
    worker = Worker()
    while True:
        method, args, kwargs = conn.recv()  # Pipe 内部已用 pickle 序列化
        if method == "__quit__":
            break

        # 真正的 RPC 分发点只有这一处
        try:
            result = execute_method(worker, method, *args, **kwargs) # 执行实例，执行方法，执行参数
            conn.send(("ok", result))
        except AttributeError:
            conn.send(("error", f"Worker 没有方法 {method!r}"))
        except Exception as e:  # noqa: BLE001
            conn.send(("error", repr(e)))


# ============================================================
# 客户端: Executor 发 RPC 请求 (类比 vLLM 的 run_method)
# ============================================================
class Executor:
    def __init__(self, conn):
        self.conn = conn

    def call_worker(self, method, *args, **kwargs):
        """发 RPC: 打包方法名 + 参数, 阻塞等结果"""
        self.conn.send((method, args, kwargs))
        status, payload = self.conn.recv()
        return status, payload

    def shutdown(self):
        """退出握手: 只发通知, 不等回包 (服务端收到后直接 break)"""
        self.conn.send(("__quit__", (), {}))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # duplex=True 的双向管道: 客户端和 worker 都能收能发
    client_conn, server_conn = multiprocessing.Pipe(duplex=True)

    proc = multiprocessing.Process(target=worker_server, args=(server_conn,))
    proc.start()
    server_conn.close()  # 客户端只用自己这一端

    executor = Executor(client_conn)

    # === 一连串 RPC 调用, 全靠 getattr 自动分发 ===
    status, r = executor.call_worker("load_model")
    print(f"RPC load_model        -> {status}: {r}")

    status, r = executor.call_worker("health_check")
    print(f"RPC health_check      -> {status}: {r}")

    status, r = executor.call_worker("execute_model", [1, 2, 3], temperature=0.7)
    print(f"RPC execute_model     -> {status}: {r}")

    status, r = executor.call_worker("get_cache_config")
    print(f"RPC get_cache_config  -> {status}: {r}")

    # 调一个不存在的方法 -> 服务端捕获 AttributeError 回传
    status, r = executor.call_worker("no_such_method")
    print(f"RPC no_such_method    -> {status}: {r}")

    executor.shutdown()
    proc.join(timeout=3)
    if proc.is_alive():
        proc.terminate()
    client_conn.close()

    # === getattr 三件套 (本地直接演示, 不跨进程) ===
    print("\n=== getattr 三件套 (本地) ===")
    w = Worker()
    print("getattr(w, 'get_cache_config')() =", getattr(w, "get_cache_config")())
    print("getattr(w, '不存在', '默认值')   =", getattr(w, "不存在", "默认值"))
    print("hasattr(w, 'health_check')      =", hasattr(w, "health_check"))
    print("hasattr(w, '不存在')            =", hasattr(w, "不存在"))
    print("getattr(w, '不存在')            ->", end=" ")
    try:
        getattr(w, "不存在")
    except AttributeError as e:
        print(f"AttributeError: {e}")
