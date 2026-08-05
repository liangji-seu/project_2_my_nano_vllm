
'''
基础使用：路径参数，查询参数
'''
from fastapi import FastAPI

app = FastAPI()

# 定义根路径的GET路由
@app.get("/")
async def root():
    return {"message": "hello world"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}






'''
post新建，put完整更新指令 + 自定义数据
'''
from pydantic import BaseModel



# 定义请求体的数据模型
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


@app.post("/items/")
async def create_item(item: Item):
    return item # 创建一个新的对象

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"item_id": item_id, "item_name": item.name}



'''
请求头+ Cookie
'''
from fastapi import Header, Cookie, Request

@app.get("/items/header/")
def test_head(user_agent: str = Header(None), session_token: str = Cookie(None)):
    return {"User-Agent": user_agent, "Session-Token": session_token}

@app.get("/items/header/all")
def all_headers(request: Request):
    return dict(request.headers)  # 返回全部请求头








"""
══════════════════════════════════════════
示例 ①：流式响应（SSE）— vLLM 逐 token 返回的核心
══════════════════════════════════════════
"""
import asyncio
from fastapi.responses import StreamingResponse


@app.post("/chat/stream")
async def chat_stream(prompt: str):
    """模拟 vLLM 的流式输出：逐 token 推给客户端"""

    async def generate():
        tokens = ["你", "好", "，", "我是", "AI", "，", prompt]
        for token in tokens:
            yield f"data: {token}\n\n"  # SSE 格式（Server-Sent Events）
            await asyncio.sleep(0.3)    # 模拟推理耗时
        yield "data: [DONE]\n\n"         # 结束标记



    '''
    可迭代对象：
        能for x in obj 的就是，列表，字符串，文件都是

    yield
        函数里面的return是一次性全给，yield是一次给一个，暂停，等你取走，继续


                def normal():
                    return [1, 2, 3]     # 一次性全给，函数结束

                def gen():
                    yield 1               # 给一个，暂停
                    yield 2               # 再给一个，暂停
                    yield 3               # 再给一个，结束

                # 用它
                for num in gen():         # for 循环逐个取
                    print(num)            # 1 → 2 → 3，每次循环取一个 yield

        内存上利差别大：return 100万个 token 先攒好；yield 每生成一个就立刻给出去，不攒。


    
        
    async for 
            普通for 在取下一个的时候必须阻塞， async for可以停下等待去处理其他请求
            async for, 以及async def generate，都可以看成是协程任务了，这样生成器不会阻塞，主线程迭代for也不会阻塞

    '''

    # 第一个参数必须是能够被迭代的对象。
    # generate()是一个异步生成器，也就是async def + yield
    return StreamingResponse(generate(), media_type="text/event-stream")




"""
══════════════════════════════════════════
示例 ②：lifespan — vLLM 的方式：启动时创建引擎，关闭时清理
══════════════════════════════════════════
"""
from contextlib import asynccontextmanager


# 用 lifespan 替代直接 FastAPI()，vLLM 就是这样做的（api_server.py:173-179）

@asynccontextmanager # 让yield从生成器切换成切分点
async def lifespan(app: FastAPI):


    '''
    启动程序 = async with 的__aenter__，  就是构造函数
    '''

    """模拟 vLLM 的 lifespan：启动建引擎 → 运行 → 关闭清理"""
    print("[lifespan] 启动中... 初始化 AsyncLLM（模拟）")
    await asyncio.sleep(1)
    app.state.engine = {"status": "ready"}    # 挂到 app.state，handler 能取
    print("[lifespan] 引擎就绪，开始对外服务")



    # 切分点
    yield                                      # ← 这里暂停，服务运行中


    '''
    结束程序 = async with 的__aexit__， 就是析构函数
    '''
    print("[lifespan] 关闭中... 清理引擎")
    await asyncio.sleep(0.5)
    print("[lifespan] 已关闭")


# 注意：用了 lifespan 后之前的 app 要改成这样，这里单独建一个演示用

# 这是一个带构造和析构逻辑的一个新的app， lifespan就是指定构造析构函数
app_with_lifespan = FastAPI(lifespan=lifespan)


@app_with_lifespan.get("/engine/status")
async def engine_status(request: Request):
    """handler 通过 app.state 取引擎状态"""
    return request.app.state.engine


# 测试: uvicorn api_server:app_with_lifespan --port 8001
#       curl http://localhost:8001/engine/status
#       按 Ctrl+C 看 lifespan 的 shutdown 日志

