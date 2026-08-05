import threading
import sys
import time
import zmq
import argparse
import uuid
from concurrent.futures import ThreadPoolExecutor



# poll多路监听+每路线程池并行处理
def server_poll_with_workers():
    context = zmq.Context.instance()
    frontend = context.socket(zmq.ROUTER) #多对一并行接收的套接字
    frontend.bind("tcp://localhost:6666")

    print("server start: 6666")

    # 定义模拟推理处理数据的过程
    def process_inference(data):
        time.sleep(0.5)
        return {
            "status": "success",
            "result": f"服务器收到消息:{data['query']}",
            "timestamp": time.time(),
        }

    poller = zmq.Poller()
    poller.register(frontend, zmq.POLLIN)

    executor = ThreadPoolExecutor(max_workers=4) #4个线程的线程池
    lock = threading.Lock()

    # 线程池的workers的工作函数
    def handle_request(identity, request):
        result = process_inference(request)

        # 构造回复response json
        response = {
            "msg_id": request['msg_id'],
            "reply": result,
            "from_engine": "server-0",
        }
        with lock:
            # 走端口返回，因为是4个worker来访问frontend这个套接字，所以需要加锁
            frontend.send_multipart([identity, zmq.utils.jsonapi.dumps(response)])


    try:
        while 1:
            socks = dict(poller.poll(1000))
            multipart = []
            if frontend in socks:
                multipart = frontend.recv_multipart()
                if not multipart:
                    continue

                identity = multipart[0]
                message = multipart[-1]

                request = zmq.utils.jsonapi.loads(message)
                print(f"server recv: {identity.decode()} request:{request['msg_id']}")

                # 现在改成向线程池提交异步任务，4个worker并行处理
                executor.submit(handle_request, identity, request)

                ''' 原来的串行处理
                result = process_inference(request)

                # 构造回复response json
                response = {
                    "msg_id": request['msg_id'],
                    "reply": result,
                    "from_engine": "server-0",
                }
                frontend.send_multipart([identity, zmq.utils.jsonapi.dumps(response)])
                '''
    except KeyboardInterrupt:
        print("server close")

    finally:
        frontend.close()





def server_poll():
    context = zmq.Context.instance()
    frontend = context.socket(zmq.ROUTER) #多对一并行接收的套接字
    frontend.bind("tcp://localhost:6666")

    print("server start: 6666")

    # 定义模拟推理处理数据的过程
    def process_inference(data):
        time.sleep(0.5)
        return {
            "status": "success",
            "result": f"服务器收到消息:{data['query']}",
            "timestamp": time.time(),
        }

    poller = zmq.Poller()
    poller.register(frontend, zmq.POLLIN)


    try:
        while 1:
            socks = dict(poller.poll(1000))
            multipart = []
            if frontend in socks:
                multipart = frontend.recv_multipart()
                if not multipart:
                    continue

                identity = multipart[0]
                message = multipart[-1]

                request = zmq.utils.jsonapi.loads(message)
                print(f"server recv: {identity.decode()} request:{request['msg_id']}")

                result = process_inference(request)

                # 构造回复response json
                response = {
                    "msg_id": request['msg_id'],
                    "reply": result,
                    "from_engine": "server-0",
                }
                frontend.send_multipart([identity, zmq.utils.jsonapi.dumps(response)])

    except KeyboardInterrupt:
        print("server close")

    finally:
        frontend.close()



def server():
    context = zmq.Context.instance()
    frontend = context.socket(zmq.ROUTER) #多对一并行接收的套接字
    frontend.bind("tcp://localhost:6666")

    print("server start: 6666")

    # 定义模拟推理处理数据的过程
    def process_inference(data):
        time.sleep(0.5)
        return {
            "status": "success",
            "result": f"服务器收到消息:{data['query']}",
            "timestamp": time.time(),
        }

    try:
        while 1:
            multipart = frontend.recv_multipart()
            if not multipart:
                continue

            identity = multipart[0]
            message = multipart[-1]

            request = zmq.utils.jsonapi.loads(message)
            print(f"server recv: {identity.decode()} request:{request['msg_id']}")

            result = process_inference(request)

            # 构造回复response json
            response = {
                "msg_id": request['msg_id'],
                "reply": result,
                "from_engine": "server-0",
            }
            frontend.send_multipart([identity, zmq.utils.jsonapi.dumps(response)])

    except KeyboardInterrupt:
        print("server close")

    finally:
        frontend.close()




def client(client_id):
    context = zmq.Context.instance()
    socket = context.socket(zmq.DEALER) # 异步双向套接字

    #id编码成字节流
    identity = f"Client-{client_id}".encode("utf-8")
    socket.setsockopt(zmq.IDENTITY, identity)

    socket.connect("tcp://localhost:6666") #连接6666端口服务
    print(f"client: {client_id} start, 连接到6666server")

    #开始用异步双向socket发送多个请求测试
    for i in range(5):
        request = {
            # UUID = Universally Unique Identifier（通用唯一标识符），类似消息的md5
            "msg_id": str(uuid.uuid4()),
            "query": f"请求{i} 来自 Client-{client_id}",
            "timestamp": time.time(),
        }
        socket.send_json(request)
        print(f"client-{client_id} 发送请求 {request['msg_id']}")
        

    
    # 异步阻塞接收
    for _ in range(5):
        response = socket.recv_json()  # 阻塞接收
        print(f"Client-{client_id} 收到响应：{response}")


    socket.close()




if __name__ == "__main__":
    t_start = time.time()

    engine_thread = threading.Thread(target=server_poll_with_workers,daemon=True)
    engine_thread.start()

    time.sleep(1)
    print(f"[{time.time() - t_start:.3f}s] 服务端启动完成")

    client_threads = []
    for i in range(3):
        t = threading.Thread(target = client, args=(i,),daemon=True)
        t.start()
        client_threads.append(t)

    print(f"[{time.time() - t_start:.3f}s] 3 个客户端已启动")

    try:
        for t in client_threads:
            t.join()
        print(f"[{time.time() - t_start:.3f}s] 所有客户端完成")
        time.sleep(2)
    except KeyboardInterrupt:
        print("over")

    t_end = time.time()
    print(f"======================================")
    print(f"  总耗时: {t_end - t_start:.3f}s")
    print(f"======================================")
