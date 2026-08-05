import zmq
import time
import thread


def echo_server():
    # 创建context对象，构造线程池，里面包含各种你需要的socket
    context = zmq.Context() 

    '''
    通信模式支持：REQ/REP, PUB/SUB, PUSH/PULL
    我们这里配置的socket套接字，是表示回复端，必须先收再发
    '''
    socket = context.socket(zmq.REP)

    # bind表示监听，等待连接
    # tcp://表示传输协议，ZMQ支持tcp,ipc,inproc,pgm/epgm
    # *表示监听所有ip地址（本机，局域网，外网）
    # 5555端口号
    socket.bind("tcp://localhost:5555")

    print("Echo 服务端启动，等待客户端连接")

    while True:
        message = socket.recv() #阻塞等待
        print(f"服务器收到：{message.decode()}")

        socket.send(message)


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
            "result": f"服务器收到消息：{data['query']}",
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




if __name__ == "__main__":




    try:
        #echo_server()
        server()
    except KeyboardInterrupt:
        print("server close")