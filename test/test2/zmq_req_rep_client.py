import zmq
import argparse
import uuid
import time


def echo_client():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    msg = "hello, im client"
    print("client send: " + msg)

    socket.send(msg.encode())
    reply = socket.recv()
    print("client recv: " + reply.decode())

def client_2(client_id):
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
        

    time.sleep(3)
    # 异步接收
    for _ in range(5):
        try:
            response = socket.recv_json(flags=zmq.NOBLOCK)#非阻塞接收
            print(f"Client-{client_id} 收到响应：{response}")
        except zmq.Again:
            time.sleep(0.1)
            continue

    socket.close()

if __name__ == "__main__":
    #echo_client()
    parser = argparse.ArgumentParser(description="ZMQ Echo Client")
    parser.add_argument("--id", type=int, default=0, help="客户端id")
    args = parser.parse_args()


    client_2(args.id)