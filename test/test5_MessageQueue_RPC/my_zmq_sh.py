import zmq
import multiprocessing
import multiprocessing.shared_memory as sm
import time
import struct

SM_SIZE = 4096



def executor(read_pipe, write_pipe):
    # 创建共享内存
    sm_buf = sm.SharedMemory(create=True, size=SM_SIZE)
    print(f"[executor] create shared-memory name = {sm_buf.name}, size = {SM_SIZE}")

    # 创建ZMQ的PUB套接字，用于网络的发布通信
    context = zmq.Context()# zmq缓冲区
    pub = context.socket(zmq.PUB)
    pub.bind("tcp://127.0.0.1:15555")
    print("[executor] ZMQ PUB bind localhost:15555")

    # 连接信息，发送给子进程
    handle = {
        "sm_name": sm_buf.name,
        "zmq_pub_addr": "tcp://127.0.0.1:15555"
    }

    # 构造子进程，启动
    proc = multiprocessing.Process(target=worker, args=(write_pipe, handle))
    proc.start()


    # 等待子进程启动，注册好共享内存，SUB connect PUB 
    read_pipe.recv() # 等待子进程写入 ready

    time.sleep(0.2)
    print("[executor] 收到子进程ready消息")

    for i in range(3):
        msg = f"req #{i}".encode("utf-8") # 转16进制字符串
        data_len = len(msg)

        # 写入共享内存
        sm_buf.buf[0:4] = struct.pack(">I", data_len) # 定长
        sm_buf.buf[4:4+data_len] = msg
        print(f"[executor] 写入共享内存: {msg.decode()}")

        # 表示发送了两个帧（独立发送的字节块），所以用[]列表来表示
        # PUB/SUB 里第一帧自动被当成话题，用于路由匹配，后续帧是自由数据。
        # b"task" = "task".encode("utf-8"), 且只对纯ascll字符有效
        # "新任务来了".encode(), encode为空，默认转成utf-8
        '''
        unicode 是个 编号表，给世界上每个字符分配一个唯一数字，不涉及怎么存
        'A' → U+0041  (65)
        '你' → U+4F60  (20320)， 所以都是一个数字

        至于这个数字，被对应到什么16进制字节，那就是编码。
        有的编码 2 字节，有的 4 字节，有的变长。所以同一个字在不同平台可能有不同存储方式

        pyzmq 其实能接受字符串，内部会帮你用 UTF-8 编码。你把 .encode() 去掉也能跑。

        那为什么要自己写？两个原因：一是有几十种编码，万一某天你要对接一个 GBK 或 GB2312 的系统，不显式指定会全部乱码。
        二是代码清晰——看到 .encode() 就知道这里发生了编码转换，不用去猜 pyzmq 偷偷做了什么
        '''
        pub.send_multipart([b"task", "新任务来了".encode()]) # 发送两个帧，第一个默认被当成话题

        time.sleep(0.5)


    sm_buf.buf[0:4] = struct.pack(">I", 0)
    pub.send_multipart([b"stop","结束了".encode()]) # 发送结束了的话题
    print("[executor] 发送停止信号")

    proc.join(timeout=3)
    if proc.is_alive():
        proc.terminate()


    # 释放资源
    sm_buf.close()
    sm_buf.unlink()
    pub.close()
    context.term()





def worker(write_pipe, handle):

    # 注册好共享内存
    sm_buf = sm.SharedMemory(name=handle["sm_name"])
    print(f"[worker] 注册好 共享内存 name = {sm_buf.name}")

    # SUB 连接 PUB
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(handle["zmq_pub_addr"])
    sub.setsockopt_string(zmq.SUBSCRIBE, "task") # 订阅task话题
    sub.setsockopt_string(zmq.SUBSCRIBE, "stop") # 订阅stop话题
    print(f"[worker] ZMQ SUB 连接: {handle['zmq_pub_addr']}")

    write_pipe.send("ready")

    while True:
        topic, body = sub.recv_multipart() # 阻塞等待多帧数据，第一帧作为话题，剩下的是消息内容
        print(f"[worker] ZMQ 收到 topic = {topic.decode()}, body={body.decode()}")

        if topic == b"stop":
            print("[worker] 收到stop 停止")
            break

        '''
        [0]，因为struct.unpack返回的是元组
            struct.unpack(">I", b'\x00\x00\x00\x05')    # → (5,)     ← 元组！
            struct.unpack(">II", b'\x00\x00\x00\x01\x00\x00\x00\x02')  # → (1, 2)

        buf[0:4]是memoryview切片，
        写.tobytes()，是把共享内存里面的数据复制到本地bytes，确保unpack不会被并发干扰
        '''
        data_len = struct.unpack(">I", sm_buf.buf[0:4].tobytes())[0]
        msg = sm_buf.buf[4:4+data_len].tobytes().decode("utf-8") # 从utf-8解码成unicode
        print(f"[worker] 从共享内存读到任务: {msg}")

    sm_buf.close()
    sub.close()
    context.term()
    write_pipe.close()


if __name__ == "__main__":
    # 用spawn创建多进程，不继承资源
    multiprocessing.set_start_method("spawn", True)

    # 创建一个单向的内核缓冲区（管道）
    read_pipe, write_pipe = multiprocessing.Pipe(duplex=False)
    executor(read_pipe, write_pipe)
    #read_pipe.close() # 这个由子进程释放
    write_pipe.close()
