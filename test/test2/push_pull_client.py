



def process_input_sockets(self,
                          input_addresses: list[str],
                          coord_input_address: Optional[str],
                          identity: bytes):
    """Input socket IO thread."""
    # Msgpack serialization decoding.
    add_request_decoder = MsgpackDecoder(EngineCoreRequest)
    generic_decoder = MsgpackDecoder()

    with ExitStack() as stack, zmq.Context() as ctx: # 创建zmq的context上下文
        # 1. 创建zmq套接字和客户端连接
        input_sockets = [
            stack.enter_context(
                make_zmq_socket(ctx,
                                input_address,
                                zmq.DEALER,
                                identity=identity,
                                bind=False))
            for input_address in input_addresses
        ]

        # Register sockets with poller.
        poller = zmq.Poller()
        ready_response = EngineCoreReadyResponse(
            max_model_len=self.vllm_config.model_config.max_model_len,
            num_gpu_blocks=self.vllm_config.cache_config.num_gpu_blocks or 0,
            dp_stats_address=self.frontend_stats_publish_address,
            dtype=str(self.vllm_config.model_config.dtype).removeprefix("torch."),
        )
        ready_payload = msgspec.msgpack.encode(ready_response)
        for input_socket in input_sockets:
            # Send initial message to each input socket - this is requ
            # before the front-end ROUTER socket can send input messag
            # back to us.
            input_socket.send(ready_payload)
            # 2. 将套接字注册到poller中，让内核关注套接字的读写事件
            poller.register(input_socket, zmq.POLLIN)
        if coord_socket is not None:
            poller.register(coord_socket, zmq.POLLIN)
        while True:
            for input_socket, _ in poller.poll():
                # (RequestType, RequestData)
                # 3. 接收来自客户端的请求数据 
                type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                request_type = EngineCoreRequestType(
                    bytes(type_frame.buffer))
                # Deserialize the request data.
                decoder = add_request_decoder if (
                    request_type
                    == EngineCoreRequestType.ADD) else generic_decoder
                # 4. 解压缩请求
                request = decoder.decode(data_frames)
                # Push to input queue for core busy loop.
                # 5. 放入到engine的待处理队列中
                self.input_queue.put_nowait((request_type, request))