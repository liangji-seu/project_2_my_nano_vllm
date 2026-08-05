import zmq
import time
def generate_tokens():
    for token in ["hello", "-world", ",", "-how"]:
        yield token
        time.sleep(0.1)

def main():
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.connect("tcp://localhost:7777")

    with context,sender:
        print("push: sending tokens...")

        for token in generate_tokens():
            sender.send_string(token)

        sender.send_string("END")
        print("PUSH: all tokens sent")

    