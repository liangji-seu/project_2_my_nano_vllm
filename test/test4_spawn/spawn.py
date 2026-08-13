import multiprocessing as mp

def f(name):
    print(name)
    print(f"child_get: {id(name)}")
    name = "456"
    print(name)
    print(f"child_write: {id(name)}")
    







if __name__ == "__main__":
    mp.set_start_method("fork")
    name = "123"
    print(f"parent write name = {name}")
    p = mp.Process(target=f, args=(name,))
    p.start()
    print(f"parent: {id(name)}")
    print(f"parent write name = {name}")
