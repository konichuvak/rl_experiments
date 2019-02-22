from queue import Queue
from threading import Thread
import _pickle
import sys
import os


class DevNull(object):
    def write(self, arg):
        pass


class NoPrint:

    def __init__(self):
        self._stdout = sys.stdout

    def __enter__(self):
        sys.stdout = DevNull()
        return

    def __exit__(self, *args):
        sys.stdout = self._stdout


def threaded(f, daemon=False):
    def wrapped_f(q, *args, **kwargs):
        ret = f(*args, **kwargs)
        q.put(ret)

    def wrap(*args, **kwargs):
        q = Queue()
        t = Thread(target=wrapped_f, args=(q,) + args, kwargs=kwargs)
        t.daemon = daemon
        t.start()
        t.result_queue = q
        return t

    return wrap


def cache(cachedir=f'/RL/codes/cached_graphs'):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            print(f'caching {func.__name__} at {cachedir}')
            filename = f'{cachedir}/{func.__name__}_{args}.pkl' if args else f'{cachedir}/{func.__name__}.pkl'
            if os.path.exists(filename):
                print(f'function {func.__name__} with arguments {args} is already cached')
                with open(filename, 'rb') as f:
                    return _pickle.load(f)

            result = func(self, *args, **kwargs)
            with open(filename, 'wb') as f:
                _pickle.dump(result, f)
            return result

        return wrapper

    return decorator



