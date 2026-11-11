"""Run a native DIRECT physics branch in its own copy of an untouched Linux process."""

import os
import pickle
import traceback


def isolated(call, *args):
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            with os.fdopen(write_fd, "wb") as sink:
                try:
                    value = (True, call(*args))
                except BaseException:
                    value = (False, traceback.format_exc())
                pickle.dump(value, sink, protocol=5)
        finally:
            os._exit(0)
    os.close(write_fd)
    try:
        with os.fdopen(read_fd, "rb") as source:
            valid, value = pickle.load(source)
    finally:
        _, status = os.waitpid(pid, 0)
    if status or not valid:
        raise RuntimeError("Physics child failed: " + str(value))
    return value
