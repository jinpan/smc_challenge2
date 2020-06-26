import contextlib
import sys
import threading
import time

import GPUtil


# Based on https://stackoverflow.com/a/14395336/1513002
# with fixes for race conditions
stdout_lock = threading.Lock()


@contextlib.contextmanager
def set_stdout_parent(parent):
  """A context manager for setting a particular parent for sys.stdout."""
  with stdout_lock:
    save_parent = sys.stdout.parent_header

    sys.stdout.parent_header = parent
    try:
        yield
    finally:
        # the flush is important, because that's when the parent_header
        # actually has its effect
        sys.stdout.flush()
        sys.stdout.parent_header = save_parent


# Asynchronously report GPU usage.  Useful for colab environment where
# we cannot easily ssh in and watch nvidia-smi.
class GPUMonitor(threading.Thread):
  def __init__(self, delay=2):
    super().__init__()

    # Used for synchronizing stdout.
    # Without, this will print into the active cell, not the cell that
    # this thread was started in.
    self.thread_parent = sys.stdout.parent_header

    self.gpu = GPUtil.getGPUs()[0]
    self.stopped = False
    self.delay = delay  # Time between calls to GPUtil
    self.start()

  def run(self):
    while not self.stopped:
      with set_stdout_parent(self.thread_parent):
        print(f"GPU load: {self.gpu.load}", end='\r')
      time.sleep(self.delay)

  def stop(self):
    self.stopped = True
