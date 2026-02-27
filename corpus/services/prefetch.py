import threading
import queue


class PrefetchBatchGenerator:
    """Wraps a BatchGenerator with a background prefetch thread.

    A worker thread continuously calls generate_batch() and fills a queue.
    The main thread pops ready batches, so GPU training never waits on I/O.
    """

    def __init__(self, batch_gen, queue_depth=3):
        self._batch_gen = batch_gen
        self._queue = queue.Queue(maxsize=queue_depth)
        self._stop_event = threading.Event()
        self._thread = None
        self._batch_size = None

    # -- proxied attributes ------------------------------------------------

    @property
    def vocabulary(self):
        return self._batch_gen.vocabulary

    @property
    def song_paths(self):
        return self._batch_gen.song_paths

    @property
    def fs(self):
        return self._batch_gen.fs

    @property
    def snippet_ticks(self):
        return self._batch_gen.snippet_ticks

    @snippet_ticks.setter
    def snippet_ticks(self, value):
        if value != self._batch_gen.snippet_ticks:
            self._batch_gen.snippet_ticks = value
            self._drain_queue()

    # -- public API --------------------------------------------------------

    def generate_batch(self, batch_size):
        """Return the next prefetched batch, starting the worker if needed."""
        if self._thread is None or self._batch_size != batch_size:
            self._start_worker(batch_size)
        try:
            return self._queue.get(timeout=60)
        except queue.Empty:
            return self._batch_gen.generate_batch(batch_size)

    def close(self):
        """Stop the background worker and drain remaining batches."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._drain_queue()

    # -- context manager ---------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- internals ---------------------------------------------------------

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                batch = self._batch_gen.generate_batch(self._batch_size)
                # Use timeout so we periodically check the stop event
                self._queue.put(batch, timeout=0.5)
            except queue.Full:
                continue
            except Exception:
                if self._stop_event.is_set():
                    break
                raise

    def _start_worker(self, batch_size):
        if self._thread is not None:
            self.close()
            self._stop_event.clear()
        self._batch_size = batch_size
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _drain_queue(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
