import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class GPUJobWorker:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[Callable[..., Any], tuple[Any, ...], asyncio.Future[Any]]] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu_worker")
        self._worker_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._process_loop())

    async def _process_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            func, args, future = await self._queue.get()
            async with self._lock:
                try:
                    result = await loop.run_in_executor(self._executor, func, *args)
                    if not future.done():
                        future.set_result(result)
                except Exception as exc:
                    if not future.done():
                        future.set_exception(exc)
                finally:
                    self._queue.task_done()

    async def submit(self, func: Callable[..., Any], *args: Any) -> Any:
        self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        await self._queue.put((func, args, future))
        return await future


gpu_worker = GPUJobWorker()
