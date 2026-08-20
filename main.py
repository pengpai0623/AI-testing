import asyncio
import threading

# 1. 这是进程（运行这个脚本本身）


async def my_coroutine(name):
    # 3. 协程执行在这里 —— 它依附于下面的主线程
    print(f"协程 {name} 正在运行，所在的线程ID: {threading.get_ident()}")
    await asyncio.sleep(1)
    return "完成"


async def main():
    # 2. 这是进程里的主线程，启动了数以万计的协程
    # 这里只有一个线程，但创建了 5 个协程“影分身”
    tasks = [my_coroutine(i) for i in range(5)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    # 启动：1个进程 -> 1个主线程 -> 异步运行多个协程
    asyncio.run(main())
