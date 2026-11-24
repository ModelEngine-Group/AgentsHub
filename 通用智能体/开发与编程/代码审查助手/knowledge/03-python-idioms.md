# Python 常见缺陷

## R-PY-01 可变默认参数

**级别**：P1

**判定**：`def f(..., x=[])`、`x={}`、`x=set()`；默认对象在函数定义时创建，调用间共享。
**反例**：`def remember(key, value, store={}): store[key] = value`
**修复**：默认 `None`，函数体内 `store = {} if store is None else store`。

## R-PY-02 裸 except 与静默失败

**级别**：P2

**判定**：`except:`、`except Exception: pass`、`except: return None` 掩盖真实错误。
**反例**：`try: parse(x) except: pass`
**修复**：捕获具体异常；记录日志；无法处理时向上抛。

## R-PY-03 资源泄漏

**级别**：P2

**判定**：`open`、`socket`、`threading.Lock`、DB 连接未用 `with`/`finally`；生成器提前退出导致清理不执行。
**反例**：`f = open(path); return f.read()`
**修复**：`with open(path) as f:`；`contextlib.ExitStack` 管理多个资源。

## R-PY-04 事务与连接

**级别**：P1

**判定**：同一请求多个 `commit`；异常分支不 `rollback`；连接未归还连接池。
**反例**：`db.execute(...); db.commit(); send_mail(); db.execute(...); db.commit()`
**修复**：一个业务一个事务；`try/except: rollback; raise`；连接池上下文。

## R-PY-05 时间与时区

**级别**：P2

**判定**：`datetime.now()` 无时区；naive 与 aware 混用；`timestamp()` 跨时区比较。
**反例**：`datetime.now() - create_time`（create_time 是 UTC）
**修复**：统一 `datetime.now(timezone.utc)`；存储 UTC，展示时转换。

## R-PY-06 金额与浮点

**级别**：P1

**判定**：`float` 表示金额；`round()` 做金额舍入；用 `==` 比较浮点。
**反例**：`total = 0.1 + 0.2; if total == 0.3:`
**修复**：`Decimal` + 明确舍入模式；整数分存储。

## R-PY-07 闭包与延迟绑定

**级别**：P2

**判定**：循环里创建 lambda/函数，引用循环变量；回调在循环结束后执行。
**反例**：`funcs = [lambda: i for i in range(3)]`
**修复**：`lambda i=i: i` 或 `functools.partial`。

## R-PY-08 `is` 与 `==`

**级别**：P2

**判定**：用 `is` 比较字符串/数字/列表；用 `==` 比较 `None`。
**反例**：`if name is "admin":`
**修复**：值比较用 `==`，`None` 用 `is None`。

## R-PY-09 用 assert 做校验

**级别**：P1

**判定**：`assert user.is_admin` 做权限校验；`assert` 在 `-O` 下被移除。
**反例**：`assert request.user.id == owner_id`
**修复**：显式 `if not ...: raise PermissionError`。

## R-PY-10 异步中阻塞

**级别**：P1

**判定**：`async def` 里调用 `requests`、`time.sleep`、同步 DB 驱动、大文件同步读。
**反例**：`async def handler(): return requests.get(url)`
**修复**：`httpx.AsyncClient`、`asyncio.sleep`、异步驱动；CPU 密集放线程池。

## R-PY-11 遍历时修改

**级别**：P2

**判定**：`for x in items: items.remove(x)`；遍历 dict 时增删 key。
**反例**：`for k in d: del d[k]`
**修复**：遍历副本 `list(d)`；或用推导式生成新对象。

## R-PY-12 编码与文件

**级别**：P2

**判定**：`open()` 不指定 `encoding`；二进制/文本混用；Windows 路径硬编码 `\`。
**反例**：`open("data.csv").read()`
**修复**：显式 `encoding="utf-8"`；用 `pathlib.Path`；跨平台路径用 `/`。

## R-PY-13 异常链与上下文

**级别**：P2

**判定**：`raise NewError()` 丢失原始异常；`except ...: raise` 之外没有 `from`。
**反例**：`except ValueError: raise RuntimeError("bad")`
**修复**：`raise RuntimeError("bad") from e`，保留堆栈。

## R-PY-14 可变对象共享

**级别**：P2

**判定**：类属性是 list/dict；`copy.copy` 浅拷贝嵌套结构；全局缓存无失效。
**反例**：`class A: items = []`
**修复**：实例属性在 `__init__` 初始化；深拷贝或显式复制；缓存加 TTL。
