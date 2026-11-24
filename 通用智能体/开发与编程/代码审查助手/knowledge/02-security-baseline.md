# 安全审查基线

> 只列「一旦出现就必须拦」的点和可判定的信号。命中即 P0/P1，不要降级。

## R-SEC-01 SQL 注入

**级别**：P0

**判定**：SQL 由 f-string、`%`、`+`、`.format()` 拼接；`execute` 的参数来自请求；动态表名/排序字段未走白名单。
**反例**：`sql = f"SELECT * FROM users WHERE id={uid}"`
**修复**：参数化查询；表名/列名用枚举白名单；ORM 的 raw 查询同样适用。

## R-SEC-02 命令注入

**级别**：P0

**判定**：`os.system`、`subprocess(..., shell=True)`、反引号拼接用户输入；`eval/exec` 处理外部数据。
**反例**：`os.system("ping " + host)`
**修复**：`subprocess.run([...], shell=False)`；参数化；禁止把用户输入拼进命令。

## R-SEC-03 路径穿越

**级别**：P1

**判定**：`open(os.path.join(base, user_path))` 未规范化；`../` 可逃逸；下载接口按文件名拼路径。
**反例**：`open(os.path.join("/data", filename))`，`filename=../../etc/passwd`
**修复**：`Path.resolve()` 后校验是否仍在 base 下；文件名用白名单/UUID。

## R-SEC-04 SSRF

**级别**：P1

**判定**：服务端按用户提供的 URL 发请求；未限制内网地址、协议、重定向。
**反例**：`requests.get(user_url)` 可访问 `http://169.254.169.254/`
**修复**：协议/域名白名单；解析后校验 IP 不在内网；禁止跟随重定向到内网。

## R-SEC-05 不安全反序列化

**级别**：P0

**判定**：`pickle.loads`、`yaml.load`（无 `SafeLoader`）、`marshal`、`eval` 处理不可信数据。
**反例**：`yaml.load(request.data)`
**修复**：`json` / `yaml.safe_load`；必须 pickle 时加签名与来源白名单。

## R-SEC-06 密钥与凭据硬编码

**级别**：P0

**判定**：代码/配置/测试里出现私钥、AK、Token、口令；示例值看起来像真实凭据。
**反例**：`AWS_ACCESS_KEY_ID = "AKIA..."`
**修复**：密钥管理/环境变量；撤销已泄露凭据；加 secret scanning。

## R-SEC-07 越权（IDOR）

**级别**：P0/P1

**判定**：按资源 id 查询未带归属条件；管理接口只靠前端隐藏；批量接口可枚举。
**反例**：`Order.objects.get(id=order_id)` 未加 `user_id`
**修复**：查询条件带 `owner_id/tenant_id`；集中鉴权；越权返回 404。

## R-SEC-08 认证与令牌

**级别**：P1

**判定**：JWT 不校验 `exp/aud/iss`；算法可降级为 `none`；Token 写日志；会话无过期。
**反例**：`jwt.decode(token, options={"verify_signature": False})`
**修复**：固定算法白名单；校验过期与受众；令牌短时效 + 刷新。

## R-SEC-09 口令存储与比较

**级别**：P1

**判定**：MD5/SHA1 存口令；明文比较；`==` 比较 HMAC；盐硬编码。
**反例**：`hashlib.md5(password.encode()).hexdigest()`
**修复**：`bcrypt`/`argon2`；`hmac.compare_digest`；每用户随机盐。

## R-SEC-10 调试开关与跨域

**级别**：P2

**判定**：`DEBUG=True` 进生产；CORS `allow_origins=["*"]` 且带凭据；堆栈直接返回前端。
**反例**：`allow_origins=["*"], allow_credentials=True`
**修复**：环境区分；CORS 白名单；统一异常处理，不泄露堆栈。

## R-SEC-11 日志与响应泄露

**级别**：P2

**判定**：日志/错误响应包含 Token、口令、身份证、手机号、内部路径、SQL。
**反例**：`return {"error": str(e), "sql": sql}`
**修复**：错误码 + 请求 id；脱敏；日志分级。

## R-SEC-12 依赖与供应链

**级别**：P1

**判定**：新增来源不明的依赖；未锁版本；许可证不兼容；安装脚本执行远程代码。
**反例**：`pip install git+https://unknown/repo`
**修复**：锁定版本与哈希；审查许可证与维护活跃度；私有源白名单。
