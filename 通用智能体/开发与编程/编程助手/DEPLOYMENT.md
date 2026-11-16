# 本地向量服务

模型 BAAI/bge-small-zh-v1.5，MIT，512维，模型输入最多512 token。FastEmbed资料：https://qdrant.github.io/fastembed/examples/Supported_Models/

在embedding目录构建：docker build -t programming-embedding:1.0 .
使用已有nexent_network网络启动：
docker run -d --name programming-embedding --restart unless-stopped --memory 1g --memory-swap 1g --cpus 2 --network nexent_network -p 127.0.0.1:8766:8766 -v programming-embedding-models:/embedding/models programming-embedding:1.0

若平台使用其他网络，请替换网络名称。平台Base URL为http://programming-embedding:8766/v1。首次启动联网下载模型，健康接口/health就绪后再使用。保留模型缓存卷。
Dockerfile是独立复现入口，本次未重新执行干净构建。实际实例复用了Nexent镜像并用独立Linux依赖卷安装FastEmbed 0.9.1。
服务仅在宿主机127.0.0.1开放，同时对所在Docker网络可达；公网使用需另配鉴权。local仅为本地兼容占位值。
实测向量服务约193MiB内存，设置1GiB上限；不代表压力测试或性能保证。
接口支持文本或文本列表、float/base64输出；不支持token ID数组，不宣称完整兼容所有OpenAI调用。
当前知识库使用5个短章节手工分块，无需启动重型解析服务。PDF/Word上传解析尚未恢复。
