# 服务器部署脚本

这些脚本默认使用 `/opt/chroniccare` 作为服务器部署目录。可通过
`CHRONICCARE_DEPLOY_ROOT` 指定任意可写目录。

脚本说明：

- `prepare_server_dirs.sh`：创建部署目录结构
- `deploy_release_to_server.sh`：把已清理的交付包复制到服务器项目目录
- `start_chroniccare.sh`：启动 `docker-compose.server.yml`
- `stop_chroniccare.sh`：停止服务
- `check_chroniccare.sh`：检查 `18088` 和 `18501` 入口
