"""
DataMate 访问适配模块。

该模块封装 DataMate HTTP 请求和数据集文件读取，供任务一、任务二服务层调用。
"""

import requests, os, subprocess, tempfile
from mcp_server.config import DATAMATE_BASE, DATAMATE_GATEWAY, DATASET_VOLUME, SUDO_PW

DM_BASE = DATAMATE_BASE
DM_GATEWAY = DATAMATE_GATEWAY

def _sudo_command(cmd):
    if SUDO_PW:
        return subprocess.run(['sudo', '-S'] + cmd, input=f'{SUDO_PW}\n', capture_output=True, text=True)
    return subprocess.run(['sudo', '-n'] + cmd, capture_output=True, text=True)

def dm_post(path: str, json_data=None, timeout=15):
    return requests.post(f'{DM_BASE}{path}', json=json_data, timeout=timeout)

def dm_get(path: str, timeout=10):
    return requests.get(f'{DM_BASE}{path}', timeout=timeout)

def gateway_post(path: str, json_data=None, timeout=15):
    return requests.post(f'{DM_GATEWAY}{path}', json=json_data, timeout=timeout)

def gateway_get(path: str, timeout=10):
    return requests.get(f'{DM_GATEWAY}{path}', timeout=timeout)

def write_temp_dataset(text: str, name: str) -> str:
    ds_dir = os.path.join(DATASET_VOLUME, name.replace(' ', '_'))
    os.makedirs(ds_dir, exist_ok=True)
    filepath = os.path.join(ds_dir, 'input.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    if DATASET_VOLUME.startswith('/home/share/'):
        _sudo_command(['chown', '-R', '1000:1000', ds_dir])
    return filepath
