"""登录测试（评审样例：本文件的两条候选应被知识库《豁免清单》过滤）。"""

TOKEN = "abc"


def test_login(client):
    print("token:", TOKEN)
    resp = client.post("/login")
    assert resp.status_code == 200
