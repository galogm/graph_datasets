"""Ding Message Utils"""
from typing import List

import requests

MSGTYPE = "actionCard"


def get_card(title, text, btn_orientation, btns):
    card = {
        "title": title,
        "text": text,
        "btnOrientation": btn_orientation,
        "btns": [*btns],
    }
    return card


def send(action_card, token, msgtype=MSGTYPE):
    # pylint: disable=invalid-name
    WEBHOOK = f"https://oapi.dingtalk.com/robot/send?access_token={token}"
    print("begin")
    res = requests.post(
        WEBHOOK,
        json={
            "msgtype": msgtype,
            "actionCard": action_card,
        },
        timeout=10000,
    )
    print("done", res.json())
    return res.json()


def notify(
    token: str = "bb68fb0c27bef0f856b72b6301d024d5fa1aaacba2d6963d27d267c673dbdf8e",
    text: str = "This is a test message",
    title: str = "Come from graph_datasets",
    btn_orientation: str = "0",
    btns: List = None,
):
    """发送钉群消息

    Args:
        text (str, optional): 消息正文，支持 markdown 语法。Defaults to "这是一条上位机消息".
        title (str, optional): 消息标题，不显示在钉群消息，但需要包含 WAAE 字样。Defaults to 'WAAE 上位机消息'.
        btn_orientation (str, optional): 交互按钮位置。Defaults to '0'.
        btns (List, optional): 交互按钮设置。Defaults to None.
    """
    if token is None:
        raise ValueError("Token should not be None!")
    if not btns:
        btns = []
        # btns = [
        #     {
        #         'title': 'Google',
        #         'actionURL': 'google.com',
        #     },
        # ]
    card = get_card(
        title=title,
        text=text,
        btn_orientation=btn_orientation,
        btns=btns,
    )
    try:
        send(
            msgtype=MSGTYPE,
            token=token,
            action_card=card,
        )
    except RuntimeError:
        return "fail"

    return "success"


# for test only
# bb68fb0c27bef0f856b72b6301d024d5fa1aaacba2d6963d27d267c673dbdf8e
if __name__ == "__main__":
    notify(
        text="This is a test message",
    )
