# ai-jury/app.py
import gradio as gr
from core.roles import load_all_roles
import base64
import requests
import os
from typing import List, Dict

SYSTEM_ROLES = load_all_roles()
DEFAULT_ROLE_NAMES = [r["name"] for r in SYSTEM_ROLES[:2]] if SYSTEM_ROLES else []


def call_qwen_vl_for_role(image_path: str, role_prompt: str, api_key: str, vl_model: str) -> str:
    """让 Qwen-VL 直接以角色身份评价图片"""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": vl_model,
        "input": {
            "messages": [{
                "role": "user",
                "content": [
                    {"image": image_b64},
                    {"text": role_prompt + "\n\n请直接评价这张图片。"}
                ]
            }]
        }
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["output"]["choices"][0]["message"]["content"]
    else:
        err = response.json().get("message", response.text)
        raise Exception(f"Qwen-VL 错误: {err}")


def call_qwen_text_for_role(content: str, role: dict, api_key: str, text_model: str) -> str:
    """调用文本模型进行角色化评审"""
    from dashscope import Generation
    os.environ["DASHSCOPE_API_KEY"] = api_key
    messages = [
        {"role": "system", "content": role["system_prompt"]},
        {"role": "user", "content": content}
    ]
    resp = Generation.call(model=text_model, messages=messages, result_format="message")
    if resp.status_code == 200:
        return resp.output.choices[0].message.content
    else:
        raise Exception(f"{resp.code}: {resp.message}")


def parse_and_validate_input(data):
    """解析输入，并确保文字和图片互斥"""
    text = ""
    image_path = None

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if "text" in item and item["text"].strip():
                    text = item["text"].strip()
                if "image" in item:
                    image_path = item["image"]
    elif isinstance(data, str):
        text = data.strip()

    # 🔒 强制互斥
    if text and image_path:
        raise ValueError("请勿同时输入文字和上传图片。请选择其中一种方式。")
    if not text and not image_path:
        raise ValueError("请提供文案或上传图片。")

    return text, image_path


def unified_review(
    multimodal_input,
    api_key: str,
    text_model: str,
    vl_model: str,
    selected_names: List[str],
    user_roles: List[Dict]
):
    if not api_key:
        return "❌ 请输入 DashScope API Key"
    if not selected_names:
        return "❌ 请至少选择一个评审角色"

    try:
        text_content, image_path = parse_and_validate_input(multimodal_input)
    except ValueError as e:
        return f"⚠️ 输入错误: {str(e)}"

    all_roles = SYSTEM_ROLES + user_roles
    role_map = {role["name"]: role for role in all_roles}
    selected_roles = [role_map[name] for name in selected_names if name in role_map]

    reviews = []
    for role in selected_roles:
        try:
            if image_path:
                # 🖼️ 图片模式：直接用 Qwen-VL 以角色身份评价
                review = call_qwen_vl_for_role(image_path, role["system_prompt"], api_key, vl_model)
            else:
                # 📝 文字模式：用文本模型评审
                review = call_qwen_text_for_role(text_content, role, api_key, text_model)
            reviews.append(f"### 👤 {role['name']}\n{review}")
        except Exception as e:
            reviews.append(f"### 👤 {role['name']}\n❌ 评审失败: {str(e)}")

    return "\n\n---\n\n".join(reviews)


# ===== UI 定义 =====
with gr.Blocks(title="AI 评审团") as demo:
    gr.Markdown("# 🎯 AI 评审团 —— 多角色智能评审")

    user_roles_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            role_selector = gr.Dropdown(
                choices=[r["name"] for r in SYSTEM_ROLES],
                multiselect=True,
                label="👥 选择评审角色",
                value=DEFAULT_ROLE_NAMES
            )

            with gr.Accordion("➕ 创建临时角色", open=False):
                new_name = gr.Textbox(label="角色名称")
                new_desc = gr.Textbox(label="角色简介（可选）")
                new_prompt = gr.TextArea(label="角色提示词（必填）", lines=5)
                save_btn = gr.Button("💾 添加到当前会话")
                status = gr.Textbox(label="状态", interactive=False)

                def add_temp_role(name, desc, prompt, current_user_roles):
                    name, prompt = name.strip(), prompt.strip()
                    if not name or not prompt:
                        return "⚠️ 名称和提示词不能为空", current_user_roles, gr.update()
                    new_role = {"name": name, "description": desc.strip() or "临时角色", "system_prompt": prompt}
                    updated = current_user_roles + [new_role]
                    all_names = [r["name"] for r in SYSTEM_ROLES] + [r["name"] for r in updated]
                    return f"✅ 角色 '{name}' 已添加！", updated, gr.update(choices=all_names)

                save_btn.click(
                    fn=add_temp_role,
                    inputs=[new_name, new_desc, new_prompt, user_roles_state],
                    outputs=[status, user_roles_state, role_selector]
                )

        with gr.Column(scale=2):
            # ✨ 关键：使用 MultimodalTextbox 实现你要的效果
            content_input = gr.MultimodalTextbox(
                file_types=["image"],
                placeholder="上传你的评审文字或点击附件上传图片",
                label="📝 内容输入",
                show_label=True
            )

            with gr.Accordion("⚙️ 模型设置", open=False):
                api_key = gr.Textbox(type="password", label="🔑 DashScope API Key")
                text_model = gr.Dropdown(["qwen-turbo", "qwen-plus", "qwen-max"], value="qwen-turbo", label="🧠 文本模型")
                vl_model = gr.Dropdown(["qwen-vl-plus", "qwen-vl-max"], value="qwen-vl-plus", label="👁️ 图像模型")

            output = gr.Markdown()
            btn = gr.Button("🚀 开始评审")

            btn.click(
                fn=unified_review,
                inputs=[content_input, api_key, text_model, vl_model, role_selector, user_roles_state],
                outputs=output
            )

    gr.Markdown("""
    💡 **说明**：
    - ⚠️ **不支持同时输入文字和图片**（系统会提示错误）
    - 图片评审：由 **Qwen-VL 直接以角色身份评价**（非先描述）
    - 所有数据仅在内存处理，不保存
    - 获取 API Key: [DashScope 控制台](https://dashscope.console.aliyun.com/apiKey)
    """)

if __name__ == "__main__":
    demo.launch()