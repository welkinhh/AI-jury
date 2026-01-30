# system_roles/__init__.py
import yaml
import os

def load_all_roles():
    """从 all_roles.yaml 加载所有角色"""
    file_path = os.path.join(os.path.dirname(__file__), "all_roles.yaml")
    with open(file_path, "r", encoding="utf-8") as f:
        roles = yaml.safe_load(f)
    return roles

# 将函数暴露给外部
__all__ = ["load_all_roles"]