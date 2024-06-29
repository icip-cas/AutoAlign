"""utils"""
import json


def read_json(data_path):
    """read json data"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(data, save_path):
    """save json data"""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
