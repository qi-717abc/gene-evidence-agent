"""Whole JSON only, optionally wrapped in one exact JSON code fence."""
import json
import re


def strip_json_fence(text):
    text=text.strip()
    match=re.fullmatch(r'```(?:json)?[ \t]*\r?\n([\s\S]+)\r?\n```',text,re.I)
    return match.group(1).strip() if match else text


def load_whole_json(text):
    return json.loads(strip_json_fence(text))


def extract_single_json_object(text):
    """Read one complete JSON object from model text without trusting surrounding prose."""
    text = strip_json_fence(text).lstrip("\ufeff")
    objects = []
    start = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if depth == 0:
            if char == "{":
                start = index
                depth = 1
                in_string = False
                escaped = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                value = json.loads(text[start:index + 1])
                if isinstance(value, dict):
                    objects.append(value)
                start = None
    if depth or len(objects) != 1:
        raise json.JSONDecodeError("expected one complete JSON object", text, len(text))
    return objects[0]
