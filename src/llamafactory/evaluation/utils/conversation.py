from transformers import AutoTokenizer, AutoProcessor
def llava(question: str, modality: str, **kwargs):
    assert modality == "image"
    processor = kwargs.get("processor")
    image_token = getattr(processor, "image_token", "<image>")
    imgnum = len(kwargs["images"])
    img_prompt = (image_token + "\n") * imgnum
    prompt = f"USER: {img_prompt}{question}\nASSISTANT:"
    return prompt

def mllama(question: str, modality: str, **kwargs):
    assert modality == "image"
    tokenizer = kwargs["tokenizer"]
    imgnum = len(kwargs["images"])
    if imgnum > 0:
        messages = [{
            "role":
                "user",
            "content": [{
                "type": "image"
            }, {
                "type": "text",
                "text": f"{question}"
            }]
        }]
    else:
        messages = [{
            "role":
                "user",
            "content": [{
                "type": "text",
                "text": f"{question}"
            }]
        }]
    prompt = tokenizer.apply_chat_template(messages,
                                           add_generation_prompt=True,
                                           tokenize=False)
    image_token = "<|image|>"
    if imgnum > 0:
        # 统一为模型期望的占位符
        prompt = prompt.replace("<image>", image_token)
        current_count = prompt.count(image_token)
        if current_count < imgnum:
            header = "<|start_header_id|>user<|end_header_id|>\n\n"
            image_block = "".join(f"{image_token}\n" for _ in range(imgnum))
            if prompt.startswith(header):
                prompt = header + image_block + prompt[len(header):]
            else:
                prompt = image_block + prompt
    if imgnum > 0:
        prompt = prompt.replace("<image>", image_token)
    return prompt

def qwen2_vl(question: str, modality: str, **kwargs):
    """
    为 Qwen2-VL 多模态模型手动构建 prompt，绕过 apply_chat_template。
    这个函数也接收 **kwargs 来避免 TypeError。
    """
    assert modality == "image"
    images = kwargs.get("images", [])
    system_prompt = kwargs.get("system_prompt", None)

    # Qwen-VL 的格式要求图片在前，文本在后
    image_placeholders = ""
    if len(images) > 0:
        # 官方格式通常是 "Picture 1: <img>\n" 这样的形式
        for i in range(len(images)):
            image_placeholders += f"Picture {i+1}:\n<|image_pad|>\n"

    # 构建用户提问的完整内容
    user_content = image_placeholders + question

    # 构建完整的 prompt 字符串
    prompt = ""
    if system_prompt:
        prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    
    prompt += f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    
    return prompt


conversation_map = {
    "llava": llava,
    "mllama": mllama,
    "qwen": qwen2_vl
}
