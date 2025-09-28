from transformers import LlamaModel, LlamaConfig

# LLaMA3本地路径
model_path = "/llama3/model/"  

# 加载LLaMA3模型配置和权重
config = LlamaConfig.from_pretrained(model_path)
model = LlamaModel.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.float16,  
    device_map="auto"
)

for param in model.parameters():
    print(param)

# 将模型保存到本地
model.save_pretrained(f"./llama3_model")  # 修改保存的文件夹名称