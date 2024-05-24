from transformers import AutoTokenizer, AutoModel


# MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
MODEL_PATH = '/Users/bytedance/projects/chatglm/model/chatglm3-6b'
# TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
TOKENIZER_PATH = '/Users/bytedance/projects/chatglm/model/chatglm3-6b'
DEFAULT_SYSTEM_PROMPT = '''
You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.
'''.strip()


# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="mps").eval()

model = model.eval()

response, history = model.chat(tokenizer, "你好", history=[])

response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)