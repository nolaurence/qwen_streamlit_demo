"""
This script is a simple web demo based on Streamlit, showcasing the use of the ChatGLM3-6B model. For a more comprehensive web demo,
it is recommended to use 'composite_demo'.

Usage:
- Run the script using Streamlit: `streamlit run web_demo_streamlit.py`
- Adjust the model parameters from the sidebar.
- Enter questions in the chat input box and interact with the ChatGLM3-6B model.

Note: Ensure 'streamlit' and 'transformers' libraries are installed and the required model checkpoints are available.
"""
import torch
import warnings
# from transformers import AutoModel, PreTrainedTokenizer
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput
from typing import List, Tuple
from copy import deepcopy
import os
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple
from modelscope import AutoModelForCausalLM, AutoTokenizer as MAutoTokenizer

BASE_PATH = '/Users/bytedance/projects/chatglm/model'
# MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
MODEL_PATH = os.path.join(BASE_PATH, 'qwen1.5-7b')
# TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
TOKENIZER_PATH = os.path.join(BASE_PATH, 'qwen1.5-7b')
DEFAULT_SYSTEM_PROMPT = '''
You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.
'''.strip()
DEVICE = "mps"

# 一些辅助类和函数
class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores
    
def process_response(self, output, history):
    content = ""
    history = deepcopy(history)
    for response in output.split("<|assistant|>"):
        metadata, content = response.split("\n", maxsplit=1)
        if not metadata.strip():
            content = content.strip()
            history.append({"role": "assistant", "metadata": metadata, "content": content})
            content = content.replace("[[训练时间]]", "2023年")
        else:
            history.append({"role": "assistant", "metadata": metadata, "content": content})
            if history[0]["role"] == "system" and "tools" in history[0]:
                content = "\n".join(content.split("\n")[1:-1])
                def tool_call(**kwargs):
                    return kwargs
                parameters = eval(content)
                content = {"name": metadata.strip(), "parameters": parameters}
            else:
                content = {"name": metadata.strip(), "content": content}
    return content, history


st.set_page_config(
    page_title="ChatGLM3-6B Streamlit Simple Demo",
    page_icon=":robot:",
    layout="wide"
)

torch.cuda.memory_allocated()


@st.cache_resource
def get_model():

    tokenizer = MAutoTokenizer.from_pretrained(TOKENIZER_PATH, device_map=DEVICE)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map=DEVICE).eval().to(DEVICE)
    return tokenizer, model


# 加载Chatglm3的model和tokenizer
tokenizer, model = get_model()

if "history" not in st.session_state:
    st.session_state.history = []
if "past_key_values" not in st.session_state:
    st.session_state.past_key_values = None

max_length = st.sidebar.slider("max_length", 0, 32768, 8192, step=1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.6, step=0.01)
system_prompt = st.text_area(
        label="System Prompt (Only for chat mode)",
        height=300,
        value=DEFAULT_SYSTEM_PROMPT,
    )

buttonClean = st.sidebar.button("清理会话历史", key="clean")
if buttonClean:
    st.session_state.history = []
    st.session_state.past_key_values = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.rerun()

for i, message in enumerate(st.session_state.history):
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            st.markdown(message["content"])
    else:
        with st.chat_message(name="assistant", avatar="assistant"):
            st.markdown(message["content"])

with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty()

prompt_text = st.chat_input("请输入您的问题")
if prompt_text:
    input_placeholder.markdown(prompt_text)
    history = st.session_state.history
    past_key_values = st.session_state.past_key_values

    # TODO tokenization
    if history is None:
        history: List[Tuple[str, str]] = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

    
    # 组装历史对话和本次对话
    history.append({"role": "user", "content": prompt_text})

    past = None

    logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())

    # 不知道干啥的变量
    eos_token_id = [tokenizer.eos_token_id]
    gen_kwargs = {"max_length": max_length, "do_sample": True, "eos_token_id": eos_token_id,
                "top_p": top_p, "temperature": temperature, "logits_processor": logits_processor}

    # tokenization
    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids

    # ==========  stream generate realm  ==========

    batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[1]
    generation_config = model.generation_config
    model_kwargs = generation_config.update(**gen_kwargs)
    model_kwargs["use_cache"] = generation_config.use_cache
    bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(DEVICE) if eos_token_id is not None else None

    has_default_max_length = model_kwargs.get("max_length") is None and generation_config.max_length is not None

    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
            "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "decoder_input_ids" if model.config.is_encoder_decoder else "input_ids"
        warnings.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`."
        )

    logits_processor = model_kwargs.get("logits_processor")
    stopping_criteria = StoppingCriteriaList()

    # 从类方法拆出来，直接调用其私有方法
    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=None,
        logits_processor=logits_processor,
    )
    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None

    # generation
    with torch.no_grad():
        while True:
            transformer_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # DL model前向传播
            outputs = model(
                **transformer_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            print(outputs)

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # 采样
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)

            if generation_config.do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            print(next_tokens)
            
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            # response = tokenizer.batch_decode(input_ids.tolist()[0][input_ids_seq_length:-1])
            # 通义qianwen编码
            response = tokenizer.batch_decode(input_ids[:, input_ids_seq_length-1:], skip_special_tokens=True)[0]
            if response and response[-1] != "�":
                response, history = process_response(response, history)

            message_placeholder.markdown(response)
            
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break
    # 展示到streamlit面板上

    st.session_state.history = history
    # st.session_state.past_key_values = past_key_values
