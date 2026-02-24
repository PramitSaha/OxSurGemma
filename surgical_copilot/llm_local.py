"""
Local LLM wrapper for surgical co-pilot.
Loads models like MedGemma (google/medgemma-4b-it) from Hugging Face for offline inference.
Requires: transformers>=4.50.0, accelerate, torch
MedGemma is gated: accept terms at https://huggingface.co/google/medgemma-4b-it, then log in via
  huggingface-cli login   or  export HF_TOKEN=hf_xxx
"""
import json
import os
import re
from typing import Any, List, Optional, Sequence, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr


def _format_tools_for_prompt(tools: Sequence[BaseTool]) -> str:
    """Format tool names and descriptions for injection into the system prompt."""
    parts = ["Available tools (use when needed):"]
    for t in tools:
        desc = (t.description or "").strip() or "(no description)"
        parts.append(f"- {t.name}: {desc}")
    parts.append(
        'To call a tool, respond with a JSON object: {"name": "<tool_name>", "args": {...}}. '
        "For simple questions, call one tool. For questions needing multiple aspects (e.g. phase + instruments, segment + instruments), call tools in sequence—you can make another tool call after seeing the first result. Synthesize results into one answer. Otherwise respond with normal text."
    )
    return "\n".join(parts)


def _parse_tool_calls_from_content(content: str, tool_names: List[str]) -> List[dict]:
    """Try to extract tool calls from model output. Returns list of {name, args, id}."""
    if not content or not tool_names:
        return []
    tool_calls = []
    # Try to find JSON object(s) with "name" and "args" (allow nested braces in args)
    i = 0
    while i < len(content):
        start = content.find('{"name"', i)
        if start == -1:
            start = content.find("{'name'", i)
        if start == -1:
            break
        depth = 0
        end = -1
        for j in range(start, len(content)):
            if content[j] == "{":
                depth += 1
            elif content[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end == -1:
            break
        try:
            obj = json.loads(content[start:end])
            name = obj.get("name") if isinstance(obj.get("name"), str) else None
            args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
            if name and name in tool_names:
                tool_calls.append({"name": name, "args": args, "id": f"call_{len(tool_calls)}"})
        except json.JSONDecodeError:
            pass
        i = end
    return tool_calls


def _messages_to_medgemma_format(messages: List[BaseMessage]) -> List[dict]:
    """Convert LangChain messages to MedGemma chat format (text-only).
    Merges consecutive ToolMessages into one user message to satisfy user/assistant alternation."""
    result = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, SystemMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            result.append({"role": "system", "content": [{"type": "text", "text": content}]})
        elif isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            result.append({"role": "user", "content": [{"type": "text", "text": content}]})
        elif isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content) if msg.content else ""
            result.append({"role": "assistant", "content": [{"type": "text", "text": content}]})
        else:
            # ToolMessage(s) - merge consecutive ones into one user message
            parts = []
            while i < len(messages) and not isinstance(messages[i], (HumanMessage, AIMessage, SystemMessage)):
                m = messages[i]
                content = getattr(m, "content", str(m))
                parts.append(f"[Tool result]: {content}")
                i += 1
            result.append({"role": "user", "content": [{"type": "text", "text": "\n\n".join(parts)}]})
            i -= 1  # compensate for outer loop increment
        i += 1
    return result


class MedGemmaChatModel(BaseChatModel):
    """
    LangChain ChatModel wrapper for MedGemma (and similar image-text-to-text models).
    Uses text-only mode for agent chat; images are not passed to the LLM in this flow.
    """

    model_id: str = Field(default="google/medgemma-4b-it", description="Hugging Face model ID")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=256, ge=1, le=8192)  # Reduced from 1024 for speed
    device_map: str = Field(default="auto", description="Device map for model loading")
    torch_dtype: Optional[str] = Field(default="bfloat16", description="torch dtype: bfloat16, float16, float32")
    cache_dir: Optional[str] = Field(default=None, description="HF cache dir for downloads (avoids home quota)")
    hf_token: Optional[str] = Field(default=None, description="HF token for gated models (or set HF_TOKEN env)")
    use_4bit: bool = Field(default=True, description="Use 4-bit quantization for 2-3x speedup")
    lora_adapter_path: Optional[str] = Field(default=None, description="Path to tool-use LoRA adapter (e.g. tool_use_lora_checkpoints/medgemma-4b-tool-use-lora)")
    # TODO: Install flash-attn on the cluster
    use_flash_attention: bool = Field(default=False, description="Use Flash Attention 2 for 2-3x speedup (requires flash-attn package)")
    use_torch_compile: bool = Field(default=True, description="Use torch.compile for 20-40% speedup after warmup")

    _model: Any = PrivateAttr(default=None)
    _processor: Any = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        cache_kw = {"cache_dir": self.cache_dir} if self.cache_dir else {}
        token = self.hf_token or os.environ.get("HF_TOKEN")
        if token:
            cache_kw["token"] = token
        
        print(f"[LLM] Loading {self.model_id} (4-bit={self.use_4bit}, flash_attn={self.use_flash_attention})...", flush=True)
        self._processor = AutoProcessor.from_pretrained(self.model_id, **cache_kw)
        
        # 4-bit quantization config for 2-3x speedup
        load_kwargs = {**cache_kw, "device_map": self.device_map}
        
        # Flash Attention 2 for 2-3x faster attention (if available)
        if self.use_flash_attention:
            try:
                load_kwargs["attn_implementation"] = "flash_attention_2"
                print("[LLM] Using Flash Attention 2 for speed", flush=True)
            except Exception:
                print("[LLM] Flash Attention 2 not available, using default", flush=True)
        
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            load_kwargs["quantization_config"] = quantization_config
            print("[LLM] Using 4-bit quantization for speed", flush=True)
        else:
            dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
            dtype = dtype_map.get(self.torch_dtype, torch.bfloat16)
            load_kwargs["torch_dtype"] = dtype
        
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            **load_kwargs,
        )
        # Load tool-use LoRA adapter if path provided (improves tool-calling behavior)
        if self.lora_adapter_path and os.path.isdir(self.lora_adapter_path):
            adapter_config = os.path.join(self.lora_adapter_path, "adapter_config.json")
            if os.path.isfile(adapter_config):
                from peft import PeftModel
                print(f"[LLM] Loading tool-use LoRA adapter: {self.lora_adapter_path}", flush=True)
                self._model = PeftModel.from_pretrained(self._model, self.lora_adapter_path, is_trainable=False)
                print("[LLM] LoRA adapter loaded", flush=True)
        self._model.eval()
        
        # torch.compile for 20-40% speedup (after first few runs)
        if self.use_torch_compile:
            try:
                import torch
                if hasattr(torch, 'compile'):
                    print("[LLM] Compiling model with torch.compile (first run will be slow)...", flush=True)
                    self._model = torch.compile(self._model, mode="reduce-overhead")
                    print("[LLM] Model compiled successfully", flush=True)
            except Exception as e:
                print(f"[LLM] torch.compile failed: {e}, using uncompiled model", flush=True)
        
        print("[LLM] Model loaded successfully", flush=True)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import torch

        hf_messages = _messages_to_medgemma_format(messages)
        inputs = self._processor.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Move to model device
        device = next(self._model.parameters()).device
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]
        
        # Limit max tokens for agent responses (shorter = faster)
        max_tokens = min(kwargs.get("max_new_tokens", self.max_new_tokens), 256)

        with torch.inference_mode():
            generation = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                top_p=0.95,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                use_cache=True,  # Enable KV cache for faster generation
                num_beams=1,  # Greedy decoding is faster than beam search
                early_stopping=True,  # Stop when EOS is generated
                repetition_penalty=1.1,  # Prevent repetition, generates faster
            )
        generation = generation[0][input_len:]
        decoded = self._processor.decode(generation, skip_special_tokens=True)

        message = AIMessage(content=decoded.strip())
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(
        self,
        tools: Sequence[Union[BaseTool, type, callable]],
        **kwargs: Any,
    ) -> "MedGemmaWithTools":
        """Bind tools for prompt-based tool use (no native tool calling)."""
        lc_tools = []
        for t in tools:
            if isinstance(t, BaseTool):
                lc_tools.append(t)
            else:
                try:
                    from langchain_core.tools import convert_to_langchain_tool
                    lc_tools.append(convert_to_langchain_tool(t))
                except Exception:
                    pass
        return MedGemmaWithTools(model=self, tools=lc_tools)

    @property
    def _llm_type(self) -> str:
        return "medgemma_local"


class MedGemmaWithTools(BaseChatModel):
    """Wraps MedGemmaChatModel with tool definitions in the prompt and parses tool calls from output."""

    model: MedGemmaChatModel = Field(description="Underlying MedGemma model")
    tools: List[BaseTool] = Field(description="Tools to make available")

    class Config:
        arbitrary_types_allowed = True

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_prompt = _format_tools_for_prompt(self.tools)
        tool_names = [t.name for t in self.tools]
        # Prepend tool definitions to the first system message, or add a system message
        new_messages: List[BaseMessage] = []
        system_appended = False
        for msg in messages:
            if isinstance(msg, SystemMessage) and not system_appended:
                content = (msg.content if isinstance(msg.content, str) else str(msg.content)) + "\n\n" + tool_prompt
                new_messages.append(SystemMessage(content=content))
                system_appended = True
            else:
                new_messages.append(msg)
        if not system_appended:
            new_messages.insert(0, SystemMessage(content=tool_prompt))

        result = self.model._generate(new_messages, stop=stop, run_manager=run_manager, **kwargs)
        gen = result.generations[0]
        msg = gen.message
        parsed = _parse_tool_calls_from_content(msg.content or "", tool_names)
        if parsed:
            msg = AIMessage(content=msg.content, tool_calls=parsed)
            result = ChatResult(generations=[ChatGeneration(message=msg)])
        return result

    @property
    def _llm_type(self) -> str:
        return "medgemma_local_with_tools"
