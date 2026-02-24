"""
SSG VQA tool: visual question answering using SFT checkpoint (surgical_vqa_sft_ssg).
Uses MedGemma + LoRA adapter trained on SSG-QA.
"""
import os
from pathlib import Path
from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from surgical_copilot.registry import register
from surgical_copilot.tools.base import SurgicalQuestionInput


_MICCAI_ROOT = Path(__file__).resolve().parent.parent.parent
_SSG_VQA_ROOT = _MICCAI_ROOT / "ssg_vqa_finetuning" / "surgical_vqa_sft_ssg" / "checkpoints"
_SSG_VQA_CHECKPOINT_CANDIDATES = [
    _SSG_VQA_ROOT / "checkpoint-7400",
    _SSG_VQA_ROOT / "checkpoint-3800",
]
_SSG_VQA_CHECKPOINT = next((p for p in _SSG_VQA_CHECKPOINT_CANDIDATES if p.is_dir()), _SSG_VQA_CHECKPOINT_CANDIDATES[0])
_BASE_MODEL = "google/medgemma-4b-it"
_PROMPT_PREFIX = "Look at the surgical image and answer the question concisely.\nQuestion: "


@register("ssg_vqa")
def make_ssg_vqa_tool(checkpoint_path: Optional[Path] = None):
    return SSGVQATool(checkpoint_path=checkpoint_path)


class SSGVQATool(BaseTool):
    """Answer natural language questions about surgical images using SSG-QA fine-tuned MedGemma."""

    name: str = "ssg_vqa"
    description: str = (
        "SSG-VQA: Visual Question Answering for surgical images. "
        "REQUIRED args: image_path (path to image) AND question (the user's exact question text—copy it verbatim). "
        "Output: short answer. Use for color, spatial relationships (top/left/right), surgical scene understanding."
    )
    args_schema: Type[BaseModel] = SurgicalQuestionInput  # type: ignore[assignment]
    checkpoint_path: Path = Field(default_factory=lambda: _SSG_VQA_CHECKPOINT)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    _model: Any = PrivateAttr(default=None)
    _processor: Any = PrivateAttr(default=None)

    def __init__(self, checkpoint_path: Optional[Path] = None, **kwargs):
        init_ckpt = Path(checkpoint_path) if checkpoint_path else _SSG_VQA_CHECKPOINT
        super().__init__(checkpoint_path=init_ckpt, **kwargs)

    def _load_model(self) -> None:
        """Lazy-load processor and model (base + LoRA adapter)."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        cache_kwargs = {}
        cache_dir = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME")
        if cache_dir:
            cache_kwargs["cache_dir"] = cache_dir

        self._processor = AutoProcessor.from_pretrained(_BASE_MODEL, **cache_kwargs)
        self._processor.tokenizer.padding_side = "left"

        model_kwargs = dict(
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **cache_kwargs,
        )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )

        model = AutoModelForImageTextToText.from_pretrained(_BASE_MODEL, **model_kwargs)
        model = PeftModel.from_pretrained(model, str(self.checkpoint_path), is_trainable=False)
        model.eval()
        self._model = model

    def _run(
        self,
        image_path: str,
        question: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not Path(image_path).exists():
            return f"Error: image not found: {image_path}"

        if not self.checkpoint_path.is_dir():
            return (
                f"Error: SSG VQA checkpoint not found: {self.checkpoint_path}. "
                "Place LoRA adapter in ssg_vqa_finetuning/surgical_vqa_sft_ssg/checkpoints/checkpoint-7400 (or checkpoint-3800)."
            )

        if self._model is None or self._processor is None:
            try:
                self._load_model()
            except Exception as e:
                return f"Error loading SSG VQA model: {e}"

        import torch
        from PIL import Image

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": _PROMPT_PREFIX + question.strip() + "\n"},
                ],
            },
        ]
        text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        pad_id = self._processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._processor.tokenizer.eos_token_id

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        pred_ids = out[0][input_len:]
        answer = self._processor.tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
        return answer
