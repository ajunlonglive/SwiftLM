# VLM (Vision-Language Model) Support Roadmap

SwiftLM currently supports VLM inference via the `--vision` flag, routing image+text requests through the OpenAI-compatible `/v1/chat/completions` endpoint with standard `base64` image payloads.

This document tracks which vision architectures are supported, which are planned, and the porting effort required.

---

## Current VLM Support Matrix

| Model Family | `model_type` | Swift VLM | Python mlx-vlm | Priority | Effort |
|---|---|---|---|---|---|
| **Qwen2-VL** | `qwen2_vl` | ✅ | ✅ | — | — |
| **Qwen2.5-VL** | `qwen2_5_vl` | ✅ | ✅ | — | — |
| **Qwen3-VL** | `qwen3_vl` | ✅ | ✅ | — | — |
| **Qwen3.5** | `qwen3_5` | ✅ | ✅ | — | — |
| **Qwen3.5 MoE** | `qwen3_5_moe` | ✅ | ✅ | — | — |
| **Gemma 3** | `gemma3` | ✅ | ✅ | — | — |
| **PaliGemma** | `paligemma` | ✅ | ✅ | — | — |
| **Idefics3** | `idefics3` | ✅ | ✅ | — | — |
| **SmolVLM2** | `smolvlm` | ✅ | ✅ | — | — |
| **FastVLM** | `fastvlm` | ✅ | ✅ | — | — |
| **Pixtral** | `pixtral` | ✅ | ✅ | — | — |
| **Mistral 3** | `mistral3` | ✅ | ✅ | — | — |
| **LFM2-VL** | `lfm2_vl` | ✅ | ✅ | — | — |
| **GLM-OCR** | `glm_ocr` | ✅ | ✅ | — | — |

**Total: 14 VLM architectures currently supported in Swift.**

---

## Planned VLM Ports (Python reference available)

### Phase 1 — High Priority (Popular models, strong community demand)

| Model Family | `model_type` | Python Ref | Notes | Est. Effort |
|---|---|---|---|---|
| **Gemma 4** | `gemma4` | ✅ `mlx-vlm` | LLM text layer already in `MLXLLM/Gemma4.swift`. Vision requires new 2D-RoPE, ClippableLinear, VisionPooler, PatchEmbedder. | ~3-4 days |
| **Llama 4** | `llama4` | ✅ `mlx-vlm` | Meta's latest multimodal. High demand. | ~2-3 days |
| **Mistral 4** | `mistral4` | ✅ `mlx-vlm` | Mistral 3 VLM already supported; likely incremental. | ~1-2 days |
| **Phi-4 SigLip** | `phi4_siglip` | ✅ `mlx-vlm` | Microsoft Phi-4 multimodal. | ~2-3 days |
| **DeepSeek-VL v2** | `deepseek_vl_v2` | ✅ `mlx-vlm` | Popular open-source VLM. | ~2-3 days |

### Phase 2 — Medium Priority (Emerging models, specialized use cases)

| Model Family | `model_type` | Python Ref | Notes | Est. Effort |
|---|---|---|---|---|
| **Gemma 3N** | `gemma3n` | ✅ `mlx-vlm` | Google's edge-optimized VLM. | ~2-3 days |
| **Kimi-VL** | `kimi_vl` | ✅ `mlx-vlm` | Moonshot AI's vision model. | ~2-3 days |
| **Molmo / Molmo2** | `molmo` / `molmo2` | ✅ `mlx-vlm` | Allen AI's multimodal series. | ~2-3 days |
| **InternVL-Chat** | `internvl_chat` | ✅ `mlx-vlm` | OpenGVLab's strong VLM. | ~2-3 days |
| **Hunyuan-VL** | `hunyuan_vl` | ✅ `mlx-vlm` | Tencent's vision model. | ~2-3 days |
| **Granite Vision** | `granite4_vision` | ✅ `mlx-vlm` | IBM's enterprise VLM. | ~2-3 days |
| **Aya Vision** | `aya_vision` | ✅ `mlx-vlm` | Cohere multilingual VLM. | ~2 days |
| **Phi-3 Vision** | `phi3_v` | ✅ `mlx-vlm` | Microsoft Phi-3 multimodal. | ~2 days |

### Phase 3 — OCR / Specialized Models

| Model Family | `model_type` | Python Ref | Notes | Est. Effort |
|---|---|---|---|---|
| **DeepSeek-OCR** | `deepseekocr` | ✅ `mlx-vlm` | Document OCR specialist. | ~2 days |
| **Falcon-OCR** | `falcon_ocr` | ✅ `mlx-vlm` | TII's OCR model. | ~2 days |
| **Florence 2** | `florence2` | ✅ `mlx-vlm` | Microsoft's dense captioner. | ~2-3 days |
| **SAM3 / SAM3.1** | `sam3` / `sam3_1` | ✅ `mlx-vlm` | Meta's segmentation. Different paradigm. | ~3-4 days |
| **Moondream 3** | `moondream3` | ✅ `mlx-vlm` | Tiny edge VLM. | ~1-2 days |
| **GLM-4V MoE** | `glm4v_moe` | ✅ `mlx-vlm` | Zhipu's MoE vision. | ~2-3 days |

### Phase 4 — Long Tail

| Model Family | `model_type` | Python Ref |
|---|---|---|
| **LLaVA** | `llava` | ✅ |
| **LLaVA-Next** | `llava_next` | ✅ |
| **LLaVA-Bunny** | `llava_bunny` | ✅ |
| **Idefics2** | `idefics2` | ✅ |
| **Jina-VLM** | `jina_vlm` | ✅ |
| **MiniCPM-O** | `minicpmo` | ✅ |
| **Qwen3-VL MoE** | `qwen3_vl_moe` | ✅ |
| **Qwen3-Omni MoE** | `qwen3_omni_moe` | ✅ |
| **PaddleOCR-VL** | `paddleocr_vl` | ✅ |
| **Falcon Perception** | `falcon_perception` | ✅ |
| **RF-DETR** | `rfdetr` | ✅ |
| **ERNIE** | `ernie4_5_moe_vl` | ✅ |
| **Dots-OCR** | `dots_ocr` | ✅ |
| **Phi-4 MM** | `phi4mm` | ✅ |

---

## Porting Methodology

Each VLM port follows the same pattern:

1. **Read** the Python `mlx-vlm` reference in `mlx_vlm/models/<model_type>/`
2. **Create** `Libraries/MLXVLM/Models/<Model>.swift` with:
   - Configuration structs (text, vision, top-level)
   - Vision encoder modules (attention, MLP, embeddings, encoder)
   - Multimodal projector
   - Top-level model that wires vision → projector → language model
3. **Create** a processor (`<Model>Processor`) for image preprocessing
4. **Register** the `model_type` string in `VLMTypeRegistry.shared` and `VLMProcessorTypeRegistry.shared` inside `VLMModelFactory.swift`
5. **Test** with `python3 test_vlm.py <model-id>`

### Shared Infrastructure Already Built
- ✅ Base64 image decoding in `Server.swift`
- ✅ `CIImage` → MLXArray pixel conversion pipeline
- ✅ OpenAI-compatible multi-part content parsing
- ✅ `--vision` CLI flag and VLM/LLM factory routing
- ✅ SSD streaming + PAPPS compatible with VLM weight loading

---

## Key Decision: Upstream Sync Strategy

> [!IMPORTANT]
> The upstream Apple `ml-explore/mlx-swift-lm` repo also adds new VLM architectures over time.
> Before porting a model, always check if Apple has already added it upstream.
> Use the `/mlx-upstream-sync` workflow to pull new models from upstream before writing custom ports.

---

## Test Validation

All VLM ports must pass the automated `test_vlm.py` pipeline:
```bash
python3 test_vlm.py "mlx-community/<model-id>"
```

This downloads a 256×256 test image, spins up SwiftLM with `--vision`, fires a base64-encoded inference request, and validates the JSON response contains a valid `choices[0].message.content` field.
