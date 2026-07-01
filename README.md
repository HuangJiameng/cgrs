# CGRS: Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression

---

## 🚀 Overview

**CGRS (Certainty-Guided Reflection Suppression)** is a lightweight, *training-free* decoding method designed to make **Large Reasoning Language Models (LRLMs)** think more efficiently.

By dynamically suppressing unnecessary *reflection behaviors* (e.g., tokens like *“Wait”*, *“But”*, *“Alternatively”*) when the model is confident in its current reasoning trajectory, **CGRS** effectively mitigates the **overthinking problem** in LRLMs — reducing redundant reasoning steps, token usage, and inference cost, **without sacrificing accuracy**.

---

## 🧩 Key Idea

Large reasoning models like *DeepSeek-R1-Distill* series, *QwQ-32B*, and *Qwen3* family often reflect excessively while reasoning — revisiting the same ideas even after reaching correct intermediate answers.

CGRS introduces a **certainty-guided decoding controller** that:

1. **Estimates certainty** at checkpoints by probing tentative answers and computing token-level entropy.
2. **Suppresses reflection triggers** probabilistically when the model exhibits high certainty.

This dynamic suppression ensures reasoning stays concise and accurate — *thinking just enough*.

---

## ✨ Highlights

* 🔹 **Training-free & model-agnostic**
  Works with any autoregressive LRLM; no fine-tuning or architectural changes required.

* 🔹 **Certainty-aware reflection control**
  Uses internal confidence signals to guide when to suppress redundant reasoning.

* 🔹 **Efficiency with accuracy**
  Achieves **18.5%–41.9%** average token reduction across four benchmarks
  (*AIME24, AMC23, MATH500, GPQA-D*) while preserving performance.

* 🔹 **Compatible with major LRLMs**
  Evaluated on DeepSeek-R1-Distill, Qwen3 (4B–32B), and QwQ-32B models.

---

## 📊 Experimental Summary

<img width="768" height="798" alt="image" src="https://github.com/user-attachments/assets/d7975860-a521-45c1-a2d1-38d896e88503" />


CGRS consistently achieves the best balance between length reduction and accuracy retention among efficient reasoning methods evaluated at the time of release.

---

## 📚 Citation

If you find this work helpful, please cite our paper:

```bibtex
@misc{huang2025efficientreasoninglargereasoning,
      title={Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression}, 
      author={Jiameng Huang and Baijiong Lin and Guhao Feng and Jierun Chen and Di He and Lu Hou},
      year={2025},
      eprint={2508.05337},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.05337}, 
}
```

---

## 🧑‍💻 Authors

* **Jiameng Huang**, *Peking University*
* **Baijiong Lin**, *HKUST (Guangzhou)*
* **Guhao Feng**, *Peking University*
* **Jierun Chen**, *Huawei Technologies Co., Ltd.*
* **Di He**, *Peking University*
* **Lu Hou**, *Huawei Technologies Co., Ltd.*
