# KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem

Official implementation of the paper **<a href="https://openreview.net/pdf?id=k5nKHWp9VC" target="_blank">KnapSpec: Knapsack-based Speculative Decoding</a>**, accepted at the **43rd International Conference on Machine Learning (ICML 2026)**.

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n knapspec python=3.10
conda activate knapspec
pip install -r requirements.txt
```

### 2. Run Benchmarks

Run comparison experiments (AR vs KnapSpec vs Baselines):

```bash
bash run_comparison_local.sh
```

### Supported Models
- Qwen/Qwen3-32B
- Qwen/Qwen3-14B (Default)
- Qwen/Qwen3-8B
- Qwen/Qwen3-4B
- Llama-3.1-70B
- Llama-3.1-8B
- Llama-3.2-1B
- Llama-3.2-3B


### Main Arguments
- `--model`: HuggingFace model ID
- `--dataset`: Dataset to evaluate (`aime24`, `mmlu_pro`, etc.)
- `--strategy`: `autoregressive`, `knapspec`, `clasp`, `del`
- `--compare-strategies`: Run all strategies sequentially for fair comparison


## 📚 Citation

```bibtex
@inproceedings{cha2026knapspec,
      title={KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem},
      author={Cha, Seongjin and Kim, Gyuwan and Han, Dongsu and Yang, Tao and Han, Insu},
      booktitle = {Proceedings of the Forty-Third International Conference on Machine Learning (ICML)},
      year={2026}
}
```
