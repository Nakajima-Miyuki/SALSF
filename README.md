# SALSF

Code for the paper on Synonym-Aware Logit Shaping Fingerprint (SALSF), a fingerprinting method for copyright authentication of Large Vision-Language Models (LVLMs).

SALSF reshapes the probability distribution of long-tail synonym clusters in the logits space while keeping the original top-1 prediction unchanged, and verifies model ownership through the Synonym Dominance Ratio (SDR).

## Dependency

+ python 3.9+
+ torch 2.2.0+
+ transformers 4.41.2+
+ numpy 1.26.4+
+ scikit-learn 1.3.0+
+ peft 0.11.0+

## Dataset & Models

+ Base model: LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`)
+ Fingerprint carriers: 100 VQA samples selected from CC3M
+ Downstream fine-tuning: V7W, ST-VQA, TextVQA, PaintingForm, MathV360k, ChEBI-20
+ Reference models: non-fingerprinted LVLMs for dynamic thresholding
+ Synonym generation: DeepSeek API (optional, can be prepared manually)

## Citation

```bibtex
@inproceedings{zheng-etal-2026-ghost,
    title = "Ghost in the Shell: Synonym-Aware Logit Shaping Fingerprint for Copyright Protection of Large Vision-Language Models",
    author = "Zheng, Xiaofan  and
      Wang, Xinghao  and
      Wan, Xiaojun",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.273/",
    doi = "10.18653/v1/2026.findings-acl.273",
    pages = "5538--5555",
    ISBN = "979-8-89176-395-1",
    abstract = "The proliferation of Large Vision-Language Models (LVLMs) has exacerbated concerns regarding model misappropriation and license violations. Malicious users may deploy open-source models as black boxes and falsely claim ownership, sparking significant community interest in fingerprinting techniques for copyright authentication. Current fingerprinting methods largely follow a backdoor-based paradigm, employing specific inputs to elicit predetermined abnormal text outputs. However, such direct distortion of the model{'}s original predictions compromises modality alignment and inevitably degrades multimodal capabilities, leading to an inherent trade-off between robustness and harmlessness. To address these challenges, we investigate whether it is possible to embed robust fingerprints while maximally preserving the original normal outputs of the model. We propose a Synonym-Aware Logit Shaping Fingerprint (SALSF). The core insight of SALSF lies in reshaping the probability distribution of semantically similar long-tail tokens within the logits space while ensuring the original top-1 prediction token and its probability remain approximately invariant. By elevating the overall prediction probability of the semantic cluster to a level distinctly higher than the natural baseline, our approach stealthily embeds the fingerprint and mitigates the disruption to modality alignment. Experimental results demonstrate that SALSF maintains multimodal performance and substantially enhances fingerprint robustness, offering a novel paradigm for the intellectual property protection of LVLMs."
}
```
