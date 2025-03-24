# Awesome Token Reduction Papers

This repository contains a list of recent papers on token reduction (token pruning, merging, clustering, etc.) for ML/AI; we categorize them based on their year and application scenarios.

We will try to make this list updated. If you found any error or any missed paper, please don't hesitate to open an issue or pull request.

## Table of Contents
- [🌁 Vision](#vision)
- [📝 Language](#language)
- [🎬 Vision-Language Model](#vision-language-model)
- [📱 Hardware Co-design](#hardware)
  
## 🔥 News
2025/03/24: Added CVPR 2025, ICLR 2025, WACV 2025, AAAI 2025

## 🌁 Vision 
<a id="vision"></a>
#### 2025
* [**CVPR'25**] Token Cropr: Faster ViTs for Quite a Few Tasks [[Paper](https://arxiv.org/pdf/2412.00965)]
* [**CVPR'25**] Attend to Not Attended: Structure-then-Detail Token Merging for Post-training DiT Acceleration
* [**CVPR'25**] MergeVQ: A Unified Framework for Visual Generation and Representation with Disentangled Token Merging and Quantization
* [**CVPR'25**] Rethinking Token Reduction with Parameter-Efficient Fine-Tuning in ViT for Pixel-Level Tasks
* [**CVPR'25**] CATANet: Efficient Content-Aware Token Aggregation for Lightweight Image Super-Resolution [[Paper](https://arxiv.org/abs/2503.06896)]
* [**CVPR'25**] VASparse: Towards Efficient Visual Hallucination Mitigation via Visual-Aware Token Sparsification [[Paper](https://arxiv.org/abs/2501.06553)][[Code](https://github.com/mengchuang123/VASparse-github)]
* [**CVPR'25**] Faster Parameter-Efficient Tuning with Token Redundancy Reduction
* [**ICLR'25**] Mutual Effort for Efficiency: A Similarity-based Token Pruning for Vision Transformers in Self-Supervised Learning [[Paper](https://openreview.net/pdf?id=GTcEe5fayC)]
* [**WACV'25**] Pruning One More Token is Enough: Leveraging Latency-Workload Non-Linearities for Vision Transformers on the Edge  [[Paper](https://arxiv.org/abs/2407.05941)]
* [**AAAI'25**] FreqTS: Frequency-Aware Token Selection for Accelerating Diffusion Models 
* [**AAAI'25**] Multimodal Promptable Token Merging for Diffusion Models 

#### 2024
* [**NeurIPS'24**] Accelerating Transformers with Spectrum-Preserving Token Merging [[Paper](https://arxiv.org/pdf/2405.16148)]
* [**ECCV'24**] Agglomerative Token Clustering [[Paper](https://arxiv.org/pdf/2409.11923)][[Code](https://github.com/JoakimHaurum/ATC)] 
* [**ECCV'24**] Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning [[Paper](https://arxiv.org/pdf/2408.06798)][[Code](https://github.com/JieShibo/ToCom)]
* [**ECCV'24**] LookupViT: Compressing visual information to a limited number of tokens [[Paper](https://arxiv.org/pdf/2407.12753)]
* [**ECCV'24**] PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation [[Paper](https://arxiv.org/abs/2403.09192)][[Code](https://github.com/THU-MIG/PYRA?tab=readme-ov-file)]
* [**CVPR'24**] vid-TLDR: Training Free Token Merging for Light-weight Video Transformer [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Choi_vid-TLDR_Training_Free_Token_Merging_for_Light-weight_Video_Transformer_CVPR_2024_paper.pdf)][[Code](https://github.com/mlvlab/vid-TLDR)]  
* [**CVPR'24**] Zero-TPrune: Zero-Shot Token Pruning through Leveraging of the Attention Graph in Pre-Trained Transformers [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Zero-TPrune_Zero-Shot_Token_Pruning_through_Leveraging_of_the_Attention_Graph_CVPR_2024_paper.pdf)][[Code](https://jha-lab.github.io/zerotprune/)] 
* [**ICLR'24**] Synergistic Patch Pruning for Vision Transformer: Unifying Intra- & Inter-Layer Patch Importance [[Paper](https://openreview.net/pdf?id=COO51g41Q4)]
* [**WACV'24**] Token Fusion: Bridging the Gap Between Token Pruning and Token Merging [[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Kim_Token_Fusion_Bridging_the_Gap_Between_Token_Pruning_and_Token_WACV_2024_paper.pdf)]
* [**WACV'24**] Revisiting Token Pruning for Object Detection and Instance Segmentation [[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Revisiting_Token_Pruning_for_Object_Detection_and_Instance_Segmentation_WACV_2024_paper.pdf)][[Code](https://github.com/uzh-rpg/svit/)]
* [arXiv] Token Pruning for Caching Better: 9 Times Acceleration on Stable Diffusion for Free [[Paper](https://arxiv.org/pdf/2501.00375)] 
* [arXiv] Vote&Mix: Plug-and-Play Token Reduction for Efficient Vision Transformer [[Paper](https://arxiv.org/pdf/2408.17062)] 
* [arXiv] Dynamic and Compressive Adaptation of Transformers From Images to Videos [[Paper](https://arxiv.org/pdf/2408.06840)]

#### 2023
* [**ICCV'23**] Dynamic Token Pruning in Plain Vision Transformers for Semantic Segmentation [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Tang_Dynamic_Token_Pruning_in_Plain_Vision_Transformers_for_Semantic_Segmentation_ICCV_2023_paper.pdf)][[Code](https://github.com/zbwxp/Dynamic-Token-Pruning)]
* [**ICCV'23**] DiffRate: Differentiable Compression Rate for Efficient Vision Transformers [[Paper](https://arxiv.org/abs/2305.17997)][[Code](https://github.com/OpenGVLab/DiffRate)]
* [**ICCV'23**] TORE: Token Reduction for Efficient Human Mesh Recovery with Transformer [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Dou_TORE_Token_Reduction_for_Efficient_Human_Mesh_Recovery_with_Transformer_ICCV_2023_paper.pdf)][[Code](https://github.com/Frank-ZY-Dou/TORE)] 
* [**ICCV'23** Workshop] Which Tokens to Use? Investigating Token Reduction in Vision Transformers [[Paper](https://arxiv.org/abs/2308.04657)][[Code](https://github.com/JoakimHaurum/TokenReduction)] 
* [**CVPR'23**] Joint Token Pruning and Squeezing Towards More Aggressive Compression of Vision Transformers [[Paper](https://arxiv.org/pdf/2304.10716)][[Code](https://github.com/megvii-research/TPS-CVPR2023)]
* [**ICLR'23**] Token Merging: Your ViT But Faster [[Paper](https://arxiv.org/pdf/2210.09461)][[Code](https://github.com/facebookresearch/ToMe)]
* [**IJCAI'23**] Adaptive Sparse ViT: Towards Learnable Adaptive Token Pruning by Fully Exploiting Self-Attention [[Paper](https://arxiv.org/pdf/2209.13802)][[Code](https://github.com/Cydia2018/AS-ViT)]
* [**TIP**] Efficient Vision Transformer via Token Merger [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10183862)]
* [arXiv] PPT: Token Pruning and Pooling for Efficient Vision Transformers [[Paper](https://arxiv.org/pdf/2310.01812)][[Code](https://github.com/xjwu1024/PPT)]

#### 2022
* [**ECCV'22**] SPViT: Enabling Faster Vision Transformers via Latency-aware Soft Token Pruning [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710618.pdf)][[Code](https://github.com/PeiyanFlying/SPViT)] 
* [**ECCV'22**] ATS: Adaptive Token Sampling For Efficient Vision Transformers [[Paper](https://arxiv.org/abs/2111.15667)][[Code](https://github.com/adaptivetokensampling/ATS)]
* [**ECCV'22**] PPT: token-Pruned Pose Transformer for monocular and multi-view human pose estimation [[Paper](https://arxiv.org/pdf/2209.08194)][[Code](https://github.com/HowieMa/PPT)]
* [**CVPR'22**] Vision Transformer Slimming: Multi-Dimension Searching in Continuous Optimization Space [[Paper](https://arxiv.org/pdf/2201.00814)][[Code](https://github.com/Arnav0400/ViT-Slim)]
* [**CVPR'22**] Patch Slimming for Efficient Vision Transformers [[Paper](https://arxiv.org/abs/2106.02852)]
* [**CVPR'22**] A-ViT: Adaptive Tokens for Efficient Vision Transformer [[Paper](https://arxiv.org/pdf/2112.07658)][[Code](https://github.com/NVlabs/A-ViT)]
* [**ICLR'22**] EViT: Expediting Vision Transformers via Token Reorganizations [[Paper](https://arxiv.org/pdf/2202.07800)][[Code](https://github.com/youweiliang/evit?tab=readme-ov-file)]
* [**AAAI'22**] Evo-ViT: Slow-Fast Token Evolution for Dynamic Vision Transformer [[Paper](https://arxiv.org/abs/2108.01390)][[Code](https://github.com/YifanXu74/Evo-ViT)]

#### 2021
* [**NeurIPS'21**] IA-RED2: Interpretability-Aware Redundancy Reduction for Vision Transformers [[Paper](https://arxiv.org/pdf/2106.12620)] 
* [**NeurIPS'21**] DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification [[Paper](https://arxiv.org/abs/2106.02034)][[Code](https://github.com/raoyongming/DynamicViT)]


## 📝 Language 
<a id="language"></a>
#### 2025
* [**ICLR'25**] MrT5: Dynamic Token Merging for Efficient Byte-level Language Models [[Paper](https://openreview.net/pdf?id=VYWBMq1L7H)][[Code](https://github.com/jkallini/mrt5)]

#### 2024
* [arXiv] LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference [[Paper](https://arxiv.org/pdf/2407.14057)] 

#### 2023
* [**EMNLP'23**] Optimizing Retrieval-augmented Reader Models via Token Elimination [[Paper](https://arxiv.org/pdf/2310.13682)][[Code](https://github.com/IntelLabs/token_elimination)]
* [**EMNLP'23**] Context Compression for Auto-regressive Transformers with Sentinel Tokens [[Paper](https://arxiv.org/pdf/2310.08152)][[Code](https://github.com/DRSY/KV_Compression)] 
* [**EMNLP'23**] Leap-of-Thought: Accelerating Transformers via Dynamic Token Routing [[Paper](https://aclanthology.org/2023.emnlp-main.976.pdf)][[Code](https://github.com/yeachan-kr/lot)]  
* [**EMNLP'23**] TLM: Token-Level Masking for Transformers [[Paper](https://arxiv.org/pdf/2310.18738)][[Code](https://github.com/Young1993/tlm)]  
* [**EMNLP'23**] Understanding the Role of Input Token Characters in Language Models: How Does Information Loss Affect Performance? [[Paper](https://aclanthology.org/2023.emnlp-main.563.pdf)]  
* [**NeurIPS'23**] Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers [[Paper](https://arxiv.org/pdf/2305.15805)]
* [**ACL'23**] Going Beyond Sentence Embeddings: A Token-Level Matching Algorithm for Calculating Semantic Textual Similarity [[Paper](https://aclanthology.org/2023.acl-short.49.pdf)]
* [**ACL'23**] Efficient Transformers with Dynamic Token Pooling [[Paper](https://aclanthology.org/2023.acl-long.353.pdf)]
* [**ACL'23**] Token-wise Decomposition of Autoregressive Language Model Hidden States for Analyzing Model Predictions [[Paper](https://aclanthology.org/2023.acl-long.562.pdf)]
* [**ACL'23**] LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models [[Paper](https://arxiv.org/pdf/2310.05736)][[Code](https://github.com/microsoft/LLMLingua)]  
* [**ACL'23**] UTC-IE: A Unified Token-pair Classification Architecture for Information Extraction [[Paper](https://aclanthology.org/2023.acl-long.226.pdf)] 
* [**ACL'23**] Revisiting Token Dropping Strategy in Efficient BERT Pretraining [[Paper](https://aclanthology.org/2023.acl-long.579.pdf)]

#### 2022
* [**ACL'22**] Pyramid-BERT: Reducing Complexity via Successive Core-set based Token Selection [[Paper](https://aclanthology.org/2022.acl-long.602.pdf)]
* [**ACL'22**] AdapLeR: Speeding up Inference by Adaptive Length Reduction [[Paper](https://aclanthology.org/2022.acl-long.1.pdf)][[Code](https://github.com/amodaresi/AdapLeR)]   
* [**KDD'22**] Learned Token Pruning for Transformers [[Paper](https://arxiv.org/pdf/2107.00910)][[Code](https://github.com/kssteven418/LTP)]    

#### 2021
* [**NeurIPS'21**] Magic Pyramid: Accelerating Inference with Early Exiting and Token Pruning [[Paper](https://arxiv.org/pdf/2111.00230)]

## 🎬 Vision-Language Model 
<a id="vision-language-model"></a>
#### 2025
* [**CVPR'25**] PACT: Pruning and Clustering-Based Token Reduction for Faster Visual Language Models
* [**CVPR'25**] DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models [[Paper](https://arxiv.org/abs/2503.02175)][[Code](https://github.com/vbdi/divprune)]
* [**CVPR'25**] SynerGen-VL: Towards Synergistic Image Understanding and Generation with Vision Experts and Token Folding [[Paper](https://arxiv.org/abs/2412.09604)]
* [**CVPR'25**] PVC: Progressive Visual Token Compression for Unified Image and Video Processing in Large Vision-Language Models [[Paper](https://arxiv.org/abs/2412.09613)]
* [**CVPR'25**] TopV: Compatible Token Pruning with Inference Time Optimization for Fast and Low-Memory Multimodal Vision Language Model
* [**CVPR'25**] Accelerating Multimodel Large Language Models by Searching Optimal Vision Token Reduction [[Paper](https://arxiv.org/abs/2412.00556)]
* [**CVPR'25**] ATP-LLaVA: Adaptive Token Pruning for Large Vision Language Models [[Paper](https://arxiv.org/abs/2412.00447)][[Code](https://yxxxb.github.io/ATP-LLaVA-page/)]
* [**CVPR'25**] DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models [[Paper](https://arxiv.org/abs/2411.15024)]
* [**ICLR'25**] LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token [[Paper](https://arxiv.org/abs/2501.03895)][[Code](https://github.com/ictnlp/LLaVA-Mini)]
* [**ICLR'25**] MrT5: Dynamic Token Merging for Efficient Byte-level Language Models [[Paper](https://openreview.net/pdf?id=VYWBMq1L7H)][[Code](https://github.com/jkallini/mrt5)]
* [**ICLR'25**] TempMe: Video Temporal Token Merging for Efficient Text-Video Retrieval [[Paper](https://arxiv.org/abs/2409.01156)][[Code](https://github.com/LunarShen/TempMe)]
* [**ICLR'25**] Inference Optimal VLMs Need Fewer Visual Tokens and More Parameters [[Paper](https://openreview.net/pdf?id=6VhDQP7WGX)]
* [**WACV'25**] VLTP: Vision-Language Guided Token Pruning for Task-Oriented Segmentation [[Paper](https://arxiv.org/pdf/2409.08464)]
* [**WACV'25**] Patch Ranking: Token Pruning as Ranking Prediction for Efficient CLIP [[Paper](https://arxiv.org/html/2409.14607v1)]
* [**AAAI'25**] Boosting Multimodal Large Language Models with Visual Tokens Withdrawal for Rapid Inference [[Paper](https://arxiv.org/pdf/2405.05803)][[Code](https://github.com/lzhxmu/VTW)]
* [**AAAI'25**] Fit and Prune: Fast and Training-free Visual Token Pruning for Multi-modal Large Language Models [[Paper](https://arxiv.org/pdf/2409.10197)][[Code](https://github.com/ywh187/FitPrune)]
* [arXiv] Dynamic Token Reduction during Generation for Vision Language Models [[Paper](https://arxiv.org/pdf/2501.14204)]
* [arXiv] Compression with Global Guidance: Towards Training-free High-Resolution MLLMs Acceleration [[Paper](https://arxiv.org/pdf/2501.05179)][[Code](https://github.com/xuyang-liu16/GlobalCom2)]
  
#### 2024
* [**NeurIPS'24**] Token Merging for Training-Free Semantic Binding in Text-to-Image Synthesis [[Paper](https://arxiv.org/abs/2411.07132)][[Code](https://github.com/hutaiHang/ToMe)]
* [**ECCV'24**] IVTP: Instruction-guided Visual Token Pruning for Large Vision-Language Models [[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02577.pdf)]
* [**ECCV'24**] An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Acceleration for VLLM Inference [[Paper](https://arxiv.org/pdf/2403.06764)][[Code](https://github.com/pkunlp-icler/FastV)]
* [**ICML'24**] CrossGET: Cross-Guided Ensemble of Tokens for Accelerating Vision-Language Transformers [[Paper](https://arxiv.org/pdf/2305.17455v4)][[Code](https://github.com/sdc17/CrossGET)]
* [**ECCV'24**] LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models [[Paper](https://arxiv.org/abs/2311.17043)][[Code](https://github.com/dvlab-research/LLaMA-VID/tree/main)]
* [**CVPR'24**] Honeybee: Locality-enhanced Projector for Multimodal LLM [[Paper](https://arxiv.org/abs/2312.06742)][[Code](https://github.com/khanrc/honeybee?tab=readme-ov-file)]
* [arXiv] Rethinking Token Reduction in MLLMs: Towards a Unified Paradigm for Training-Free Acceleration [[Paper](https://arxiv.org/pdf/2411.17686)][[Code](https://github.com/kawhiiiileo/FiCoCo)]
* [OpenReview] LVP: Language-guide Visual Projector for Efficient Multimodal LLM [[Paper](https://openreview.net/pdf?id=PxBzxO02Ef)]
* [arXiv] FrameFusion: Combining Similarity and Importance for Video Token Reduction on Large Visual Language Models  [[Paper](https://arxiv.org/abs/2501.01986)][[Code](https://github.com/thu-nics/FrameFusion)]
* [arXiv] SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference [[Paper](https://arxiv.org/pdf/2410.04417)][[Code](https://github.com/gumpest/sparsevlms)]
* [arXiv] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning [[Paper](https://arxiv.org/pdf/2412.03248)][[Code](https://github.com/LaVi-Lab/AIM)]
* [arXiv] VisionZip: Longer is Better but Not Necessary in Vision Language Models [[Paper](https://arxiv.org/pdf/2412.04467)][[Code](https://github.com/dvlab-research/VisionZip)]
* [arXiv] TokenPacker: Efficient Visual Projector for Multimodal LLM [[Paper](https://arxiv.org/pdf/2407.02392)][[Code](https://github.com/CircleRadon/TokenPacker)]
* [arXiv] Less is More: A Simple yet Effective Token Reduction Method for Efficient Multi-modal LLMs [[Paper](https://arxiv.org/pdf/2409.10994)]
* [arXiv] mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding [[Paper](https://arxiv.org/pdf/2409.03420)][[Code](https://github.com/X-PLUG/mPLUG-DocOwl)]
* [arXiv] TempMe: Video Temporal Token Merging for Efficient Text-Video Retrieval [[Paper](https://arxiv.org/pdf/2409.01156)][[Code](https://github.com/X-PLUG/mPLUG-DocOwl)]
* [arXiv] Recoverable Compression: A Multimodal Vision Token Recovery Mechanism Guided by Text Information [[Paper](https://arxiv.org/pdf/2409.01179)]
* [arXiv] HiRED: Attention-Guided Token Dropping for Efficient Inference of High-Resolution Vision-Language Models in Resource-Constrained Environments [[Paper](https://arxiv.org/pdf/2408.10945)][[Code](https://github.com/hasanar1f/HiRED)]
* [arXiv] Token-level Correlation-guided Compression for Efficient Multimodal Document Understanding [[Paper](https://arxiv.org/pdf/2407.14439)][[Code](https://github.com/JiuTian-VL/TokenCorrCompressor)]
* [arXiv] VoCo-LLaMA: Towards Vision Compression with Large Language Models [[Paper](https://arxiv.org/pdf/2406.12275)][[Code](https://github.com/Yxxxb/VoCo-LLaMA)]
* [arXiv] DeCo: Decoupling Token Compression from Semantic Abstraction in Multimodal Large Language Models [[Paper](https://arxiv.org/pdf/2405.20985)][[Code](https://github.com/yaolinli/DeCo)]
* [arXiv] CATP: Cross-Attention Token Pruning for Accuracy Preserved Multimodal Model Inference [[Paper](https://arxiv.org/pdf/2404.08567)]
* [arXiv] MobileVLM V2: Faster and Stronger Baseline for Vision Language Model [[Paper](https://arxiv.org/abs/2402.03766.pdf)][[Code](https://github.com/Meituan-AutoML/MobileVLM)]
* [arXiv] LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models [[Paper](https://arxiv.org/abs/2403.15388.pdf)][[Code](https://github.com/42Shawn/LLaVA-PruMerge)]
#### 2023
* [**ACL'23**] PuMer: Pruning and Merging Tokens for Efficient Vision Language Models [[Paper](https://aclanthology.org/2023.acl-long.721.pdf)][[Code](https://github.com/csarron/PuMer)]  

## 🐍 State Space Models 
* [**EMNLP'24**] Rethinking Token Reduction for State Space Models [[Paper](https://arxiv.org/pdf/2410.14725)][[Code](https://github.com/wuyushuwys/ToR_SSM)]
* [**NeurIPS'24**] Exploring Token Pruning in Vision State Space Models [[Paper](https://arxiv.org/pdf/2409.18962)]
* [**ECCV'24** Workshop] Famba-V: Fast Vision Mamba with Cross-Layer Token Fusion [[Paper](https://arxiv.org/pdf/2409.09808)][[Code](https://github.com/AIoT-MLSys-Lab/Famba-V)]


## 📱 Hardware Co-design
<a id="hardware"></a>
* [**DATE'24**] ViT-ToGo : Vision Transformer Accelerator with Grouped Token Pruning [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10546804)]
* [**HPCA'23**] HeatViT: Hardware-Efficient Adaptive Token Pruning for Vision Transformers [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10071047)]
* [**HPCA'21**] SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning [[Paper](https://arxiv.org/pdf/2012.09852)][[Code](https://github.com/mit-han-lab/spatten)]
