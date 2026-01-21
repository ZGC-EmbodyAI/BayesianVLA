<div align="center">

<h1>
    <span
        style="
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            display: inline-block;
        "
    >BayesianVLA</span>
    : Bayesian Decomposition of Vision Language Action Models via Latent Action Queries
</h1>

<a href="https://github.com/ZGC-EmbodyAI/TwinBrainVLA">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-ZGC--EmbodyAI%2FTwinBrainVLA-blue?logo=github">
</a>
<a href="https://arxiv.org/">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2601.xxxxx-b31b1b.svg">
</a>
<a href="https://github.com/ZGC-EmbodyAI/TwinBrainVLA/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
</a>

**Shijie Lian**<sup>1,2,\*</sup> **Bin Yu**<sup>2,4,\*</sup> **Xiaopeng Lin**<sup>2,5,\*</sup> **Laurence T. Yang**<sup>6,1,†</sup> **Zhaolong Shen**<sup>2,7</sup><br>
**Changti Wu**<sup>2,8</sup> **Yuzhuo Miao**<sup>2,4</sup> **Cong Huang**<sup>2,3</sup> **Kai Chen**<sup>2,3,9,†</sup>

<sup>1</sup>HUST, <sup>2</sup>ZGCA, <sup>3</sup>ZGCI, <sup>4</sup>HIT, <sup>5</sup>HKUST(GZ), <sup>6</sup>ZZU, <sup>7</sup>BUAA, <sup>8</sup>ECNU, <sup>9</sup>DeepCybo

<sup>*</sup>Equal contribution, <sup>†</sup>Corresponding author

<img src="./assets/ZGCA-logo.png" alt="ZGCA" style="vertical-align: middle; height: 16px; margin-right: 4px; position: relative; top: -2px;" />[Zhongguancun Academy](https://www.bjzgca.edu.cn/) & <img src="./assets/ZGCI-logo.png" alt="ZGCI" style="vertical-align: middle; height: 16px; margin-right: 4px; position: relative; top: -2px;" />[Zhongguancun Institute of Artificial Intelligence](https://www.zgci.ac.cn/)

</div>

---

## 📖 Abstract

Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term `Information Collapse`. Consequently, models degenerate into vision-only policies that ignore language constraints and fail in out-of-distribution (OOD) settings. To address this, we propose **BayesianVLA:**, a novel framework that enforces instruction following via Bayesian decomposition. By introducing learnable **Latent Action Queries**, we construct a dual-branch architecture to estimate both a vision-only prior $p(a \mid v)$ and a language-conditioned posterior $\pi(a \mid v, \ell)$. We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Without requiring new data, BayesianVLA significantly improves generalization. Extensive experiments across on SimplerEnv and RoboCasa demonstrate substantial gains, including an **11.3\%** improvement on the challenging OOD SimplerEnv benchmark, validating the ability of our approach to robustly ground language in action.

## 🏗️ Architecture

<div align="center">
  <img src="./assets/arch.png" alt="BayesianVLA Framework" width="100%">
</div>

## 🙏 Acknowledgements

We would like to thank the [starVLA](https://github.com/starVLA/starVLA) project for its inspiring work and open-source contributions. At the same time, we also express our gratitude to the following projects:

- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [LeRobot](https://github.com/huggingface/lerobot/)
- [SimplerEnv](https://github.com/simpler-env/SimplerEnv)
- [Franka Teleop](https://github.com/Shenzhaolong1330/lerobot_franka_teleop)
