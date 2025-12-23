<div align="center" style="font-family: charter;">

<h1><br>From Indoor to Open World:<br>Revealing the Spatial Reasoning Gap in MLLMs</br></h1>

<a href="https://arxiv.org/abs/2512.19683" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-OpenBench-red?logo=arxiv" height="20" />
</a>
<a href="https://harmlesssr.github.io/openbench/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Project Page-OpenBench-blue.svg" height="20" />
</a>
<a href="https://huggingface.co/datasets/HarmlessSR07/OpenBench" target="_blank">
    <img alt="HF Dataset: OpenBench" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Benchmark-OpenBench-ffc107?color=ffc107&logoColor=white" height="20" />
</a>

<div>
<a href="https://scholar.google.com/citations?user=Y1TtPL8AAAAJ&hl=en">Mingrui Wu</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=CkDanj8AAAAJ&hl=en">Zhaozhi Wang</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=ysTmrEsAAAAJ&hl=en">Fangjinhua Wang</a><sup>2</sup>,
<a href="https://scholar.google.com/citations?user=GuqoolgAAAAJ&hl=en">Jiaolong Yang</a><sup>3</sup>,
<a href="https://scholar.google.com/citations?user=YYH0BjEAAAAJ&hl=en">Marc Pollefeys</a><sup>2</sup>,
<a href="https://scholar.google.com/citations?user=kCy8JG8AAAAJ&hl=en">Tong Zhang</a><sup>1,*</sup>

</div>

<div>
    <sup>1</sup>University of Chinese Academy of Sciences, UCAS&emsp;
    <sup>2</sup>ETH Zürich&emsp;
    <sup>3</sup>Microsoft Research Asia&emsp;
</div>

<div align="center">
    <small>* Corresponding Author</small>
</div>

</br>
<img src="static/teaser.png" width="80%"/>
</div>

- **A benchmark and dataset for spatial intelligence**. We introduce OpenBench, a metrically precise outdoor benchmark built from multi-sensor pedestrian-view data, with 8,736 question-answer pairs.
- **Comprehensive evaluation of state-of-the-art (SoTA) MLLMs.** We conduct an extensive analysis of leading open- and closed-source models, offering the first unified assessment of spatial reasoning across static, relational, and dynamic tasks under real-world conditions.
- **Current spatial intelligence is fragile.** Our findings reveal that existing MLLMs lack generalizable spatial intelligence—their gains on indoor benchmarks do not transfer to open-world settings.


## News

- `2025-12` : Our paper 'From Indoor to Open World: Revealing the Spatial Reasoning Gap in MLLMs' is released on <a href="https://arxiv.org/abs/2512.19683">Arxiv</a>!
- `2025-12` : We released OpenBench and corresponding evaluation code!

## OpenBench

We introduce OpenBench, a metrically precise outdoor benchmark built from multi-sensor pedestrian-view data, with 8,736 question-answer pairs.

The tasks in OpenBench are structured as a hierarchy of three tiers, representing a progression of spatial reasoning capabilities, covering static, relational, and dynamic tasks under real-world conditions.

<div align="center" style="font-family: charter;">
<img src="static/tasks.png" width="90%"/>
</div>


## Evaluation on OpenBench

We utilize **[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)**, an open-source evaluation toolkit for large vision-language models (LVLMs), to conduct evaluations on OpenBench. Our codebase is adapted to support the metrics and data format of OpenBench.

### 1. Installation

First, clone our repository and set up the environment.

```bash
git clone https://github.com/harmlessSR/OpenBench.git
cd OpenBench/VLMEvalKit

conda create -n openbench python=3.10
conda activate openbench

pip install -e .
```

> **Note on Transformers Versions:**
> Different MLLMs rely on specific versions of the `transformers` library. Please refer to the list below to install the recommended version for your target model.

<details>
<summary>Click to expand: Recommended Transformers Versions for different models</summary>

- **transformers==4.33.0**: Qwen series, Monkey series, InternLM-XComposer Series, mPLUG-Owl2, OpenFlamingo v2, IDEFICS series, VisualGLM, MMAlaya, ShareCaptioner, MiniGPT-4 series, InstructBLIP series, PandaGPT, VXVERSE.
- **transformers==4.36.2**: Moondream1.
- **transformers==4.37.0**: LLaVA series, ShareGPT4V series, TransCore-M, LLaVA (XTuner), CogVLM Series, EMU2 Series, Yi-VL Series, MiniCPM-[V1/V2], OmniLMM-12B, DeepSeek-VL series, InternVL series, Cambrian Series, VILA Series, Llama-3-MixSenseV1_1, Parrot-7B, PLLaVA Series.
- **transformers==4.40.0**: IDEFICS2, Bunny-Llama3, MiniCPM-Llama3-V2.5, 360VL-70B, Phi-3-Vision, WeMM.
- **transformers==4.42.0**: AKI.
- **transformers==4.44.0**: Moondream2, H2OVL series.
- **transformers==4.45.0**: Aria.
- **transformers==latest**: LLaVA-Next series, PaliGemma-3B, Chameleon series, Video-LLaVA-7B-HF, Ovis series, Mantis series, MiniCPM-V2.6, OmChat-v2.0-13B-sinlge-beta, Idefics-3, GLM-4v-9B, VideoChat2-HD, RBDash_72b, Llama-3.2 series, Kosmos series.

</details>

### 2. Data Preparation & Configuration

Our evaluation code will automatically download and uncompress the full benchmark from HuggingFace upon the first run.

**Storage Requirement:** The full dataset (video + metadata) requires at least **160GB** of free disk space. Please ensure your target directory has sufficient storage.

You can specify the download location by modifying the `config/openbench.json` file. By default, it will check if the data exists; if not, it will start downloading.

```json
// config/openbench.json
{
    "model": {
        "Qwen2.5-VL-3B-Instruct": {}
    },
    "data": {
        "OpenBench": {
            "class": "OpenBench",
            // MODIFY THIS PATH to your preferred storage directory
            "data_path": "./OpenBench_HF_Cache_full",
            "dataset": "OpenBench",
            "nframe": 32,
            "download": true,      
            "force_unzip": false  
        }
    }
}
```

### 3. Running Evaluation

To start the evaluation, simply run the following command with your configuration file:

```bash
python run.py --config config/openbench.json
```

The script will:
1.  Check `data_path` for the OpenBench dataset.
2.  Automatically download and unzip the data if it's missing (and `download` is set to `true`).
3.  Load the model specified in the config.
4.  Run inference and generate results.

For advanced usage (e.g., multi-GPU parallel evaluation, launching different models), please refer to the original [VLMEvalKit Documentation](https://github.com/open-compass/VLMEvalKit).

## Acknowledgement
VLMEvalKit serves as foundation for our evaluation code repository. Thanks for their wonderful work!

## Citation

If you find our paper and code useful in your research, please consider giving us a star and citing our work :)
```
@article{wu2025indoor,
    title={From Indoor to Open World: Revealing the Spatial Reasoning Gap in MLLMs},
    author={Mingrui Wu and Zhaozhi Wang and Fangjinhua Wang and Jiaolong Yang and Marc Pollefeys and Tong Zhang},
    journal={arXiv preprint arXiv:2512.19683},
    year={2025}}
```
