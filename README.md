# Guiding Medical Vision-Language Models with Explicit Visual Prompts

We introduce MedVP, an visual prompt generation and fine-tuning framework, which involves extract medical entities, generate visual prompts, and adapt datasets for visual prompt guided fine-tuning.

**Guiding Medical Vision-Language Models with Explicit Visual Prompts: Framework Design and Comprehensive Exploration of Prompt Variations** [[Paper](https://arxiv.org/pdf/2501.02385)] <br>

## 🌟 Requirements
1. Clone this repository and navigate to MedVP folder
```bash
git clone https://github.com/Kelvinz-89757/MedVP.git
cd MedVP
```

2. Install Package: Create conda environment

```Shell
conda create -n MedVP python=3.9 -y
conda activate MedVP
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

```

## 📖 Data Description

3. For all the medical datasets, you may need to firstly apply for the right of access and then download the dataset.
- [SLAKE](https://www.med-vqa.com/slake/)
- [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)
- [PMC-VQA](https://huggingface.co/datasets/RadGenome/PMC-VQA)

## 📦 Model Checkpoints

We provide the following pretrained and finetuned MedVP-LLaVA and MedTriAligned-LLaVA checkpoints for download:

**MedTriAligned-LLaVA** ([Google Drive](https://drive.google.com/drive/folders/1RHGNqs7kFyfTxfLTCm_Lzya5gNy5ih4i?usp=drive_link))
- Based on [ViP-LLaVA](https://github.com/WisconsinAIVision/ViP-LLaVA), Partially finetuned on [MedTrinity-25M](https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M).


**MedVP-LLaVA (SLAKE)**([Google Drive](https://drive.google.com/drive/folders/16mPfdzN3i3G4P2XgPUMoOP96dXYz2Tb4?usp=drive_link))
- Based on MedTriAligned-LLaVA, finetuned on the SLAKE training set using generated visual prompts.

**MedVP-LLaVA (VQA-RAD)** ([Google Drive](https://drive.google.com/drive/folders/1hSde7t1FQr1mWHZ3YtN3n0bCYQyu3Tt5?usp=drive_link))
- Based on MedTriAligned-LLaVA, finetuned on the VQA-RAD training set using generated visual prompts.

**MedVP-LLaVA (PMC-VQA)** ([Google Drive](https://drive.google.com/drive/folders/12bn4JpeEd4gRHfnD7ILavFilsIEwe-fN?usp=drive_link))
- Based on MedTriAligned-LLaVA, finetuned on the PMC-VQA training set using generated visual prompts.

## 🧭 Grounding

We utilize Grounding DINO within the MMDetection framework. To get started, please follow the official [MMDetection installation guide](https://mmdetection.readthedocs.io/en/latest/get_started.html).

**🔧 Finetuning**

We finetune Grounding DINO on a subset of the [SAMed-2D]( https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M ) and SLAKE training datasets. The training pipeline is implemented using [MMDetection](https://github.com/open-mmlab/mmdetection). We provide the training script, which is located at `mm_grounding_dino/finetune`. A grounding DINO checkpoint is provided in [DINO_ck](https://drive.google.com/file/d/1o43K0aHxSFe-O9wTX7sEVXtDyq_8LIkk/view?usp=drive_link).


**🗺️ Grounding**

Using the entities extracted from questions via ChatGLM, Grounding DINO is used to localize corresponding image regions. We provide preprocessed JSON files with generated visual prompts for training and evaluation:
- [🔗 Training JSON](https://drive.google.com/drive/folders/1RzkLMrjPZJt35zKu9dd41sDxNYZTSgPW?usp=drive_link)
- [🔗 Testing JSON](https://drive.google.com/drive/folders/1Gutn102szQF9jXo31ylpGrj6zPLhqY0r?usp=drive_link)


## 🏋️ Training
- In the final step, the model is trained utilizing the generated visual grounding information. The training script of the final step is at `scripts/finetune_MedVP`. Make sure to rewrite necessary path in it.

## 🔍 Inference

- For test dataset inference, we provide two inference scripts, one for Closed/Open-form VQA, another for Multiple-choice VQA (e.g., PMC-VQA). They are located in the `inference` folder.
- A sh script is at `scripts/inference_MedVP.sh`. Before running, ensure that all paths to data and checkpoints are correctly set.



## 📚 Citation

```bibtex
@article{zhu2025guiding,
  title={Guiding Medical Vision-Language Models with Explicit Visual Prompts: Framework Design and Comprehensive Exploration of Prompt Variations},
  author={Zhu, Kangyu and Qin, Ziyuan and Yi, Huahui and Jiang, Zekun and Lao, Qicheng and Zhang, Shaoting and Li, Kang},
  journal={arXiv preprint arXiv:2501.02385},
  year={2025}
}
```

## 🙏 Acknowledgement
We use code from [ViP-LLaVA](https://github.com/WisconsinAIVision/ViP-LLaVA), [mmdetection](https://github.com/open-mmlab/mmdetection). We thank the authors for releasing their code.
