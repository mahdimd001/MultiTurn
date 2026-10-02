---
language:
- en
license: cc-by-nc-4.0
pretty_name: MultiVerse
task_categories:
- image-text-to-text
tags:
- benchmark
- multimodal
- vision-language-model
- conversation
- multi-turn conversation
- evaluation
dataset_info:
  features:
  - name: index
    dtype: string
  - name: image
    dtype: image
  - name: character
    dtype: string
  - name: scenario
    dtype: string
  - name: goal
    dtype: string
  - name: conversation
    sequence:
    - name: speaker
      dtype: string
    - name: utterance
      dtype: string
    - name: checklist
      sequence:
      - name: question
        dtype: string
      - name: main_criteria
        dtype: string
      - name: sub_criteria
        dtype: string
  - name: main_goal_category
    dtype: string
  - name: sub_goal_category
    dtype: string
  - name: main_task_category
    dtype: string
  - name: sub_task_category
    dtype: string
  - name: image_main_category
    dtype: string
  - name: image_sub_category
    dtype: string
  splits:
  - name: train
    num_bytes: 370337877.0
    num_examples: 647
  download_size: 364287875
  dataset_size: 370337877.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Dataset Card for **MultiVerse**

[💻 GitHub](https://github.com/passing2961/MultiVerse) | [📄 Arxiv](https://arxiv.org/abs/2510.16641) | [📕 PDF](https://arxiv.org/pdf/2510.16641)

![MultiVerse](multiverse_teaser.png)


## Dataset Description

- **Repository:** [Code](https://github.com/passing2961/MultiVerse)  
- **Paper:** [MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models](https://huggingface.co/papers/2510.16641)  
- **Point of Contact:** [Young-Jun Lee](mailto:passing2961@gmail.com)


## Dataset Summary

**MultiVerse** is a novel multi-turn conversation benchmark containing **647 dialogues**, each averaging **four turns**, derived from **12 popular VLM evaluation benchmarks**. The dataset includes **484 distinct tasks** and **484 interaction goals**, covering a wide range of topics—from factual knowledge and perception to advanced reasoning tasks such as mathematics and coding. For detailed descriptions of these categories, please refer to our accompanying paper.

## Dataset Structure

| Field | Type | Description |
|-------|------|-------------|
| `index` | `string` | Unique identifier for each dialogue, in the format `idx:source_dataset_name:original_index` |
| `image` | `image` | Source image associated with the dialogue |
| `character` | `string` | Persona or role of the main character in the dialogue |
| `scenario` | `string` | Narrative or situational context for the dialogue |
| `goal` | `string` | Target objective or conversational intent |
| `conversation` | `sequence` | List of multi-turn conversational exchanges |
| ├── `speaker` | `string` | Name or role of the speaker (e.g., *Speaker A*, *AI Assistant*) |
| ├── `utterance` | `string` | Spoken text for that turn |
| └── `checklist` | `sequence` | Evaluation checklist per turn |
| &nbsp;&nbsp;&nbsp;&nbsp; ├── `question` | `string` | Assessment question |
| &nbsp;&nbsp;&nbsp;&nbsp; ├── `main_criteria` | `string` | Primary evaluation dimension |
| &nbsp;&nbsp;&nbsp;&nbsp; └── `sub_criteria` | `string` | Secondary evaluation sub-dimension |
| `main_goal_category` | `string` | High-level goal classification |
| `sub_goal_category` | `string` | Fine-grained goal classification |
| `main_task_category` | `string` | High-level task category |
| `sub_task_category` | `string` | Fine-grained task category |
| `image_main_category` | `string` | High-level image domain |
| `image_sub_category` | `string` | Fine-grained image domain |

**Splits**

| Split | # Examples | Size |
|--------|-------------|------|
| `train` | 647 | 370 MB |

- **Download size:** 364 MB  
- **Dataset size:** 370 MB  
- **License:** CC-BY-NC-4.0  


## Dataset Creation

We first collect source images from 12 widely used VLM evaluation benchmarks (e.g., *MegaBench*, *CharXiv*, *MMMU*), spanning diverse domains such as nature, science, and mathematics. After selecting high-quality seed images, we adopt a **personal background-to-conversation** approach to ensure lexically diverse and contextually rich multi-turn dialogues. Finally, all samples undergo meticulous manual review to remove unnatural, factually incorrect, or visually inaccessible cases—ensuring **natural conversation flow**, **factual accuracy**, and **blindness robustness**.


## Limitations and Further Details

For more information on dataset limitations and evaluation methodologies, please refer to our [paper](https://arxiv.org/abs/2510.16641).


## License

The **MultiVerse** dataset is distributed under the **CC-BY-NC-4.0** license.


## Acknowledgement

This work was partly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (RS-2022-II220641, XVoice: Multi-Modal Voice Meta Learning)


## Citation

If you find this dataset useful, please cite:

```bibtex
@misc{lee2025multiversemultiturnconversationbenchmark,
      title={MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models}, 
      author={Young-Jun Lee and Byung-Kwan Lee and Jianshu Zhang and Yechan Hwang and Byungsoo Ko and Han-Gyu Kim and Dongyu Yao and Xuankun Rong and Eojin Joo and Seung-Ho Han and Bowon Ko and Ho-Jin Choi},
      year={2025},
      eprint={2510.16641},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.16641}, 
}
```