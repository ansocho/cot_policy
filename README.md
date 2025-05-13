<h1 align="center">Fast Flow-based Visuomotor Policies via Conditional Optimal Transport</h1>

<p align="center">
  <a href="https://www.arxiv.org/pdf/2505.01179"><img src="https://img.shields.io/badge/arXiv-2505.01179-b31b1b.svg?style=flat-square)"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

<p align="center">
<b><a target="_blank" href="https://scholar.google.com/citations?user=6w9786sAAAAJ&hl=en">Andreas Sochopoulos</a><sup>1,2</sup>,
<b><a target="_blank" href="https://malkin1729.github.io/">Nikolay Malkin</a><sup>1</sup>,
<a target="_blank" href="https://tsagkas.github.io/">Nikolaos Tsagkas</a><sup>1</sup>,<br>
 <a target="_blank" href="https://scholar.google.co.uk/citations?user=1L5kTRcAAAAJ&hl=en">Jo&atilde;o Moura</a><sup>1</sup>,
<a target="_blank" href="https://scholar.google.de/citations?user=oU2jyxMAAAAJ&hl=de">Michael Gienger</a><sup>2</sup>,
<a target="_blank" href="https://homepages.inf.ed.ac.uk/svijayak/"> Sethu Vijayakumar</a><sup>1</sup>
</b></p>
<p align="center">
<sup>1</sup>University of Edinburgh,
<sup>2</sup>Honda Research Institute Europe,
</p>
</p>

<p align="center">
  <a href="https://ansocho.github.io/cot-policy/">🌐 Website</a> | 
  <a href="https://arxiv.org/abs/2505.01179">📝 Paper</a>
</p>


<p align="center">
  <img src="cot_policy/media/moons_animation.gif" alt="Moons animation" />
</p>

## Abstract
Diffusion and flow matching policies have recently demonstrated remarkable performance in robotic applications by accurately capturing multimodal robot trajectory distributions. However, their computationally expensive inference, due to the numerical integration of an ODE or SDE, limits their applicability as real-time controllers for robots. We introduce a methodology that utilizes conditional Optimal Transport couplings between noise and samples to enforce straight solutions in the flow ODE for robot action generation tasks. We show that naively coupling noise and samples fails in conditional tasks and propose incorporating condition variables into the coupling process to improve few-step performance. The proposed few-step policy achieves a 4% higher success rate with a 10x speed-up compared to Diffusion Policy on a diverse set of simulation tasks. Moreover, it produces high-quality and diverse action trajectories within 1-2 steps on a set of real-world robot tasks. Our method also retains the same training complexity as Diffusion Policy and vanilla Flow Matching, in contrast to distillation-based approaches.

## Installation

### 1. Dependencies
```
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```
### 2. Set up Environment
```
conda env create -f environment_<suite>.yml
conda activate cot-policy-<suite>
```
where `<suite>` needs to be replaced be `mimicgen, metaworld` or `mujoco`. To train on D4RL Maze tasks, you have to manually install D4RL according to the instructions in the [official repository](https://github.com/Farama-Foundation/D4RL).




## How to use?

### Get expert demonstrations
| **Environment**   | **Instructions**                                                                                                                                                          | **Link**                                                                                   | **Command to Generate Data**                   |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|------------------------------------------------|
| **Metaworld**     | Use the built-in functionality of the suite to generate expert demos.         | [GitHub](https://github.com/tsagkas/pvrobo)                                                                                           | `python -m expert_demos.generate`* |
| **MimicGen**      | Install from source and follow official instructions for dataset generation.                                                                                              | [GitHub](https://github.com/NVlabs/mimicgen), [Docs](https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html) | `python download_datasets.py --dataset_type core --tasks <task>`               |
| **Mujoco tasks**  | Download pre-generated data for ball-in-cup and maze tasks. Custom mazes require changes in D4RL.                                                                         | [Download Link](https://osf.io/6ezsg/?view_only=6f4f132715b347d7949c161b5197ff60)                                                                          | —                                              |
| **Real-world**    | Download teleoperated data for three real-world tasks used in the paper.                                                                                                  | [Download Link](https://osf.io/6ezsg/?view_only=6f4f132715b347d7949c161b5197ff60)                                                                          | —                                              |



### Train COT Policy and baselines
Example for training a COT Policy on the `push-t` task. 
```
./train.sh pusht_image cot_policy_full_pca 
```
:warning: Before training on any task, make sure you modify the paths in the config files and inside `train.sh`.

You can evaluate the trained policy using variations of the following script:

```
python eval.py --checkpoint /path/to/checkpoint/ -o /output/path/ -gs 10000 -es 100 -is desired_inference_steps
```
The `-is` argument determines the number time steps the interval [0,1] is partioned with and it does not directly translate to neural function evaluations (NFE). If using the `euler` solver then <em>NFE= desired_inference_steps - 1</em> and when using the `midpoint` solver then <em>NFE=2*(desired_inference_steps) - 1</em>.

To calculate the <em>Trajectory Variance (TV)</em> metric simply run variations of the following command:

```
python eval_tv.py --checkpoint /path/to/checkpoint/ -o /output/path/ -gs 10000 -es 100 -is desired_inference_steps
```

## Acknowledgments
This codebase is based on the [Diffusion Policy repository](https://github.com/real-stanford/diffusion_policy). Our work and code has also drawn inspiration from many other excellent works such as [AdaFlow](https://arxiv.org/abs/2402.04292), [torchcfm](https://github.com/atong01/conditional-flow-matching), [OT Conditional Flow Matching](https://arxiv.org/abs/2302.00482) and more. 

## Citation
```latex
@article{sochopoulos2025cot,
  title     = {Fast Flow-based Visuomotor Policies via Conditional Optimal Transport Couplings},
  author    = {Sochopoulos, Andreas and Malkin, Nikolay and Tsagkas, Nikolaos and Moura, João and Gienger, Michael and Vijayakumar, Sethu},
  journal   = {arXiv preprint arXiv:2505.01179},
  year      = {2025}
}
```