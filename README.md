# Making VITS efficient (wip)

## Goals
 - [ ] Try to implement LoRA Finetuning on VITS by modifying attentions.py as described in the LoRA Paper
   - [x] Doesnt work as the generator doesnt get updated, need to research more
 - [x] Try to implement 8-bit training using bitsandbytes to reduce vram usage
   - [x] 8 bit optimisers
 - [ ] Write paper explaining steps?

## Plan
 - [x] Extract discriminator from hifigan
 - [x] Modify models.py to match hifigan implementation
 - [x] Make sure training loop runs
 - [x] Clean up logging, maybe integrate WandB to track training progress
   - [x] WandB integrated! Run ```train_wandb.py``` to use it 
 - [x] Test finetune as-is on test dataset to make sure patched discriminator works
   - Wont be releasing weights as model is not that good. See w&b report underneath
 - [ ] implement LoRA
 - [ ] Test finetune on lora using same test dataset
 - [x] Implement 8-bit optimizers
 - [x] Create webpage to show results (see w&b report)

## Finetuning and Pretrained Models
 - The generator for LJSpeech is [here](https://drive.google.com/file/d/1T-u3OV49W6Lv3bDxh-EA63ALZKHqyy0t/view?usp=share_link)
 - The discriminator(extracted from hifi-gan) is [here](https://drive.google.com/file/d/118ffn807Eqlu891qbNRQP7O9E0-aMPxM/view?usp=share_link)
 - A notebook with a running training loop is [here](https://colab.research.google.com/drive/1rtbhcfxwRRHPkFJT_u7M_slo8_s_PYcK?usp=sharing)
 - W&B [Report](https://api.wandb.ai/links/nivibilla/q1ncpudm) on finetuning with 8bit optimisers

## 8 bit AdamW
<img src="resources/bitsandbytes.png" alt="VITS at training with 8bit AdamW" height="150">
By using 8bit AdamW for the loss. The training now uses approx 8.5gb of vram, wheras before it was using 12gb!!

## Ethical Concerns
 - I realise that if this works, anyone remotely knowledgeable about machine learning will be able to finetune VITS on large datasets quite quickly to achieve pretty good voice cloning. However, this is already a thing(tortoise-tts: mrq fork, DLAS fork), but it takes a long time to finetune and generate. 
 - However, sparks of efficient finetuning for TTS systems are already [here](https://paperswithcode.com/paper/evaluating-parameter-efficient-transfer), its only a matter of time before someone like me will do it for other models.
 - My initial plan is to only provide a comparison of results, and any good and ethical finetunes. I believe the real value lies in the dataset creation so will not be opensourcing any scripts to clean and curate the data.

# Readme from Original Repo

# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

### Jaehyeon Kim, Jungil Kong, and Juhee Son

In our recent [paper](https://arxiv.org/abs/2106.06103), we propose VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.

Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.

Visit our [demo](https://jaywalnut310.github.io/vits-demo/index.html) for audio samples.

We also provide the [pretrained models](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2?usp=sharing).

** Update note: Thanks to [Rishikesh (ऋषिकेश)](https://github.com/jaywalnut310/vits/issues/1), our interactive TTS demo is now available on [Colab Notebook](https://colab.research.google.com/drive/1CO61pZizDj7en71NQG_aqqKdGaA_SaBf?usp=sharing).

<table style="width:100%">
  <tr>
    <th>VITS at training</th>
    <th>VITS at inference</th>
  </tr>
  <tr>
    <td><img src="resources/fig_1a.png" alt="VITS at training" height="400"></td>
    <td><img src="resources/fig_1b.png" alt="VITS at inference" height="400"></td>
  </tr>
</table>


## Pre-requisites
0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak-ng`
0. Download datasets
    1. Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
    1. For mult-speaker setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJ Speech and VCTK have been already provided.
# python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt 
# python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```


## Training Exmaple
```sh
# LJ Speech
python train.py -c configs/ljs_base.json -m ljs_base

# VCTK
python train_ms.py -c configs/vctk_base.json -m vctk_base
```


## Inference Example
See [inference.ipynb](inference.ipynb)
