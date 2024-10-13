# Physical Vocoder and Adversarial Attack
Here are some codes for paper "An Initial Investigation of Neural Replay Simulator for Over-The-Air Adversarial Perturbations to Automatic Speaker Verification".
- adversarial attack scripts: `./attack`
- physical vocoder (the UNet model to simulate over-the-air process, specified in paper): `./phys_vocoder`
- model checkpoints (the pretrained models specified in paper): `./pretrained_models`
- utils for audio clipping (how we collect real-world recording datasets and align frame to frame): `./audio_clipper`
- generate PGD adversarial examples: `./attack.py` and `init_attack.py`
- test attack accuracy on ASV models: `./test.py` and `init_test.py`

## Guidelines to recovering audios
First, convert the device recording to `.wav` format with 16000 resample rate using ffmpeg: 
```bash
ffmpeg -i "input.wav" -osr 16000 output.wav
```

Set up the source audio and the folder you want to recover into in the script `audio_clipper/recover_pulse.py`.


Then, set the `offset` parameter to the observed first pulse point, as shown in the image, the first pulse is between 44-45 seconds, so set it to 44 * 16000:
![Alt text](image.png)
