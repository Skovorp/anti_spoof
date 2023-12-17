# Anti Spoofing with RawNet2
Model weights that achieve <5% EER on eval ASVspoof 2019 are here: https://drive.google.com/file/d/1_MI3A3SPWcpsboGKdQeooWxJfzb-JVXt/view?usp=drive_link 


you can load them with gdown. also install everything from requirements
```
pip install -r ./requirements.txt
pip install gdown
gdown https://drive.google.com/file/d/1_MI3A3SPWcpsboGKdQeooWxJfzb-JVXt/view
```
- You can compute test accuracy with test_on_eval.py
- Run predictions on audios from /test_audios with test.py
- I trained winner model with train.py
- I ran ablations on configs from /ablation_configs with run_config_ablations.py
- my winner config is config.yaml

REPORT https://wandb.ai/sposiboh/anti_spoof/reports/Anti-spoofing--Vmlldzo2Mjg1MjYy?accessToken=y5wlnm9ky3apkp74mvgdsynn5yj0fzww1swixnnlpkb5hxl1wfrtwi1p0xjg9crd
