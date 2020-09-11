# pytorch-imageClassification 

### 1. Update log
- basic version (support resnet-18 and ImageFolder datasets)

### 2. Usages
 #### Edit config.yaml
 #### training: `python train.py`
 #### validation: `python eval.py`

### 3. Support functions 
- [x] pytorch model(from torchvision)
- [x] pytorch pre-defined multiple optimizer: Adam, SGD, Adagrad etc.
- [x] multiple lr_scheduler: step, multistep, reduce, explr


### 4. TODO
- [ ] multiple models: pre-trained models、Res2Net、EfficientNet etc.
- [ ] multiple lr_scheduler: coslr, warmup etc.
- [ ] fp16
- [ ] Learning rate scheduler
