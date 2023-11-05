# Experiment on DTD Dataset

## Experiment Setting

| config | use pretrained | freeze backbone | freeze until epoch | gradient clip | augmentation | use scheduler |
|:------:|:--------------:|:---------------:|:------------------:|:-------------:|:------------:|:-------------:|
|  exp1  |       √        |        √        |         15         |       √       |      √       |       √       |
|  exp2  |       √        |        √        |         15         |       √       |      x       |       √       |
|  exp3  |       √        |        √        |         15         |       √       |      √       |       x       |
|  exp4  |       √        |        √        |         15         |       x       |      √       |       √       |
|  exp5  |       √        |        √        |         -          |       √       |      √       |       √       |
|  exp6  |       √        |        x        |         -          |       √       |      √       |       √       |
|  exp7  |       x        |        -        |         -          |       √       |      √       |       √       |
|  exp8  |       x        |        -        |         -          |       x       |      √       |       √       |
|  exp9  |       x        |        -        |         -          |       x       |      x       |       √       |
| exp10  |       x        |        -        |         -          |       x       |      x       |       x       |

> **freeze until epoch:** freeze backbone until this epoch, then unfreeze
> **gradient clip:** clip gradient norm to 10
> **augmentation:** random rotation, random horizontal flip, color jitter

- exp1: full
- exp2: w/o augmentation
- exp3: w/o scheduler
- exp4: w/o gradient clip
- exp5: w/o unfreeze (keep backbone frozen during entire training)
- exp6: w/o freeze backbone
- exp7: w/o pretrained
- exp8: w/o pretrained, w/o gradient clip
- exp9: w/o pretrained, w/o gradient clip, w/o augmentation
- exp10: w/o pretrained, w/o gradient clip, w/o augmentation, w/o scheduler

**Other hyperparameters settings:**

- batch size: 64
- optimizer: SGD
- learning rate: 0.01
- momentum: 0.9
- weight decay: 0.0001
- scheduler: StepLR (if used)
- lr_scheduler_step_size: 10
- lr_scheduler_gamma: 0.1

## Experiment Result

See the full wandb log [here](https://wandb.ai/lcf235/ResNet18_DTD_full_experiment).

## To Reproduce

Go to the root directory of this project, then run the following command to reproduce full experiment:

```bash
ls -l config | awk '{print $9}' | grep .yml | xargs -n 1  -I {} python train.py config/{}
```