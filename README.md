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

- exp1: full
- exp2: w/o augmentation
- exp3: w/o scheduler
- exp4: w/o gradient clip
- exp5: w/o unfreeze
- exp6: w/o freeze backbone
- exp7: w/o pretrained
- exp8: w/o pretrained, w/o gradient clip
- exp9: w/o pretrained, w/o gradient clip, w/o augmentation
- exp10: w/o pretrained, w/o gradient clip, w/o augmentation, w/o scheduler