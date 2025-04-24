# livecc-utils

LiveCC Utils is a supplement to qwen-vl-utils, which contains a set of helper functions for processing and integrating visual language information with LiveCC Model.

## Install

```bash
pip install qwen-vl-utils livecc-utils
```

## Feature

### Decord Video Reader Plus
Compared to ```_read_video_decord``` in ```qwen-vl-utils```, we provide ```_read_video_decord_plus``` that can handle video with ```video_start```, ```video_end```, and support both smart nframe and strict fps. Please refer to [src/livecc_utils/video_process_patch.py](src/livecc_utils/video_process_patch.py)

#### Usage
Easy. Just put the import of  ```livecc-utils``` before ```qwen_vl_utils```.

```
import livecc_utils
from qwen_vl_utils import ...
```

### Easy KV Cache for Multi-turn
Original Qwen2VL has some bugs during multi-turn conversation with past_key_values. We provide a patch ```prepare_multiturn_multimodal_inputs_for_generation``` at [src/livecc_utils/generation_patch.py](src/livecc_utils/generation_patch.py). During inference, using ```model.prepare_inputs_for_generation = functools.partial(prepare_multiturn_multimodal_inputs_for_generation, model)``` then do faster inference with past_key_values!

#### Usage
Easy. Just let ```model.prepare_inputs_for_generation = functools.partial(prepare_multiturn_multimodal_inputs_for_generation, model)```.

