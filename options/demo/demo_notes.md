## 💡 Notes on Demo Settings

The demo script ([`basicsr/demo.py`](../../basicsr/demo.py)) uses a specific set of fixed settings, which can be found in the [options/demo](../../options/demo/) folder. This document highlights the settings that differ from those used in our experiments ([`options/experiments`](../../options/experiments/)).

### 1. Test-Time Local Converter (TLC) Layers
In our paper, we reported results for the RGBD ShiftNet architecture using the [Test-Time Local Converter (TLC)](https://github.com/megvii-research/TLC) introduced by [Chu et al. (ECCV 2022)](https://arxiv.org/abs/2112.04491). At that time, only the local average pooling layer ([`AvgPool2d`](../../basicsr/models/archs/tlc_utils.py#L26)) was replaced. 

For the demo script, we implemented an additional local cross-attention fusion block ([`LocalCrossAttentionFusionBlock`](../../basicsr/models/archs/tlc_utils.py#L126)). Using the TLC strategy with `LocalCrossAttentionFusionBlock` in the RGBD ShiftNet architecture during testing improves deblurring performance.

To enable both `local_avgpool` and `local_crossattn` layers with TLC, update the testing settings in the configuration YAML file as follows:

```yaml
# Testing settings
test:
  use_TLC: true
  tlc_layers: [local_avgpool, local_crossattn]
```

If `tlc_layers` is not specified, only `local_avgpool` will be used by default.

### 2. Temporal Context Window Length for Inference
To reduce GPU memory usage in the demo, we decreased the number of frames in the temporal context window for inference. This adjustment applies to video-based architectures such as `Shiftnet_RGBD`, `Shiftnet_baseRGB`, `VRT`, and `RVRT`.

You can modify the number of frames based on your available computing resources by updating the following variables in the configuration YAML file. For example, to use 16 frames:

```yaml
datasets:
  test:
    num_frame_testing: 16

# Testing settings
test:
  num_frame_testing: 16
```

### 3. ShiftNet Models in SOTA Comparison
The base RGB ShiftNet and the extended RGBD ShiftNet architectures were each trained three times. For the SOTA comparison experiment presented in our paper, we reported the average performance metrics across these training runs.

In this demo, we showcase results using the following checkpoints, which provide robust performance:
- `Shiftnet_baseRGB`: Checkpoint with `seed: 10`
- `Shiftnet_RGBD`: Checkpoint with `seed: 13`