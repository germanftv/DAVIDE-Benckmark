# License and Acknowledgements


This project is released under the [MIT license](../LICENSE.txt). It builds upon the following projects:


- [BasicSR](https://github.com/XPixelGroup/BasicSR)

    The project BasicSR (Basic Super Restoration) is an open-source image and video restoration toolbox based on PyTorch. Our project utilizes the code structure of BasicSR and makes use of many utility functions provided in their code. We have also incorporated the EDVR architecture for comparison with state-of-the-art video deblurring methods. You can find the license of BasicSR [here](./LICENSE_BasicSR).

- [Shift-Net](https://github.com/dasongli1/Shift-Net)

    The Shift-Net project offers a PyTorch implementation of the research paper titled "A Simple Baseline for Video Restoration with Spatial-temporal Shift". Their code is built upon the BasicSR toolbox. We have adopted their Shift-Net architecture as our foundational deblurring model. We have refactored their original implementation of Shift-Net to improve readability, and we have expanded this architecture to incorporate depth information for depth-aware video deblurring. Additionally, we have utilized and made modifications to some of their utility functions.

- [KAIR](https://github.com/cszn/KAIR/)

    The KAIR repository provides training and testing codes for several learning-based models for image and video restoration problems. We have incorporated the VRT and RVRT architectures in our code for comparison with state-of-the-art video deblurring methods. In addition, we have utilized and made modifications to some of their utility functions. The license of KAIR is [here](./LICENSE_KAIR).

- [Restormer](https://github.com/swz30/Restormer)

    The Restormer project is the official implementation of the "Restormer: Efficient Transformer for High-Resolution Image Restoration" paper. Our extended RGBD Shift-Net architecture incorporates the proposed DaT block, which draws inspiration from the transformer blocks in the Restormer architecture. We have integrated and modified their PyTorch implementation into our Shift-Net implementation. The license of Restormer is [here](./LICENSE_Restormer.md).

- [TLC](https://github.com/megvii-research/TLC)

    The TLC repository is the official implementation of the paper "Improving Image Restoration by Revisiting Global Information Aggregation". Their code is built upon the BasicSR toolbox. We have incorporated the Test-time Local Converter (TLC) strategy into the Shift-Net architecture to enhance the performance during inference. The license of TLC is [here](./LICENSE_TLC).

- [HINet](https://github.com/megvii-model/HINet)

    The HINet project provides an implementation of the paper: "HINet: Half Instance Normalization Network for Image Restoration". Their code is based on the BasicSR toolbox, and we have incorporated some of their code simplifications compared to the original BasicSR code. The license of HINet is [here](./LICENSE_HINet).