## README

The metadata of the **DAVIDE dataset** is organized into two main categories: **Meta-Info** and **Annotations**. These files support dataset loading and evaluation with our PyTorch-based [Benchmark Repository](https://github.com/germanftv/DAVIDE-Benckmark).

---

### 📁 Meta-Info

These files provide essential information used by the dataset classes for the training, validation, and testing splits. They define the partitioning of the dataset and serve as entry points for loading video clips.

- **Train Split Meta-Info**  
  `meta_info/meta_info_DAVIDE_train.txt`  
  Contains the list of video clips used for training, along with basic metadata needed by the dataset loader.

- **Validation Split Partition**  
  `meta_info/meta_info_DAVIDE_val_partition.txt`  
  Defines the subset of training clips used for validation purposes.

- **Test Split Meta-Info**  
  `meta_info/meta_info_DAVIDE_test.txt`  
  Lists all test clips along with the necessary information for testing.

---

### 📝 Annotations

These files provide detailed annotations for the **test** split, including environmental context, motion characteristics, proximity estimations, and depth map confidence metrics. For further details, refer to Section 3.4 of our [paper](https://arxiv.org/abs/2409.01274).

- **Environment and Motion Labels**  
  `annotations/test_env_motion.csv`  
  Annotates each test clip with environment type (e.g., indoor/outdoor) and motion patterns (e.g. CM: Camera Motion, CM+MO: Camera Motion + Moving Objects).

- **Proximity Labels**  
    `annotations/test_proximity.csv`  
    Contains proximity level annotations for each test clip, categorized into three distinct ranges: close, mid, and far.

- **Average Confidence of Depth Maps**  
    `annotations/test_avg_conf_depth.csv`  
    Provides the average confidence score for the depth measurements per frame.
