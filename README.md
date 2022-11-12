# ldf_kitti_object_eval_python

Long-Distance-Focused KITTI object detection evaluation in Python (2d/bev/3d/aos)

## Acknowledgement
 - This repository is developed based on [traveller59/kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python).

## Dependencies
 - Only support python 3.6+, need `numpy`, `skimage`, `numba`, `fire`. If you have Anaconda, just install `cudatoolkit` in Anaconda.

## Usage
 - Evaluate your detection results
   ```
   python evaluate.py evaluate --result_path=/path/to/your_result_folder --label_path=/path/to/your_gt_label_folder --label_split_file=/path/to/val.txt --current_classes=0,1,2
   ```
