# Probabilistic-Segmentation
 Implementation of probabilistic segmentation for robot trajectories.

Corresponding paper can be found for free [here](https://arxiv.org/abs/2404.18383), please read for method details.

Several methods exist for teaching robots, with one of the most prominent being Learning from Demonstration (LfD). If a primitive skill is demonstrated, it can be immediately encoded. However, demonstrations may be more complex and are required to be broken down into motion primitives. To do this, we apply segmentation to the captured demonstrations. However, we consider multiple modes of interaction in our segmentation, and combine segmented modes probabilistically.

<img src="https://github.com/brenhertel/Probabilistic-Segmentation/blob/main/pictures/motion_prim_library_sequence_crop.png" alt="" width="800"/>

This repository implements the method described in the paper above using Python. All code can be found in `scripts\segmentation.py` which implements probabilistic segmentation as well as examples. If you have any questions, please contact Brendan Hertel (brendan_hertel@student.uml.edu).

If you use the code present in this repository, please cite the following paper:
```
@inproceedings{hertel2023reusable_skills,
  title={A Framework for Learning and Reusing Robotic Skills},
  author={Hertel, Brendan and Tran, Nhu and Elkoudi, Meriem and Azadeh, Reza},
  booktitle={21st International Conference on Ubiquitous Robots (UR)},
  year={2024}
}
```
