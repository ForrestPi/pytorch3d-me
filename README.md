# Pytorch3D-Me

A [Pytorch3D](https://github.com/facebookresearch/pytorch3d) 0.6.1 extension with features introduced in [FitMe](https://github.com/lattas/FitMe) (CVPR 2023) and [AvatarMe++](https://github.com/lattas/avatarme) (TPAMI 2021),
which introduces additional functionality in texturing and shading. In detail we add:
- A renderer object for rendering directly in UV-space,
- A blinn-phong based shader,
- The option to use multiple reflectance textures with a single mesh, including Diffuse Albedo, Specular Albedo, Diffuse Normals, Specular Normals and Occlusion Shadow,
- Spatially-varying specular shininess,
- Subsurface-scattering approximation with spatially-varying translucency,
- Multiple Point and Directional lights per rendered batch item.

<br></br>
Below we show the skin shading comparison between 
a) `Pytorch3d` `TexturedSoftPhongShader` with the albedo texture and shape normals,
b) our `Pytorch3d-Me` Blinn-Phong shader, with separate textures for diffuse and specular albedo and normals
c) previous with additional subsurface scattering approximation and
d) previous with additional occlusion shadow. Additional discussion is in included in the [AvatarMe++](https://arxiv.org/abs/2112.05957) paper and the qualitative comparison is shown below:

![AvatarMe Rendering Comparisons](media/rendering-comparisons.png)

# Installation
To install `Pytorch3d-Me` you need to build this repo from source
following the standard installation instructions at [INSTALL.md](./INSTALL.md).
In short, first install the prerequisites:
```
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install -c pytorch pytorch=1.9.1 torchvision cudatoolkit=10.2
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub

# Demos and examples
conda install jupyter
pip install scikit-image matplotlib imageio plotly opencv-python
```
And then build and install the project:
```
cd pytorch3d-me
pip install -e .
```

# Citations
 
If you find this extension useful in your research consider citing the works below:
```bibtex
@inproceedings{lattas2023fitme,
  title={FitMe: Deep Photorealistic 3D Morphable Model Avatars},
  author={Lattas, Alexandros and Moschoglou, Stylianos and Ploumpis, Stylianos
          and Gecer, Baris and Deng, Jiankang and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8629--8640},
  year={2023}
}

@article{lattas2021avatarme++,
  title={Avatarme++: Facial shape and brdf inference with photorealistic rendering-aware gans},
  author={Lattas, Alexandros and Moschoglou, Stylianos and Ploumpis, Stylianos
          and Gecer, Baris and Ghosh, Abhijeet and Zafeiriou, Stefanos},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={12},
  pages={9269--9284},
  year={2021},
  publisher={IEEE}
}
```
as well as the main Pytorch3D project:
```bibtex
@article{ravi2020pytorch3d,
    author = {Nikhila Ravi and Jeremy Reizenstein and David Novotny and Taylor Gordon
                  and Wan-Yen Lo and Justin Johnson and Georgia Gkioxari},
    title = {Accelerating 3D Deep Learning with PyTorch3D},
    journal = {arXiv:2007.08501},
    year = {2020},
}
```


Below is the official README

---


# Introduction

PyTorch3D provides efficient, reusable components for 3D Computer Vision research with [PyTorch](https://pytorch.org).

Key features include:

- Data structure for storing and manipulating triangle meshes
- Efficient operations on triangle meshes (projective transformations, graph convolution, sampling, loss functions)
- A differentiable mesh renderer

PyTorch3D is designed to integrate smoothly with deep learning methods for predicting and manipulating 3D data.
For this reason, all operators in PyTorch3D:

- Are implemented using PyTorch tensors
- Can handle minibatches of hetereogenous data
- Can be differentiated
- Can utilize GPUs for acceleration

Within FAIR, PyTorch3D has been used to power research projects such as [Mesh R-CNN](https://arxiv.org/abs/1906.02739).

## Installation

For detailed instructions refer to [INSTALL.md](INSTALL.md).

## License

PyTorch3D is released under the [BSD License](LICENSE).

## Tutorials

Get started with PyTorch3D by trying one of the tutorial notebooks.

|<img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/dolphin_deform.gif" width="310"/>|<img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/bundle_adjust.gif" width="310"/>|
|:-----------------------------------------------------------------------------------------------------------:|:--------------------------------------------------:|
| [Deform a sphere mesh to dolphin](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/deform_source_mesh_to_target_mesh.ipynb)| [Bundle adjustment](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/bundle_adjustment.ipynb) |

| <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/render_textured_mesh.gif" width="310"/> | <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/camera_position_teapot.gif" width="310" height="310"/>
|:------------------------------------------------------------:|:--------------------------------------------------:|
| [Render textured meshes](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/render_textured_meshes.ipynb)| [Camera position optimization](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/camera_position_optimization_with_differentiable_rendering.ipynb)|

| <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/pointcloud_render.png" width="310"/> | <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/cow_deform.gif" width="310" height="310"/>
|:------------------------------------------------------------:|:--------------------------------------------------:|
| [Render textured pointclouds](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/render_colored_points.ipynb)| [Fit a mesh with texture](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/fit_textured_mesh.ipynb)|

| <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/densepose_render.png" width="310"/> | <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/shapenet_render.png" width="310" height="310"/>
|:------------------------------------------------------------:|:--------------------------------------------------:|
| [Render DensePose data](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/render_densepose.ipynb)| [Load & Render ShapeNet data](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/dataloaders_ShapeNetCore_R2N2.ipynb)|

| <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/fit_textured_volume.gif" width="310"/> | <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/fit_nerf.gif" width="310" height="310"/>
|:------------------------------------------------------------:|:--------------------------------------------------:|
| [Fit Textured Volume](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/fit_textured_volume.ipynb)| [Fit A Simple Neural Radiance Field](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/fit_simple_neural_radiance_field.ipynb)|




## Documentation

Learn more about the API by reading the PyTorch3D [documentation](https://pytorch3d.readthedocs.org/).

We also have deep dive notes on several API components:

- [Heterogeneous Batching](https://github.com/facebookresearch/pytorch3d/tree/main/docs/notes/batching.md)
- [Mesh IO](https://github.com/facebookresearch/pytorch3d/tree/main/docs/notes/meshes_io.md)
- [Differentiable Rendering](https://github.com/facebookresearch/pytorch3d/tree/main/docs/notes/renderer_getting_started.md)

### Overview Video

We have created a short (~14 min) video tutorial providing an overview of the PyTorch3D codebase including several code examples. Click on the image below to watch the video on YouTube:

<a href="http://www.youtube.com/watch?v=Pph1r-x9nyY"><img src="http://img.youtube.com/vi/Pph1r-x9nyY/0.jpg" height="225" ></a>

## Development

We welcome new contributions to PyTorch3D and we will be actively maintaining this library! Please refer to [CONTRIBUTING.md](./.github/CONTRIBUTING.md) for full instructions on how to run the code, tests and linter, and submit your pull requests.

## Development and Compatibility

- `main` branch: actively developed, without any guarantee, Anything can be broken at any time
  - REMARK: this includes nightly builds which are built from `main`
  - HINT: the commit history can help locate regressions or changes
- backward-compatibility between releases: no guarantee. Best efforts to communicate breaking changes and facilitate migration of code or data (incl. models).

## Contributors

PyTorch3D is written and maintained by the Facebook AI Research Computer Vision Team.

In alphabetical order:

* Amitav Baruah
* Steve Branson
* Luya Gao
* Georgia Gkioxari
* Taylor Gordon
* Justin Johnson
* Patrick Labatut
* Christoph Lassner
* Wan-Yen Lo
* David Novotny
* Nikhila Ravi
* Jeremy Reizenstein
* Dave Schnizlein
* Roman Shapovalov
* Olivia Wiles

## Citation

If you find PyTorch3D useful in your research, please cite our tech report:

```bibtex
@article{ravi2020pytorch3d,
    author = {Nikhila Ravi and Jeremy Reizenstein and David Novotny and Taylor Gordon
                  and Wan-Yen Lo and Justin Johnson and Georgia Gkioxari},
    title = {Accelerating 3D Deep Learning with PyTorch3D},
    journal = {arXiv:2007.08501},
    year = {2020},
}
```

If you are using the pulsar backend for sphere-rendering (the `PulsarPointRenderer` or `pytorch3d.renderer.points.pulsar.Renderer`), please cite the tech report:

```bibtex
@article{lassner2020pulsar,
    author = {Christoph Lassner and Michael Zollh\"ofer},
    title = {Pulsar: Efficient Sphere-based Neural Rendering},
    journal = {arXiv:2004.07484},
    year = {2020},
}
```

## News

Please see below for a timeline of the codebase updates in reverse chronological order. We are sharing updates on the releases as well as research projects which are built with PyTorch3D. The changelogs for the releases are available under [`Releases`](https://github.com/facebookresearch/pytorch3d/releases),  and the builds can be installed using `conda` as per the instructions in [INSTALL.md](INSTALL.md).

**[Oct 6th 2021]:**   PyTorch3D [v0.6.0](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.6.0) released

**[Aug 5th 2021]:**   PyTorch3D [v0.5.0](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.5.0) released

**[Feb 9th 2021]:** PyTorch3D [v0.4.0](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.4.0) released with support for implicit functions, volume rendering and a [reimplementation of NeRF](https://github.com/facebookresearch/pytorch3d/tree/main/projects/nerf).

**[November 2nd 2020]:** PyTorch3D [v0.3.0](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.3.0) released, integrating the pulsar backend.

**[Aug 28th 2020]:**   PyTorch3D [v0.2.5](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.2.5) released

**[July 17th 2020]:**   PyTorch3D tech report published on ArXiv: https://arxiv.org/abs/2007.08501

**[April 24th 2020]:**   PyTorch3D [v0.2.0](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.2.0) released

**[March 25th 2020]:**   [SynSin](https://arxiv.org/abs/1912.08804) codebase released using PyTorch3D: https://github.com/facebookresearch/synsin

**[March 8th 2020]:**   PyTorch3D [v0.1.1](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.1.1) bug fix release

**[Jan 23rd 2020]:**   PyTorch3D [v0.1.0](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.1.0) released. [Mesh R-CNN](https://arxiv.org/abs/1906.02739) codebase released: https://github.com/facebookresearch/meshrcnn
