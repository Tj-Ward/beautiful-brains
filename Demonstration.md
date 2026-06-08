# Beautiful Brains 
## Basic Guide

beautiful-brains is an visualization tool for presentations

Features:

 * Image registration (rigid and linear warp)
 * Mask images based on intensity or through skullstripping
 * Smooth images
 * Multi-slice panels
 * Easily automate png/jpeg creation

Future Features:

 * Template figures for QC checks and seeing your data
 * Intensity normalization

*Powered by [ANTsPy](https://antspy.readthedocs.io)*


```python
import beautiful_brains as bb
import inspect
import numpy as np
```

### Define the path to your images


```python
MRI_PATH = 'MPRAGE.nii.gz'
PET_PATH = 'FBB.nii'
PET_DYNAMIC_PATH = 'FBB_dynamic.nii'

TEMPLATE_PATH = 'MNI152_T1_1mm.nii.gz'
```

### Here, we load the two nifti images into the BBImage 


```python
print(inspect.signature(bb.load))
```

    (path: 'str | Path', *, LUT: 'str | Path | None' = None, colormap: 'Any | None' = None, threshold: 'tuple[float, float] | None' = None, indices: 'Sequence[int] | str | None' = None, scale: 'float | None' = None, interp: 'Any | None' = None, sharpen: 'int | None' = None, bias_correction: 'bool' = False, debug: 'bool' = False) -> 'BBImage'



```python
MRI = bb.load(
  MRI_PATH,
  colormap="gray",
  interp='NEAREST',
  bias_correction=True,
  scale=1,
  sharpen=0,
  threshold=(100,1900)
)

PET = bb.load(
  PET_PATH,
  colormap="nih",
  interp='NEAREST',
  scale=1,
  sharpen=0,
  threshold=(100,10000)
)

PET_dynamic = bb.load(
  PET_DYNAMIC_PATH,
  colormap="nih",
  interp='NEAREST',
  scale=1,
  sharpen=0,
  threshold=(100,10000)
)
```

## Transformations

First, I will register the PET to the MRI using rigid body transformation


```python
PET = bb.transform(source=PET, target=MRI, warp='Rigid')
```

Next, I'll warp the MRI to my template using 12 degrees of freedom


```python
MRI = bb.transform(source=MRI, target=TEMPLATE_PATH, warp='Affine')
```


```python
PET = bb.apply_transform(source=PET, matrix=MRI.transform_info)
```

I can then apply that transformation matrix to other images in the same space


```python
PET = bb.apply_transform(source=PET,matrix=MRI.transform_info)
```

I can also align all the images in a given timeseries.


```python
PET_dynamic = bb.align_timeseries(source=PET_dynamic,warp='Rigid')
```

    Caution: align_timeseries() can be slow because it computes one rigid-body alignment for every non-reference frame in the timeseries.


## Metadata
This stores useful information about the image


```python
MRI.metadata()
```

    BBImage metadata
    name: MPRAGE.nii.gz
    path: MPRAGE.nii.gz
    kind: intensity
    frames: 1
    volume:
      data: matrix shape=(208, 256, 256), dtype=float32
      affine: matrix shape=(4, 4), dtype=float64
      voxel_sizes: (1.0, 1.0, 1.0)
    display:
      scale: 1.0
      interp: NEAREST
      sharpen: 0
      colormap: gray
      LUT: None
      threshold: (100.0, 1900.0)
    indices: None
    crop_info: None
    mask: none
    timeseries_transform_info: none
    transform:
      status: pending
      warp: Affine
      template: MNI152_T1_1mm.nii.gz
      forward_transforms: 1 file(s)
      inverse_transforms: 1 file(s)
      template_volume: data matrix shape=(182, 218, 182), dtype=float32, affine matrix shape=(4, 4), dtype=float64
      template_shape: (182, 218, 182)
      template_affine: matrix shape=(4, 4), dtype=float64



```python
print(PET.volume.path)
print(type(PET.volume.image))
print(type(PET.volume.data), PET.volume.data.shape)
print(type(PET.volume.affine), PET.volume.affine.shape)
```

    FBB.nii
    <class 'nibabel.nifti1.Nifti1Image'>
    <class 'numpy.ndarray'> (256, 256, 89)
    <class 'numpy.ndarray'> (4, 4)



```python
PET_dynamic.metadata()
```

    BBImage metadata
    name: FBB_dynamic.nii
    path: FBB_dynamic.nii
    kind: intensity
    frames: 8
    volume:
      data: matrix shape=(192, 192, 89, 8), dtype=float32
      affine: matrix shape=(4, 4), dtype=float64
      voxel_sizes: (1.333333, 1.333333, 2.779999)
    display:
      scale: 1.0
      interp: NEAREST
      sharpen: 0
      colormap: nih
      LUT: None
      threshold: (100.0, 10000.0)
    indices: None
    crop_info: None
    mask: none
    timeseries_transform_info: 7 transform(s) for 8 frame(s)
    transform:
      status: none


## Reslicing

I want both images to have the same dimensions so that they can share a mask. This is not necessary. Reslice is its own function so that you can transform the image multiple times without needing to interpolate the data multiple times. You can also choose to not reslice at all. the slice function will select the correct voxels based on the affine matrix, even if the logical slice is oblique to the 3 axes.

Here, I am going to choose LANCZOS interpolation which is visually pleasing.

I could also select NEAREST and BICUBIC


```python
PET = PET.reslice(interp='LANCZOS')
MRI = MRI.reslice(interp='LANCZOS')
```

I can take a slice of my images to see what they look like

I changed the scale here because I wanted small pictures of the data. It is recommended to upscale 2x or 3x when making figures which improves visualization.


```python
MRI.scale=1
MRI.slice(80)
```




    
![png](Demonstration_files/Demonstration_26_0.png)
    




```python
PET.scale=1
PET.colormap = bb.colormaps.as_listed_colormap('nih')
PET.slice(80)
```




    
![png](Demonstration_files/Demonstration_27_0.png)
    




```python
PET.colormap = bb.colormaps.as_listed_colormap('soft-nih')
PET.slice(80)
```




    
![png](Demonstration_files/Demonstration_28_0.png)
    



## Masking

I have two options to create masks

bb.create_mask 
bb.create_brainmask

The former masks the entire image and the later masks the brain 

Both are powered by antspy


```python
mask = bb.create_mask(
  source=MRI,
  pad=4,
  cleanup=0,
)
```


```python
MRI.mask = mask
MRI.slice(80)
```




    
![png](Demonstration_files/Demonstration_32_0.png)
    



Here is why I chose to reslice the PET image. Now I can use the same mask for the PET as I made for the MRI.


```python
PET.mask = mask
```


```python
PET.slice(80)
```




    
![png](Demonstration_files/Demonstration_35_0.png)
    



## Smoothing


```python
PET_smoo = bb.smooth_image(source=PET,kernel=(10,10,10))
```


```python
PET_smoo.scale=1
PET_smoo.slice(80)
```




    
![png](Demonstration_files/Demonstration_38_0.png)
    



## Cropping / bounding box


```python
# Auto-crop or define the bounding box yourself
MRI = bb.crop(
  MRI,
  how="otsu",
  pad=0
)
```


```python
print(MRI.volume.data.shape)
print(MRI.crop_info)
```

    (182, 218, 182)
    ((0, 182), (7, 218), (0, 173))



```python
MRI.crop_info = ((20, 160), (20, 200), (0, 182))
```


```python
MRI.slice(80)
```




    
![png](Demonstration_files/Demonstration_43_0.png)
    




```python
# Resetting the crop
MRI.crop_info = ((0, 182), (0, 218), (0, 182))
```


```python
MRI.slice(80)
```




    
![png](Demonstration_files/Demonstration_45_0.png)
    



## Modifying visualization info

All the visualization metadata can be defined as such.

These values define how the slice function renders the image.


```python
MRI.colormap = (bb.colormaps.as_listed_colormap('grey'))
MRI.threshold = (0,1900)
```


```python
PET.colormap = (bb.colormaps.as_listed_colormap('nih'))
PET.threshold = (100,10000)
```


```python
MRI.interp='NEAREST'
MRI.scale=3
```


```python
MRI.sharpen=50
```


```python
PET.interp = 'NEAREST'
PET.scale=3
```

## Creating figure panels


```python
import numpy as np
Ratio = MRI.slice(120,'axial').size
fig = bb.bbfigure(
  size=1,
  grid=(5, 1),
  dpi=2400,
    box_ratio=(Ratio[0]/Ratio[1]),
  background=(0, 0, 0, 0),
)
for i,slice_i in enumerate(np.round(np.linspace(50,100,5),0)):
    fig[i, 0] = MRI.slice(int(slice_i), "axial",alpha=1)
img = fig.render()
img.save('./beautiful-brains/figures/5x1_MRI_axial.png')
img
```




    
![png](Demonstration_files/Demonstration_54_0.png)
    




```python
Ratio = MRI.slice(120,'axial').size
fig = bb.bbfigure(
  size=1,
  grid=(5, 1),
  dpi=2400,
    box_ratio=(Ratio[0]/Ratio[1]),
  background=(0, 0, 0, 0),
)
for i,slice_i in enumerate(np.round(np.linspace(50,100,5),0)):
    fig[i, 0] = PET.slice(int(slice_i), "axial",alpha=1)
img = fig.render()
img.save('./beautiful-brains/figures/5x1_PET_axial.png')
img
```




    
![png](Demonstration_files/Demonstration_55_0.png)
    



## Overlaying PET on MRI


```python
Ratio = MRI.slice(120,'axial').size
fig = bb.bbfigure(
  size=1,
  grid=(5, 1),
  dpi=2400,
    box_ratio=(Ratio[0]/Ratio[1]),
  background=(0, 0, 0, 0),
)
for i,slice_i in enumerate(np.round(np.linspace(50,100,5),0)):
    fig[i, 0] = MRI.slice(int(slice_i), "axial")
    fig[i, 0].append(PET.slice(int(slice_i), "axial",alpha=.55))
img = fig.render()
img.save('./beautiful-brains/figures/5x1_MRI_PET_axial.png')
img
```




    
![png](Demonstration_files/Demonstration_57_0.png)
    




```python
Ratio = MRI.slice(120,'coronal').size
fig = bb.bbfigure(
  size=1,
  grid=(5, 1),
  dpi=2400,
    box_ratio=(Ratio[0]/Ratio[1]),
  background=(0, 0, 0, 0),
)
for i,slice_i in enumerate(np.round(np.linspace(70,140,5)[::-1],0)):
    fig[i, 0] = MRI.slice(int(slice_i), "coronal")
    fig[i, 0].append(PET.slice(int(slice_i), "coronal",alpha=.55))
img = fig.render()
img.save('./beautiful-brains/figures/5x1_MRI_PET_coronal.png')
img
```




    
![png](Demonstration_files/Demonstration_58_0.png)
    




```python

```


```python
Ratio = MRI.slice(120,'axial').size
fig = bb.bbfigure(
  size=1,
  grid=(5, 2),
  dpi=2400,
    box_ratio=(Ratio[0]/Ratio[1]),
  background=(0, 0, 0, 0),
)
for i,slice_i in enumerate(np.round(np.linspace(50,100,5),0)):
    fig[i, 0] = MRI.slice(int(slice_i), "axial")
for i,slice_i in enumerate(np.round(np.linspace(50,100,5),0)[::1]):
    fig[i, 1] = (PET.slice(int(slice_i), "axial"))
img = fig.render()
img.save('./beautiful-brains/figures/5x2_MRI_PET_3.png')
```


```python
img
```




    
![png](Demonstration_files/Demonstration_61_0.png)
    




```python

```

### Create movies 


```python
print(inspect.signature(bb.make_video))
```

    (frames: 'Iterable[Any]', timing: 'float', filepath: 'str | Path') -> 'Path'


I'm going to create a short animation using 8 frames from a PET scan. 

Before creating the movie, this scan needs to be preprocessed for visualization purposes.

 - Align frames (already done above)
 - Smooth
 - Crop


```python
PET_dynamic.slice(42,frame=i, plane="axial",alpha=1)
```




    
![png](Demonstration_files/Demonstration_66_0.png)
    




```python
PET_dynamic = bb.crop(
  PET_dynamic,
  how="mean",
  pad=4
)
```


```python
PET_dynamic.slice(42,frame=i, plane="axial",alpha=1)
```




    
![png](Demonstration_files/Demonstration_68_0.png)
    




```python
PET_dynamic = bb.smooth_image(source=PET_dynamic,kernel=8)
PET_dynamic.slice(42,frame=i, plane="axial",alpha=1)
```




    
![png](Demonstration_files/Demonstration_69_0.png)
    




```python
pet_mask = bb.create_mask(
  source=PET_dynamic,
  pad=2,
  cleanup=0,
)
```


```python
PET_dynamic.mask = pet_mask
```


```python
Ratio = PET_dynamic.slice(50,'axial').size
frame_count = PET_dynamic.volume.data.shape[-1]
fig = bb.bbfigure(
  size=1,
  grid=(frame_count, 1),
  dpi=2400,
    box_ratio=(Ratio[0]/Ratio[1]),
  background=(0, 0, 0, 0),
)
for i in np.arange(0,frame_count):
    fig[i, 0] = PET_dynamic.slice(42,frame=i, plane="axial",alpha=1)
img = fig.render()
img
```




    
![png](Demonstration_files/Demonstration_72_0.png)
    




```python
Ratio = PET_dynamic.slice(50,'axial').size
slices = list(np.arange(0,frame_count))
#slices.extend(slices[:-1:][::-1])
time = 0.2
frames = []
for i in np.arange(0,frame_count):
    fig = bb.bbfigure(
      size=1,
      grid=(1, 1),
      dpi=300,
        box_ratio=(Ratio[0]/Ratio[1]),
      background=(0, 0, 0, 0),
    )
    fig[0, 0] = PET_dynamic.slice(42,frame=i, plane="axial",alpha=1)
    frames.append(fig)

bb.make_video(frames,time,'pet_dynamic_video.webp')
```




    PosixPath('pet_dynamic_video.webp')




```python

```

### Here, I create a small animated webp file for the github README.md


```python
Ratio = PET.slice(120,'axial').size
slices = list(np.arange(50,111,1))
slices.extend(slices[:-1:][::-1])
time = 0.1
frames = []
for i,slice_i in enumerate(slices):
    fig = bb.bbfigure(
      size=0.1,
      grid=(1, 1),
      dpi=600,
        box_ratio=(Ratio[0]/Ratio[1]),
      background=(0, 0, 0, 0),
    )
    fig[0, 0] = PET.slice(int(slice_i), "axial",interp='LANCZOS')
    frames.append(fig)

bb.make_video(frames,time,'video.webp')
```




    PosixPath('video.webp')



### You can use any matplotlib ListedColormap or make your own


```python
type(PET.colormap)
```




    matplotlib.colors.ListedColormap




```python
bb.colorbar(colormap='nih', height=50, length=500)
```




    
![png](Demonstration_files/Demonstration_79_0.png)
    




```python
# 'soft' colormaps have a small transparent portion at low intensities
bb.colorbar(colormap='soft-nih', height=50, length=500)
```




    
![png](Demonstration_files/Demonstration_80_0.png)
    




```python
bb.colorbar(colormap='gray', height=50, length=500)
```




    
![png](Demonstration_files/Demonstration_81_0.png)
    




```python
bb.colorbar(colormap='viridis', height=50, length=500)
```




    
![png](Demonstration_files/Demonstration_82_0.png)
    




```python
bb.colorbar(colormap='plasma', height=50, length=500)
```




    
![png](Demonstration_files/Demonstration_83_0.png)
    




```python
bb.colorbar(colormap='jet', height=50, length=500)
```




    
![png](Demonstration_files/Demonstration_84_0.png)
    




```python
bb.colorbar(colormap='hot', height=50, length=500)
```




    
![png](Demonstration_files/Demonstration_85_0.png)
    




```python
bb.colorbar(colormap='bone', height=50, length=500)
```




    
![png](Demonstration_files/Demonstration_86_0.png)
    




```python
bb.colorbar(colormap='copper', height=50, length=500)
```




    
![png](Demonstration_files/Demonstration_87_0.png)
    


