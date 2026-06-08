[![Beautiful-Brains logo](https://raw.githubusercontent.com/Tj-Ward/beautiful-brains/main/figures/logo.png)](https://github.com/Tj-Ward/beautiful-brains)

# Beautiful Brains 


  [![PyPI version](https://img.shields.io/pypi/v/beautiful-brains.svg)](https://pypi.org/project/beautiful-brains/)
  [![License](https://img.shields.io/pypi/l/beautiful-brains.svg)](https://pypi.org/project/beautiful-brains/)
  [![Python versions](https://img.shields.io/pypi/pyversions/beautiful-brains.svg)](https://pypi.org/project/beautiful-brains/)
  [![Downloads](https://img.shields.io/pypi/dm/beautiful-brains.svg)](https://pypi.org/project/beautiful-brains/)

Beautiful-Brains creates notebook-friendly NIfTI figures from intensity images, masks, and label maps.

Beautiful-Brains provides a Python toolkit for loading, smoothing, transforming, coregistering, slicing, and compositing NIfTI images in Jupyter notebooks and Python scripts. Images are read with [NiBabel](https://nipy.org/nibabel/), processed with [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/), and rendered with [Pillow](https://pillow.readthedocs.io/en/stable/). Registration, masking, and bias-field correction are performed with [ANTsPy](https://antspy.readthedocs.io/en/stable/).

![Beautiful-Brains](https://github.com/Tj-Ward/beautiful-brains/blob/main/figures/video.webp)

Beautiful-Brains was created from a collection of personal scripts. These tools saved me a lot of time making figures for presentations that are visually pleasing. I also used these features to create lightbox-style QC images for automated neuroimaging pipelines.

## Installation

Install the current [Beautiful-Brains release](https://pypi.org/project/beautiful-brains/) with `pip`:

```bash
pip install beautiful-brains
```

## Usage

See the [demonstration notebook](https://github.com/Tj-Ward/beautiful-brains/blob/main/Demonstration.md).

## AI Disclaimer

This project originated as a collection of human-coded scripts. Claude Code (Sonnet 4.6) and Codex (GPT-5.5) were used to convert a working codebase into a Python package. These AI tools restructured and rewrote nearly everything. Future development will continue using AI-assisted coding tools.

## Support

Please send questions, bug reports, and feature requests to the [issue tracker](https://github.com/Tj-Ward/beautiful-brains/issues).

Documentation will be expanded after the project leaves alpha.

## License

Beautiful-Brains is licensed under the terms of the [GNU General Public License version 3](https://www.gnu.org/licenses/gpl-3.0.en.html). For more information, see the [LICENSE](https://github.com/Tj-Ward/beautiful-brains/blob/main/LICENSE) file.
:::
