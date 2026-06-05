[![Beautiful-Brains logo](https://raw.githubusercontent.com/Tj-Ward/beautiful-brains/main/figures/logo.png)](https://github.com/Tj-Ward/beautiful-brains)

# Beautiful Brains

Beautiful-Brains creates notebook-friendly NIfTI figures from intensity images, masks, and label maps.

Beautiful-Brains provides a Python toolkit for loading, smoothing, transforming, coregistering, slicing, and compositing NIfTI images in Jupyter notebooks and Python scripts. Images are read with [NiBabel](https://nipy.org/nibabel/), processed with [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/), and rendered with [Pillow](https://pillow.readthedocs.io/en/stable/). Registration, masking, and bias-field correction are performed with ANTsPy.

Beautiful-Brains began as a series of personal utilities and scripts.

## To Do Before Beta Release

* [x] Complete basic feature set
* [ ] Design starter templates
* [ ] Add documentation
* [ ] Set up PyPI
* [ ] Complete human code review

## AI Disclaimer

This project began as a set of human-coded utilities for private use. Claude Code (Sonnet 4.6) and Codex (GPT-5.5) were used to convert a working codebase into a Python package. These tools helped redesign the code and build the package as of the first alpha release. Future development will continue using AI-assisted coding tools.

All code merged into a stable release must undergo human review.

## Installation

Install the current [Beautiful-Brains release](https://pypi.org/project/beautiful-brains/) with `pip`:

```bash
pip install beautiful-brains
```

## Usage

See the [demonstration notebook](https://github.com/Tj-Ward/beautiful-brains/blob/main/Demonstration.md).

## Support

Please send questions, bug reports, and feature requests to the [issue tracker](https://github.com/Tj-Ward/beautiful-brains/issues).

Documentation will be expanded after the project leaves alpha.

## License

Beautiful-Brains is licensed under the terms of the [GNU General Public License version 3](https://www.gnu.org/licenses/gpl-3.0.en.html). For more information, see the [LICENSE](https://github.com/Tj-Ward/beautiful-brains/blob/main/LICENSE) file.
::: 

