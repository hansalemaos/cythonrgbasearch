import os
import subprocess
import sys
import numpy as np

def _dummyimport():
    import Cython

try:
    from .cythonrgba import rgbapic,rgbpic

except Exception as e:
    cstring = r"""# distutils: language=c
# distutils: extra_compile_args=/openmp /O2
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3
# cython: initializedcheck=False

from cython.parallel cimport prange
cimport cython
import numpy as np
cimport numpy as np
import cython
cpdef void  rgbapic(cython.uint[:] sfara,cython.uint[:] vara, Py_ssize_t rgba_shape1, Py_ssize_t rgba_shape0,cython.uint[:] listresu ):
    cdef Py_ssize_t sha1,v
    cdef Py_ssize_t varalen=vara.shape[0]
    cdef Py_ssize_t rgbatotal=rgba_shape1*rgba_shape0
    for sha1 in prange(rgbatotal,nogil=True):
        for v in range(varalen):
            if sfara[sha1] == vara[v]:
                listresu[sha1]=v+1
                break

cpdef void  rgbpic(cython.uchar[:] b,cython.uchar[:] g,cython.uchar[:] r,cython.uchar[:,:] vara,cython.uint[:] listresu,Py_ssize_t rgbatotal ):
    cdef Py_ssize_t sha1,v
    cdef Py_ssize_t varalen=vara.shape[0]
    for sha1 in prange(rgbatotal,nogil=True):
        for v in range(varalen):
            if b[sha1] == vara[v][0] and g[sha1] == vara[v][1] and r[sha1] == vara[v][2]:
                listresu[sha1]=v+1
                break

"""
    pyxfile = f"cythonrgba.pyx"
    pyxfilesetup = f"cythonrgbacompiled_setup.py"

    dirname = os.path.abspath(os.path.dirname(__file__))
    pyxfile_complete_path = os.path.join(dirname, pyxfile)
    pyxfile_setup_complete_path = os.path.join(dirname, pyxfilesetup)

    if os.path.exists(pyxfile_complete_path):
        os.remove(pyxfile_complete_path)
    if os.path.exists(pyxfile_setup_complete_path):
        os.remove(pyxfile_setup_complete_path)
    with open(pyxfile_complete_path, mode="w", encoding="utf-8") as f:
        f.write(cstring)
    numpyincludefolder = np.get_include()
    compilefile = (
            """
	from setuptools import Extension, setup
	from Cython.Build import cythonize
	ext_modules = Extension(**{'py_limited_api': False, 'name': 'cythonrgba', 'sources': ['cythonrgba.pyx'], 'include_dirs': [\'"""
            + numpyincludefolder
            + """\'], 'define_macros': [], 'undef_macros': [], 'library_dirs': [], 'libraries': [], 'runtime_library_dirs': [], 'extra_objects': [], 'extra_compile_args': [], 'extra_link_args': [], 'export_symbols': [], 'swig_opts': [], 'depends': [], 'language': None, 'optional': None})

	setup(
		name='cythonrgba',
		ext_modules=cythonize(ext_modules),
	)
			"""
    )
    with open(pyxfile_setup_complete_path, mode="w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [x.lstrip().replace(os.sep, "/") for x in compilefile.splitlines()]
            )
        )
    subprocess.run(
        [sys.executable, pyxfile_setup_complete_path, "build_ext", "--inplace"],
        cwd=dirname,
        shell=True,
        env=os.environ.copy(),
    )
    try:
        from .cythonrgba import rgbapic, rgbpic

    except Exception as fe:
        sys.stderr.write(f'{fe}')
        sys.stderr.flush()



def get_pointer_array(original):
    """
    Get a pointer array from a NumPy array.

    Parameters:
    - original: NumPy array

    Returns:
    - Pointer array
    """
    dty = np.ctypeslib.as_ctypes_type(original.dtype)

    b = original.ctypes.data
    buff = (dty * original.size).from_address(b)

    aflat = np.frombuffer(buff, dtype=original.dtype)
    return aflat


def find_rgba_colors(pic, colors, dummy_alpha=0):
    """
    Find RGBA colors in a picture.

    Parameters:
    - pic: Input picture (NumPy array)
    - colors: Colors to find (NumPy array)
    - dummy_alpha: Dummy alpha value (default is 0)

    Returns:
    - NumPy recarray with color information
    """
    if not colors.flags['C_CONTIGUOUS']:
        colors = np.ascontiguousarray(colors)
    if pic.shape[-1] == 3:
        rgba = np.ascontiguousarray(np.dstack([pic, np.full(pic.shape[:2], dummy_alpha, dtype=np.uint8)]))
    else:
        if not pic.flags['C_CONTIGUOUS']:
            rgba = np.ascontiguousarray(pic)
        else:
            rgba = pic
    sfara = get_pointer_array(colors).view(np.uint32)
    vara = get_pointer_array(rgba).view(np.uint32)
    listresu = np.zeros((len(vara)), dtype=np.uint32)
    rgbapic(vara, sfara, rgba.shape[1], rgba.shape[0], listresu)
    nonz = np.nonzero(listresu)
    x,y = (np.divmod(nonz, pic.shape[0]))
    allcolors = colors[listresu[nonz] - 1]
    return np.rec.fromarrays(
        [allcolors[..., 2], allcolors[..., 1], allcolors[..., 0], allcolors[..., 3], x[0], y[0]],
        dtype=[('r', np.uint8), ('g', np.uint8), ('b', np.uint8), ('a', np.uint8), ('x', np.int64), ('y', np.int64)])

def find_rgb_colors(pic,colors):
    """
    Find RGB colors in a picture.

    Parameters:
    - pic: Input picture (NumPy array)
    - colors: Colors to find (NumPy array)

    Returns:
    - NumPy recarray with color information
    """
    b = np.ascontiguousarray(pic[..., 0].ravel())
    g = np.ascontiguousarray(pic[..., 1].ravel())
    r = np.ascontiguousarray(pic[..., 2].ravel())
    if not colors.flags['C_CONTIGUOUS']:
        colors = np.ascontiguousarray(colors)
    listresu = np.zeros(len(b), dtype=np.uint32)
    rgbatotal = len(b)
    rgbpic(b, g, r, colors, listresu, rgbatotal)
    nonz = np.nonzero(listresu)
    x, y = (np.divmod(nonz, pic.shape[0]))
    allcolors = colors[listresu[nonz] - 1]
    return np.rec.fromarrays(
        [allcolors[..., 2], allcolors[..., 1], allcolors[..., 0],  x[0], y[0]],
        dtype=[('r', np.uint8), ('g', np.uint8), ('b', np.uint8), ('x', np.int64), ('y', np.int64)])
