import enum

import anyio.to_thread
import numpy
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Security
from h5py._hl.dataset import Dataset as H5Dataset
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT
from tiled.server.authentication import (  # type: ignore
    check_scopes,
    get_current_access_tags,
    get_current_principal,
    get_current_scopes,
    get_session_state,
)
from tiled.server.core import NoEntry
from tiled.server.dependencies import get_root_tree  # type: ignore
from tiled.server.schemas import Principal
from tiled.type_aliases import AccessTags, Scopes

# from tiled.server.router import *


class ScanType(enum.Enum):
    StepScan = "StepScan"
    FlyScan = "FlyScan"


visr_router = APIRouter()


@visr_router.get("/test-lookup")
async def test_lookup(request: Request):
    root = request.app.state.root_tree
    adapter = await root.lookup_adapter(
        ["aaf5e459-eb01-487c-8f90-a9468d4a2852", "primary", "data", "sample_stage-x"]
    )
    data = await anyio.to_thread.run_sync(adapter.read)
    return {"shape": list(data.shape), "dtype": str(data.dtype)}


@visr_router.get("/debug-tree/{path:path}")
async def debug_tree(path: str, request: Request):
    root = request.app.state.root_tree
    segments = [s for s in path.strip("/").split("/") if s]

    try:
        adapter = await root.lookup_adapter(segments)
    except Exception as e:
        return {"error": type(e).__name__, "detail": str(e), "segments": segments}

    adapter_type = type(adapter).__name__

    if hasattr(adapter, "keys_range"):
        try:
            keys = await adapter.keys_range(0, 100)
            return {"adapter_type": adapter_type, "children": list(keys)}
        except Exception as e:
            return {
                "error": type(e).__name__,
                "detail": str(e),
                "adapter_type": adapter_type,
            }
    else:
        # Leaf node — read it
        try:
            data = await adapter.read()
            return {
                "adapter_type": adapter_type,
                "shape": list(data.shape),
                "dtype": str(data.dtype),
            }
        except Exception as e:
            return {
                "error": type(e).__name__,
                "detail": str(e),
                "adapter_type": adapter_type,
            }


async def get_data(root, segments) -> H5Dataset | numpy.ndarray | dict:
    try:
        adapter = await root.lookup_adapter(segments)
    except Exception:
        raise

    adapter_type = type(adapter).__name__

    if hasattr(adapter, "keys_range"):
        try:
            keys = await adapter.keys_range(0, 100)
            return {"adapter_type": adapter_type, "children": list(keys)}
        except Exception:
            raise
    else:
        # Leaf node — read it
        try:
            data = await adapter.read()
            return data
        except Exception:
            raise


async def fill_data(root, segments, shape=None, fill_value=numpy.nan):
    try:
        return await get_data(root, segments)
    except NoEntry:
        if shape is None:
            raise
        return numpy.full(shape, fill_value)


@visr_router.get("/binned/{path:path}")
async def binned(  # type: ignore
    path: str,
    request: Request,
    x_dim_index: int = 0,
    y_dim_index: int = 1,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    width: int | None = None,
    height: int | None = None,
    slice_dim: list[str] | None = Query(  # noqa: B008
        None, description="Repeatable: dim:center:thickness"
    ),
    principal: Principal | None = Depends(get_current_principal),  # type: ignore  # noqa: B008
    root_tree=Depends(get_root_tree),  # type: ignore  # noqa: B008
    session_state: dict = Depends(get_session_state),  # type: ignore  # noqa: B008
    authn_access_tags: AccessTags | None = Depends(get_current_access_tags),  # type: ignore  # noqa: B008
    authn_scopes: Scopes = Depends(get_current_scopes),  # type: ignore  # noqa: B008
    _=Security(check_scopes, scopes=["read:data"]),  # noqa: B008
):
    """Fetch a folded representation of an array dataset."""
    root = request.app.state.root_tree
    segments = [s for s in path.strip("/").split("/") if s]
    uid = segments[0]
    # data = await get_data(root, segments)

    # entry = await get_entry(
    #     path,
    #     ["read:data"],
    #     principal,
    #     authn_access_tags,
    #     authn_scopes,
    #     root_tree,
    #     session_state,
    #     request.state.metrics,
    #     None,
    #     getattr(request.app.state, "access_policy", None),
    # )

    # # Only allow array-like adapters (must have .read)
    # if not callable(getattr(entry, "read", None)):
    #     raise HTTPException(
    #         status_code=400,
    #         detail=f"Entry at path '{path}' is not array-like and cannot be binned.",
    #     )
    # array_entry = cast(ArrayAdapter, entry)
    # try:
    #     with record_timing(request.state.metrics, "read"):
    #       data = await ensure_awaitable(array_entry.read) # type: ignore[attr-defined]
    # except Exception as e:
    #     raise HTTPException(
    #         status_code=500,
    #         detail=f"Error reading array data from entry at path '{path}': {e}",
    #     ) from e

    try:
        red_total = await get_data(root, [uid, "primary", "RedTotal"])
        green_total = await get_data(root, [uid, "primary", "GreenTotal"])
        blue_total = await get_data(root, [uid, "primary", "BlueTotal"])
        data = {
            "RedTotal": red_total,
            "GreenTotal": green_total,
            "BlueTotal": blue_total,
        }
        assert isinstance(red_total, H5Dataset) or isinstance(red_total, numpy.ndarray)
        assert isinstance(green_total, H5Dataset) or isinstance(
            green_total, numpy.ndarray
        )
        assert isinstance(blue_total, H5Dataset) or isinstance(
            blue_total, numpy.ndarray
        )
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(f"Could not find data channels for '{uid = }': {e}"),
        ) from None

    try:
        readback_x = await get_data(root, [uid, "primary", "sample_stage-x"])  # noqa: F841
        scan_type = ScanType.FlyScan
    except NoEntry:
        readback_x = await get_data(root, [uid, "primary", "X"])  # noqa: F841
        scan_type = ScanType.StepScan
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(f"Could not find readback data for '{uid = }': {e}"),
        ) from None

    assert isinstance(readback_x, H5Dataset) or isinstance(readback_x, numpy.ndarray)

    if scan_type == ScanType.FlyScan:
        readback_y = fill_data(
            root, [uid, "primary", "sample_stage-y"], readback_x.shape
        )  # noqa: F841
        readback_z = fill_data(
            root, [uid, "primary", "sample_stage-z"], readback_x.shape
        )  # noqa: F841
    else:
        readback_y = await fill_data(root, [uid, "primary", "Y"], readback_x.shape)  # noqa: F841
        readback_z = await fill_data(root, [uid, "primary", "Z"], readback_x.shape)  # noqa: F841

    readbacks = numpy.array(  # noqa: F841
        [
            readback_x,
            readback_y,
            readback_z,
        ]
    )

    print(uid)

    # mask out the points that lie outside the slice
    mask = numpy.ones(readbacks.size, dtype=bool)
    if slice_dim is not None:
        for slice_spec in slice_dim:
            try:
                dim_str, center_str, thick_str = slice_spec.split(":")
                dim = int(dim_str)
                center = float(center_str)
                thickness = float(thick_str)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid slice_dim format: {slice_spec}."
                        " Expected dim:center:thickness"
                    ),
                ) from None
            if dim < 0 or dim >= readbacks.shape[0]:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"slice_dim index {dim} is out of range"
                        f" (0-{readbacks.shape[0] - 1})"
                    ),
                )
            if dim in (x_dim_index, y_dim_index):
                raise HTTPException(
                    status_code=400,
                    detail=f"slice_dim cannot contain x or y dimension {dim}",
                )
            mask &= numpy.abs(readbacks[dim, :] - center) <= thickness

        readbacks = readbacks[:, mask]
        data = {channel: d[mask] for channel, d in data.items()}

    print(f"After slicing: {readbacks.shape}, {red_total.shape}")

    # readback_x = readbacks[x, :]
    # readback_y = readbacks[y, :]

    # # bundle the kwargs
    # histogram2d_kwargs = {}
    # if all(opt is not None for opt in (width, height)):
    #     histogram2d_kwargs["bins"] = (width, height)
    # if all(opt is not None for opt in (xmin, xmax, ymin, ymax)):
    #     histogram2d_kwargs["range"] = ((xmin, xmax), (ymin, ymax))

    # binned_output = {
    #     channel: compute_binned_image(
    #         data[channel], readback_x, readback_y, **histogram2d_kwargs
    #     )
    #     for channel in ("RedTotal", "GreenTotal", "BlueTotal")
    # }

    # return {"data": binned_output}


def compute_binned_image(data, readback_x, readback_y, **kwargs):
    counts, edges_x, edges_y = numpy.histogram2d(readback_x, readback_y, **kwargs)
    weights, _, _ = numpy.histogram2d(readback_x, readback_y, weights=data, **kwargs)
    img = numpy.divide(weights, counts, out=numpy.zeros_like(weights), where=counts > 0)
    return {"img": img, "x": edges_x, "y": edges_y}
