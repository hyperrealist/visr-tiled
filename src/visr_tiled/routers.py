import enum
import inspect
import logging
import os

import anyio.to_thread
import numpy
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Security
from h5py._hl.dataset import Dataset as H5Dataset
from scanspec.core import stack2dimension
from scanspec.specs import Spec
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

logger = logging.getLogger(__name__)

CROSS_CHANNEL_RETRIES = int(os.getenv("VISR_TILED_CROSS_CHANNEL_RETRIES", "5"))
CROSS_CHANNEL_RETRY_DELAY = float(
    os.getenv("VISR_TILED_CROSS_CHANNEL_RETRY_DELAY", "0.1")
)


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
            if inspect.iscoroutinefunction(adapter.read):
                data = await adapter.read()
            else:
                data = await anyio.to_thread.run_sync(adapter.read)
            result: dict = {"adapter_type": adapter_type}
            if hasattr(data, "shape"):
                result["shape"] = list(data.shape)
            if hasattr(data, "dtype"):
                result["dtype"] = str(data.dtype)
            return result
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
            if inspect.iscoroutinefunction(adapter.read):
                data = await adapter.read()
            else:
                data = await anyio.to_thread.run_sync(adapter.read)
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


def _channel_length(array: H5Dataset | numpy.ndarray | dict) -> int:
    assert isinstance(array, (H5Dataset, numpy.ndarray))
    return array.shape[-1]


async def get_setpoints(root, uid):
    """Return setpoints from the bluesky start document stored in a node's metadata."""
    adapter = await root.lookup_adapter([uid])
    metadata = adapter.metadata()
    try:
        spec = Spec.deserialize(metadata["start"]["spec"])
    except KeyError as e:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Could not find 'start.spec' in metadata for '{uid}': {e}",
        ) from None

    midpoints = list(stack2dimension(spec.calculate()).midpoints.values())
    x = midpoints[0]
    y = midpoints[1] if len(midpoints) > 1 else numpy.full(x.shape, numpy.nan)
    z = midpoints[2] if len(midpoints) > 2 else numpy.full(x.shape, numpy.nan)
    return numpy.array([x, y, z])


async def _fetch_readback_x(root, uid) -> tuple[H5Dataset | numpy.ndarray, ScanType]:
    try:
        x = await get_data(root, [uid, "primary", "internal", "sample_stage-x"])
        scan_type = ScanType.FlyScan
    except NoEntry:
        try:
            x = await get_data(root, [uid, "primary", "sample_stage-x"])
            scan_type = ScanType.FlyScan
        except NoEntry:
            x = await get_data(root, [uid, "primary", "X"])
            scan_type = ScanType.StepScan
    assert isinstance(x, (H5Dataset, numpy.ndarray))
    return x, scan_type


async def _fetch_readback_yz(root, uid, scan_type, x_shape):
    if scan_type == ScanType.FlyScan:
        try:
            y = await get_data(root, [uid, "primary", "internal", "sample_stage-y"])
        except NoEntry:
            y = await fill_data(root, [uid, "primary", "sample_stage-y"], x_shape)
        try:
            z = await get_data(root, [uid, "primary", "internal", "sample_stage-z"])
        except NoEntry:
            z = await fill_data(root, [uid, "primary", "sample_stage-z"], x_shape)
    else:
        y = await fill_data(root, [uid, "primary", "Y"], x_shape)
        z = await fill_data(root, [uid, "primary", "Z"], x_shape)
    return y, z


async def get_readbacks(root, uid, readback_x):
    """
    Utility function to load readback positions (x, y, z) and detect scan type.

    Args:
        root: The root tree to fetch data from.
        uid: Unique identifier for the dataset.
        readback_x: The x readback data.

    Returns:
        A tuple containing:
            - A numpy array with readback positions (x, y, z).
            - The detected scan type (FlyScan or StepScan).
    """
    try:
        readback_x, scan_type = await _fetch_readback_x(root, uid)
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(f"Could not find readback data for '{uid = }': {e}"),
        ) from None

    assert isinstance(readback_x, H5Dataset) or isinstance(readback_x, numpy.ndarray)
    readback_y, readback_z = await _fetch_readback_yz(
        root, uid, scan_type, readback_x.shape
    )

    # x, y, and z are each fetched independently and can race a live write,
    # landing at different lengths. Target the longest length seen on the
    # first read and retry re-fetching all three until they reach it,
    # falling back to truncating to the shortest if they don't converge
    # within the retry budget -- same strategy as the cross-channel
    # reconciliation in binned().
    lengths = {
        "x": _channel_length(readback_x),
        "y": _channel_length(readback_y),
        "z": _channel_length(readback_z),
    }
    target_len = max(lengths.values())
    for attempt in range(CROSS_CHANNEL_RETRIES):
        if min(lengths.values()) >= target_len:
            break
        logger.info(
            "Cross-axis length mismatch for '%s' (attempt %d/%d): %s, "
            "waiting to reach %d",
            uid,
            attempt + 1,
            CROSS_CHANNEL_RETRIES,
            lengths,
            target_len,
        )
        await anyio.sleep(CROSS_CHANNEL_RETRY_DELAY)
        try:
            new_x, new_scan_type = await _fetch_readback_x(root, uid)
            new_y, new_z = await _fetch_readback_yz(
                root, uid, new_scan_type, new_x.shape
            )
        except Exception as e:
            logger.info(
                "Re-fetch failed while reconciling readback lengths for '%s': %s",
                uid,
                e,
            )
            break
        readback_x, readback_y, readback_z, scan_type = (
            new_x,
            new_y,
            new_z,
            new_scan_type,
        )
        lengths = {
            "x": _channel_length(readback_x),
            "y": _channel_length(readback_y),
            "z": _channel_length(readback_z),
        }
    else:
        logger.info(
            "Giving up on cross-axis reconciliation for '%s' after %d attempt(s), "
            "falling back to shortest: %s",
            uid,
            CROSS_CHANNEL_RETRIES,
            lengths,
        )

    common_len = min(target_len, min(lengths.values()))
    if any(length != common_len for length in lengths.values()):
        readback_x = readback_x[:common_len]
        readback_y = readback_y[:common_len]
        readback_z = readback_z[:common_len]

    return numpy.array([readback_x, readback_y, readback_z]), scan_type


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
    setpoints: bool = False,
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
    """Fetch a folded representation of an array dataset.

    Args:
        x_dim_index: Index into the position array to use as the x axis (default 0).
        y_dim_index: Index into the position array to use as the y axis (default 1).
        xmin: Lower bound of the x histogram range. When combined with xmax, ymin, and
            ymax, passed as the ``range`` argument to ``numpy.histogram2d``.
        xmax: Upper bound of the x histogram range.
        ymin: Lower bound of the y histogram range.
        ymax: Upper bound of the y histogram range.
        width: Number of bins along the x axis. Requires ``height`` to also be set.
        height: Number of bins along the y axis. Requires ``width`` to also be set.
        setpoints: If ``True``, derive positions from the ScanSpec setpoints stored in
            the run's start document rather than from the recorded readback values.
        slice_dim: Repeatable query parameter that restricts which data points
            contribute to the image by filtering along a dimension that is neither x
            nor y.  Each value must be formatted as ``dim:center:thickness``, where
            *dim* is the integer dimension index, *center* is the centre of the slice,
            and *thickness* is the half-width (points with
            ``|position - center| <= thickness`` are kept).
    """
    root = request.app.state.root_tree
    segments = [s for s in path.strip("/").split("/") if s]
    uid = segments[0]

    # load data
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

    # Get positions either from setpoints (spec) or readbacks
    try:
        if setpoints:
            readbacks = await get_setpoints(root, uid)
        else:
            readbacks, _ = await get_readbacks(root, uid, None)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(f"Could not find position data for '{uid = }': {e}"),
        ) from None

    # The three totals and the readbacks are each fetched independently and can
    # race a live write, landing at different lengths. Target the longest length
    # seen on the first read (an indication the others will reach it shortly),
    # and retry re-fetching everything until they do. If they don't catch up
    # within the retry budget, fall back to truncating to whatever the shortest
    # currently is, so we always return a consistent (if possibly stale) image.
    lengths = {
        "RedTotal": _channel_length(red_total),
        "GreenTotal": _channel_length(green_total),
        "BlueTotal": _channel_length(blue_total),
        "readbacks": readbacks.shape[-1],
    }
    target_len = max(lengths.values())
    for attempt in range(CROSS_CHANNEL_RETRIES):
        if min(lengths.values()) >= target_len:
            break
        logger.info(
            "Cross-channel length mismatch for '%s' (attempt %d/%d): %s, "
            "waiting to reach %d",
            uid,
            attempt + 1,
            CROSS_CHANNEL_RETRIES,
            lengths,
            target_len,
        )
        await anyio.sleep(CROSS_CHANNEL_RETRY_DELAY)
        try:
            new_red_total = await get_data(root, [uid, "primary", "RedTotal"])
            new_green_total = await get_data(root, [uid, "primary", "GreenTotal"])
            new_blue_total = await get_data(root, [uid, "primary", "BlueTotal"])
            if setpoints:
                new_readbacks = await get_setpoints(root, uid)
            else:
                new_readbacks, _ = await get_readbacks(root, uid, None)
        except Exception as e:
            logger.info(
                "Re-fetch failed while reconciling channel lengths for '%s': %s",
                uid,
                e,
            )
            break
        # Only commit the re-fetch once all four succeed together, so
        # red_total/green_total/blue_total/readbacks and `lengths` never
        # drift out of sync with each other.
        red_total, green_total, blue_total, readbacks = (
            new_red_total,
            new_green_total,
            new_blue_total,
            new_readbacks,
        )
        lengths = {
            "RedTotal": _channel_length(red_total),
            "GreenTotal": _channel_length(green_total),
            "BlueTotal": _channel_length(blue_total),
            "readbacks": readbacks.shape[-1],
        }
    else:
        logger.info(
            "Giving up on cross-channel reconciliation for '%s' after %d attempt(s), "
            "falling back to shortest: %s",
            uid,
            CROSS_CHANNEL_RETRIES,
            lengths,
        )

    common_len = min(target_len, min(lengths.values()))
    if any(length != common_len for length in lengths.values()):
        red_total = red_total[:common_len]
        green_total = green_total[:common_len]
        blue_total = blue_total[:common_len]
        readbacks = readbacks[:, :common_len]

    data = {
        "RedTotal": red_total,
        "GreenTotal": green_total,
        "BlueTotal": blue_total,
    }

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

    logger.debug(
        "readbacks shape %s, red_total shape %s (sliced=%s)",
        readbacks.shape,
        red_total.shape,
        slice_dim is not None,
    )

    x_positions = readbacks[x_dim_index, :]
    y_positions = readbacks[y_dim_index, :]

    # bundle the kwargs
    histogram2d_kwargs = {}
    if all(opt is not None for opt in (width, height)):
        histogram2d_kwargs["bins"] = (width, height)
    if all(opt is not None for opt in (xmin, xmax, ymin, ymax)):
        histogram2d_kwargs["range"] = ((xmin, xmax), (ymin, ymax))

    binned_output = {}
    for channel in ("RedTotal", "GreenTotal", "BlueTotal"):
        binned_channel = compute_binned_image(
            data[channel], x_positions, y_positions, **histogram2d_kwargs
        )
        binned_output[channel] = binned_channel["img"].tolist()
    binned_output["x_limits"] = binned_channel["x"].tolist()
    binned_output["y_limits"] = binned_channel["y"].tolist()

    return binned_output


def compute_binned_image(data, readback_x, readback_y, **kwargs):
    counts, edges_x, edges_y = numpy.histogram2d(readback_x, readback_y, **kwargs)
    weights, _, _ = numpy.histogram2d(readback_x, readback_y, weights=data, **kwargs)
    img = numpy.divide(weights, counts, out=numpy.zeros_like(weights), where=counts > 0)
    return {"img": img, "x": edges_x, "y": edges_y}
