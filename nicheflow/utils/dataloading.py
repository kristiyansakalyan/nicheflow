import torch

from nicheflow.datasets.st_dataset_base import STTrainDataBatch, STTrainDataItem, STValDataItem


def pad_tensor(tensor: torch.Tensor, target_length: int, dim: int) -> torch.Tensor:
    """
    Pad a tensor with zeros along a specified dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to pad.
    target_length : int
        Desired size along the dimension `dim`.
    dim : int
        The dimension along which to pad.

    Returns
    -------
    torch.Tensor
        Tensor padded to shape where `tensor.shape[dim] == target_length`.
        If the input length is already greater than or equal to `target_length`,
        the input tensor is returned unchanged.
    """
    current_length = tensor.shape[dim]
    pad_size = target_length - current_length

    if pad_size <= 0:
        return tensor

    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_size
    pad = torch.zeros(*pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=dim)


def make_mask(lengths: list[int], max_len: int) -> torch.Tensor:
    """
    Create a binary mask indicating valid entries and padded positions.

    Parameters
    ----------
    lengths : list of int
        Lengths of each sequence in the batch.
    max_len : int
        Target length to pad sequences to.

    Returns
    -------
    torch.Tensor
        A binary mask of shape ``(len(lengths), max_len)``,
        where 1 marks valid elements and 0 marks padding.
    """
    return torch.arange(max_len, device=torch.device("cpu")).expand(
        len(lengths), max_len
    ) < torch.tensor(lengths).unsqueeze(1)


def pad_and_stack_field(key: str, max_len: int, batch: list[STTrainDataItem]) -> torch.Tensor:
    """
    Pad and stack tensors from a batch under a given key.

    Parameters
    ----------
    key : str
        Key in the batch dictionaries to extract tensors from.
    max_len : int
        Target length to pad to along ``dim=1``.
    batch : list of MicroEnvTrainItem
        A list of training items (dict-like) containing tensors.

    Returns
    -------
    torch.Tensor
        A stacked tensor of shape ``(bs, n_microenvs, max_len, feat_dim)``.
    """
    padded = [pad_tensor(sample[key], max_len, dim=1) for sample in batch]
    return torch.stack(padded, dim=0)


def stack_and_expand_ohe(
    batch: list[STTrainDataItem],
    key: str,
    n_microenvs: int,
    n_points: int,
) -> torch.Tensor:
    """
    Stack and expand a one-hot tensor from (bs, D) to (bs * n_microenvs, n_points, D)
    """
    bs = len(batch)
    ohe = torch.stack([sample[key] for sample in batch], dim=0)  # (bs, D)
    return (
        ohe[:, None, None, :]  # (bs, 1, 1, D)
        .expand(bs, n_microenvs, n_points, -1)  # (bs, n_microenvs, n_points, D)
        .reshape(bs * n_microenvs, n_points, -1)  # (bs * n_microenvs, n_points, D)
    )


def microenv_train_collate(batch: list[STTrainDataItem]) -> STTrainDataBatch:
    """
    Collate function for Transformer-style models on microenvironments.

    This function:
      - Pads point clouds within each microenvironment to the maximum
        number of points in the batch.
      - Stacks features, positions, and timepoint encodings across samples.
      - Creates binary masks to distinguish valid points from padded ones.

    Parameters
    ----------
    batch : list of MicroEnvTrainItem
        A list of training items. Each item is expected to have:
            - ``X_t1`` : torch.Tensor, shape (n_microenvs, n_points_t1, feat_dim)
            - ``X_t2`` : torch.Tensor, shape (n_microenvs, n_points_t2, feat_dim)
            - ``pos_t1`` : torch.Tensor, shape (n_microenvs, n_points_t1, 2)
            - ``pos_t2`` : torch.Tensor, shape (n_microenvs, n_points_t2, 2)
            - ``t1_ohe`` : torch.Tensor, shape (d_t1,)
            - ``t2_ohe`` : torch.Tensor, shape (d_t2,)

    Returns
    -------
    MicroEnvTrainBatch
        A batch dictionary containing:
            - ``X_t1`` : (bs * n_microenvs, max_n_points_t1, feat_dim)
            - ``X_t2`` : (bs * n_microenvs, max_n_points_t2, feat_dim)
            - ``pos_t1`` : (bs * n_microenvs, max_n_points_t1, 2)
            - ``pos_t2`` : (bs * n_microenvs, max_n_points_t2, 2)
            - ``t1_ohe`` : (bs * n_microenvs, max_n_points_t1, d_t1)
            - ``t2_ohe`` : (bs * n_microenvs, max_n_points_t2, d_t2)
            - ``mask_t1`` : (bs * n_microenvs, max_n_points_t1)
            - ``mask_t2`` : (bs * n_microenvs, max_n_points_t2)
    """
    bs = len(batch)
    n_microenvs = batch[0]["X_t1"].shape[0]

    # Determine maximum points across the batch
    max_n_points_t1 = max(sample["X_t1"].shape[1] for sample in batch)
    max_n_points_t2 = max(sample["X_t2"].shape[1] for sample in batch)

    # Pad and stack fields
    X_t1 = pad_and_stack_field("X_t1", max_n_points_t1, batch).reshape(
        bs * n_microenvs, max_n_points_t1, -1
    )
    X_t2 = pad_and_stack_field("X_t2", max_n_points_t2, batch).reshape(
        bs * n_microenvs, max_n_points_t2, -1
    )
    pos_t1 = pad_and_stack_field("pos_t1", max_n_points_t1, batch).reshape(
        bs * n_microenvs, max_n_points_t1, -1
    )
    pos_t2 = pad_and_stack_field("pos_t2", max_n_points_t2, batch).reshape(
        bs * n_microenvs, max_n_points_t2, -1
    )

    # Stack and expand one-hot encodings
    t1_ohe = stack_and_expand_ohe(batch, "t1_ohe", n_microenvs, max_n_points_t1)
    t2_ohe = stack_and_expand_ohe(batch, "t2_ohe", n_microenvs, max_n_points_t2)

    # Build masks
    mask_t1 = torch.stack(
        [make_mask([sample["X_t1"].shape[1]] * n_microenvs, max_n_points_t1) for sample in batch]
    ).reshape(bs * n_microenvs, max_n_points_t1)

    mask_t2 = torch.stack(
        [make_mask([sample["X_t2"].shape[1]] * n_microenvs, max_n_points_t2) for sample in batch]
    ).reshape(bs * n_microenvs, max_n_points_t2)

    return {
        "X_t1": X_t1,
        "X_t2": X_t2,
        "pos_t1": pos_t1,
        "pos_t2": pos_t2,
        "t1_ohe": t1_ohe,
        "t2_ohe": t2_ohe,
        "mask_t1": mask_t1,
        "mask_t2": mask_t2,
    }


def microenv_val_collate(batch: list[STValDataItem]) -> STValDataItem:
    if len(batch) != 1:
        raise ValueError(
            f"The validation batch should always contain one item. Got {len(batch)} items.F"
        )

    sample = batch[0]

    t1_ohe = sample["t1_ohe"][None, None, :].expand(
        sample["X_t1"].size(0), sample["X_t1"].size(1), -1
    )
    t2_ohe = sample["t2_ohe"][None, None, :].expand(
        sample["X_t2"].size(0), sample["X_t2"].size(1), -1
    )

    return {
        "X_t1": sample["X_t1"],
        "X_t2": sample["X_t2"],
        "pos_t1": sample["pos_t1"],
        "pos_t2": sample["pos_t2"],
        "t1_ohe": t1_ohe,
        "t2_ohe": t2_ohe,
        "global_pos_t2": sample["global_pos_t2"],
        "global_ct_t2": sample["global_ct_t2"],
    }


def sp_rpc_train_collate(batch: list[STTrainDataItem]) -> STTrainDataItem:
    # We know that we would have the same number of points in both
    # time 1 and time 2 => We just need to stack.
    X_t1 = torch.stack([el["X_t1"] for el in batch], dim=0)
    X_t2 = torch.stack([el["X_t2"] for el in batch], dim=0)

    pos_t1 = torch.stack([el["pos_t1"] for el in batch], dim=0)
    pos_t2 = torch.stack([el["pos_t2"] for el in batch], dim=0)

    # We also need to expand the t1 and t2 ohes
    bs, n_cells = X_t1.shape[:2]

    t1_ohe = torch.stack([el["t1_ohe"] for el in batch], dim=0)[:, None, :].expand(bs, n_cells, -1)
    t2_ohe = torch.stack([el["t2_ohe"] for el in batch], dim=0)[:, None, :].expand(bs, n_cells, -1)

    return {
        "X_t1": X_t1,
        "X_t2": X_t2,
        "pos_t1": pos_t1,
        "pos_t2": pos_t2,
        "t1_ohe": t1_ohe,
        "t2_ohe": t2_ohe,
    }


def sp_rpc_val_collate(batch: list[STValDataItem]) -> STValDataItem:
    if len(batch) != 1:
        raise ValueError(
            f"The validation batch should always contain one item. Got {len(batch)} items.F"
        )

    sample = batch[0]
    X_t1 = sample["X_t1"].unsqueeze(0)
    X_t2 = sample["X_t2"].unsqueeze(0)

    pos_t1 = sample["pos_t1"].unsqueeze(0)
    pos_t2 = sample["pos_t2"].unsqueeze(0)

    t1_ohe = sample["t1_ohe"][None, None, :].expand(X_t1.size(0), X_t1.size(1), -1)
    t2_ohe = sample["t2_ohe"][None, None, :].expand(X_t2.size(0), X_t2.size(1), -1)

    return {
        "X_t1": X_t1,
        "X_t2": X_t2,
        "pos_t1": pos_t1,
        "pos_t2": pos_t2,
        "t1_ohe": t1_ohe,
        "t2_ohe": t2_ohe,
        "global_pos_t2": sample["global_pos_t2"],
        "global_ct_t2": sample["global_ct_t2"],
    }
