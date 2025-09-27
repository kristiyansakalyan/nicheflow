import torch

from nicheflow.datasets.microenv_dataset import MicroEnvTrainBatch, MicroEnvTrainItem


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


def pad_and_stack_field(key: str, max_len: int, batch: list[MicroEnvTrainItem]) -> torch.Tensor:
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


def collate_function_transformer(batch: list[MicroEnvTrainItem]) -> MicroEnvTrainBatch:
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
            - ``X_t1`` : (bs, n_microenvs, max_n_points_t1, feat_dim)
            - ``X_t2`` : (bs, n_microenvs, max_n_points_t2, feat_dim)
            - ``pos_t1`` : (bs, n_microenvs, max_n_points_t1, 2)
            - ``pos_t2`` : (bs, n_microenvs, max_n_points_t2, 2)
            - ``t1_ohe`` : (bs, d_t1)
            - ``t2_ohe`` : (bs, d_t2)
            - ``mask_t1`` : (bs, n_microenvs, max_n_points_t1)
            - ``mask_t2`` : (bs, n_microenvs, max_n_points_t2)
    """
    n_microenvs = batch[0]["X_t1"].shape[0]

    # Determine maximum points across the batch
    max_n_points_t1 = max(sample["X_t1"].shape[1] for sample in batch)
    max_n_points_t2 = max(sample["X_t2"].shape[1] for sample in batch)

    # Pad and stack fields
    X_t1 = pad_and_stack_field("X_t1", max_n_points_t1, batch)
    X_t2 = pad_and_stack_field("X_t2", max_n_points_t2, batch)
    pos_t1 = pad_and_stack_field("pos_t1", max_n_points_t1, batch)
    pos_t2 = pad_and_stack_field("pos_t2", max_n_points_t2, batch)

    # Stack one-hot encodings
    t1_ohe = torch.stack([sample["t1_ohe"] for sample in batch], dim=0)  # (bs, d_t1)
    t2_ohe = torch.stack([sample["t2_ohe"] for sample in batch], dim=0)  # (bs, d_t2)

    # Build masks
    mask_t1 = torch.stack(
        [make_mask([sample["X_t1"].shape[1]] * n_microenvs, max_n_points_t1) for sample in batch]
    )  # (bs, n_microenvs, max_n_points_t1)

    mask_t2 = torch.stack(
        [make_mask([sample["X_t2"].shape[1]] * n_microenvs, max_n_points_t2) for sample in batch]
    )  # (bs, n_microenvs, max_n_points_t2)

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
