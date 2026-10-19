import argparse
import torch
import os


def to_1d_bool(x):
    if x.dim() == 2 and x.size(-1) == 1:
        x = x.squeeze(-1)
    return x.detach().cpu().bool()


def to_2d_bool(x):
    return x.detach().cpu().bool()


def count_positive_bonds(bond_labels, mask_i, mask_j):
    n = bond_labels.size(0)

    pair_mask = (
        mask_i[:, None] & mask_j[None, :]
    ) | (
        mask_j[:, None] & mask_i[None, :]
    )

    upper = torch.triu(
        torch.ones(n, n, dtype=torch.bool),
        diagonal=1,
    )

    return int((bond_labels & pair_mask & upper).sum().item())


def is_bad_sample(sample, anchor_high=1.9):
    pos = sample["positions"].detach().cpu().float()

    anchor1 = to_1d_bool(sample["anchor1_mask"])
    anchor2 = to_1d_bool(sample["anchor2_mask"])
    linker = to_1d_bool(sample["linker_mask"])

    bond_labels = to_2d_bool(sample["bond_labels"])
    bond_mask = to_2d_bool(sample["bond_mask"])
    valid_bond_labels = bond_labels & bond_mask

    linker_idx = torch.where(linker)[0]
    anchor1_idx = torch.where(anchor1)[0]
    anchor2_idx = torch.where(anchor2)[0]

    reasons = []

    if len(anchor1_idx) != 1:
        reasons.append("bad_anchor1_count")

    if len(anchor2_idx) != 1:
        reasons.append("bad_anchor2_count")

    if len(linker_idx) <= 0:
        reasons.append("no_linker_atoms")

    if len(reasons) > 0:
        return True, reasons

    d1 = torch.cdist(pos[anchor1_idx], pos[linker_idx]).min().item()
    d2 = torch.cdist(pos[anchor2_idx], pos[linker_idx]).min().item()

    n_a1_linker_bond = count_positive_bonds(
        valid_bond_labels,
        anchor1,
        linker,
    )

    n_a2_linker_bond = count_positive_bonds(
        valid_bond_labels,
        anchor2,
        linker,
    )

    # 主要过滤条件：
    # 1. anchor-linker 真实距离太远
    # 2. bond label 中没有 anchor-linker 正样本
    if d1 > anchor_high:
        reasons.append(f"anchor1_linker_too_far:{d1:.3f}")

    if d2 > anchor_high:
        reasons.append(f"anchor2_linker_too_far:{d2:.3f}")

    if n_a1_linker_bond <= 0:
        reasons.append("no_positive_anchor1_linker_bond_label")

    if n_a2_linker_bond <= 0:
        reasons.append("no_positive_anchor2_linker_bond_label")

    return len(reasons) > 0, reasons


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="./datasets/protac_pretrain_train.pt",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./datasets/protac_pretrain_train_clean.pt",
    )

    parser.add_argument(
        "--anchor_high",
        type=float,
        default=1.9,
    )

    args = parser.parse_args()

    data = torch.load(args.input, map_location="cpu")

    clean_data = []
    bad_records = []

    for idx, sample in enumerate(data):
        bad, reasons = is_bad_sample(
            sample,
            anchor_high=args.anchor_high,
        )

        if bad:
            name = sample.get("name", f"sample_{idx}")
            bad_records.append((idx, name, reasons))
        else:
            clean_data.append(sample)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    torch.save(clean_data, args.output)

    report_path = args.output.replace(".pt", "_removed.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"input: {args.input}\n")
        f.write(f"output: {args.output}\n")
        f.write(f"n_input: {len(data)}\n")
        f.write(f"n_clean: {len(clean_data)}\n")
        f.write(f"n_removed: {len(bad_records)}\n\n")

        for idx, name, reasons in bad_records:
            f.write(f"[REMOVED] idx={idx} name={name} reasons={reasons}\n")

    print("========== FILTER RESULT ==========")
    print(f"input:     {args.input}")
    print(f"output:    {args.output}")
    print(f"n_input:   {len(data)}")
    print(f"n_clean:   {len(clean_data)}")
    print(f"n_removed: {len(bad_records)}")
    print(f"report:    {report_path}")
    print("===================================")


if __name__ == "__main__":
    main()