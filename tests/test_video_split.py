import pandas as pd

from uav_vit.data.video_to_coco import _assign_splits


def test_assign_splits_produces_all_required_labels() -> None:
    rows = []
    for idx in range(30):
        rows.append(
            {
                "video_name": f"v_{idx // 10}.mp4",
                "frame_idx": idx,
                "weather": "clear" if idx % 2 == 0 else "fog",
                "quality": "high" if idx % 3 == 0 else "low",
                "maneuver": "straight",
            }
        )
    frame_table = pd.DataFrame(rows)
    split_df = _assign_splits(frame_table, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42)
    assert set(split_df["split"].unique()) == {"train", "val", "test"}
