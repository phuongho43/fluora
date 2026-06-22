from pathlib import Path

from fluora.extract import extract_intensities
from fluora.io import load_images
from fluora.segment import segment_frames
from fluora.track import track_cells

IMGS_DIR = Path("imgs")
OUTPUT_PATH = Path("traces.csv")
DEVICE = "cpu"  # set to "cuda" if a GPU is available


def main():
    print("Loading images...")
    timestamps, images = load_images(IMGS_DIR)
    print(f"  {len(images)} frames loaded")

    print("Segmenting cells...")
    masks = segment_frames(images, device=DEVICE)

    print("Tracking cells across frames...")
    track_df = track_cells(masks)

    print("Extracting fluorescence intensities...")
    traces = extract_intensities(timestamps, images, masks, track_df)

    traces.to_csv(OUTPUT_PATH, index=False)
    n_cells = traces["cell_id"].nunique()
    print(f"Done. {n_cells} cell traces saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
