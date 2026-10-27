from pathlib import Path

import streamlit as st


RESULTS_DIR = Path(__file__).parent / "results" / "notebook_outputs"

st.set_page_config(page_title="StyleGAN CelebA Results", page_icon="🎨", layout="wide")
st.title("StyleGAN CelebA Results")
st.write("Browse generated face samples and experiment visualizations.")

if not RESULTS_DIR.is_dir():
    st.error(f"Results directory not found: {RESULTS_DIR}")
    st.stop()

images = sorted(RESULTS_DIR.glob("*.png"))
if not images:
    st.error(f"No PNG results found in {RESULTS_DIR}")
    st.stop()

samples = [path for path in images if path.name.startswith("epoch_")]
visualizations = [path for path in images if not path.name.startswith("epoch_")]
collections = {
    "All results": images,
    "Generated samples": samples,
    "Experiment visualizations": visualizations,
}

selection = st.selectbox("Collection", list(collections))
selected_images = collections[selection]
st.caption(f"{len(selected_images)} image(s)")

columns = st.columns(3)
for index, image_path in enumerate(selected_images):
    with columns[index % len(columns)]:
        st.image(
            str(image_path),
            caption=image_path.stem.replace("_", " ").title(),
            use_container_width=True,
        )
