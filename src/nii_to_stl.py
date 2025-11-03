import os
import numpy as np
import nibabel as nib
from skimage import measure
import trimesh

# === CONFIGURATION ===
input_folder = r"C:\Users\Youssef\Desktop\Mpr visualization\Dataset\heart"
output_folder = r"C:\Users\Youssef\Desktop\heartstl"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# === HELPER FUNCTION ===
def nifti_to_mesh(nifti_path, level=0.5):
    """Convert a NIfTI file to a Trimesh mesh using marching cubes."""
    nii = nib.load(nifti_path)
    data = nii.get_fdata()

    # Convert to binary (thresholding)
    data = (data > level).astype(np.uint8)

    # Run marching cubes to extract surface
    verts, faces, normals, _ = measure.marching_cubes(data, level=0.5)

    # Apply affine transformation to match spatial orientation
    verts = nib.affines.apply_affine(nii.affine, verts)

    # Return as trimesh
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

# === MAIN LOGIC ===
for file in os.listdir(input_folder):
    if file.endswith(".nii") or file.endswith(".nii.gz"):
        full_path = os.path.join(input_folder, file)
        print(f"Processing {file} ...")

        mesh = nifti_to_mesh(full_path)

        # Create STL filename matching input file
        base_name = os.path.splitext(os.path.splitext(file)[0])[0]  # handle .nii.gz
        stl_path = os.path.join(output_folder, f"{base_name}.stl")

        # Export individual STL
        mesh.export(stl_path)
        print(f"✅ Saved: {stl_path}")

print("🎉 All NIfTI files converted to individual STL files!")
