//! Weight loading from safetensors format.

use crate::MimiError;
use half::f16;
use ndarray::{Array1, Array2, Array3};
use safetensors::{Dtype, SafeTensors};

/// Parse a safetensors file and extract tensors.
pub fn load_safetensors(data: &[u8]) -> Result<SafeTensors<'_>, MimiError> {
    SafeTensors::deserialize(data).map_err(|e| MimiError::WeightLoad(e.to_string()))
}

/// Helper to get a tensor view from safetensors.
pub fn get_tensor<'a>(
    tensors: &'a SafeTensors,
    name: &str,
) -> Result<safetensors::tensor::TensorView<'a>, MimiError> {
    tensors
        .tensor(name)
        .map_err(|e| MimiError::WeightLoad(format!("Tensor '{}' not found: {}", name, e)))
}

/// Decode raw bytes from a tensor view into f32 values, supporting f32 and f16 dtypes.
fn decode_floats(view: &safetensors::tensor::TensorView) -> Result<Vec<f32>, MimiError> {
    match view.dtype() {
        Dtype::F32 => Ok(view
            .data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()),
        Dtype::F16 => Ok(view
            .data()
            .chunks_exact(2)
            .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
            .collect()),
        other => Err(MimiError::WeightLoad(format!(
            "Unsupported tensor dtype: {:?}",
            other
        ))),
    }
}

/// Convert a tensor view to Array1 (1D vector).
pub fn to_array1(view: safetensors::tensor::TensorView) -> Result<Array1<f32>, MimiError> {
    let shape = view.shape();
    if shape.len() != 1 {
        return Err(MimiError::WeightLoad(format!(
            "Expected 1D tensor, got shape {:?}",
            shape
        )));
    }

    let data = decode_floats(&view)?;
    Ok(Array1::from_vec(data))
}

/// Convert a tensor view to Array2 (2D matrix).
pub fn to_array2(view: safetensors::tensor::TensorView) -> Result<Array2<f32>, MimiError> {
    let shape = view.shape();
    if shape.len() != 2 {
        return Err(MimiError::WeightLoad(format!(
            "Expected 2D tensor, got shape {:?}",
            shape
        )));
    }

    let data = decode_floats(&view)?;
    Array2::from_shape_vec((shape[0], shape[1]), data)
        .map_err(|e| MimiError::WeightLoad(e.to_string()))
}

/// Convert a tensor view to Array3 (3D tensor).
pub fn to_array3(view: safetensors::tensor::TensorView) -> Result<Array3<f32>, MimiError> {
    let shape = view.shape();
    if shape.len() != 3 {
        return Err(MimiError::WeightLoad(format!(
            "Expected 3D tensor, got shape {:?}",
            shape
        )));
    }

    let data = decode_floats(&view)?;
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), data)
        .map_err(|e| MimiError::WeightLoad(e.to_string()))
}

/// Get optional bias tensor (returns None if not found).
pub fn get_optional_bias(
    tensors: &SafeTensors,
    name: &str,
) -> Result<Option<Array1<f32>>, MimiError> {
    match tensors.tensor(name) {
        Ok(view) => Ok(Some(to_array1(view)?)),
        Err(_) => Ok(None),
    }
}
