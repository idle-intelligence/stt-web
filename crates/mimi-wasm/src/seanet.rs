//! SEANet encoder for Mimi codec.
//!
//! Convolutional encoder that downsamples audio 320x (from 24kHz to 75Hz internal).

use crate::conv::Conv1d;
use crate::tensor::Tensor3;
use ndarray::Array1;

/// Residual block in SEANet.
#[derive(Clone, Debug)]
pub struct ResidualBlock {
    pub conv1: Conv1d,
    pub conv2: Conv1d,
    pub shortcut: Option<Conv1d>,
}

impl ResidualBlock {
    pub fn forward(&self, x: &Tensor3) -> Tensor3 {
        // SEANet residual block: ELU → Conv1 → ELU → Conv2
        let residual = x.elu(1.0);
        let residual = self.conv1.forward(&residual);
        let residual = residual.elu(1.0);
        let residual = self.conv2.forward(&residual);

        // Add shortcut connection
        match &self.shortcut {
            Some(shortcut) => {
                let shortcut_out = shortcut.forward(x);
                &residual + &shortcut_out
            }
            None => &residual + x,
        }
    }

    pub fn reset(&mut self) {
        self.conv1.reset();
        self.conv2.reset();
        if let Some(ref mut s) = self.shortcut {
            s.reset();
        }
    }
}

/// Single encoder layer (residual blocks + downsampling).
#[derive(Clone, Debug)]
pub struct EncoderLayer {
    pub residual_blocks: Vec<ResidualBlock>,
    pub downsample: Conv1d,
}

impl EncoderLayer {
    pub fn forward(&self, x: &Tensor3) -> Tensor3 {
        let mut x = x.clone();

        // Apply residual blocks
        for block in &self.residual_blocks {
            x = block.forward(&x);
        }

        // ELU before downsample
        x = x.elu(1.0);
        self.downsample.forward(&x)
    }

    pub fn reset(&mut self) {
        for block in &mut self.residual_blocks {
            block.reset();
        }
        self.downsample.reset();
    }
}

/// SEANet encoder.
///
/// Architecture:
/// - Initial conv (channels=1 → 64, kernel=7)
/// - 4 encoder layers with ratios [8, 6, 5, 4] (downsample 960x total)
/// - Final conv (kernel=3)
/// - Output: (batch, 512, time/960)
#[derive(Clone, Debug)]
pub struct SeaNetEncoder {
    pub init_conv: Conv1d,
    pub layers: Vec<EncoderLayer>,
    pub final_conv: Conv1d,
}

impl SeaNetEncoder {
    pub fn forward(&self, x: &Tensor3) -> Tensor3 {
        // Initial convolution
        let mut x = self.init_conv.forward(x);
        x = x.elu(1.0);

        // Encoder layers
        for layer in &self.layers {
            x = layer.forward(&x);
        }

        // ELU before final convolution
        x = x.elu(1.0);
        self.final_conv.forward(&x)
    }

    pub fn reset(&mut self) {
        self.init_conv.reset();
        for layer in &mut self.layers {
            layer.reset();
        }
        self.final_conv.reset();
    }
}
