//! SEANet encoder for Mimi codec.
//!
//! Convolutional encoder that downsamples audio 320x (from 24kHz to 75Hz internal).

use crate::conv::Conv1d;
use crate::tensor::Tensor3;

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
        let mut residual = x.elu(1.0);
        residual = self.conv1.forward(&residual);
        residual.elu_inplace(1.0);
        let mut residual = self.conv2.forward(&residual);

        // Add shortcut connection (in-place to avoid allocation)
        match &self.shortcut {
            Some(shortcut) => {
                let shortcut_out = shortcut.forward(x);
                residual.add_assign(&shortcut_out);
                residual
            }
            None => {
                residual.add_assign(x);
                residual
            }
        }
    }

    /// Streaming step: ELU → conv1.step → ELU → conv2.step + shortcut.
    pub fn step(&mut self, x: &Tensor3) -> Option<Tensor3> {
        let residual = x.elu(1.0);
        let residual = self.conv1.step(&residual)?;
        let mut residual_elu = residual;
        residual_elu.elu_inplace(1.0);
        let mut residual = self.conv2.step(&residual_elu)?;

        match &mut self.shortcut {
            Some(shortcut) => {
                let sc = shortcut.step(x)?;
                residual.add_assign(&sc);
                Some(residual)
            }
            None => {
                // For no-shortcut case, the residual path through conv1+conv2
                // (both stride=1 causal) will buffer on the first call(s) and
                // then produce output time == input time in steady state.
                // We need to align x with the output by taking the last N samples.
                let (_, _, res_time) = residual.shape();
                let (_, _, x_time) = x.shape();
                if res_time == x_time {
                    residual.add_assign(x);
                } else if res_time < x_time {
                    let trimmed = x.slice_time(x_time - res_time, res_time);
                    residual.add_assign(&trimmed);
                }
                Some(residual)
            }
        }
    }

    pub fn init_streaming(&mut self) {
        self.conv1.init_streaming();
        self.conv2.init_streaming();
        if let Some(ref mut s) = self.shortcut {
            s.init_streaming();
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
        // Apply residual blocks (each returns new tensor, no need to clone input)
        let mut x = if self.residual_blocks.is_empty() {
            x.clone()
        } else {
            let mut result = self.residual_blocks[0].forward(x);
            for block in &self.residual_blocks[1..] {
                result = block.forward(&result);
            }
            result
        };

        // ELU before downsample
        x.elu_inplace(1.0);
        self.downsample.forward(&x)
    }

    /// Streaming step: residual blocks → ELU → downsample.step().
    pub fn step(&mut self, x: &Tensor3) -> Option<Tensor3> {
        let mut x_owned = if self.residual_blocks.is_empty() {
            x.clone()
        } else {
            let mut result = self.residual_blocks[0].step(x)?;
            for block in &mut self.residual_blocks[1..] {
                result = block.step(&result)?;
            }
            result
        };
        x_owned.elu_inplace(1.0);
        self.downsample.step(&x_owned)
    }

    pub fn init_streaming(&mut self) {
        for block in &mut self.residual_blocks {
            block.init_streaming();
        }
        self.downsample.init_streaming();
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
        x.elu_inplace(1.0);

        // Encoder layers
        for layer in &self.layers {
            x = layer.forward(&x);
        }

        // ELU before final convolution
        x.elu_inplace(1.0);
        self.final_conv.forward(&x)
    }

    /// Streaming step: init_conv.step → ELU → layers.step → ELU → final_conv.step.
    pub fn step(&mut self, x: &Tensor3) -> Option<Tensor3> {
        let mut x = self.init_conv.step(x)?;
        x.elu_inplace(1.0);
        for layer in &mut self.layers {
            x = layer.step(&x)?;
        }
        x.elu_inplace(1.0);
        self.final_conv.step(&x)
    }

    /// Initialize all convolution buffers for streaming with batch_size=1.
    pub fn init_streaming(&mut self) {
        self.init_conv.init_streaming();
        for layer in &mut self.layers {
            layer.init_streaming();
        }
        self.final_conv.init_streaming();
    }

    pub fn reset(&mut self) {
        self.init_conv.reset();
        for layer in &mut self.layers {
            layer.reset();
        }
        self.final_conv.reset();
    }
}
