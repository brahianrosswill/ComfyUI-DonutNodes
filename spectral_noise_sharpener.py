#!/usr/bin/env python3
"""
Spectral Noise Sharpener - Reference-Guided Frequency Domain Enhancement

Based on 2024 research in dual-domain feature fusion and reversible frequency decomposition.
Implements spectral noise sharpening to match reference image frequency characteristics.

Key innovations from recent papers:
- Amplitude-phase decomposition for brightness preservation
- Reversible frequency decomposition to prevent information loss
- Multi-band interactive processing for natural enhancement
- Reference-guided spectral matching
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Dict, List, Tuple, Optional
import json


class SpectralNoiseSharpener:
    """
    Advanced spectral noise sharpening using reference-guided frequency domain enhancement.
    
    Implements state-of-the-art techniques from 2024 research:
    1. Dual-domain (spatial-frequency) processing
    2. Amplitude-phase decomposition for brightness preservation
    3. Reversible frequency decomposition
    4. Multi-band interactive enhancement
    """
    
    def __init__(self, num_bands: int = 16):
        """
        Initialize the spectral noise sharpener.
        
        Args:
            num_bands: Number of frequency bands for spectral analysis
        """
        self.num_bands = num_bands
        self.band_filters = None
        self.reference_spectrum = None
        
    def amplitude_phase_decomposition(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose image into amplitude and phase components using FFT.
        
        Based on 2024 research: "Fourier transform can decouple the degradation of 
        low-light images, breaking down the complex coupling problem into two 
        relatively easier-to-solve sub-problems."
        
        Args:
            image: Input grayscale image (normalized to [0,1])
            
        Returns:
            Tuple of (amplitude, phase) components
        """
        # Compute 2D FFT
        fft_image = fft2(image)
        fft_shifted = fftshift(fft_image)
        
        # Extract amplitude and phase
        amplitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        
        return amplitude, phase
    
    def create_reversible_frequency_bands(self, image_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Create reversible frequency band filters that ensure proper frequency separation.
        
        Args:
            image_shape: Shape of the image (height, width)
            
        Returns:
            List of frequency band filters
        """
        h, w = image_shape
        center_y, center_x = h // 2, w // 2
        
        # Create coordinate matrices
        y, x = np.ogrid[:h, :w]
        
        # Calculate normalized distance from center
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = min(center_x, center_y)
        normalized_distances = distances / max_distance
        
        # Create overlapping Gaussian band filters for smooth frequency separation
        band_filters = []
        band_width = 1.0 / self.num_bands
        
        for i in range(self.num_bands):
            center_freq = (i + 0.5) * band_width  # Center of band
            
            # Create smooth Gaussian band filter
            sigma = band_width * 0.6  # Overlap for smooth transitions
            filter_mask = np.exp(-((normalized_distances - center_freq) / sigma)**2)
            
            # Special handling for first and last bands
            if i == 0:
                # Low-pass component for DC band
                filter_mask = np.maximum(filter_mask, 
                                       np.exp(-((normalized_distances) / (band_width * 0.3))**2))
            elif i == self.num_bands - 1:
                # High-pass component for highest band
                high_freq_mask = np.zeros_like(normalized_distances)
                high_freq_mask[normalized_distances > center_freq] = 1.0
                filter_mask = np.maximum(filter_mask, high_freq_mask * 0.5)
            
            band_filters.append(filter_mask)
        
        # Verify reconstruction capability
        total_filter = np.sum(band_filters, axis=0)
        reconstruction_error = np.mean(np.abs(total_filter - 1.0))
        
        if reconstruction_error > 0.1:
            # Normalize for better reconstruction if needed
            total_filter = np.maximum(total_filter, 1e-10)
            for i in range(len(band_filters)):
                band_filters[i] = band_filters[i] / total_filter
        
        self.band_filters = band_filters
        return band_filters
    
    def analyze_reference_spectrum(self, reference_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze reference image to extract target spectral characteristics.
        
        Args:
            reference_image: Reference image (grayscale, normalized to [0,1])
            
        Returns:
            Dictionary with reference spectral characteristics
        """
        # Decompose reference into amplitude and phase
        ref_amplitude, ref_phase = self.amplitude_phase_decomposition(reference_image)
        
        # Create band filters if not already created
        if self.band_filters is None:
            self.create_reversible_frequency_bands(reference_image.shape)
        
        # Extract spectral noise characteristics per band (not normalized by total energy)
        band_energies = []
        
        for band_filter in self.band_filters:
            # Calculate absolute energy in this frequency band
            band_energy = np.sum((ref_amplitude * band_filter)**2)
            # Normalize by band area to get energy density (noise level per frequency)
            band_area = np.sum(band_filter**2)
            noise_density = band_energy / (band_area + 1e-10)
            band_energies.append(noise_density)
        
        # Extract phase characteristics per band
        band_phase_stats = []
        for band_filter in self.band_filters:
            # Weight phase by amplitude in this band
            weighted_phase = ref_phase * band_filter * ref_amplitude
            phase_variance = np.var(weighted_phase)
            band_phase_stats.append(phase_variance)
        
        # Store reference spectrum
        self.reference_spectrum = {
            'band_energies': np.array(band_energies),
            'band_phase_stats': np.array(band_phase_stats),
            'total_amplitude': ref_amplitude,
            'total_phase': ref_phase
        }
        
        return self.reference_spectrum
    
    def calculate_spectral_enhancement_factors(self, input_image: np.ndarray) -> List[float]:
        """
        Calculate enhancement factors for each frequency band based on reference spectrum.
        
        Args:
            input_image: Input image to be enhanced
            
        Returns:
            List of enhancement factors for each frequency band
        """
        if self.reference_spectrum is None:
            raise ValueError("Reference spectrum not analyzed. Call analyze_reference_spectrum first.")
        
        # Analyze input image spectrum
        input_amplitude, input_phase = self.amplitude_phase_decomposition(input_image)
        
        # Extract input spectral noise characteristics (same method as reference)
        input_band_energies = []
        
        for band_filter in self.band_filters:
            # Calculate absolute energy in this frequency band
            band_energy = np.sum((input_amplitude * band_filter)**2)
            # Normalize by band area to get noise density per frequency
            band_area = np.sum(band_filter**2)
            noise_density = band_energy / (band_area + 1e-10)
            input_band_energies.append(noise_density)
        
        input_band_energies = np.array(input_band_energies)
        ref_band_energies = self.reference_spectrum['band_energies']
        
        # Calculate enhancement factors based on reference spectrum comparison
        enhancement_factors = []
        for i in range(self.num_bands):
            target_energy = ref_band_energies[i]
            current_energy = input_band_energies[i]
            
            # Calculate enhancement based on frequency band position
            freq_position = i / (self.num_bands - 1)  # 0 to 1
            
            # Always use reference spectrum for comparison
            if current_energy > 1e-10:
                # Calculate the actual ratio between target and current energy
                energy_ratio = target_energy / current_energy
                
                # Apply frequency-dependent enhancement scaling based on ratio
                if freq_position < 0.1:
                    # Very low frequencies (DC, large structures) - minimal change
                    enhancement_factor = 1.0 + (energy_ratio - 1.0) * 0.1
                elif freq_position < 0.3:
                    # Low-mid frequencies - gentle enhancement
                    enhancement_factor = 1.0 + (energy_ratio - 1.0) * 0.3
                elif freq_position < 0.7:
                    # Mid frequencies - moderate enhancement (textures, details)
                    enhancement_factor = 1.0 + (energy_ratio - 1.0) * 0.6
                else:
                    # High frequencies - full ratio enhancement (no scaling)
                    enhancement_factor = energy_ratio
            else:
                # Handle very small current energy - calculate based on reference
                if target_energy > 1e-6:  # Reference has meaningful energy in this band
                    # Use full reference energy as enhancement factor (no current energy to compare)
                    enhancement_factor = 1.0 + target_energy / 1e6  # Scale appropriately
                else:
                    # Even with minimal energy, apply minimal enhancement based on frequency position
                    enhancement_factor = 1.0 + 0.01 * (freq_position + 0.1)  # Small progressive enhancement
            
            enhancement_factors.append(enhancement_factor)
        
        # Enhancement factors calculated successfully
        
        return enhancement_factors
    
    def apply_spectral_noise_sharpening(self, image: np.ndarray, 
                                      enhancement_factors: List[float],
                                      strength: float = 1.0) -> np.ndarray:
        """
        Apply spectral noise sharpening using dual-domain processing.
        
        Based on 2024 research: Uses amplitude-phase decomposition with brightness
        preservation and reversible frequency enhancement.
        
        Args:
            image: Input image (normalized to [0,1])
            enhancement_factors: Enhancement factors for each frequency band
            strength: Overall enhancement strength (0.0 to 1.0)
            
        Returns:
            Enhanced image with matched spectral noise characteristics
        """
        # Decompose into amplitude and phase
        amplitude, phase = self.amplitude_phase_decomposition(image)
        
        # Create band filters if needed
        if self.band_filters is None:
            self.create_reversible_frequency_bands(image.shape)
        
        # Apply band-wise enhancement by amplifying existing frequencies
        enhanced_amplitude = np.zeros_like(amplitude)
        
        for i, (band_filter, enhancement_factor) in enumerate(zip(self.band_filters, enhancement_factors)):
            # Extract amplitude in this frequency band
            band_amplitude = amplitude * band_filter
            
            # Apply enhancement with strength control - amplify existing content
            actual_enhancement = 1.0 + (enhancement_factor - 1.0) * strength
            enhanced_band = band_amplitude * actual_enhancement
            
            # Add to enhanced amplitude
            enhanced_amplitude += enhanced_band
        
        # Preserve original phase completely (maintains texture structure)
        # Phase modification causes psychedelic artifacts, so we avoid it
        enhanced_phase = phase.copy()
        
        # Reconstruct enhanced image
        enhanced_fft = enhanced_amplitude * np.exp(1j * enhanced_phase)
        enhanced_fft_shifted = ifftshift(enhanced_fft)
        enhanced_image = np.real(ifft2(enhanced_fft_shifted))
        
        # Brightness preservation: match original image mean (but allow some variation)
        original_mean = np.mean(image)
        enhanced_mean = np.mean(enhanced_image)
        
        if enhanced_mean > 1e-10:
            # Only apply brightness correction if the change is significant
            brightness_ratio = original_mean / enhanced_mean
            if abs(brightness_ratio - 1.0) > 0.1:  # Only correct if >10% change
                enhanced_image = enhanced_image * brightness_ratio
        
        # Ensure valid range
        enhanced_image = np.clip(enhanced_image, 0.0, 1.0)
        
        return enhanced_image
    
    def apply_reference_guided_unsharp_mask(self, input_image: np.ndarray, 
                                          reference_image: np.ndarray,
                                          strength: float = 1.0) -> np.ndarray:
        """
        Apply reference-guided unsharp masking for visible texture enhancement.
        
        Args:
            input_image: Input image (grayscale, normalized)
            reference_image: Reference image (grayscale, normalized)
            strength: Enhancement strength
            
        Returns:
            Enhanced image with reference-guided sharpening
        """
        # Allow all strength values including 0.0 - let the processing handle it
        # Allow identical images - still process through enhancement pipeline
        
        from scipy import ndimage
        
        # Calculate reference image characteristics
        ref_variance = np.var(reference_image)
        ref_edges = ndimage.sobel(reference_image)
        ref_edge_strength = np.var(ref_edges)
        
        # Calculate target enhancement based on reference characteristics
        if ref_variance > 0.01:  # High detail reference
            base_enhancement = 0.8
        elif ref_variance > 0.005:  # Medium detail reference
            base_enhancement = 0.5
        else:  # Low detail reference
            base_enhancement = 0.3
        
        # Multi-scale unsharp masking
        enhanced_image = input_image.copy()
        
        # Scale 1: Fine details (small radius)
        blur_1 = ndimage.gaussian_filter(input_image, sigma=0.8)
        detail_1 = input_image - blur_1
        enhanced_image += detail_1 * base_enhancement * strength
        
        # Scale 2: Medium details
        blur_2 = ndimage.gaussian_filter(input_image, sigma=2.0)
        detail_2 = input_image - blur_2
        enhanced_image += detail_2 * base_enhancement * 0.7 * strength
        
        # Scale 3: Coarse details (large radius)
        blur_3 = ndimage.gaussian_filter(input_image, sigma=4.0)
        detail_3 = input_image - blur_3
        enhanced_image += detail_3 * base_enhancement * 0.4 * strength
        
        return np.clip(enhanced_image, 0.0, 1.0)
    
    def enhance_to_match_reference(self, input_image: np.ndarray,
                                 reference_image: np.ndarray,
                                 strength: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """
        Main enhancement function: match input image spectral characteristics to reference.
        
        Args:
            input_image: Image to be enhanced (BGR or RGB)
            reference_image: Reference image with target characteristics
            strength: Enhancement strength (0.0 to 1.0)
            
        Returns:
            Tuple of (enhanced_image, enhancement_info)
        """
        # Convert to grayscale for spectral analysis
        if len(input_image.shape) == 3:
            input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            input_gray = input_image.astype(np.float32) / 255.0
            
        if len(reference_image.shape) == 3:
            ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            ref_gray = reference_image.astype(np.float32) / 255.0
        
        # Resize reference to match input if needed
        if ref_gray.shape != input_gray.shape:
            ref_gray = cv2.resize(ref_gray, (input_gray.shape[1], input_gray.shape[0]))
        
        # Reset band filters to ensure they match input image size
        self.band_filters = None
        
        # Analyze reference spectrum (this will create band filters for the correct size)
        reference_spectrum = self.analyze_reference_spectrum(ref_gray)
        
        # Calculate enhancement factors
        enhancement_factors = self.calculate_spectral_enhancement_factors(input_gray)
        
        # Check if strong spectral enhancement is needed (high enhancement factors)
        max_enhancement = max(enhancement_factors)
        if max_enhancement > 2.5:
            # Strong enhancement needed - prioritize spectral processing
            spectral_enhanced = self.apply_spectral_noise_sharpening(
                input_gray, enhancement_factors, strength
            )
            # Minimal unsharp masking to avoid diluting the effect
            unsharp_enhanced = self.apply_reference_guided_unsharp_mask(
                input_gray, ref_gray, strength * 0.2
            )
            # Weighted blend favoring spectral
            enhanced_gray = input_gray + (spectral_enhanced - input_gray) * 0.8 + (unsharp_enhanced - input_gray) * 0.2
        else:
            # Mild enhancement - use hybrid approach
            spectral_enhanced = self.apply_spectral_noise_sharpening(
                input_gray, enhancement_factors, strength * 0.6
            )
            unsharp_enhanced = self.apply_reference_guided_unsharp_mask(
                input_gray, ref_gray, strength * 0.4
            )
            # Balanced blend
            enhanced_gray = input_gray + (spectral_enhanced - input_gray) + (unsharp_enhanced - input_gray)
        
        enhanced_gray = np.clip(enhanced_gray, 0.0, 1.0)
        
        # Apply enhancement to color channels if input is color
        if len(input_image.shape) == 3:
            input_float = input_image.astype(np.float32) / 255.0
            enhanced_image = np.zeros_like(input_float)
            
            # Calculate enhancement ratio for each pixel
            ratio = enhanced_gray / (input_gray + 1e-8)
            ratio = np.clip(ratio, 0.5, 2.0)
            
            # Apply ratio to each color channel
            for channel in range(3):
                enhanced_image[:, :, channel] = input_float[:, :, channel] * ratio
            
            enhanced_image = np.clip(enhanced_image * 255, 0, 255).astype(np.uint8)
        else:
            enhanced_image = np.clip(enhanced_gray * 255, 0, 255).astype(np.uint8)
        
        # Compile enhancement info
        enhancement_info = {
            'reference_spectrum': reference_spectrum,
            'enhancement_factors': enhancement_factors,
            'strength': strength,
            'num_bands': self.num_bands,
            'method': 'spectral_noise_sharpening'
        }
        
        return enhanced_image, enhancement_info


def main():
    """Example usage of the spectral noise sharpener."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Spectral noise sharpening with reference guidance')
    parser.add_argument('input_image', help='Input image path')
    parser.add_argument('reference_image', help='Reference image path')
    parser.add_argument('--output', '-o', help='Output image path')
    parser.add_argument('--strength', '-s', type=float, default=1.0, help='Enhancement strength')
    parser.add_argument('--bands', '-b', type=int, default=16, help='Number of frequency bands')
    
    args = parser.parse_args()
    
    # Load images
    input_img = cv2.imread(args.input_image)
    reference_img = cv2.imread(args.reference_image)
    
    if input_img is None:
        raise ValueError(f"Could not load input image: {args.input_image}")
    if reference_img is None:
        raise ValueError(f"Could not load reference image: {args.reference_image}")
    
    # Initialize sharpener
    sharpener = SpectralNoiseSharpener(num_bands=args.bands)
    
    # Enhance image
    enhanced_img, enhancement_info = sharpener.enhance_to_match_reference(
        input_img, reference_img, strength=args.strength
    )
    
    # Save result
    if args.output:
        cv2.imwrite(args.output, enhanced_img)
        print(f"Enhanced image saved to: {args.output}")
    
    print(f"Spectral noise sharpening completed with {args.bands} frequency bands")
    print(f"Enhancement factors range: {min(enhancement_info['enhancement_factors']):.3f} - {max(enhancement_info['enhancement_factors']):.3f}")


if __name__ == '__main__':
    main()