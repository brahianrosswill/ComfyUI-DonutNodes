#!/usr/bin/env python3
"""
Image Noise Analysis Tool for AI vs Real Photo Detection

Analyzes frequency domain characteristics to detect AI-generated images
and score them based on high/low energy noise patterns.

Based on recent research (2024-2025) showing that:
- AI images exhibit different frequency distributions than real photos
- Fake images show higher energy at mid-high frequencies
- Real photos have characteristic spectral decay patterns
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from scipy import signal
from scipy.fft import fft2, fftshift, fftfreq
from pathlib import Path
import argparse
from datetime import datetime


class ImageNoiseAnalyzer:
    """
    Comprehensive image noise analysis using frequency domain techniques.
    
    This class implements state-of-the-art methods for analyzing the frequency
    characteristics of images to distinguish between AI-generated and real photos.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """
        Initialize the noise analyzer.
        
        Args:
            target_size: Target image size for analysis (width, height)
        """
        self.target_size = target_size
        self.analysis_results = {}
        
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image for frequency analysis.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed grayscale image as numpy array
        """
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size for consistent analysis
        image = cv2.resize(image, self.target_size)
        
        # Convert to grayscale for frequency analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Normalize to [0, 1] range
        gray = gray.astype(np.float32) / 255.0
        
        # Remove DC component (mean) for better FFT analysis
        gray = gray - np.mean(gray)
        
        return gray
    
    def compute_2d_fft(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 2D FFT and return magnitude and phase.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Tuple of (magnitude_spectrum, phase_spectrum)
        """
        # Apply 2D FFT
        fft_result = fft2(image)
        fft_shifted = fftshift(fft_result)
        
        # Get magnitude and phase
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        
        return magnitude, phase
    
    def compute_power_spectral_density(self, image: np.ndarray) -> np.ndarray:
        """
        Compute power spectral density using Welch's method.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Power spectral density
        """
        # Flatten image for 1D PSD analysis per row/column
        psd_rows = []
        
        # Compute PSD for each row
        for row in image:
            freqs, psd = signal.welch(row, nperseg=min(64, len(row)//4))
            psd_rows.append(psd)
        
        # Average PSD across all rows
        avg_psd = np.mean(psd_rows, axis=0)
        
        return avg_psd
    
    def compute_radial_power_spectrum(self, magnitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute radial power spectrum for rotational invariance.
        
        Args:
            magnitude: FFT magnitude spectrum
            
        Returns:
            Tuple of (radial_frequencies, radial_power)
        """
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Create coordinate matrices
        y, x = np.ogrid[:h, :w]
        
        # Calculate distance from center
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
        
        # Get maximum radius
        max_radius = min(center_x, center_y)
        
        # Compute radial average
        radial_power = []
        radial_freqs = []
        
        for radius in range(max_radius):
            mask = (r == radius)
            if np.any(mask):
                power = np.mean(magnitude[mask]**2)
                radial_power.append(power)
                radial_freqs.append(radius)
        
        return np.array(radial_freqs), np.array(radial_power)
    
    def analyze_frequency_bands(self, magnitude: np.ndarray, num_bands: int = 12) -> Dict[str, float]:
        """
        Analyze energy distribution across multiple frequency bands for granular analysis.
        
        Args:
            magnitude: FFT magnitude spectrum
            num_bands: Number of frequency bands to analyze (default: 12 for detailed analysis)
            
        Returns:
            Dictionary with energy percentages for different bands
        """
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Create distance matrix from center
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = min(center_x, center_y)
        
        # Normalize distances to [0, 1]
        normalized_distances = distances / max_distance
        
        # Calculate energy in each band
        power = magnitude**2
        total_energy = np.sum(power)
        
        band_energies = {}
        
        # Create granular frequency bands
        band_width = 1.0 / num_bands
        
        for i in range(num_bands):
            band_start = i * band_width
            band_end = (i + 1) * band_width
            
            # Create mask for this frequency band
            if i == 0:
                # First band includes DC component
                band_mask = normalized_distances <= band_end
            elif i == num_bands - 1:
                # Last band includes all remaining high frequencies
                band_mask = normalized_distances > band_start
            else:
                band_mask = (normalized_distances > band_start) & (normalized_distances <= band_end)
            
            # Calculate energy percentage
            band_energy = np.sum(power[band_mask]) / total_energy * 100
            
            # Create descriptive band name
            freq_percent = int((band_start + band_end) / 2 * 100)
            band_name = f'band_{i:02d}_{freq_percent:02d}pct'
            band_energies[band_name] = band_energy
        
        # Also compute traditional broad bands for compatibility
        traditional_bands = {
            'dc_component': np.sum(power[normalized_distances <= 0.02]) / total_energy * 100,
            'very_low_freq': np.sum(power[(normalized_distances > 0.02) & (normalized_distances <= 0.08)]) / total_energy * 100,
            'low_freq': np.sum(power[(normalized_distances > 0.08) & (normalized_distances <= 0.20)]) / total_energy * 100,
            'low_mid_freq': np.sum(power[(normalized_distances > 0.20) & (normalized_distances <= 0.35)]) / total_energy * 100,
            'mid_freq': np.sum(power[(normalized_distances > 0.35) & (normalized_distances <= 0.50)]) / total_energy * 100,
            'mid_high_freq': np.sum(power[(normalized_distances > 0.50) & (normalized_distances <= 0.65)]) / total_energy * 100,
            'high_freq': np.sum(power[(normalized_distances > 0.65) & (normalized_distances <= 0.80)]) / total_energy * 100,
            'very_high_freq': np.sum(power[(normalized_distances > 0.80) & (normalized_distances <= 0.95)]) / total_energy * 100,
            'ultra_high_freq': np.sum(power[normalized_distances > 0.95]) / total_energy * 100,
        }
        
        # Combine granular and traditional bands
        band_energies.update(traditional_bands)
        
        return band_energies
    
    def compute_detailed_spectral_metrics(self, magnitude: np.ndarray, radial_power: np.ndarray) -> Dict[str, float]:
        """
        Compute detailed spectral metrics for comprehensive noise analysis.
        
        Args:
            magnitude: FFT magnitude spectrum
            radial_power: Radial power spectrum
            
        Returns:
            Dictionary of detailed spectral metrics
        """
        freqs = np.arange(len(radial_power))
        
        # Basic spectral features
        spectral_centroid = np.sum(freqs * radial_power) / np.sum(radial_power)
        
        # Spectral moments for distribution analysis
        spectral_variance = np.sum(((freqs - spectral_centroid) ** 2) * radial_power) / np.sum(radial_power)
        spectral_skewness = np.sum(((freqs - spectral_centroid) ** 3) * radial_power) / (np.sum(radial_power) * (spectral_variance ** 1.5))
        spectral_kurtosis = np.sum(((freqs - spectral_centroid) ** 4) * radial_power) / (np.sum(radial_power) * (spectral_variance ** 2))
        
        # Multiple rolloff points for detailed analysis
        cumulative_energy = np.cumsum(radial_power)
        total_energy = cumulative_energy[-1]
        
        rolloff_points = {}
        for threshold in [0.50, 0.75, 0.85, 0.90, 0.95, 0.99]:
            rolloff_energy = threshold * total_energy
            rolloff_idx = np.where(cumulative_energy >= rolloff_energy)[0]
            rolloff_points[f'rolloff_{int(threshold*100)}'] = rolloff_idx[0] if len(rolloff_idx) > 0 else len(radial_power)
        
        # Spectral entropy (measure of spectral complexity)
        normalized_power = radial_power / (np.sum(radial_power) + 1e-10)
        spectral_entropy = -np.sum(normalized_power * np.log2(normalized_power + 1e-10))
        
        # Multiple spectral slopes for different frequency ranges
        spectral_slopes = {}
        ranges = [
            ('low', 0.1, 0.3),
            ('mid', 0.3, 0.6), 
            ('high', 0.6, 0.9)
        ]
        
        for range_name, start_pct, end_pct in ranges:
            start_idx = int(start_pct * len(radial_power))
            end_idx = int(end_pct * len(radial_power))
            
            if end_idx > start_idx:
                freq_range = freqs[start_idx:end_idx]
                power_range = radial_power[start_idx:end_idx]
                
                if len(freq_range) > 1 and np.all(power_range > 0):
                    log_freqs = np.log(freq_range + 1)
                    log_power = np.log(power_range + 1e-10)
                    slope = np.polyfit(log_freqs, log_power, 1)[0]
                    spectral_slopes[f'slope_{range_name}'] = slope
                else:
                    spectral_slopes[f'slope_{range_name}'] = 0.0
        
        # Peak analysis
        from scipy.signal import find_peaks
        peaks, peak_properties = find_peaks(radial_power, height=np.mean(radial_power))
        
        peak_metrics = {
            'num_peaks': len(peaks),
            'peak_prominence_mean': np.mean(peak_properties['peak_heights']) if len(peaks) > 0 else 0.0,
            'peak_prominence_std': np.std(peak_properties['peak_heights']) if len(peaks) > 0 else 0.0,
        }
        
        # High frequency energy ratios at different thresholds
        hf_ratios = {}
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            split_point = int(threshold * len(radial_power))
            low_energy = np.sum(radial_power[:split_point])
            high_energy = np.sum(radial_power[split_point:])
            hf_ratios[f'hf_ratio_{int(threshold*100)}'] = high_energy / (low_energy + 1e-10)
        
        # Compile all metrics
        detailed_metrics = {
            'spectral_centroid': spectral_centroid,
            'spectral_variance': spectral_variance,
            'spectral_skewness': spectral_skewness,
            'spectral_kurtosis': spectral_kurtosis,
            'spectral_entropy': spectral_entropy,
            **rolloff_points,
            **spectral_slopes,
            **peak_metrics,
            **hf_ratios,
        }
        
        return detailed_metrics
    
    def compute_spectral_features(self, magnitude: np.ndarray, radial_power: np.ndarray) -> Dict[str, float]:
        """
        Compute advanced spectral features for AI detection.
        
        Args:
            magnitude: FFT magnitude spectrum
            radial_power: Radial power spectrum
            
        Returns:
            Dictionary of spectral features
        """
        # 1. Spectral centroid (center of mass of spectrum)
        freqs = np.arange(len(radial_power))
        spectral_centroid = np.sum(freqs * radial_power) / np.sum(radial_power)
        
        # 2. Spectral rolloff (frequency below which 85% of energy is contained)
        cumulative_energy = np.cumsum(radial_power)
        total_energy = cumulative_energy[-1]
        rolloff_threshold = 0.85 * total_energy
        spectral_rolloff = np.where(cumulative_energy >= rolloff_threshold)[0][0] if len(np.where(cumulative_energy >= rolloff_threshold)[0]) > 0 else len(radial_power)
        
        # 3. Spectral flatness (Wiener entropy)
        # Measure of how flat the spectrum is (white noise = 1, pure tone = 0)
        geometric_mean = np.exp(np.mean(np.log(radial_power + 1e-10)))
        arithmetic_mean = np.mean(radial_power)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        # 4. High frequency ratio
        mid_point = len(radial_power) // 2
        high_freq_energy = np.sum(radial_power[mid_point:])
        low_freq_energy = np.sum(radial_power[:mid_point])
        high_freq_ratio = high_freq_energy / (low_freq_energy + 1e-10)
        
        # 5. Spectral slope (how quickly energy decreases with frequency)
        log_freqs = np.log(freqs[1:] + 1)  # Avoid log(0)
        log_power = np.log(radial_power[1:] + 1e-10)
        spectral_slope = np.polyfit(log_freqs, log_power, 1)[0]
        
        # 6. Spectral irregularity
        spectral_irregularity = np.sum(np.abs(np.diff(radial_power)))
        
        return {
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'spectral_flatness': spectral_flatness,
            'high_freq_ratio': high_freq_ratio,
            'spectral_slope': spectral_slope,
            'spectral_irregularity': spectral_irregularity,
        }
    
    def compute_ai_detection_score(self, band_energies: Dict[str, float], 
                                  spectral_features: Dict[str, float]) -> Dict[str, float]:
        """
        Compute AI detection score based on frequency characteristics.
        
        Based on research findings:
        - AI images have higher mid-high frequency energy
        - Real photos have more natural spectral decay
        - AI images lack certain high-frequency details
        
        Args:
            band_energies: Energy distribution across frequency bands
            spectral_features: Advanced spectral features
            
        Returns:
            Dictionary with various AI detection scores
        """
        scores = {}
        
        # 1. Mid-high frequency anomaly score
        # AI images typically have higher energy in mid-high frequencies
        expected_mid_high = 15.0  # Expected percentage for real photos
        mid_high_anomaly = max(0, band_energies['mid_high_freq'] - expected_mid_high) / expected_mid_high
        scores['mid_high_anomaly'] = min(mid_high_anomaly, 1.0)
        
        # 2. Spectral decay score
        # Real photos should have steeper spectral slope (more negative)
        expected_slope = -2.0  # Expected slope for real photos
        slope_anomaly = max(0, spectral_features['spectral_slope'] - expected_slope) / abs(expected_slope)
        scores['spectral_decay_anomaly'] = min(slope_anomaly, 1.0)
        
        # 3. High frequency deficiency score
        # AI images often lack very high frequency details
        expected_high_freq = 5.0  # Expected percentage for real photos
        high_freq_deficiency = max(0, expected_high_freq - band_energies['high_freq']) / expected_high_freq
        scores['high_freq_deficiency'] = min(high_freq_deficiency, 1.0)
        
        # 4. Spectral flatness anomaly
        # AI images often have more uniform spectral distribution
        expected_flatness = 0.3  # Expected flatness for real photos
        flatness_anomaly = max(0, spectral_features['spectral_flatness'] - expected_flatness) / expected_flatness
        scores['spectral_flatness_anomaly'] = min(flatness_anomaly, 1.0)
        
        # 5. Overall AI detection score (weighted average)
        weights = {
            'mid_high_anomaly': 0.3,
            'spectral_decay_anomaly': 0.25,
            'high_freq_deficiency': 0.25,
            'spectral_flatness_anomaly': 0.2,
        }
        
        overall_score = sum(scores[key] * weight for key, weight in weights.items())
        scores['overall_ai_score'] = overall_score
        scores['confidence'] = 'high' if overall_score > 0.7 else 'medium' if overall_score > 0.4 else 'low'
        
        return scores
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Complete analysis of an image for AI detection.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Complete analysis results
        """
        print(f"Analyzing image: {os.path.basename(image_path)}")
        
        # Load and preprocess image
        image = self.load_and_preprocess_image(image_path)
        
        # Compute FFT
        magnitude, phase = self.compute_2d_fft(image)
        
        # Compute PSD
        psd = self.compute_power_spectral_density(image)
        
        # Compute radial spectrum
        radial_freqs, radial_power = self.compute_radial_power_spectrum(magnitude)
        
        # Analyze frequency bands (with configurable granularity)
        band_energies = self.analyze_frequency_bands(magnitude, num_bands=16)  # More granular analysis
        
        # Compute detailed spectral metrics
        detailed_metrics = self.compute_detailed_spectral_metrics(magnitude, radial_power)
        
        # Compute spectral features (keep for compatibility)
        spectral_features = self.compute_spectral_features(magnitude, radial_power)
        
        # Compute AI detection scores
        ai_scores = self.compute_ai_detection_score(band_energies, spectral_features)
        
        # Compile results
        results = {
            'image_path': image_path,
            'image_size': self.target_size,
            'timestamp': datetime.now().isoformat(),
            'frequency_bands': band_energies,
            'spectral_features': spectral_features,
            'detailed_spectral_metrics': detailed_metrics,
            'ai_detection_scores': ai_scores,
            'raw_data': {
                'radial_frequencies': radial_freqs.tolist(),
                'radial_power': radial_power.tolist(),
                'psd': psd.tolist(),
            }
        }
        
        return results
    
    def create_visualization(self, results: Dict, output_path: str = None):
        """
        Create comprehensive visualization of the analysis results.
        
        Args:
            results: Analysis results from analyze_image()
            output_path: Path to save the visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Frequency Domain Analysis: {os.path.basename(results['image_path'])}", fontsize=16)
        
        # 1. Granular frequency band energy distribution
        ax1 = axes[0, 0]
        
        # Separate granular bands from traditional bands
        all_bands = results['frequency_bands']
        granular_bands = {k: v for k, v in all_bands.items() if k.startswith('band_')}
        traditional_bands = {k: v for k, v in all_bands.items() if not k.startswith('band_')}
        
        # Show granular bands
        if granular_bands:
            band_names = list(granular_bands.keys())
            band_energies = list(granular_bands.values())
            
            # Create color gradient for granular bands
            colors = plt.cm.viridis(np.linspace(0, 1, len(band_names)))
            
            bars = ax1.bar(range(len(band_names)), band_energies, color=colors, alpha=0.8)
            ax1.set_xlabel('Granular Frequency Bands (by percentage)')
            ax1.set_ylabel('Energy (%)')
            ax1.set_title(f'Granular Energy Distribution ({len(band_names)} bands)')
            
            # Simplified x-axis labels (show every 2nd or 3rd label to avoid crowding)
            step = max(1, len(band_names) // 8)
            tick_positions = range(0, len(band_names), step)
            tick_labels = [band_names[i].split('_')[-1] for i in tick_positions]  # Extract percentage
            
            ax1.set_xticks(tick_positions)
            ax1.set_xticklabels(tick_labels, rotation=45)
            
            # Add peak energy annotation
            max_energy_idx = np.argmax(band_energies)
            max_energy = band_energies[max_energy_idx]
            ax1.annotate(f'Peak: {max_energy:.1f}%', 
                        xy=(max_energy_idx, max_energy),
                        xytext=(max_energy_idx, max_energy + max(band_energies) * 0.1),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        ha='center', color='red', fontweight='bold')
        else:
            # Fallback to traditional bands if granular not available
            band_names = list(traditional_bands.keys())
            band_energies = list(traditional_bands.values())
            colors = plt.cm.viridis(np.linspace(0, 1, len(band_names)))
            
            bars = ax1.bar(range(len(band_names)), band_energies, color=colors, alpha=0.7)
            ax1.set_xlabel('Traditional Frequency Bands')
            ax1.set_ylabel('Energy (%)')
            ax1.set_title('Traditional Band Energy Distribution')
            ax1.set_xticks(range(len(band_names)))
            ax1.set_xticklabels([b.replace('_', ' ').title() for b in band_names], rotation=45)
        
        # 2. Radial power spectrum
        ax2 = axes[0, 1]
        radial_freqs = np.array(results['raw_data']['radial_frequencies'])
        radial_power = np.array(results['raw_data']['radial_power'])
        
        ax2.semilogy(radial_freqs, radial_power, 'b-', linewidth=2)
        ax2.set_xlabel('Radial Frequency')
        ax2.set_ylabel('Power (log scale)')
        ax2.set_title('Radial Power Spectrum')
        ax2.grid(True, alpha=0.3)
        
        # 3. AI detection scores
        ax3 = axes[0, 2]
        ai_scores = results['ai_detection_scores']
        score_names = [k for k in ai_scores.keys() if k not in ['confidence', 'overall_ai_score']]
        score_values = [ai_scores[k] for k in score_names]
        
        bars = ax3.barh(range(len(score_names)), score_values, color='red', alpha=0.7)
        ax3.set_xlabel('Anomaly Score (0=Real, 1=AI)')
        ax3.set_title('AI Detection Scores')
        ax3.set_yticks(range(len(score_names)))
        ax3.set_yticklabels([s.replace('_', ' ').title() for s in score_names])
        ax3.set_xlim(0, 1)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, score_values)):
            ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', ha='left', va='center')
        
        # 4. Power Spectral Density
        ax4 = axes[1, 0]
        psd = np.array(results['raw_data']['psd'])
        freqs = np.linspace(0, 0.5, len(psd))  # Normalized frequencies
        
        ax4.semilogy(freqs, psd, 'g-', linewidth=2)
        ax4.set_xlabel('Normalized Frequency')
        ax4.set_ylabel('Power Spectral Density (log scale)')
        ax4.set_title('Power Spectral Density (1D)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Traditional frequency bands (for reference)
        ax5 = axes[1, 1]
        
        if traditional_bands:
            trad_names = list(traditional_bands.keys())
            trad_energies = list(traditional_bands.values())
            
            # Color code by frequency level
            colors = ['darkblue', 'blue', 'lightblue', 'green', 'yellow', 'orange', 'red', 'darkred', 'purple']
            if len(colors) < len(trad_names):
                colors = plt.cm.plasma(np.linspace(0, 1, len(trad_names)))
            
            bars = ax5.bar(range(len(trad_names)), trad_energies, color=colors[:len(trad_names)], alpha=0.8)
            ax5.set_xlabel('Traditional Frequency Bands')
            ax5.set_ylabel('Energy (%)')
            ax5.set_title('Traditional Band Distribution')
            ax5.set_xticks(range(len(trad_names)))
            ax5.set_xticklabels([n.replace('_', ' ').replace('freq', '').title() for n in trad_names], rotation=45)
            
            # Add value labels on bars for key bands
            for i, (bar, energy) in enumerate(zip(bars, trad_energies)):
                if energy > 2.0:  # Only label significant bands
                    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                            f'{energy:.1f}%', ha='center', va='bottom', fontsize=8)
        else:
            ax5.text(0.5, 0.5, 'No traditional\nbands available', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12)
        
        # 6. Overall assessment
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        overall_score = ai_scores['overall_ai_score']
        confidence = ai_scores['confidence']
        
        # Get detailed metrics for assessment
        detailed_metrics = results.get('detailed_spectral_metrics', {})
        
        # Find peak frequency band
        granular_bands = {k: v for k, v in all_bands.items() if k.startswith('band_')}
        if granular_bands:
            peak_band = max(granular_bands.items(), key=lambda x: x[1])
            peak_freq_info = f"Peak at {peak_band[0].split('_')[-1]} ({peak_band[1]:.1f}%)"
        else:
            peak_freq_info = "No granular data"
        
        # Create assessment text
        assessment_text = f"""
COMPREHENSIVE FREQUENCY ANALYSIS

AI Detection Score: {overall_score:.3f}
Confidence: {confidence.upper()}
Classification: {'AI-Generated' if overall_score > 0.5 else 'Likely Real Photo'}

GRANULAR ANALYSIS ({len(granular_bands)} bands):
{peak_freq_info}

DETAILED METRICS:
• Spectral Centroid: {detailed_metrics.get('spectral_centroid', 0):.1f}
• Spectral Entropy: {detailed_metrics.get('spectral_entropy', 0):.2f}
• Spectral Variance: {detailed_metrics.get('spectral_variance', 0):.1f}
• Peak Count: {detailed_metrics.get('num_peaks', 0)}

HIGH FREQUENCY RATIOS:
• 50% Split: {detailed_metrics.get('hf_ratio_50', 0):.3f}
• 70% Split: {detailed_metrics.get('hf_ratio_70', 0):.3f}

ROLLOFF POINTS:
• 85% Energy: {detailed_metrics.get('rolloff_85', 0)}
• 95% Energy: {detailed_metrics.get('rolloff_95', 0)}

TRADITIONAL BANDS:
• DC Component: {traditional_bands.get('dc_component', 0):.1f}%
• Very Low: {traditional_bands.get('very_low_freq', 0):.1f}%
• Low-Mid: {traditional_bands.get('low_mid_freq', 0):.1f}%
• Mid-High: {traditional_bands.get('mid_high_freq', 0):.1f}%
• Ultra High: {traditional_bands.get('ultra_high_freq', 0):.1f}%
        """
        
        ax6.text(0.05, 0.95, assessment_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Analyze images for AI detection using frequency domain analysis')
    parser.add_argument('image_path', help='Path to the image file to analyze')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--size', default='512,512', help='Target image size (width,height)')
    parser.add_argument('--visualize', '-v', action='store_true', help='Create visualization')
    
    args = parser.parse_args()
    
    # Parse target size
    width, height = map(int, args.size.split(','))
    target_size = (width, height)
    
    # Initialize analyzer
    analyzer = ImageNoiseAnalyzer(target_size=target_size)
    
    # Analyze image
    results = analyzer.analyze_image(args.image_path)
    
    # Set up output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path('.')
    
    # Save results
    image_name = Path(args.image_path).stem
    results_file = output_dir / f'{image_name}_noise_analysis.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis results saved to: {results_file}")
    
    # Print summary
    ai_score = results['ai_detection_scores']['overall_ai_score']
    confidence = results['ai_detection_scores']['confidence']
    classification = 'AI-Generated' if ai_score > 0.5 else 'Likely Real Photo'
    
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Image: {os.path.basename(args.image_path)}")
    print(f"AI Detection Score: {ai_score:.3f}")
    print(f"Confidence: {confidence.upper()}")
    print(f"Classification: {classification}")
    
    # Create visualization if requested
    if args.visualize:
        viz_file = output_dir / f'{image_name}_noise_analysis.png'
        analyzer.create_visualization(results, str(viz_file))


if __name__ == '__main__':
    main()