#!/usr/bin/env python3
"""
DonutFrequencyAnalysis - ComfyUI Node for AI Image Detection via Frequency Analysis

This node analyzes images using frequency domain techniques to detect AI-generated content
and provides comprehensive diagnostics about the image's spectral characteristics.
"""

import torch
import numpy as np
import cv2
import json
from typing import Dict, Tuple, Any
from PIL import Image
import io
import base64

from .image_noise_analyzer import ImageNoiseAnalyzer


class DonutFrequencyAnalysis:
    """
    ComfyUI node that analyzes images for AI detection using frequency domain analysis.
    
    Takes an image input and outputs detailed frequency analysis diagnostics.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "analysis_bands": ("INT", {"default": 16, "min": 8, "max": 32, "step": 1}),
                "target_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "output_format": (["detailed", "summary", "scores_only"], {"default": "detailed"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("diagnostics_json", "summary_report", "ai_assessment", "ai_score", "frequency_profile")
    FUNCTION = "analyze_image"
    CATEGORY = "donut/analysis"

    def __init__(self):
        self.analyzer = None
    
    def tensor_to_pil(self, tensor_image):
        """
        Convert ComfyUI tensor image to PIL Image.
        
        Args:
            tensor_image: ComfyUI image tensor (B, H, W, C)
            
        Returns:
            PIL Image
        """
        # ComfyUI images are in format [batch, height, width, channels] with values in [0, 1]
        if len(tensor_image.shape) == 4:
            # Take first image from batch
            tensor_image = tensor_image[0]
        
        # Convert from tensor to numpy
        if hasattr(tensor_image, 'cpu'):
            np_image = tensor_image.cpu().numpy()
        else:
            np_image = tensor_image
        
        # Convert from [0, 1] to [0, 255] and ensure uint8
        np_image = (np_image * 255).astype(np.uint8)
        
        # Convert to PIL Image
        if np_image.shape[2] == 3:  # RGB
            pil_image = Image.fromarray(np_image, 'RGB')
        elif np_image.shape[2] == 4:  # RGBA
            pil_image = Image.fromarray(np_image, 'RGBA')
        else:
            # Convert to RGB if other format
            pil_image = Image.fromarray(np_image[:, :, :3], 'RGB')
        
        return pil_image
    
    def pil_to_cv2(self, pil_image):
        """
        Convert PIL Image to OpenCV format for analysis.
        
        Args:
            pil_image: PIL Image
            
        Returns:
            OpenCV image array
        """
        # Convert PIL to RGB if not already
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        cv2_image = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
        
        return cv2_image
    
    def create_temp_image_file(self, cv2_image, target_size):
        """
        Create a temporary image file for analysis.
        
        Args:
            cv2_image: OpenCV image array
            target_size: Target size for analysis
            
        Returns:
            Path to temporary image file
        """
        import tempfile
        import os
        
        # Resize image to target size
        resized_image = cv2.resize(cv2_image, (target_size, target_size))
        
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
        
        try:
            # Write image to temporary file
            cv2.imwrite(temp_path, resized_image)
            os.close(temp_fd)
            return temp_path
        except:
            os.close(temp_fd)
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    
    def analyze_image_direct(self, cv2_image, analysis_bands, target_size):
        """
        Analyze image directly without temporary file.
        
        Args:
            cv2_image: OpenCV image array
            analysis_bands: Number of frequency bands to analyze
            target_size: Target size for analysis
            
        Returns:
            Analysis results dictionary
        """
        # Initialize analyzer if needed
        if self.analyzer is None:
            self.analyzer = ImageNoiseAnalyzer(target_size=(target_size, target_size))
        
        # Resize image
        resized_image = cv2.resize(cv2_image, (target_size, target_size))
        
        # Convert to RGB and then grayscale for analysis
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # Normalize to [0, 1] range
        gray_image = gray_image.astype(np.float32) / 255.0
        
        # Remove DC component (mean) for better FFT analysis
        gray_image = gray_image - np.mean(gray_image)
        
        # Perform analysis using analyzer methods
        magnitude, phase = self.analyzer.compute_2d_fft(gray_image)
        psd = self.analyzer.compute_power_spectral_density(gray_image)
        radial_freqs, radial_power = self.analyzer.compute_radial_power_spectrum(magnitude)
        
        # Analyze frequency bands with custom band count
        band_energies = self.analyzer.analyze_frequency_bands(magnitude, num_bands=analysis_bands)
        
        # Compute detailed spectral metrics
        detailed_metrics = self.analyzer.compute_detailed_spectral_metrics(magnitude, radial_power)
        
        # Compute spectral features
        spectral_features = self.analyzer.compute_spectral_features(magnitude, radial_power)
        
        # Compute AI detection scores
        ai_scores = self.analyzer.compute_ai_detection_score(band_energies, spectral_features)
        
        # Compile results
        results = {
            'image_info': {
                'original_size': cv2_image.shape[:2],
                'analysis_size': (target_size, target_size),
                'analysis_bands': analysis_bands,
            },
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
    
    def format_detailed_output(self, results):
        """
        Format detailed analysis output.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Formatted detailed output string
        """
        ai_scores = results['ai_detection_scores']
        band_energies = results['frequency_bands']
        detailed_metrics = results['detailed_spectral_metrics']
        
        # Separate granular and traditional bands
        granular_bands = {k: v for k, v in band_energies.items() if k.startswith('band_')}
        traditional_bands = {k: v for k, v in band_energies.items() if not k.startswith('band_')}
        
        output = f"""╔═══════════════════════════════════════════════════════════════════════════════╗
║                          FREQUENCY DOMAIN ANALYSIS REPORT                      ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ AI DETECTION ASSESSMENT                                                         ║
║ • Overall Score: {ai_scores['overall_ai_score']:.3f} ({ai_scores['confidence'].upper()})                                    ║
║ • Classification: {'AI-Generated Image' if ai_scores['overall_ai_score'] > 0.5 else 'Likely Real Photo'}                ║
║                                                                                 ║
║ DETECTION METRICS:                                                              ║
║ • Mid-High Frequency Anomaly: {ai_scores['mid_high_anomaly']:.3f}                              ║
║ • Spectral Decay Anomaly: {ai_scores['spectral_decay_anomaly']:.3f}                                 ║
║ • High Frequency Deficiency: {ai_scores['high_freq_deficiency']:.3f}                              ║
║ • Spectral Flatness Anomaly: {ai_scores['spectral_flatness_anomaly']:.3f}                            ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ GRANULAR FREQUENCY ANALYSIS ({len(granular_bands)} bands)                                  ║"""

        # Add top 5 energy bands
        if granular_bands:
            sorted_bands = sorted(granular_bands.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (band, energy) in enumerate(sorted_bands):
                freq_pct = band.split('_')[-1]
                output += f"""
║ • Band {i+1}: {freq_pct} frequency - {energy:.2f}% energy                       ║"""

        output += f"""
╠═══════════════════════════════════════════════════════════════════════════════╣
║ TRADITIONAL FREQUENCY BANDS                                                     ║"""
        
        # Add traditional bands
        for band, energy in traditional_bands.items():
            band_name = band.replace('_', ' ').title()[:20]  # Truncate long names
            output += f"""
║ • {band_name:<20}: {energy:>6.2f}%                                     ║"""

        output += f"""
╠═══════════════════════════════════════════════════════════════════════════════╣
║ ADVANCED SPECTRAL METRICS                                                       ║
║ • Spectral Centroid: {detailed_metrics.get('spectral_centroid', 0):>8.1f}                                    ║
║ • Spectral Entropy: {detailed_metrics.get('spectral_entropy', 0):>9.3f}                                     ║
║ • Spectral Variance: {detailed_metrics.get('spectral_variance', 0):>8.1f}                                    ║
║ • Spectral Skewness: {detailed_metrics.get('spectral_skewness', 0):>8.3f}                                    ║
║ • Spectral Kurtosis: {detailed_metrics.get('spectral_kurtosis', 0):>8.3f}                                    ║
║ • Peak Count: {detailed_metrics.get('num_peaks', 0):>12}                                           ║
║                                                                                 ║
║ HIGH FREQUENCY ENERGY RATIOS:                                                   ║
║ • 50% Split Ratio: {detailed_metrics.get('hf_ratio_50', 0):>8.3f}                                      ║
║ • 70% Split Ratio: {detailed_metrics.get('hf_ratio_70', 0):>8.3f}                                      ║
║ • 80% Split Ratio: {detailed_metrics.get('hf_ratio_80', 0):>8.3f}                                      ║
║                                                                                 ║
║ SPECTRAL ROLLOFF POINTS:                                                        ║
║ • 85% Energy Rolloff: {detailed_metrics.get('rolloff_85', 0):>6}                                      ║
║ • 95% Energy Rolloff: {detailed_metrics.get('rolloff_95', 0):>6}                                      ║
║ • 99% Energy Rolloff: {detailed_metrics.get('rolloff_99', 0):>6}                                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝"""

        return output
    
    def format_summary_output(self, results):
        """
        Format summary analysis output.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Formatted summary output string
        """
        ai_scores = results['ai_detection_scores']
        band_energies = results['frequency_bands']
        detailed_metrics = results['detailed_spectral_metrics']
        
        # Get key metrics
        overall_score = ai_scores['overall_ai_score']
        confidence = ai_scores['confidence']
        classification = 'AI-Generated' if overall_score > 0.5 else 'Real Photo'
        
        # Find peak frequency band
        granular_bands = {k: v for k, v in band_energies.items() if k.startswith('band_')}
        if granular_bands:
            peak_band = max(granular_bands.items(), key=lambda x: x[1])
            peak_info = f"{peak_band[0].split('_')[-1]} ({peak_band[1]:.1f}%)"
        else:
            peak_info = "N/A"

        summary = f"""FREQUENCY ANALYSIS SUMMARY
═══════════════════════════

Classification: {classification}
AI Score: {overall_score:.3f} ({confidence.upper()} confidence)

Peak Energy: {peak_info}
Spectral Centroid: {detailed_metrics.get('spectral_centroid', 0):.1f}
High Freq Ratio (70%): {detailed_metrics.get('hf_ratio_70', 0):.3f}

Key Anomalies:
• Mid-High Freq: {ai_scores['mid_high_anomaly']:.3f}
• Spectral Decay: {ai_scores['spectral_decay_anomaly']:.3f}
• High Freq Deficiency: {ai_scores['high_freq_deficiency']:.3f}

Traditional Bands (Top 3):
• DC Component: {band_energies.get('dc_component', 0):.1f}%
• Low Frequency: {band_energies.get('low_freq', 0):.1f}%
• Mid Frequency: {band_energies.get('mid_freq', 0):.1f}%"""

        return summary
    
    def format_ai_assessment(self, results):
        """
        Format AI assessment output.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            AI assessment string
        """
        ai_scores = results['ai_detection_scores']
        overall_score = ai_scores['overall_ai_score']
        confidence = ai_scores['confidence']
        
        if overall_score >= 0.8:
            assessment = f"HIGHLY LIKELY AI-GENERATED ({overall_score:.3f})"
        elif overall_score >= 0.6:
            assessment = f"LIKELY AI-GENERATED ({overall_score:.3f})"
        elif overall_score >= 0.4:
            assessment = f"UNCERTAIN - MIXED CHARACTERISTICS ({overall_score:.3f})"
        elif overall_score >= 0.2:
            assessment = f"LIKELY REAL PHOTO ({overall_score:.3f})"
        else:
            assessment = f"HIGHLY LIKELY REAL PHOTO ({overall_score:.3f})"
        
        return f"{assessment} - {confidence.upper()} CONFIDENCE"
    
    def format_frequency_profile(self, results):
        """
        Format frequency profile for parameter optimization.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Frequency profile string
        """
        band_energies = results['frequency_bands']
        detailed_metrics = results['detailed_spectral_metrics']
        
        # Get granular bands for profile
        granular_bands = {k: v for k, v in band_energies.items() if k.startswith('band_')}
        
        profile = "FREQUENCY PROFILE FOR OPTIMIZATION:\n"
        profile += "═" * 40 + "\n"
        
        # Energy distribution profile
        if granular_bands:
            sorted_bands = sorted(granular_bands.items(), key=lambda x: int(x[0].split('_')[1]))
            profile += "Energy Distribution:\n"
            for band, energy in sorted_bands:
                freq_pct = band.split('_')[-1]
                bar_length = int(energy * 2)  # Scale for visual bar
                bar = "█" * min(bar_length, 20)
                profile += f"{freq_pct:>6}: {energy:>5.1f}% {bar}\n"
        
        profile += f"\nSpectral Characteristics:\n"
        profile += f"Centroid: {detailed_metrics.get('spectral_centroid', 0):.1f}\n"
        profile += f"Entropy: {detailed_metrics.get('spectral_entropy', 0):.2f}\n"
        profile += f"HF Ratio: {detailed_metrics.get('hf_ratio_70', 0):.3f}\n"
        profile += f"Rolloff 95%: {detailed_metrics.get('rolloff_95', 0)}\n"
        
        return profile

    def analyze_image(self, image, analysis_bands=16, target_size=512, output_format="detailed"):
        """
        Main analysis function for ComfyUI node.
        
        Args:
            image: Input image tensor from ComfyUI
            analysis_bands: Number of frequency bands to analyze
            target_size: Target size for analysis
            output_format: Output format (detailed, summary, scores_only)
            
        Returns:
            Tuple of (diagnostics_json, summary_report, ai_assessment, ai_score, frequency_profile)
        """
        try:
            # Convert ComfyUI tensor to PIL Image
            pil_image = self.tensor_to_pil(image)
            
            # Convert PIL to OpenCV format
            cv2_image = self.pil_to_cv2(pil_image)
            
            # Perform analysis
            results = self.analyze_image_direct(cv2_image, analysis_bands, target_size)
            
            # Format outputs based on requested format
            if output_format == "detailed":
                summary_report = self.format_detailed_output(results)
            elif output_format == "summary":
                summary_report = self.format_summary_output(results)
            else:  # scores_only
                ai_scores = results['ai_detection_scores']
                summary_report = f"AI Score: {ai_scores['overall_ai_score']:.3f} ({ai_scores['confidence']})"
            
            # Create outputs
            diagnostics_json = json.dumps(results, indent=2)
            ai_assessment = self.format_ai_assessment(results)
            ai_score = float(results['ai_detection_scores']['overall_ai_score'])
            frequency_profile = self.format_frequency_profile(results)
            
            return (diagnostics_json, summary_report, ai_assessment, ai_score, frequency_profile)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            return (
                f'{{"error": "{error_msg}"}}',
                error_msg,
                "ANALYSIS FAILED",
                0.0,
                "Error occurred during analysis"
            )


NODE_CLASS_MAPPINGS = {
    "Donut Frequency Analysis": DonutFrequencyAnalysis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Donut Frequency Analysis": "Donut Frequency Analysis",
}