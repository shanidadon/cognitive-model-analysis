"""
Hick's Law Analysis on CIFAR-10H Dataset.

This module analyzes the relationship between response uncertainty (entropy)
and reaction time for the CIFAR-10H dataset, testing Hick's law which predicts
that higher uncertainty should be associated with longer reaction times.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlretrieve
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import (
    entropy as scipy_entropy,
    mannwhitneyu,
    pearsonr,
    spearmanr,
    ttest_ind,
)


class HicksLawAnalyzer:
    """
    Analyzes Hick's law on CIFAR-10H dataset.
    
    Hick's law predicts that reaction time increases with the number of choices
    or uncertainty in the stimulus. We test this by examining whether images with
    higher label entropy (more disagreement among annotators) have longer mean
    reaction times.
    """
    
    # Official CIFAR-10H download URLs
    CIFAR10H_URLS = {
        "counts": (
            "https://github.com/jcpeterson/cifar-10h/raw/master/data/"
            "cifar10h-counts.npy"
        ),
        "probs": (
            "https://github.com/jcpeterson/cifar-10h/raw/master/data/"
            "cifar10h-probs.npy"
        ),
        "raw": (
            "https://github.com/jcpeterson/cifar-10h/raw/master/data/"
            "cifar10h-raw.zip"
        ),
    }
    
    def __init__(
        self,
        cifar10h_counts_path: str = "data/cifar10h-counts.npy",
        cifar10h_probs_path: str = "data/cifar10h-probs.npy",
        cifar10h_raw_zip_path: str = "data/cifar10h-raw.zip",
        output_dir: str = "results/hicks_law",
        auto_download: bool = True,
    ):
        """
        Initialize the Hick's Law analyzer.

        Args:
            cifar10h_counts_path: Path to CIFAR-10H label counts array.
            cifar10h_probs_path: Path to CIFAR-10H label probabilities array.
            cifar10h_raw_zip_path: Path to CIFAR-10H raw data ZIP file
                containing cifar10h-raw.csv with reaction_time field.
            output_dir: Directory to save results and plots.
            auto_download: Automatically download CIFAR-10H if not found.
        """
        self.cifar10h_counts_path = Path(cifar10h_counts_path)
        self.cifar10h_probs_path = Path(cifar10h_probs_path)
        self.cifar10h_raw_zip_path = Path(cifar10h_raw_zip_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.auto_download = auto_download

        self.counts = None
        self.probs = None
        self.cifar10h_data = None
        self.reaction_times = None
        self.entropies = None
        self.mean_rts = None
    
    def _download_cifar10h(self) -> None:
        """Download CIFAR-10H dataset from official repository."""
        print("Downloading CIFAR-10H dataset...")

        # Ensure data directory exists
        self.cifar10h_counts_path.parent.mkdir(parents=True, exist_ok=True)        

        
        # Download counts file if it doesn't exist (skips if already present).
        if not self.cifar10h_counts_path.exists():
            print(f"  Downloading counts to "f"{self.cifar10h_counts_path}...")
            urlretrieve(self.CIFAR10H_URLS["counts"],self.cifar10h_counts_path,)
            print("  ✓ Downloaded counts")
        else:
            print(f"  ✓ Counts already exist at "f"{self.cifar10h_counts_path}")

       
        # Download probs file if it doesn't exist (skips if already present).
        if not self.cifar10h_probs_path.exists():
            print(f"  Downloading probabilities to "f"{self.cifar10h_probs_path}...")
            urlretrieve(self.CIFAR10H_URLS["probs"],self.cifar10h_probs_path,)
            print("  ✓ Downloaded probabilities")
        else:
            print(f"  ✓ Probabilities already exist at "f"{self.cifar10h_probs_path}")

        
        # Download raw ZIP file if it doesn't exist (skips if already present).
        if not self.cifar10h_raw_zip_path.exists():
            print(f"  Downloading raw data to "f"{self.cifar10h_raw_zip_path}...")
            print("  (This may take a while, ~150MB)...")
            urlretrieve(self.CIFAR10H_URLS["raw"],self.cifar10h_raw_zip_path,)
            print("  ✓ Downloaded raw data ZIP")
        else:
            print(f"  ✓ Raw data already exists at "f"{self.cifar10h_raw_zip_path}")

       
        print("CIFAR-10H download complete!\n")

    
    def load_data(self) -> None:
        """Load CIFAR-10H data including reaction times."""
        print("Loading CIFAR-10H data...\n")

        # Auto-download if files don't exist
        if self.auto_download and (
            not self.cifar10h_counts_path.exists()
            or not self.cifar10h_raw_zip_path.exists()
        ):
            self._download_cifar10h()

        # Load label counts (shape: [n_images, n_classes])
        if self.cifar10h_counts_path.exists():
            self.counts = np.load(self.cifar10h_counts_path)
            print(f"Loaded counts: {self.counts.shape}")
        else:
            raise FileNotFoundError(
                f"CIFAR-10H counts file not found at {self.cifar10h_counts_path}. "
                f"Set auto_download=True to download automatically."
            )

        # Load label probabilities (optional, for validation)
        if self.cifar10h_probs_path.exists():
            self.probs = np.load(self.cifar10h_probs_path)
            print(f"Loaded probabilities: {self.probs.shape}")


        # Load CIFAR-10H raw data with reaction times
        self._load_reaction_times_from_cifar10h()
        print()
    
    def _load_reaction_times_from_cifar10h(self) -> None:
        """Load reaction times from CIFAR-10H raw data ZIP file."""
        print(
            f"Loading raw CIFAR-10H data from "
            f"{self.cifar10h_raw_zip_path}..."
        )

        # Check ZIP file exists
        if not self.cifar10h_raw_zip_path.exists():
            raise FileNotFoundError(
                f"CIFAR-10H raw data ZIP not found at "
                f"{self.cifar10h_raw_zip_path}. "
                f"Enable auto_download=True or download manually from "
                f"https://github.com/jcpeterson/cifar-10h"
            )

        # Extract and read CSV from ZIP
        with ZipFile(self.cifar10h_raw_zip_path, "r") as zip_ref:
            # List files in ZIP to find the CSV
            file_list = zip_ref.namelist()
            csv_files = [
                f for f in file_list if f.endswith(".csv")
            ]

            if not csv_files:
                raise ValueError(
                    f"No CSV files found in "
                    f"{self.cifar10h_raw_zip_path}. "
                    f"Files in ZIP: {file_list}"
                )

            csv_file = csv_files[0]
            print(f"  Extracting {csv_file} from ZIP...")
            self.cifar10h_data = pd.read_csv(
                zip_ref.open(csv_file)
            )

        print(f"  ✓ Loaded raw data: {self.cifar10h_data.shape}")
        print(
            f"  Columns: {self.cifar10h_data.columns.tolist()}"
        )

        # Check that reaction_time field exists
        if "reaction_time" not in self.cifar10h_data.columns:
            raise ValueError(
                "reaction_time field not found in CIFAR-10H raw data. "
                f"Available columns: "
                f"{self.cifar10h_data.columns.tolist()}"
            )

        # Check for image_id column (may be named differently)
        image_id_col = None
        for col in ["image_id", "cifar10_test_test_idx", "image_idx", "idx"]:
            if col in self.cifar10h_data.columns:
                image_id_col = col
                break

        if not image_id_col:
            raise ValueError(
                "image_id field not found in CIFAR-10H raw data. "
                f"Available columns: "
                f"{self.cifar10h_data.columns.tolist()}"
            )

        # Rename to standard image_id if needed
        if image_id_col != "image_id":
            print(
                f"  Renaming '{image_id_col}' to 'image_id' for "
                f"consistency..."
            )
            self.cifar10h_data = self.cifar10h_data.rename(
                columns={image_id_col: "image_id"}
            )

        # Filter out attention check trials (image_id == -99999)
        n_before = len(self.cifar10h_data)
        self.cifar10h_data = self.cifar10h_data[
            self.cifar10h_data["image_id"] >= 0
        ]
        n_attn_checks = n_before - len(self.cifar10h_data)
        if n_attn_checks > 0:
            print(f"  Filtered out {n_attn_checks} attention check trials")

        # Apply log transformation to reaction times (in milliseconds)
        # This handles the right-skewed distribution of RTs and mitigates outlier bias
        print("Reaction times will be log-transformed during storage")

        # Group by image_id to get reaction times per image
        n_images = self.counts.shape[0]
        max_annotators = (
            self.cifar10h_data.groupby("image_id").size().max()
        )

        # Create reaction times array
        self.reaction_times = np.full(
            (n_images, max_annotators), np.nan
        )

        for image_id, group in self.cifar10h_data.groupby(
            "image_id"
        ):
            rts = group["reaction_time"].values
            self.reaction_times[image_id, : len(rts)] = np.log(rts)

        print(
            f"Extracted reaction times for {n_images} images "
            f"(max {max_annotators} annotators per image)"
        )
        print(
            f"Mean log(reaction time): "
            f"{np.nanmean(self.reaction_times):.3f} "
            f"(range: [{np.nanmin(self.reaction_times):.3f}, "
            f"{np.nanmax(self.reaction_times):.3f}])"
        )
    
    def _compute_entropies(self) -> None:
        """
        Compute Shannon entropy with Miller-Madow bias correction.
        
        The plug-in (empirical) entropy estimator is downward biased with
        finite samples. Miller-Madow correction adds (K-1)/(2*N) where
        K = number of classes and N = number of votes per image.
        
        Reference:
            Miller, G. A. (1955). "Note on the bias of information estimates."
            Information and Control, 2(4), 386-390.
        """
        print("Computing label distribution entropies (Miller-Madow corrected)...")
        
        # Normalize counts to get probabilities
        probs = self.counts / self.counts.sum(axis=1, keepdims=True)
        
        # Compute plug-in entropy (downward biased) using log base 2
        plugin_entropy = np.array(
            [scipy_entropy(probs[i, :], base=2) for i in range(len(probs))]
        )
        
        # Miller-Madow bias correction: (K-1) / (2*N)
        # K = number of classes
        n_classes = self.counts.shape[1]
        # N = total votes per image
        n_votes = self.counts.sum(axis=1)
        miller_madow_correction = (n_classes - 1) / (2 * n_votes)
        
        # Bias-corrected entropy
        self.entropies = plugin_entropy + miller_madow_correction
        
        print(f"Entropy range (bias-corrected): [{self.entropies.min():.3f}, {self.entropies.max():.3f}] bits")
        print(f"  Plugin estimator range: [{plugin_entropy.min():.3f}, {plugin_entropy.max():.3f}] bits")
        print(f"  Mean bias correction: {miller_madow_correction.mean():.6f} bits")
        
        
    def compute_statistics(self) -> pd.DataFrame:
        """
        Compute mean reaction time and entropy for each image.

        Returns:
            DataFrame with columns: image_id, mean_rt, entropy
        """
        if self.entropies is None:
            self._compute_entropies()

        print("Computing mean reaction times per image...")
        # Compute mean RT per image, ignoring NaN values
        self.mean_rts = np.nanmean(self.reaction_times, axis=1)

        # Create results dataframe
        df = pd.DataFrame(
            {
                "image_id": np.arange(len(self.mean_rts)),
                "mean_rt": self.mean_rts,
                "entropy": self.entropies,
            }
        )

        print(f"Statistics computed for {len(df)} images")
        print(
            f"Mean log(RT) range: [{df['mean_rt'].min():.3f}, "
            f"{df['mean_rt'].max():.3f}]"
        )

        return df
    
    def plot_hicks_law(self, df: pd.DataFrame, save_path: Optional[Path] = None) -> Tuple[float, float]:
        """
        Plot mean reaction time as a function of entropy (log base 2 scale).
        
        Args:
            df: DataFrame with mean_rt and entropy columns
            save_path: Optional path to save the figure
        """
        print("Generating Hick's law plot...")
        
        df_filtered = df.copy()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot with regression line
        ax1 = axes[0]
        ax1.scatter(
            df_filtered["entropy"],
            df_filtered["mean_rt"],
            alpha=0.5,
            s=20,
            edgecolors="k",
            linewidths=0.5,
        )
        
        # Add regression line
        z = np.polyfit(df_filtered["entropy"], df_filtered["mean_rt"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(
            df_filtered["entropy"].min(),
            df_filtered["entropy"].max(),
            100,
        )
        ax1.plot(
            x_line,
            p(x_line),
            "r--",
            linewidth=2,
            label=f"y = {z[0]:.3f}x + {z[1]:.3f}",
        )
        
        # Compute correlation
        r_pearson, p_pearson = pearsonr(
            df_filtered["entropy"], df_filtered["mean_rt"]
        )
        r_spearman, p_spearman = spearmanr(
            df_filtered["entropy"], df_filtered["mean_rt"]
        )
        
        ax1.set_xlabel("Entropy (bits)", fontsize=12)
        ax1.set_ylabel("Mean log(Reaction Time)", fontsize=12)
        ax1.set_title(
            f"Hick's Law: log(RT) vs Entropy\n"
            f"Pearson r={r_pearson:.3f} (p={p_pearson:.2e})\n"
            f"Spearman ρ={r_spearman:.3f} (p={p_spearman:.2e})\n"
            f"n={len(df_filtered)} images",
            fontsize=12,
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Hexbin plot for density
        ax2 = axes[1]
        hexbin = ax2.hexbin(
            df_filtered["entropy"],
            df_filtered["mean_rt"],
            gridsize=30,
            cmap="YlOrRd",
            mincnt=1,
        )
        ax2.set_xlabel("Entropy (bits)", fontsize=12)
        ax2.set_ylabel("Mean log(Reaction Time)", fontsize=12)
        ax2.set_title("Density: log(RT) vs Entropy", fontsize=12)
        plt.colorbar(hexbin, ax=ax2, label="Count")
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "hicks_law_plot.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}\n")
        
        plt.close()
        
        return (r_pearson, r_spearman)
    
    def split_by_entropy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split images into two groups based on entropy with threshold.
        
        Due to Miller-Madow bias correction, we use a threshold instead of 
        exact zero. The correction adds (K-1)/(2*N) ≈ 0.09 nats to all 
        entropies, so entropy ≤ 0.09 corresponds to original entropy = 0 
        (near-unanimous votes).
        
        Args:
            df: DataFrame with Miller-Madow corrected entropy column
            
        Returns:
            Tuple of (zero_entropy_df, positive_entropy_df)
        """
        print("\nSplitting images by entropy...")
        
        # Entropy threshold: Miller-Madow correction for unanimous votes
        # K=10 classes, N≈50 votes → correction ≈ (10-1)/(2*50) = 0.09
        entropy_threshold = 0.09
        
        print(f"Using entropy threshold = {entropy_threshold} bits")
        print("(corresponds to Miller-Madow corrected entropy of unanimously voted images)")
        
        zero_entropy = df[df["entropy"] <= entropy_threshold]
        positive_entropy = df[df["entropy"] > entropy_threshold]
        
        print(f"\nEntropy Split Results:")
        print(
            f"  Images with entropy ≤ {entropy_threshold} (near-unanimous): {len(zero_entropy)} "
            f"({len(zero_entropy)/len(df)*100:.1f}%)"
        )
        print(
            f"  Images with entropy > {entropy_threshold} (disagreement): {len(positive_entropy)} "
            f"({len(positive_entropy)/len(df)*100:.1f}%)"
        )
        
        if len(zero_entropy) > 0:
            print(f"\n  Zero entropy group (near-unanimous):")
            print(
                f"    Mean log(RT): {zero_entropy['mean_rt'].mean():.3f} ± "
                f"{zero_entropy['mean_rt'].std():.3f}"
            )
            print(
                f"    Mean entropy: {zero_entropy['entropy'].mean():.3f} ± "
                f"{zero_entropy['entropy'].std():.3f} bits"
            )
        
        if len(positive_entropy) > 0:
            print(f"\n  Positive entropy group (disagreement):")
            print(
                f"    Mean log(RT): {positive_entropy['mean_rt'].mean():.3f} ± "
                f"{positive_entropy['mean_rt'].std():.3f}"
            )
            print(
                f"    Mean entropy: {positive_entropy['entropy'].mean():.3f} ± "
                f"{positive_entropy['entropy'].std():.3f} bits"
            )
        
        # Statistical test if both groups exist
        if len(zero_entropy) > 0 and len(positive_entropy) > 0:
            t_stat, t_pval = ttest_ind(zero_entropy["mean_rt"], positive_entropy["mean_rt"])
            u_stat, u_pval = mannwhitneyu(zero_entropy["mean_rt"], positive_entropy["mean_rt"])
            
            print(f"\n  Statistical Tests (Zero vs Positive Entropy):")
            print(f"    Independent t-test: t = {t_stat:.3f}, p = {t_pval:.4e}")
            print(f"    Mann-Whitney U: U = {u_stat:.1f}, p = {u_pval:.4e}")
        
        return zero_entropy, positive_entropy
    
    def plot_hicks_law_by_group(
        self,
        zero_entropy_df: pd.DataFrame,
        positive_entropy_df: pd.DataFrame,
    ) -> None:
        """
        Plot separate Hick's law graphs for zero and positive entropy groups.
        
        Args:
            zero_entropy_df: DataFrame with entropy == 0
            positive_entropy_df: DataFrame with entropy > 0
        """
        print("Generating separate Hick's law plots by entropy group...")
        
        # Filter both groups (no RT filtering needed now that data is log-transformed)
        zero_filtered = zero_entropy_df.copy()
        positive_filtered = positive_entropy_df.copy()
        
        print(
            f"  Zero entropy: {len(zero_entropy_df)} images"
        )
        print(
            f"  Positive entropy: {len(positive_entropy_df)} images"
        )
        
        # Create figure with 2 subplots (one per group)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Zero entropy group
        ax1 = axes[0]
        if len(zero_filtered) > 0:
            ax1.scatter(
                [0] * len(zero_filtered),
                zero_filtered["mean_rt"],
                alpha=0.6,
                s=50,
                c="green",
                edgecolors="darkgreen",
                linewidths=0.5,
                label="Zero Entropy Images",
            )
            ax1.set_xlabel("Entropy (bits)", fontsize=14)
            ax1.set_ylabel("mean reaction time\nlog(ms)", fontsize=14)
            ax1.set_title(
                "Zero Entropy Group",
                fontsize=16,
            )
            ax1.set_xlim(-1, 1)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Print zero entropy group info
            print(f"\n  Zero Entropy Group:")
            print(f"    n={len(zero_filtered)} images")
            print(f"    Mean log(RT): {zero_filtered['mean_rt'].mean():.3f} ± {zero_filtered['mean_rt'].std():.3f}")
        else:
            ax1.text(0.5, 0.5, "No data after filtering", ha="center", va="center")
            ax1.set_title("Zero Entropy Group", fontsize=12)
        
        # Plot 2: Positive entropy group
        ax2 = axes[1]
        if len(positive_filtered) > 0:
            ax2.scatter(
                positive_filtered["entropy"],
                positive_filtered["mean_rt"],
                alpha=0.6,
                s=50,
                c="orange",
                edgecolors="darkorange",
                linewidths=0.5,
                label="Positive Entropy Images",
            )
            
            # Add regression line
            z = np.polyfit(
                positive_filtered["entropy"],
                positive_filtered["mean_rt"],
                1,
            )
            p = np.poly1d(z)
            x_line = np.linspace(
                positive_filtered["entropy"].min(),
                positive_filtered["entropy"].max(),
                100,
            )
            ax2.plot(
                x_line,
                p(x_line),
                "r--",
                linewidth=2,
                label=f"y = {z[0]:.3f}x + {z[1]:.3f}",
            )
            
            # Compute correlation
            r_pearson, p_pearson = pearsonr(
                positive_filtered["entropy"],
                positive_filtered["mean_rt"],
            )
            r_spearman, p_spearman = spearmanr(
                positive_filtered["entropy"],
                positive_filtered["mean_rt"],
            )
            
            # Print group-specific correlations
            print(f"\n  Positive Entropy Group Correlations:")
            print(f"    Pearson r={r_pearson:.4f} (p={p_pearson:.4e})")
            print(f"    Spearman ρ={r_spearman:.4f} (p={p_spearman:.4e})")
            print(f"    n={len(positive_filtered)} images")
            
            ax2.set_xlabel("Entropy (bits)", fontsize=14)
            ax2.set_ylabel("mean reaction time\nlog(ms)", fontsize=14)
            ax2.set_title(
                "Positive Entropy Group",
                fontsize=16,
            )
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "No data after filtering", ha="center", va="center")
            ax2.set_title("Positive Entropy Group", fontsize=12)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "hicks_law_by_group.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}\n")
        
        plt.close()
    
    def plot_entropy_distribution(self, df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
        """
        Plot the distribution of entropy values.
        
        Args:
            df: DataFrame with entropy column
            save_path: Optional path to save the figure
        """
        print("Generating entropy distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        ax1 = axes[0]
        ax1.hist(df["entropy"], bins=50, edgecolor="black", alpha=0.7)
        ax1.axvline(0, color="red", linestyle="--", linewidth=2, label="Entropy = 0")
        ax1.set_xlabel("Entropy (bits)", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_title("Distribution of Label Entropy", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative distribution
        ax2 = axes[1]
        sorted_entropy = np.sort(df["entropy"])
        cumulative = np.arange(1, len(sorted_entropy) + 1) / len(sorted_entropy)
        ax2.plot(sorted_entropy, cumulative, linewidth=2)
        ax2.axvline(0, color="red", linestyle="--", linewidth=2, label="Entropy = 0")
        ax2.set_xlabel("Entropy (bits)", fontsize=12)
        ax2.set_ylabel("Cumulative Proportion", fontsize=12)
        ax2.set_title("Cumulative Distribution of Entropy", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "entropy_distribution.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        
        plt.close()
    
    def save_results(self, df: pd.DataFrame, filename: str = "hicks_law_results.csv") -> None:
        """
        Save analysis results to CSV.
        
        Args:
            df: DataFrame with results
            filename: Output filename
        """
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    
    def run_full_analysis(self) -> pd.DataFrame:
        """Run complete Hick's law analysis pipeline."""
        print("=" * 60)
        print("HICK'S LAW ANALYSIS ON CIFAR-10H")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Compute statistics
        df = self.compute_statistics()
        
        # Generate plots
        self.plot_hicks_law(df)
        self.plot_entropy_distribution(df)
        
        # Split by entropy
        zero_entropy_df, positive_entropy_df = self.split_by_entropy(df)
        
        # Generate separate plots by group
        self.plot_hicks_law_by_group(zero_entropy_df, positive_entropy_df)
        
        # Save results
        self.save_results(df)
        self.save_results(zero_entropy_df, "zero_entropy_images.csv")
        self.save_results(positive_entropy_df, "positive_entropy_images.csv")
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
        return df


def main() -> None:
    """Main entry point for running the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze Hick's law on CIFAR-10H dataset"
    )
    parser.add_argument(
        "--counts-path",
        type=str,
        default="data/cifar10h-counts.npy",
        help="Path to CIFAR-10H label counts file",
    )
    parser.add_argument(
        "--probs-path",
        type=str,
        default="data/cifar10h-probs.npy",
        help="Path to CIFAR-10H label probabilities file",
    )
    parser.add_argument(
        "--raw-zip-path",
        type=str,
        default="data/cifar10h-raw.zip",
        help="Path to CIFAR-10H raw data ZIP file with reaction_time",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hicks_law",
        help="Directory to save results",
    )
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Disable automatic download of CIFAR-10H",
    )

    args = parser.parse_args()

    # Create analyzer and run
    analyzer = HicksLawAnalyzer(
        cifar10h_counts_path=args.counts_path,
        cifar10h_probs_path=args.probs_path,
        cifar10h_raw_zip_path=args.raw_zip_path,
        output_dir=args.output_dir,
        auto_download=not args.no_auto_download,
    )

    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
