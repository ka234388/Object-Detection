#!/usr/bin/env python3
import json, os, sys, argparse, logging
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

# I USED THESE DIRECTORY LCOATION HENCE IM TRYIGN TO DISPLAY ONLY IMAGES SO NOT REQUIRD TO ADD IT
DEFAULT_EVALUATION_DIR = "/kaggle/working/evaluation"
DEFAULT_LOGS_DIR       = "/kaggle/working/logs"

class ModelEvaluator:
    def __init__(self, results_file, evaluation_dir=DEFAULT_EVALUATION_DIR, logs_dir=DEFAULT_LOGS_DIR, log_level=logging.INFO):
        os.makedirs(evaluation_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        self.results_file  = results_file
        self.evaluation_dir = evaluation_dir
        self.logs_dir       = logs_dir
        self.logger         = self._setup_logging(log_level)

        if not os.path.exists(results_file):
            self.logger.error(f"Results file not found: {results_file}")
            raise FileNotFoundError(results_file)

        self.data            = self._load_results()
        self.results         = self.data.get('results', [])
        self.metadata        = self.data.get('metadata', {})
        self.summary_metrics = self.data.get('summary_metrics', {})
        self.logger.info(f"Loaded {len(self.results)} results")

    # ---- logging ----
    def _setup_logging(self, log_level):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filepath = os.path.join(self.logs_dir, f"evaluation_{ts}.log")

        logger = logging.getLogger(f"evaluation_{ts}")
        logger.setLevel(log_level)
        for h in list(logger.handlers):
            logger.removeHandler(h)

        fh = logging.FileHandler(log_filepath);  fh.setLevel(log_level)
        ch = logging.StreamHandler(sys.stdout);  ch.setLevel(logging.INFO)

        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
        fh.setFormatter(fmt); ch.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(ch)
        logger.info(f"Log file: {log_filepath}")
        return logger

    def _load_results(self):
        with open(self.results_file, 'r') as f:
            return json.load(f)


    def print_summary(self):
        m = self.metadata; s = self.summary_metrics
        self.logger.info("="*60)
        self.logger.info("MODEL EVALUATION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Model: {m.get('model_name', 'Unknown')}")
        self.logger.info(f"Timestamp: {m.get('timestamp', 'Unknown')}")
        self.logger.info(f"Device: {m.get('device', 'Unknown')}")
        self.logger.info(f"Confidence Threshold: {m.get('confidence_threshold', 'Unknown')}")
        self.logger.info(f"Total Images: {m.get('total_images', 0)}")
        self.logger.info(f"Total Detections: {m.get('total_detections', 0)}")
        self.logger.info("")
        self.logger.info("PERFORMANCE METRICS:")
        self.logger.info(f"  Average Time / Image: {s.get('average_time_per_image', 0):.4f}s")
        self.logger.info(f"  FPS: {s.get('fps', 0):.2f}")
        self.logger.info(f"  Avg Detections / Image: {s.get('average_detections_per_image', 0):.2f}")
        self.logger.info(f"  Mean Confidence: {s.get('mean_confidence', 0):.3f}")
        self.logger.info("="*60)

    def analyze_class_distribution(self):
        class_counts = Counter()
        class_conf   = defaultdict(list)
        for r in self.results:
            for d in r.get('detections', []):
                class_counts[d.get('class_name','unknown')] += 1
                class_conf[d.get('class_name','unknown')].append(float(d.get('score', 0.0)))

        rows = []
        for c, n in class_counts.items():
            confs = class_conf[c]
            rows.append({
                "class_name": c, "count": n,
                "mean_confidence": np.mean(confs) if confs else 0.0,
                "std_confidence":  np.std(confs)  if confs else 0.0,
                "min_confidence":  np.min(confs)  if confs else 0.0,
                "max_confidence":  np.max(confs)  if confs else 0.0,
            })
        df = pd.DataFrame(rows).sort_values("count", ascending=False)
        if not df.empty:
            self.logger.info("\nCLASS DISTRIBUTION:\n" + df.head(20).to_string(index=False))
        else:
            self.logger.warning("No class stats available.")
        return df, class_counts, class_conf

    def analyze_inference_times(self):
        times = [float(r['inference_time']) for r in self.results if 'error' not in r and 'inference_time' in r]
        if not times:
            self.logger.warning("No valid inference times.")
            return None, None
        stats = {
            'mean': np.mean(times), 'std': np.std(times),
            'min': np.min(times), 'max': np.max(times),
            'median': np.median(times),
            'p25': np.percentile(times, 25), 'p75': np.percentile(times, 75),
            'p95': np.percentile(times, 95), 'p99': np.percentile(times, 99),
        }
        self.logger.info(f"Latency mean: {stats['mean']:.4f}s | median: {stats['median']:.4f}s")
        return times, stats

    def analyze_confidence_distribution(self):
        scores = [float(d['score']) for r in self.results for d in r.get('detections', []) if 'score' in d]
        if not scores:
            self.logger.warning("No confidence scores.")
            return None, None
        stats = {
            'mean': np.mean(scores), 'std': np.std(scores), 'min': np.min(scores),
            'max': np.max(scores), 'median': np.median(scores), 'count': len(scores)
        }
        self.logger.info(f"Conf mean: {stats['mean']:.3f} | count: {stats['count']}")
        return scores, stats

    # ---- visualizatioN
    def sample_image(self, index=0, score_thr=0.5):
        if not self.results or index < 0 or index >= len(self.results):
            self.logger.warning(f"sample_image index {index} out of range"); return
        sample = self.results[index]
        img_path = sample.get("image_path")
        if not img_path or not os.path.exists(img_path):
            self.logger.warning(f"Image not found on disk: {img_path}"); return

        try:
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            for det in sample.get("detections", []):
                sc = float(det.get("score", 0.0))
                if sc < score_thr: continue
                x1,y1,x2,y2 = [int(v) for v in det.get("bbox", [0,0,0,0])]
                draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=3)
                draw.text((x1+2,y1+2), f"{det.get('class_name','?')} {sc:.2f}", fill=(255,255,255))

            out_name = f"{self.metadata.get('model_name','model')}_sample_{index}.jpg"
            out_path = os.path.join(self.evaluation_dir, out_name)
            img.save(out_path)
            self.logger.info(f"Saved sample visualization: {out_path}")
        except Exception as e:
            self.logger.error(f"Failed to render sample {index}: {e}")

    def create_visualizations(self):
        plt.style.use('default'); sns.set_palette("husl")
        df, class_counts, class_conf = self.analyze_class_distribution()
        times, _ = self.analyze_inference_times()
        scores, _ = self.analyze_confidence_distribution()

        fig = plt.figure(figsize=(20,16))

        # 1) Top-10 classes
        ax1 = plt.subplot(2,3,1)
        if df is not None and not df.empty:
            top = df.head(10)
            ax1.bar(range(len(top)), top['count'])
            ax1.set_xticks(range(len(top))); ax1.set_xticklabels(top['class_name'], rotation=45)
            ax1.set_title("Top 10 Detected Classes"); ax1.set_ylabel("Detections")
        else:
            ax1.set_title("No class data")

        # 2) Confidence histogram
        ax2 = plt.subplot(2,3,2)
        if scores:
            ax2.hist(scores, bins=30, alpha=0.7, edgecolor='black')
            ax2.axvline(np.mean(scores), color='red', linestyle='--', label=f"Mean: {np.mean(scores):.3f}")
            ax2.legend(); ax2.set_title("Confidence Score Distribution")
        else:
            ax2.set_title("No confidence data")

        # 3) Inference time histogram
        ax3 = plt.subplot(2,3,3)
        if times:
            ax3.hist(times, bins=30, alpha=0.7, edgecolor='black')
            ax3.axvline(np.mean(times), color='red', linestyle='--', label=f"Mean: {np.mean(times):.4f}s")
            ax3.legend(); ax3.set_title("Inference Time Distribution")
        else:
            ax3.set_title("No timing data")

        # 4) Detections per image
        ax4 = plt.subplot(2,3,4)
        det_counts = [len(r.get('detections', [])) for r in self.results]
        if det_counts:
            ax4.hist(det_counts, bins=20, alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(det_counts), color='red', linestyle='--', label=f"Mean: {np.mean(det_counts):.1f}")
            ax4.legend(); ax4.set_title("Detections per Image")
        else:
            ax4.set_title("No detections")

        # 5) Confidence by class (top 8)
        ax5 = plt.subplot(2,3,5)
        if df is not None and not df.empty:
            top_names = df['class_name'].tolist()[:8]
            conf_data = [class_conf[c] for c in top_names if c in class_conf and class_conf[c]]
            if conf_data:
                ax5.boxplot(conf_data, labels=top_names)
                ax5.set_title("Confidence by Class (Top 8)")
            else:
                ax5.set_title("No per-class confidence data")
        else:
            ax5.set_title("No class data")

        # 6) Performance summary bars
        ax6 = plt.subplot(2,3,6)
        s = self.summary_metrics
        metrics = ['FPS', 'Avg Det/Img', 'Mean Conf', 'Avg Time (ms)']
        values  = [s.get('fps',0), s.get('average_detections_per_image',0),
                   s.get('mean_confidence',0), s.get('average_time_per_image',0)*1000]
        bars = ax6.bar(metrics, values, color=['skyblue','lightgreen','lightcoral','lightyellow'])
        for b,v in zip(bars, values):
            ax6.text(b.get_x()+b.get_width()/2., b.get_height(), f"{v:.2f}", ha='center', va='bottom')
        ax6.set_title("Performance Summary")

        plt.tight_layout()
        out = os.path.join(self.evaluation_dir, f"{self.metadata.get('model_name','model')}_evaluation_{self.metadata.get('timestamp','unknown')}.png")
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Visualization saved to: {out}")

    def generate_detailed_report(self):
        model = self.metadata.get('model_name','model')
        ts    = self.metadata.get('timestamp','unknown')
        out   = os.path.join(self.evaluation_dir, f"{model}_detailed_report_{ts}.txt")
        df, _, _ = self.analyze_class_distribution()
        scores,_ = self.analyze_confidence_distribution()
        times,_  = self.analyze_inference_times()
        with open(out, 'w') as f:
            f.write("="*80 + "\nDETAILED MODEL EVALUATION REPORT\n" + "="*80 + "\n\n")
            f.write("METADATA:\n")
            for k,v in self.metadata.items(): f.write(f"  {k}: {v}\n")
            f.write("\nSUMMARY METRICS:\n")
            for k,v in self.summary_metrics.items(): f.write(f"  {k}: {v}\n")
            f.write("\nCLASS DISTRIBUTION (top 50):\n")
            if df is not None and not df.empty: f.write(df.head(50).to_string(index=False) + "\n")
            f.write("\nNotes:\n")
            f.write(" - Confidence stats and latency histograms computed from JSON.\n")
        self.logger.info(f"Detailed report saved to: {out}")

# ---- runner ----
def run_evaluation(json_file_path, evaluation_dir=None, logs_dir=None, make_plots=True, make_report=True, num_samples=0, score_thr=0.5, log_level=logging.INFO):
    print("="*60)
    print("STARTING MODEL EVALUATION")
    print("="*60)
    print(f"JSON File: {json_file_path}")

    evaluation_dir = evaluation_dir or DEFAULT_EVALUATION_DIR
    logs_dir       = logs_dir or DEFAULT_LOGS_DIR
    print(f"Evaluation Dir: {evaluation_dir}")
    print(f"Logs Dir: {logs_dir}")
    print("="*60)

    evaluator = ModelEvaluator(json_file_path, evaluation_dir, logs_dir, log_level)
    evaluator.print_summary()
    evaluator.analyze_class_distribution()
    evaluator.analyze_inference_times()
    evaluator.analyze_confidence_distribution()

    if make_plots:
        evaluator.create_visualizations()

    if num_samples > 0:
        for i in range(min(num_samples, len(evaluator.results))):
            evaluator.sample_image(index=i, score_thr=score_thr)

    if make_report:
        evaluator.generate_detailed_report()

    print("="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    return True

def main():
    p = argparse.ArgumentParser(description="Model Evaluation Script")
    p.add_argument("json_file", help="Path to a JSON results file")
    p.add_argument("--eval_dir",  default=None, help="Output directory for plots/samples/reports")
    # p.add_argument("--logs_dir",  default=None, help="Directory for log files")
    # p.add_argument("--no_plots",  action="store_true", help="Disable metric plots")
    # p.add_argument("--no_report", action="store_true", help="Disable text report")
    # p.add_argument("--samples",   type=int, default=0, help="How many sample images to draw")
    # p.add_argument("--score_thr", type=float, default=0.5, help="Score threshold for sample overlays")
    # p.add_argument("--quiet",     action="store_true", help="Reduce logging")
    args = p.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    run_evaluation(
        json_file_path=args.json_file,
        evaluation_dir=args.eval_dir,
        logs_dir=args.logs_dir,
        make_plots=not args.no_plots,
        make_report=not args.no_report,
        num_samples=args.samples,
        score_thr=args.score_thr,
        log_level=log_level
    )

if __name__ == "__main__":
    main()
